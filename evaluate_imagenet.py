# evaluate_imagenet.py

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.models import ResNet50_Weights
from transformers import CLIPTokenizer

from configs import load_config
from data.transforms import get_val_transform
from models import CLIPModel

logging.basicConfig(
    level=logging.INFO,
    format="[PRTS %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(cfg, checkpoint_path: str, device: torch.device) -> CLIPModel:
    model = CLIPModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=False,
        embed_dim=cfg.embed_dim,
        text_encoder_name=getattr(
            cfg, "text_encoder_name", "openai/clip-vit-base-patch32"
        ),
        text_max_length=cfg.text_max_length,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"已加载权重: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def normalize_label(label) -> str:
    if isinstance(label, (tuple, list)):
        label = label[0]
    label = str(label).replace("_", " ")
    return label.split(",")[0].strip()


def load_class_index_json(path: str) -> dict[str, str]:
    """Load common imagenet_class_index.json: idx -> [wnid, readable label]."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = {}
    for _, value in data.items():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            mapping[str(value[0])] = normalize_label(value[1])
    return mapping


def infer_imagenet_root(root: Path) -> tuple[Path, str]:
    """Return dataset root and backend for common layouts."""
    if (root / "meta.bin").exists():
        return root, "torchvision"
    if (root / "val").is_dir():
        val_children = [path for path in (root / "val").iterdir() if path.is_dir()]
        if len(val_children) >= 1000 and all(path.name.startswith("n") for path in val_children[:10]):
            return root / "val", "imagefolder"
        return root, "torchvision"
    return root, "imagefolder"


def build_dataset(root: str, image_size: int, backend: str, class_index_json: str):
    root_path = Path(root)
    if backend == "auto":
        root_path, backend = infer_imagenet_root(root_path)

    transform = get_val_transform(image_size)
    if backend == "torchvision":
        dataset = ImageNet(str(root_path), split="val", transform=transform)
        labels = [normalize_label(label) for label in dataset.classes]
        logger.info("数据后端: torchvision.datasets.ImageNet")
        return dataset, labels

    dataset = ImageFolder(str(root_path), transform=transform)
    wnid_to_label = load_class_index_json(class_index_json)
    if wnid_to_label:
        labels = [wnid_to_label.get(cls, normalize_label(cls)) for cls in dataset.classes]
    elif len(dataset.classes) == 1000 and all(cls.startswith("n") for cls in dataset.classes):
        raise ValueError(
            "ImageFolder 类目录看起来是 ImageNet WNID。请传入 "
            "--class-index-json 指向 imagenet_class_index.json，否则 prompt 只能是 n014... "
            "这种编号，评估不是真正的语义 zero-shot。"
        )
    elif len(dataset.classes) == 1000:
        labels = [
            normalize_label(label)
            for label in ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
        ]
        logger.warning(
            "ImageFolder 有 1000 类但未提供 class-index json；假设目录顺序与 "
            "torchvision ImageNet-1K 类别顺序一致。"
        )
    else:
        labels = [normalize_label(cls) for cls in dataset.classes]

    logger.info("数据后端: torchvision.datasets.ImageFolder")
    return dataset, labels


def build_prompts(labels: list[str], templates: list[str]) -> list[list[str]]:
    return [[template.format(label=label) for template in templates] for label in labels]


@torch.no_grad()
def encode_class_texts(
    model: CLIPModel,
    labels: list[str],
    templates: list[str],
    tokenizer: CLIPTokenizer,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    per_class_prompts = build_prompts(labels, templates)
    flat_prompts = [prompt for prompts in per_class_prompts for prompt in prompts]
    inputs = tokenizer(
        flat_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    embeds = model.encode_text(input_ids, attention_mask)
    embeds = embeds.view(len(labels), len(templates), -1).mean(dim=1)
    return torch.nn.functional.normalize(embeds, dim=-1)


@torch.no_grad()
def evaluate(
    model: CLIPModel,
    loader: DataLoader,
    class_text_embeds: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    total = 0
    top1_correct = 0
    top5_correct = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        image_embeds = model.encode_image(images)
        logits = image_embeds @ class_text_embeds.t()
        max_k = min(5, logits.size(1))
        pred = logits.topk(max_k, dim=1).indices

        total += targets.size(0)
        top1_correct += pred[:, :1].eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        top5_correct += pred[:, :max_k].eq(targets.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "top1_accuracy": 100.0 * top1_correct / max(total, 1),
        "top5_accuracy": 100.0 * top5_correct / max(total, 1),
        "num_samples": float(total),
    }


def main():
    parser = argparse.ArgumentParser(description="[PRTS] ImageNet zero-shot evaluation")
    parser.add_argument("--config", default="configs/coco_3gpu_cliptext.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imagenet-root", required=True)
    parser.add_argument(
        "--backend",
        choices=["auto", "torchvision", "imagefolder"],
        default="auto",
    )
    parser.add_argument("--class-index-json", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="logs/imagenet_zero_shot.txt")
    parser.add_argument(
        "--templates",
        nargs="+",
        default=["a photo of a {label}."],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.imagenet_root).exists():
        message = f"ImageNet root not found: {args.imagenet_root}"
        logger.error(message)
        output_path.write_text(
            "\n".join(
                [
                    "dataset: imagenet-val",
                    "status: missing_data",
                    f"root: {args.imagenet_root}",
                    f"checkpoint: {args.checkpoint}",
                    f"error: {message}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(2)

    dataset, labels = build_dataset(
        args.imagenet_root,
        cfg.image_size,
        args.backend,
        args.class_index_json,
    )
    logger.info(f"ImageNet val 样本数: {len(dataset)} | 类别数: {len(labels)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(cfg, args.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )
    class_text_embeds = encode_class_texts(
        model,
        labels,
        args.templates,
        tokenizer,
        cfg.text_max_length,
        device,
    )
    metrics = evaluate(model, loader, class_text_embeds, device)

    logger.info("=" * 50)
    logger.info("ImageNet Zero-shot Classification")
    logger.info("=" * 50)
    for name, value in metrics.items():
        if name == "num_samples":
            logger.info(f"  {name:>15s}: {int(value)}")
        else:
            logger.info(f"  {name:>15s}: {value:.2f}")
    logger.info("=" * 50)

    lines = [
        "dataset: imagenet-val",
        "status: completed",
        f"root: {args.imagenet_root}",
        f"checkpoint: {args.checkpoint}",
        f"templates: {args.templates}",
    ]
    for name, value in metrics.items():
        if name == "num_samples":
            lines.append(f"{name}: {int(value)}")
        else:
            lines.append(f"{name}: {value:.2f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
