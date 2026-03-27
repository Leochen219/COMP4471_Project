# evaluate_transfer.py

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
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


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
}


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
    logger.info(
        f"已加载权重: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})"
    )
    return model


def build_dataset(name: str, root: str, image_size: int, download: bool):
    dataset_cls = DATASETS[name]
    dataset = dataset_cls(
        root=root,
        train=False,
        transform=get_val_transform(image_size),
        download=download,
    )
    return dataset


def build_prompts(labels: list[str]) -> list[str]:
    prompts = []
    for label in labels:
        clean_label = label.replace("_", " ")
        prompts.append(f"a photo of a {clean_label}.")
    return prompts


@torch.no_grad()
def encode_class_texts(
    model: CLIPModel,
    labels: list[str],
    tokenizer: CLIPTokenizer,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    prompts = build_prompts(labels)
    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return model.encode_text(input_ids, attention_mask)


@torch.no_grad()
def evaluate_classifier(
    model: CLIPModel,
    loader: DataLoader,
    class_text_embeds: torch.Tensor,
    device: torch.device,
    topk: tuple[int, int] = (1, 5),
) -> dict[str, float]:
    total = 0
    correct_at_k = {k: 0 for k in topk}

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_embeds = model.encode_image(images)
        logits = image_embeds @ class_text_embeds.t()

        max_k = min(max(topk), logits.size(1))
        topk_indices = logits.topk(max_k, dim=1).indices

        total += targets.size(0)
        for k in topk:
            effective_k = min(k, logits.size(1))
            correct = topk_indices[:, :effective_k].eq(targets.unsqueeze(1))
            correct_at_k[k] += correct.any(dim=1).sum().item()

    metrics = {}
    for k in topk:
        metrics[f"top{k}_accuracy"] = 100.0 * correct_at_k[k] / max(total, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="[PRTS] Zero-shot transfer evaluation"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="cifar100",
    )
    parser.add_argument(
        "--data-root",
        default="/data/ydongbd/datasets",
        help="torchvision dataset root",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    dataset = build_dataset(
        args.dataset,
        args.data_root,
        cfg.image_size,
        download=args.download,
    )
    logger.info(f"数据集: {args.dataset} | 样本数: {len(dataset)}")

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
        dataset.classes,  # type: ignore[attr-defined]
        tokenizer,
        cfg.text_max_length,
        device,
    )
    logger.info(f"类别数: {class_text_embeds.size(0)}")

    metrics = evaluate_classifier(model, loader, class_text_embeds, device)

    logger.info("=" * 50)
    logger.info(f"{args.dataset} Zero-shot Classification")
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info(f"  {name:>15s}: {value:.2f}")
    logger.info("=" * 50)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"dataset: {args.dataset}",
            f"checkpoint: {args.checkpoint}",
        ]
        for name, value in metrics.items():
            lines.append(f"{name}: {value:.2f}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
