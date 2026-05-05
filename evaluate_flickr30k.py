# evaluate_flickr30k.py

import argparse
import io
import logging
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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


class Flickr30kWebDataset(Dataset):
    """Small random-access Flickr30k WDS reader for evaluation splits."""

    def __init__(self, root: str, split: str, transform):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(f"No Flickr30k samples found in {self.root / split}")

    def _load_samples(self) -> list[dict]:
        split_dir = self.root / self.split
        shards = sorted(split_dir.glob("*.tar"))
        samples = []
        for shard in shards:
            grouped: dict[str, dict[str, bytes]] = {}
            with tarfile.open(shard, "r:") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = Path(member.name)
                    key = name.stem
                    suffix = name.suffix.lower()
                    if suffix not in {".jpg", ".jpeg", ".png", ".txt"}:
                        continue
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    grouped.setdefault(key, {})[suffix] = fileobj.read()

            for key in sorted(grouped):
                item = grouped[key]
                image_bytes = (
                    item.get(".jpg") or item.get(".jpeg") or item.get(".png")
                )
                text_bytes = item.get(".txt")
                if not image_bytes or not text_bytes:
                    continue
                captions = [
                    line.strip()
                    for line in text_bytes.decode("utf-8", errors="ignore").splitlines()
                    if line.strip()
                ]
                if captions:
                    samples.append(
                        {
                            "image": image_bytes,
                            "captions": captions,
                            "key": f"{shard.name}:{key}",
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(io.BytesIO(sample["image"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "captions": sample["captions"],
            "key": sample["key"],
        }


def collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "captions": [item["captions"] for item in batch],
        "key": [item["key"] for item in batch],
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
    logger.info(f"已加载权重: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def extract_embeddings(model, loader, tokenizer, max_length: int, device: torch.device):
    image_embeds = []
    text_embeds = []
    text_to_image = []
    image_index = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        image_embeds.append(model.encode_image(images).cpu())

        captions = []
        owners = []
        for local_idx, sample_captions in enumerate(batch["captions"]):
            global_image_idx = image_index + local_idx
            for caption in sample_captions:
                captions.append(caption)
                owners.append(global_image_idx)

        inputs = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        text_embeds.append(model.encode_text(input_ids, attention_mask).cpu())
        text_to_image.extend(owners)
        image_index += len(batch["captions"])

    return (
        torch.cat(image_embeds, dim=0),
        torch.cat(text_embeds, dim=0),
        torch.tensor(text_to_image, dtype=torch.long),
    )


@torch.no_grad()
def compute_multi_caption_retrieval(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    text_to_image: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    sim = image_embeds @ text_embeds.t()
    num_images = image_embeds.size(0)

    metrics: dict[str, float] = {}

    for k in ks:
        topk_texts = sim.topk(min(k, sim.size(1)), dim=1).indices
        hits = text_to_image[topk_texts].eq(
            torch.arange(num_images).unsqueeze(1)
        ).any(dim=1)
        metrics[f"i2t_R@{k}"] = 100.0 * hits.float().mean().item()

    text_sim = sim.t()
    for k in ks:
        topk_images = text_sim.topk(min(k, text_sim.size(1)), dim=1).indices
        hits = topk_images.eq(text_to_image.unsqueeze(1)).any(dim=1)
        metrics[f"t2i_R@{k}"] = 100.0 * hits.float().mean().item()

    metrics["mean_recall"] = sum(
        value for key, value in metrics.items() if "R@" in key
    ) / (2 * len(ks))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="[PRTS] Flickr30k retrieval eval")
    parser.add_argument("--config", default="configs/coco_cc3m_3gpu_cliptext.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="data/flickr30k_wds")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="logs/flickr30k_retrieval.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    dataset = Flickr30kWebDataset(
        root=args.root,
        split=args.split,
        transform=get_val_transform(cfg.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    logger.info(f"Flickr30k {args.split} 样本数: {len(dataset)}")

    model = load_model(cfg, args.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )

    image_embeds, text_embeds, text_to_image = extract_embeddings(
        model, loader, tokenizer, cfg.text_max_length, device
    )
    logger.info(
        f"图像嵌入: {tuple(image_embeds.shape)}  "
        f"文本嵌入: {tuple(text_embeds.shape)}"
    )

    metrics = compute_multi_caption_retrieval(
        image_embeds, text_embeds, text_to_image
    )

    logger.info("=" * 50)
    logger.info("Flickr30k Retrieval")
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info(f"  {name:>15s}: {value:.2f}")
    logger.info("=" * 50)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "dataset: flickr30k",
        f"split: {args.split}",
        f"root: {args.root}",
        f"checkpoint: {args.checkpoint}",
        f"num_images: {image_embeds.size(0)}",
        f"num_texts: {text_embeds.size(0)}",
    ]
    for name, value in metrics.items():
        lines.append(f"{name}: {value:.2f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
