# baseline/evaluate_transfer.py
#
# Zero-shot transfer evaluation (CIFAR-10/100) for ResNet-50 baseline model.
# Adapted from evaluate_transfer.py - imports BaselineModel instead of CLIPModel.

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from transformers import CLIPTokenizer

# Allow importing from project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs import load_config
from data.transforms import get_val_transform
from baseline import BaselineModel

logging.basicConfig(
    level=logging.INFO,
    format="[BASELINE %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CIFAR class names (hardcoded fallback in case meta file is missing) ──
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── Custom dataset that loads CIFAR from raw pickle files ──
class CIFARPickleDataset(Dataset):
    """Load CIFAR-10/100 from the extracted pickle files (no torchvision dependency)."""

    def __init__(self, root: str, name: str, transform=None):
        self.transform = transform
        self.name = name

        if name == "cifar10":
            inner_dir = "cifar-10-batches-py"
            test_file = "test_batch"
            meta_file = "batches.meta"
            self.classes = CIFAR10_CLASSES
        else:
            inner_dir = "cifar-100-python"
            test_file = "test"
            meta_file = "meta"
            self.classes = CIFAR100_CLASSES

        data_dir = os.path.join(root, inner_dir)

        # Try to load class names from meta file
        meta_path = os.path.join(data_dir, meta_file)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f, encoding="bytes")
                if name == "cifar10":
                    self.classes = [
                        label.decode("utf-8") for label in meta[b"label_names"]
                    ]
                else:
                    self.classes = [
                        label.decode("utf-8") for label in meta[b"fine_label_names"]
                    ]
                logger.info(f"Loaded class names from {meta_path}")
            except Exception as e:
                logger.warning(f"Could not load meta file: {e}, using defaults")

        # Load test batch
        test_path = os.path.join(data_dir, test_file)
        if not os.path.isfile(test_path):
            raise FileNotFoundError(
                f"Test file not found at {test_path}. "
                f"Run: python baseline/download_cifar.py {name}"
            )

        with open(test_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        if name == "cifar10":
            self.images = batch[b"data"]
            self.labels = batch[b"labels"]
        else:
            self.images = batch[b"data"]
            self.labels = batch[b"fine_labels"]

        # Reshape images: [N, 3072] -> [N, 3, 32, 32]
        self.images = self.images.reshape(-1, 3, 32, 32).astype(np.uint8)
        self.labels = np.array(self.labels, dtype=np.int64)

        logger.info(
            f"Loaded {name} test set: {len(self.images)} images, {len(self.classes)} classes"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        # Convert to [H, W, C] for torchvision transforms
        img = img.transpose(1, 2, 0)  # [3, 32, 32] -> [32, 32, 3]
        label = self.labels[idx]

        if self.transform is not None:
            # Need to convert to PIL for torchvision transforms
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, label


def load_model(cfg, checkpoint_path: str, device: torch.device) -> BaselineModel:
    model = BaselineModel(
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
        f"Loaded weights: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})"
    )
    return model


# ── Dataset builder ──
DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
}


def build_dataset(name: str, root: str, image_size: int, download: bool):
    """Try torchvision first, fall back to pickle-based loader."""
    dataset_cls = DATASETS[name]

    # Try torchvision with download=False first
    try:
        dataset = dataset_cls(
            root=root,
            train=False,
            transform=get_val_transform(image_size),
            download=False,
        )
        logger.info(f"Loaded {name} via torchvision from {root}")
        return dataset
    except (RuntimeError, Exception) as e:
        logger.info(f"torchvision could not load {name}: {e}")

    # Try torchvision with download=True
    if download:
        try:
            dataset = dataset_cls(
                root=root,
                train=False,
                transform=get_val_transform(image_size),
                download=True,
            )
            logger.info(f"Downloaded {name} via torchvision to {root}")
            return dataset
        except (RuntimeError, Exception) as e:
            logger.warning(f"torchvision download failed: {e}")

    # Fall back to pickle-based loader
    logger.info(f"Falling back to pickle-based CIFAR loader...")
    try:
        dataset = CIFARPickleDataset(
            root=root,
            name=name,
            transform=get_val_transform(image_size),
        )
        logger.info(f"Loaded {name} via pickle loader from {root}")
        return dataset
    except FileNotFoundError as e:
        logger.error(
            f"CIFAR data not found. Please download it first:\n"
            f"  python baseline/download_cifar.py {name}"
        )
        raise


def build_prompts(labels: list[str]) -> list[str]:
    prompts = []
    for label in labels:
        clean_label = label.replace("_", " ")
        prompts.append(f"a photo of a {clean_label}.")
    return prompts


@torch.no_grad()
def encode_class_texts(
    model: BaselineModel,
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
    model: BaselineModel,
    loader: DataLoader,
    class_text_embeds: torch.Tensor,
    device: torch.device,
    topk: tuple[int, int] = (1, 5),
) -> dict[str, float]:
    total = 0
    correct_at_k = {k: 0 for k in topk}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        image_embeds = model.encode_image(images)  # [B, D]

        # Similarity: [B, num_classes]
        sim = image_embeds @ class_text_embeds.t()

        for k in topk:
            pred_topk = sim.topk(k, dim=1).indices  # [B, k]
            # Ground truth labels
            correct_at_k[k] += (
                (pred_topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            )

        total += images.size(0)

    results = {}
    for k in topk:
        acc = 100.0 * correct_at_k[k] / total
        results[f"top{k}_accuracy"] = acc
        logger.info(f"Top-{k} accuracy: {acc:.2f}%")

    results["num_samples"] = total
    return results


def main():
    parser = argparse.ArgumentParser(
        description="[BASELINE] Zero-shot Transfer Evaluation"
    )
    parser.add_argument("--config", default="baseline/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- Model ----------
    model = load_model(cfg, args.checkpoint, device)

    # ---------- Dataset ----------
    dataset = build_dataset(args.dataset, args.data_root, cfg.image_size, args.download)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(
        f"Dataset: {args.dataset} | Samples: {len(dataset)} | Classes: {len(dataset.classes)}"
    )

    # ---------- Encode class texts ----------
    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )
    class_text_embeds = encode_class_texts(
        model, dataset.classes, tokenizer, cfg.text_max_length, device
    )
    logger.info(f"Class text embeddings: {class_text_embeds.shape}")

    # ---------- Evaluate ----------
    results = evaluate_classifier(model, loader, class_text_embeds, device)

    # ---------- Save ----------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
