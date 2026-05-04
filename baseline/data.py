# baseline/data.py
#
# Custom dataloader for baseline training.
# Wraps the existing CleanCOCODataset with random subsampling support.
#
# IMPORTANT: On Windows, PyTorch DataLoader with num_workers=0 can hang
# after the first batch due to internal iterator state issues. We work
# around this by:
#   1. Pre-tokenizing all captions in the dataset constructor (avoids
#      tokenizer calls in __getitem__ which can trigger the hang)
#   2. Using num_workers=0 (safest on Windows - no multiprocessing issues)
#   3. Avoiding pin_memory (not needed with num_workers=0)
#   4. Adding a timeout on image loading to prevent hangs on corrupted
#      or slow-to-read files

import logging
import random
import json
import os
import threading

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPTokenizer

import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.transforms import get_train_transform, get_val_transform

logger = logging.getLogger(__name__)

# ====================================================================
# Timeout helper for image loading (prevents hangs on Windows)
# ====================================================================
class ImageLoadTimeoutError(Exception):
    """Raised when image loading exceeds the timeout."""
    pass


def _load_image_with_timeout(img_path: str, timeout_sec: float = 10.0) -> Image.Image:
    """Load an image with a timeout to prevent indefinite hangs on Windows.

    Uses a daemon thread to enforce the timeout. If the image cannot be
    loaded within timeout_sec, raises ImageLoadTimeoutError.
    """
    result = [None]
    exception = [None]
    event = threading.Event()

    def _load():
        try:
            result[0] = Image.open(img_path).convert("RGB")
        except Exception as e:
            exception[0] = e
        finally:
            event.set()

    t = threading.Thread(target=_load, daemon=True)
    t.start()

    if not event.wait(timeout=timeout_sec):
        # Thread still alive after timeout -> treat as hang
        raise ImageLoadTimeoutError(
            f"Image loading timed out after {timeout_sec}s: {img_path}"
        )

    if exception[0] is not None:
        raise exception[0]

    return result[0]


# ====================================================================
# PreTokenizedCOCODataset
#
# A self-contained COCO dataset that pre-tokenizes ALL captions in the
# constructor. This avoids calling the tokenizer inside __getitem__,
# which is the primary cause of DataLoader hangs on Windows.
# ====================================================================
class PreTokenizedCOCODataset(Dataset):
    """COCO Captions dataset with pre-tokenized captions.

    Pre-tokenizes all captions once during __init__, so __getitem__
    is extremely fast (just image loading + tensor indexing). This
    avoids the Windows DataLoader hang issue with num_workers=0.

    Supports optional subsampling: pass max_samples to limit the
    number of images used. Subsampling happens BEFORE tokenization
    to avoid unnecessary work.
    """

    def __init__(
        self,
        json_path: str,
        img_dir: str,
        transform=None,
        tokenizer=None,
        max_length: int = 77,
        eval_mode: bool = False,
        max_samples: int | None = None,
        subsample_seed: int = 4471,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length
        self.eval_mode = eval_mode

        # ---------- Read JSON ----------
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Group captions by image_id
        img_to_captions: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            img_id: int = ann["image_id"]
            caption = self._clean_text(ann["caption"])
            if img_id not in img_to_captions:
                img_to_captions[img_id] = []
            img_to_captions[img_id].append(caption)

        # Build image list (filter missing files)
        all_img_data: list[dict] = []
        skipped = 0
        for img in data["images"]:
            path = os.path.join(self.img_dir, img["file_name"])
            if img["id"] in img_to_captions and os.path.exists(path):
                all_img_data.append(
                    {"file_name": img["file_name"], "id": img["id"]}
                )
            else:
                skipped += 1

        # ---------- Subsampling (BEFORE tokenization) ----------
        if max_samples is not None and max_samples < len(all_img_data):
            rng = random.Random(subsample_seed)
            indices = rng.sample(range(len(all_img_data)), max_samples)
            self.img_data = [all_img_data[i] for i in indices]
            logger.info(
                f"[BASELINE] Subsampled from {len(all_img_data)} to "
                f"{max_samples} samples (seed={subsample_seed})"
            )
        else:
            self.img_data = all_img_data

        # ---------- Pre-tokenize ALL captions ----------
        # Store as list of (input_ids, attention_mask) tensors per image
        self.tokenized_captions: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for img_info in self.img_data:
            img_id = img_info["id"]
            caps = img_to_captions[img_id]
            tokenized_list = []
            for cap in caps:
                encoded = tokenizer(
                    cap,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokenized_list.append(
                    (
                        encoded["input_ids"].squeeze(0).clone(),
                        encoded["attention_mask"].squeeze(0).clone(),
                    )
                )
            self.tokenized_captions.append(tokenized_list)

        logger.info(
            f"[BASELINE] Dataset ready: {len(self.img_data)} valid samples, "
            f"{skipped} skipped (pre-tokenized)"
        )

    @staticmethod
    def _clean_text(text) -> str:
        if not isinstance(text, str):
            text = str(text)
        return " ".join(text.strip().lower().split())

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        img_info = self.img_data[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # Read image with timeout to prevent hangs on Windows
        try:
            image = _load_image_with_timeout(img_path, timeout_sec=10.0)
        except ImageLoadTimeoutError:
            logger.warning(
                f"[BASELINE] Image load timed out (10s): {img_path}, skipping"
            )
            return self.__getitem__((idx + 1) % len(self))
        except Exception:
            logger.warning(f"[BASELINE] Corrupted image: {img_path}, skipping")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # Select caption (pre-tokenized, just index into list)
        captions = self.tokenized_captions[idx]
        if self.eval_mode:
            input_ids, attention_mask = captions[0]
        else:
            input_ids, attention_mask = random.choice(captions)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


# ====================================================================
# DataLoader builders
# ====================================================================

def build_baseline_dataloaders(cfg):
    """Build train/val dataloaders with optional random subsampling.

    Uses num_workers=0 (safest on Windows) with pre-tokenized dataset
    to avoid the Windows DataLoader hang issue.
    Subsampling happens BEFORE tokenization for efficiency.
    """

    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )

    max_train_samples = getattr(cfg, "max_train_samples", None)

    train_dataset = PreTokenizedCOCODataset(
        json_path=cfg.train_json,
        img_dir=cfg.train_img,
        transform=get_train_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=False,
        max_samples=max_train_samples,
    )

    val_dataset = PreTokenizedCOCODataset(
        json_path=cfg.val_json,
        img_dir=cfg.val_img,
        transform=get_val_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=False,
    )

    # num_workers=0 is safest on Windows (no multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader


def build_baseline_eval_dataloader(cfg):
    """Build evaluation dataloader (always uses first caption)."""

    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )

    eval_dataset = PreTokenizedCOCODataset(
        json_path=cfg.val_json,
        img_dir=cfg.val_img,
        transform=get_val_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return eval_loader
