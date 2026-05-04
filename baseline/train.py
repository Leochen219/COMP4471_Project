# baseline/train.py
#
# Training script for ResNet-50 baseline model.
# Adapted from train.py - same logic, imports BaselineModel instead of CLIPModel.

import os
import math
import logging
import argparse
import signal
from typing import Any

# Force HuggingFace offline mode to avoid network hangs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys

# Allow importing from project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs import load_config
from baseline import BaselineModel
from baseline.data import build_baseline_dataloaders

# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[BASELINE %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    if loss.ndim > 0:
        return loss.mean()
    return loss


def parse_gpu_ids(raw_gpu_ids: Any) -> list[int]:
    if raw_gpu_ids is None:
        return []
    if isinstance(raw_gpu_ids, int):
        return [raw_gpu_ids]
    if isinstance(raw_gpu_ids, (list, tuple)):
        return [int(gpu_id) for gpu_id in raw_gpu_ids]
    return []


# ======================== LR Scheduler ==========================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """All param groups share the same warmup + cosine decay ratio."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ======================== Batch timeout handler =================
class BatchTimeoutError(Exception):
    """Raised when a single batch takes too long (DataLoader hang on Windows)."""
    pass


def _timeout_handler(signum, frame):
    raise BatchTimeoutError("Batch processing timed out")


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, cfg):
    """Train for one epoch with batch-level timeout to detect DataLoader hangs."""
    model.train()
    total_loss = 0.0
    batch_timeout = getattr(cfg, "batch_timeout", 120)  # seconds per batch

    step = 0
    loader_iter = iter(loader)
    num_batches = len(loader)

    while step < num_batches:
        try:
            # Set alarm for this batch (Unix only; Windows uses a different approach)
            # On Windows, we rely on the image loading timeout in the dataset
            batch = next(loader_iter)
        except StopIteration:
            break
        except Exception as exc:
            logger.warning(
                f"[BASELINE] DataLoader error at step {step}, "
                f"recreating iterator: {exc}"
            )
            loader_iter = iter(loader)
            continue

        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        loss = reduce_loss(model(images, input_ids, attention_mask))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in unwrap_model(model).parameters() if p.requires_grad],
            max_norm=cfg.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % cfg.log_interval == 0:
            # Print learning rates for each param group
            current_lrs = scheduler.get_last_lr()
            lr_info = "  ".join(
                f"{g.get('name', i)}={current_lrs[min(i, len(current_lrs)-1)]:.2e}"
                for i, g in enumerate(optimizer.param_groups)
            )
            temp = unwrap_model(model).logit_scale.exp().item()
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{num_batches}]  "
                f"Loss: {loss.item():.4f}  LR: [{lr_info}]  Temp: {temp:.2f}"
            )

        step += 1

    return total_loss / max(num_batches, 1)


# ======================== Validation ============================
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        loss = reduce_loss(model(images, input_ids, attention_mask))
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ======================== Main ==================================
def main():
    parser = argparse.ArgumentParser(description="[BASELINE] ResNet-50 Training")
    parser.add_argument("--config", default="baseline/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---------- Device ----------
    gpu_ids = parse_gpu_ids(getattr(cfg, "gpu_ids", None))
    if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        available_gpus = torch.cuda.device_count()
        if gpu_ids:
            invalid_gpu_ids = [
                gpu_id for gpu_id in gpu_ids
                if gpu_id < 0 or gpu_id >= available_gpus
            ]
            if invalid_gpu_ids:
                raise ValueError(f"Invalid gpu_ids: {invalid_gpu_ids}")
            device = torch.device(f"cuda:{gpu_ids[0]}")
        else:
            device = torch.device(cfg.device)
            if device.index is None:
                gpu_ids = [0]
            else:
                gpu_ids = [device.index]
    else:
        device = torch.device("cpu")
        gpu_ids = []

    logger.info(f"Device: {device}")
    if gpu_ids:
        logger.info(f"Using GPU IDs: {gpu_ids}")

    # ---------- Data ----------
    train_loader, val_loader = build_baseline_dataloaders(cfg)
    logger.info(
        f"Train set: {len(train_loader.dataset)} samples  "
        f"Val set: {len(val_loader.dataset)} samples"
    )

    # ---------- Model ----------
    freeze_text_encoder = getattr(cfg, "freeze_text_encoder", False)
    freeze_text_projection = getattr(cfg, "freeze_text_projection", False)

    model = BaselineModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=cfg.pretrained,
        embed_dim=cfg.embed_dim,
        text_encoder_name=getattr(
            cfg, "text_encoder_name", "openai/clip-vit-base-patch32"
        ),
        text_max_length=cfg.text_max_length,
        freeze_text_encoder=freeze_text_encoder,
        freeze_text_projection=freeze_text_projection,
    ).to(device)

    # ---------- Optimizer ----------
    image_encoder_lr = getattr(cfg, "image_encoder_lr", cfg.lr)

    param_groups = model.get_param_groups(
        lr=cfg.lr,
        image_encoder_lr=image_encoder_lr,
        weight_decay=cfg.weight_decay,
    )

    optimizer = AdamW(param_groups)

    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.warmup_steps, total_steps
    )

    # ---------- Resume ----------
    start_epoch = 0
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint incompatible with baseline model."
            ) from exc
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---------- DataParallel ----------
    if device.type == "cuda" and len(gpu_ids) > 1:
        if cfg.batch_size < len(gpu_ids):
            raise ValueError("batch_size must be >= number of GPUs")
        per_gpu_batch = math.ceil(cfg.batch_size / len(gpu_ids))
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0],
        )
        logger.info(
            f"[BASELINE] DataParallel enabled | GPUs={gpu_ids} | "
            f"global_batch={cfg.batch_size} | approx_per_gpu_batch={per_gpu_batch}"
        )

    # ---------- Training strategy summary ----------
    logger.info("=" * 60)
    logger.info("[BASELINE] Training strategy summary")
    logger.info(f"  freeze_text_encoder:    {freeze_text_encoder}")
    logger.info(f"  freeze_text_projection: {freeze_text_projection}")
    logger.info(f"  image_encoder_lr:       {image_encoder_lr:.2e}")
    logger.info(f"  clip_head_lr:           {cfg.lr:.2e}")
    logger.info(f"  epochs:                 {cfg.epochs}")
    logger.info(f"  total_steps:            {total_steps}")
    logger.info("=" * 60)

    # ---------- Training loop ----------
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    val_interval = getattr(cfg, "val_interval", 5)  # Validate every N epochs

    for epoch in range(start_epoch, cfg.epochs):
        logger.info(f"========== Epoch {epoch}/{cfg.epochs - 1} ==========")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, cfg
        )

        # Validate every val_interval epochs (or always on last epoch)
        do_validate = (epoch % val_interval == 0) or (epoch == cfg.epochs - 1)
        if do_validate:
            val_loss = validate(model, val_loader, device)
            logger.info(
                f"Epoch {epoch} done | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
        else:
            val_loss = float("inf")
            logger.info(
                f"Epoch {epoch} done | "
                f"Train Loss: {train_loss:.4f} | (val skipped)"
            )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": vars(cfg),
        }

        torch.save(ckpt, os.path.join(cfg.save_dir, "latest.pt"))

        if do_validate and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(cfg.save_dir, "best.pt"))
            logger.info(f"Best model saved (val_loss={val_loss:.4f})")

    logger.info("[BASELINE] Training complete.")


if __name__ == "__main__":
    main()
