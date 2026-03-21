# train.py

import os
import math
import logging
import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from configs import load_config
from models import CLIPModel
from data import build_dataloaders

# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[PRTS %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================== 学习率调度 ============================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """所有参数组共享同一 warmup + cosine 衰减比例"""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ======================== 训练一个 epoch ========================
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, cfg):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        loss = model(images, input_ids, attention_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=cfg.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % cfg.log_interval == 0:
            # 打印各参数组学习率
            lr_info = "  ".join(
                f"{g.get('name', i)}={g['lr'] * scheduler.get_last_lr()[0] / g['lr']:.2e}"
                if len(scheduler.get_last_lr()) == 1
                else f"{g.get('name', i)}={scheduler.get_last_lr()[min(i, len(scheduler.get_last_lr())-1)]:.2e}"
                for i, g in enumerate(optimizer.param_groups)
            )
            temp = model.logit_scale.exp().item()
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}]  "
                f"Loss: {loss.item():.4f}  LR: [{lr_info}]  Temp: {temp:.2f}"
            )

    return total_loss / len(loader)


# ======================== 验证 ==================================
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        loss = model(images, input_ids, attention_mask)
        total_loss += loss.item()

    return total_loss / len(loader)


# ======================== 主函数 ================================
def main():
    parser = argparse.ArgumentParser(description="[PRTS] CLIP Training")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---------- 设备 ----------
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # ---------- 数据 ----------
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        f"训练集: {len(train_loader.dataset)} 样本  " # type: ignore[arg-type]
        f"验证集: {len(val_loader.dataset)} 样本" # type: ignore[arg-type]
    ) 

    # ---------- 模型（含冻结策略）----------
    # 获取冻结配置，兼容旧配置文件（默认不冻结）
    freeze_text_encoder = getattr(cfg, "freeze_text_encoder", False)
    freeze_text_projection = getattr(cfg, "freeze_text_projection", False)

    model = CLIPModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=cfg.pretrained,
        embed_dim=cfg.embed_dim,
        vocab_size=cfg.vocab_size,
        text_hidden_dim=cfg.text_hidden_dim,
        text_num_layers=cfg.text_num_layers,
        text_num_heads=cfg.text_num_heads,
        text_max_length=cfg.text_max_length,
        freeze_text_encoder=freeze_text_encoder,
        freeze_text_projection=freeze_text_projection,
    ).to(device)

    # ---------- 优化器：使用分组学习率 ----------
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

    # ---------- 断点续训 ----------
    start_epoch = 0
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"从 epoch {start_epoch} 恢复训练")

    # ---------- 打印训练策略 ----------
    logger.info("=" * 60)
    logger.info("[PRTS] 训练策略摘要")
    logger.info(f"  freeze_text_encoder:    {freeze_text_encoder}")
    logger.info(f"  freeze_text_projection: {freeze_text_projection}")
    logger.info(f"  image_encoder_lr:       {image_encoder_lr:.2e}")
    logger.info(f"  clip_head_lr:           {cfg.lr:.2e}")
    logger.info(f"  epochs:                 {cfg.epochs}")
    logger.info(f"  total_steps:            {total_steps}")
    logger.info("=" * 60)

    # ---------- 训练循环 ----------
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.epochs):
        logger.info(f"========== Epoch {epoch}/{cfg.epochs - 1} ==========")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, cfg
        )
        val_loss = validate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch} 完毕 | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # 保存 checkpoint（保存完整模型，含冻结参数以便后续切换策略）
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": vars(cfg),
        }

        torch.save(ckpt, os.path.join(cfg.save_dir, "latest.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(cfg.save_dir, "best.pt"))
            logger.info(f"✅ Best model saved (val_loss={val_loss:.4f})")

    logger.info("[PRTS] 训练流程结束。")


if __name__ == "__main__":
    main()