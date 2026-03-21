# data/build.py

from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from .dataset import CleanCOCODataset
from .transforms import get_train_transform, get_val_transform


def build_dataloaders(cfg):
    """训练 + 验证 DataLoader（验证集随机选 caption，用于算 loss）"""

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = CleanCOCODataset(
        json_path=cfg.train_json,
        img_dir=cfg.train_img,
        transform=get_train_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=False,
    )

    val_dataset = CleanCOCODataset(
        json_path=cfg.val_json,
        img_dir=cfg.val_img,
        transform=get_val_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=False,
    )

    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=pw,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=pw,
    )

    return train_loader, val_loader


def build_eval_dataloader(cfg):
    """评估专用 DataLoader（eval_mode=True → 始终取第一条 caption）"""

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    eval_dataset = CleanCOCODataset(
        json_path=cfg.val_json,
        img_dir=cfg.val_img,
        transform=get_val_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=True,
    )

    pw = cfg.num_workers > 0

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=pw,
    )

    return eval_loader