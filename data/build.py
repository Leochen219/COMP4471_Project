# data/build.py

import logging

from torch.utils.data import DataLoader, IterableDataset
from transformers import CLIPTokenizer

from .dataset import (
    CC3MDataset,
    CC3MWebDataset,
    CleanCOCODataset,
    ImageTextManifestDataset,
    RandomMixDataset,
    StreamingCOCOCC3MDataset,
)
from .transforms import get_train_transform, get_val_transform

logger = logging.getLogger(__name__)


def _build_dataset_from_spec(spec, cfg, tokenizer, transform, default_eval_mode=False):
    name = str(spec.get("name", "coco")).lower()
    max_samples = int(spec.get("max_samples", 0) or 0)

    if name == "coco":
        return CleanCOCODataset(
            json_path=spec.get(
                "json_path",
                spec.get("train_json", spec.get("val_json", cfg.train_json)),
            ),
            img_dir=spec.get(
                "img_dir",
                spec.get("train_img", spec.get("val_img", cfg.train_img)),
            ),
            transform=transform,
            tokenizer=tokenizer,
            max_length=cfg.text_max_length,
            eval_mode=bool(spec.get("eval_mode", default_eval_mode)),
        )

    if name == "cc3m":
        return CC3MDataset(
            manifest_path=spec["manifest"],
            img_root=spec.get("img_root", "data/cc3m"),
            transform=transform,
            tokenizer=tokenizer,
            max_length=cfg.text_max_length,
            max_samples=max_samples,
        )

    if name in {"cc3m_wds", "cc3m_webdataset"}:
        return CC3MWebDataset(
            root=spec.get("root", "data/cc3m_wds"),
            split=spec.get("split", "train"),
            transform=transform,
            tokenizer=tokenizer,
            max_length=cfg.text_max_length,
            max_samples=max_samples,
        )

    if name in {"manifest", "image_text_manifest"}:
        return ImageTextManifestDataset(
            manifest_path=spec["manifest"],
            img_root=spec.get("img_root", ""),
            transform=transform,
            tokenizer=tokenizer,
            max_length=cfg.text_max_length,
            max_samples=max_samples,
            dataset_name=name,
        )

    raise ValueError(f"[PRTS] 不支持的数据集类型: {name}")


def _build_train_dataset(cfg, tokenizer):
    train_specs = getattr(cfg, "train_datasets", None)
    if not train_specs:
        return CleanCOCODataset(
            json_path=cfg.train_json,
            img_dir=cfg.train_img,
            transform=get_train_transform(cfg.image_size),
            tokenizer=tokenizer,
            max_length=cfg.text_max_length,
            eval_mode=False,
        )

    transform = get_train_transform(cfg.image_size)
    spec_names = [str(spec.get("name", "")).lower() for spec in train_specs]
    if "coco" in spec_names and any(
        name in {"cc3m_wds", "cc3m_webdataset"} for name in spec_names
    ):
        coco_spec = train_specs[spec_names.index("coco")]
        cc3m_idx = next(
            i
            for i, name in enumerate(spec_names)
            if name in {"cc3m_wds", "cc3m_webdataset"}
        )
        cc3m_spec = train_specs[cc3m_idx]
        coco_dataset = _build_dataset_from_spec(
            coco_spec, cfg, tokenizer, transform
        )
        return StreamingCOCOCC3MDataset(
            coco_dataset=coco_dataset,
            cc3m_root=cc3m_spec.get("root", "data/cc3m_wds"),
            split=cc3m_spec.get("split", "train"),
            transform=transform,
            tokenizer=tokenizer,
            max_length=cfg.text_max_length,
            mix_weights=getattr(cfg, "train_mix_weights", [1.0, 3.0]),
            epoch_size=getattr(cfg, "train_epoch_size", 240000),
        )

    datasets = [
        _build_dataset_from_spec(spec, cfg, tokenizer, transform)
        for spec in train_specs
    ]
    if len(datasets) == 1:
        return datasets[0]

    weights = getattr(cfg, "train_mix_weights", None)
    epoch_size = getattr(cfg, "train_epoch_size", None)
    mixed = RandomMixDataset(
        datasets=datasets,
        weights=weights,
        epoch_size=epoch_size,
    )
    logger.info(
        "[PRTS] 混合训练集: "
        + " | ".join(
            f"{spec.get('name', 'dataset')}={len(dataset)}"
            for spec, dataset in zip(train_specs, datasets)
        )
        + f" | epoch_size={len(mixed)}"
    )
    return mixed


def _build_val_dataset(cfg, tokenizer):
    val_specs = getattr(cfg, "val_datasets", None)
    if val_specs:
        transform = get_val_transform(cfg.image_size)
        datasets = [
            _build_dataset_from_spec(
                spec, cfg, tokenizer, transform, default_eval_mode=True
            )
            for spec in val_specs
        ]
        if len(datasets) == 1:
            return datasets[0]
        return RandomMixDataset(
            datasets=datasets,
            weights=getattr(cfg, "val_mix_weights", None),
            epoch_size=getattr(cfg, "val_epoch_size", None),
        )

    return CleanCOCODataset(
        json_path=cfg.val_json,
        img_dir=cfg.val_img,
        transform=get_val_transform(cfg.image_size),
        tokenizer=tokenizer,
        max_length=cfg.text_max_length,
        eval_mode=False,
    )


def build_dataloaders(cfg):
    """训练 + 验证 DataLoader（验证集随机选 caption，用于算 loss）"""

    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )

    train_dataset = _build_train_dataset(cfg, tokenizer)
    val_dataset = _build_val_dataset(cfg, tokenizer)

    pw = cfg.num_workers > 0

    is_iterable = isinstance(train_dataset, IterableDataset)
    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": pw,
    }
    if not is_iterable:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(**train_loader_kwargs)

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

    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )

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
