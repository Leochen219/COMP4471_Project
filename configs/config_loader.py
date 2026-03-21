# configs/config_loader.py

import yaml
from types import SimpleNamespace


def load_config(path: str = "configs/default.yaml") -> SimpleNamespace:
    """
    读取 YAML 配置，展平为单层 SimpleNamespace
    使用方式: cfg.batch_size, cfg.lr, ...
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    flat: dict = {}
    for section_name, section_val in raw.items():
        if isinstance(section_val, dict):
            flat.update(section_val)
        else:
            # 顶层非 dict 字段直接保留
            flat[section_name] = section_val

    return SimpleNamespace(**flat)