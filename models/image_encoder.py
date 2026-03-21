# models/image_encoder.py

import torch.nn as nn
from torchvision import models as tv_models


def build_image_encoder(
    name: str = "vit_base_patch16_224",
    pretrained: bool = True,
):
    """
    返回 (backbone, feature_dim)
    backbone 输出: [B, feature_dim]
    """

    # ---------- ViT-B/16 ----------
    if name == "vit_base_patch16_224":
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)

        assert isinstance(model.heads.head, nn.Linear)
        feature_dim = model.heads.head.in_features  # 768

        model.heads.head = nn.Identity()
        return model, feature_dim

    # ---------- ResNet-50 ----------
    elif name == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)

        assert isinstance(model.fc, nn.Linear)
        feature_dim = model.fc.in_features  # 2048

        setattr(model, "fc", nn.Identity())
        return model, feature_dim

    # ---------- ResNet-101 ----------
    elif name == "resnet101":
        weights = tv_models.ResNet101_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet101(weights=weights)

        assert isinstance(model.fc, nn.Linear)
        feature_dim = model.fc.in_features  # 2048

        setattr(model, "fc", nn.Identity())
        return model, feature_dim

    else:
        raise ValueError(f"[PRTS] 不支持的图像编码器: {name}")