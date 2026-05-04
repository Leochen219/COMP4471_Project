# baseline/dual_encoder_model.py
#
# Baseline model: ResNet-50 dual-encoder (CLIP-style).
#
# Architecture:
#   - ResNet-50 image encoder (fine-tuned, ImageNet-pretrained)
#   - Frozen CLIP text encoder (openai/clip-vit-base-patch32)
#   - Trainable projection heads -> 256-dim shared embedding
#   - Symmetric InfoNCE loss
#
# This is the same paradigm as the main CLIPModel (models/clip_model.py),
# but uses ResNet-50 instead of ViT-B/16 as the image backbone.
#
# Interface (matching CLIPModel conventions):
#   encode_image(images)                        -> [B, embed_dim] L2-normalized
#   encode_text(input_ids, attention_mask)       -> [B, embed_dim] L2-normalized
#   forward(images, input_ids, attention_mask)   -> scalar loss
#   get_param_groups(lr, image_encoder_lr)       -> list[dict]

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Allow importing from project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.image_encoder import build_image_encoder
from models.text_encoder import TextEncoder
from utils.losses import clip_info_nce_loss

logger = logging.getLogger(__name__)


class BaselineModel(nn.Module):
    """
    ResNet-50 dual-encoder model (CLIP-style baseline).

    Uses the same frozen CLIP text encoder and InfoNCE loss as the main model,
    but replaces ViT-B/16 with ResNet-50 as the image backbone.
    """

    def __init__(
        self,
        image_encoder_name: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 256,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        text_max_length: int = 77,
        freeze_text_encoder: bool = True,
        freeze_text_projection: bool = False,
    ):
        super().__init__()

        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_text_projection = freeze_text_projection
        self.text_max_length = text_max_length

        # ---- Image branch (fine-tune ResNet-50) ----
        self.image_encoder, img_feat_dim = build_image_encoder(
            image_encoder_name, pretrained
        )
        # ResNet-50 outputs 2048-dim features -> project to embed_dim
        self.image_projection = nn.Sequential(
            nn.Linear(img_feat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- Text branch (frozen CLIP) ----
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
        )
        text_feat_dim = self.text_encoder.output_dim
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable temperature (same as CLIPModel)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Apply freeze strategy
        self._apply_freeze()

    # =============================================================
    #  Freeze logic
    # =============================================================
    def _apply_freeze(self):
        """Freeze specified modules based on config."""
        if self.freeze_text_encoder:
            frozen_count = 0
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
            logger.info(
                f"[BASELINE] text_encoder frozen ({frozen_count / 1e6:.1f}M params)"
            )

        if self.freeze_text_projection:
            for param in self.text_projection.parameters():
                param.requires_grad = False
            logger.info("[BASELINE] text_projection frozen")

        if self.freeze_text_encoder:
            self.text_encoder.eval()

        # Parameter statistics
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        logger.info(
            f"[BASELINE] Parameter stats - Total: {total/1e6:.1f}M | "
            f"Trainable: {trainable/1e6:.1f}M | Frozen: {frozen/1e6:.1f}M"
        )

    # =============================================================
    #  Parameter groups (different learning rates)
    # =============================================================
    def get_param_groups(
        self,
        lr: float = 3e-4,
        image_encoder_lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """
        Return parameter groups:
          - group 0: image_encoder (fine-tune, small LR)
          - group 1: image_projection + text_projection + logit_scale (normal LR)
        Frozen params are automatically skipped.
        """
        # Image encoder params (trainable only)
        image_encoder_params = [
            p for p in self.image_encoder.parameters() if p.requires_grad
        ]

        # Projection heads + logit_scale
        clip_head_params = []
        for module in [self.image_projection, self.text_projection]:
            clip_head_params.extend(
                [p for p in module.parameters() if p.requires_grad]
            )
        if self.logit_scale.requires_grad:
            clip_head_params.append(self.logit_scale)

        # Verify no params are missed
        grouped_ids = {id(p) for p in image_encoder_params + clip_head_params}
        all_trainable = [p for p in self.parameters() if p.requires_grad]

        orphan_params = [p for p in all_trainable if id(p) not in grouped_ids]
        if orphan_params:
            logger.warning(
                f"[BASELINE] Found {len(orphan_params)} ungrouped trainable params, "
                f"added to clip_head group"
            )
            clip_head_params.extend(orphan_params)

        param_groups = []

        if image_encoder_params:
            param_groups.append(
                {
                    "params": image_encoder_params,
                    "lr": image_encoder_lr,
                    "weight_decay": weight_decay,
                    "name": "image_encoder",
                }
            )

        if clip_head_params:
            param_groups.append(
                {
                    "params": clip_head_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "name": "clip_head",
                }
            )

        # Log
        for g in param_groups:
            cnt = sum(p.numel() for p in g["params"])
            logger.info(
                f"[BASELINE] Param group [{g['name']}] "
                f"{cnt/1e6:.1f}M params, lr={g['lr']:.2e}"
            )

        return param_groups

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    # =============================================================
    #  Forward
    # =============================================================
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] -> [B, embed_dim]"""
        feat = self.image_encoder(images)
        return F.normalize(self.image_projection(feat), dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """input_ids: [B, 77], attention_mask: [B, 77] -> [B, embed_dim]"""
        if self.freeze_text_encoder:
            with torch.no_grad():
                feat = self.text_encoder(input_ids, attention_mask)
        else:
            feat = self.text_encoder(input_ids, attention_mask)

        return F.normalize(self.text_projection(feat), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return symmetric InfoNCE loss (scalar)"""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return clip_info_nce_loss(image_embeds, text_embeds, self.logit_scale)
