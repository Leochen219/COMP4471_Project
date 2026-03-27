# models/clip_model.py

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import build_image_encoder
from .text_encoder import TextEncoder
from utils.losses import clip_info_nce_loss

logger = logging.getLogger(__name__)


class CLIPModel(nn.Module):
    """
    接口约定:
        encode_image(images)                        → [B, embed_dim] L2-normalized
        encode_text(input_ids, attention_mask)       → [B, embed_dim] L2-normalized
        forward(images, input_ids, attention_mask)   → scalar loss
        get_param_groups(lr, image_encoder_lr)       → list[dict]  用于 optimizer
    """

    def __init__(
        self,
        image_encoder_name: str = "vit_base_patch16_224",
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

        # ---- 图像分支（fine-tune）----
        self.image_encoder, img_feat_dim = build_image_encoder(
            image_encoder_name, pretrained
        )
        self.image_projection = nn.Sequential(
            nn.Linear(img_feat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- 文本分支 ----
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
        )
        text_feat_dim = self.text_encoder.output_dim
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 可学习温度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ---- 执行冻结 ----
        self._apply_freeze()

    # =============================================================
    #  冻结逻辑
    # =============================================================
    def _apply_freeze(self):
        """根据配置冻结指定模块"""

        if self.freeze_text_encoder:
            frozen_count = 0
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
            logger.info(
                f"[PRTS] text_encoder 已冻结 ({frozen_count / 1e6:.1f}M 参数)"
            )

        if self.freeze_text_projection:
            for param in self.text_projection.parameters():
                param.requires_grad = False
            logger.info("[PRTS] text_projection 已冻结")

        if self.freeze_text_encoder:
            self.text_encoder.eval()

        # 统计总体情况
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        logger.info(
            f"[PRTS] 参数统计 — 总计: {total/1e6:.1f}M | "
            f"可训练: {trainable/1e6:.1f}M | 冻结: {frozen/1e6:.1f}M"
        )

    # =============================================================
    #  参数分组（不同学习率）
    # =============================================================
    def get_param_groups(
        self,
        lr: float = 3e-4,
        image_encoder_lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """
        返回参数分组:
          - group 0: image_encoder（fine-tune，小学习率）
          - group 1: image_projection + text_projection + logit_scale（正常学习率）
        冻结参数自动跳过（requires_grad=False 不加入任何组）
        """

        # image_encoder 参数（仅可训练的）
        image_encoder_params = [
            p for p in self.image_encoder.parameters() if p.requires_grad
        ]

        # CLIP 投影层 + logit_scale 参数
        clip_head_params = []
        for module in [self.image_projection, self.text_projection]:
            clip_head_params.extend(
                [p for p in module.parameters() if p.requires_grad]
            )
        if self.logit_scale.requires_grad:
            clip_head_params.append(self.logit_scale)

        # 用 id 集合做校验，确保无遗漏无重复
        grouped_ids = {id(p) for p in image_encoder_params + clip_head_params}
        all_trainable = [p for p in self.parameters() if p.requires_grad]

        orphan_params = [p for p in all_trainable if id(p) not in grouped_ids]
        if orphan_params:
            logger.warning(
                f"[PRTS] 发现 {len(orphan_params)} 个未分组的可训练参数，"
                f"已归入 clip_head 组"
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

        # 日志
        for g in param_groups:
            cnt = sum(p.numel() for p in g["params"])
            logger.info(
                f"[PRTS] 参数组 [{g['name']}] "
                f"{cnt/1e6:.1f}M 参数, lr={g['lr']:.2e}"
            )

        return param_groups

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    # =============================================================
    #  前向
    # =============================================================
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] → [B, embed_dim]"""
        feat = self.image_encoder(images)
        return F.normalize(self.image_projection(feat), dim=-1)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """input_ids: [B, 77], attention_mask: [B, 77] → [B, embed_dim]"""
        # text_encoder 已冻结时仍需前向传播，但不计算梯度
        if self.freeze_text_encoder:
            with torch.no_grad():
                feat = self.text_encoder(input_ids, attention_mask)
            # 从 no_grad 出来后，feat 是 detached 的
            # text_projection 仍可训练，梯度从 projection 开始回传
        else:
            feat = self.text_encoder(input_ids, attention_mask)

        return F.normalize(self.text_projection(feat), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回对称 InfoNCE loss (scalar)"""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return clip_info_nce_loss(image_embeds, text_embeds, self.logit_scale)
