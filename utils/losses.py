# utils/losses.py

import torch
import torch.nn.functional as F


def clip_info_nce_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    对称 InfoNCE 损失
    image_embeds: [B, D] (已 L2 归一化)
    text_embeds:  [B, D] (已 L2 归一化)
    logit_scale:  scalar parameter (log-scale)
    return:       scalar loss
    """
    scale = logit_scale.exp().clamp(max=100.0)

    # 相似度矩阵 [B, B]
    logits_per_image = scale * (image_embeds @ text_embeds.t())
    logits_per_text = logits_per_image.t()

    # 正样本在对角线
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2.0