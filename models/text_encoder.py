# models/text_encoder.py

import torch
import torch.nn as nn
from transformers import CLIPTextModel


class TextEncoder(nn.Module):
    """
    预训练 CLIP 文本编码器包装。

    输入:
      input_ids: [B, seq_len]
      attention_mask: [B, seq_len]

    输出:
      text_features: [B, hidden_dim]
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.pooler_output
