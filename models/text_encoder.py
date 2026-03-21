# models/text_encoder.py

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    pe: torch.Tensor
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, d_model]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Transformer 文本编码器
    输入:  input_ids [B, seq_len], attention_mask [B, seq_len]
    输出:  text_features [B, hidden_dim]
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 77,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids:      [B, seq_len]
        attention_mask:  [B, seq_len]  (1=有效, 0=padding)
        return:          [B, hidden_dim]
        """
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        # Transformer 需要 True=屏蔽
        key_padding_mask = attention_mask == 0  # [B, seq_len]

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)

        # EOS pooling: 取每个样本最后一个有效 token
        eos_indices = attention_mask.sum(dim=1).long() - 1  # [B]
        eos_indices = eos_indices.clamp(min=0)
        batch_indices = torch.arange(x.size(0), device=x.device)
        text_features = x[batch_indices, eos_indices]  # [B, hidden_dim]

        return text_features
