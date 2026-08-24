"""模型共用的 Transformer 基础模块，attention 使用 PyTorch 原生 SDPA。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Attention(nn.Module):
    """多头自注意力。

    在 CUDA、数据类型和 tensor 布局都满足条件时，PyTorch 会自动选择
    Flash Attention；不满足时会自动回退，不需要我们维护两套实现。
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("attention dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.output = nn.Linear(dim, dim, bias=bias)

    def forward(self, inputs: Tensor, *, is_causal: bool = False) -> Tensor:
        batch, tokens, dim = inputs.shape
        # 先得到 Q/K/V，再拆成 [batch, heads, tokens, head_dim]。
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.output(attended)


class TransformerBlock(nn.Module):
    """encoder 和 predictor 共用的 Pre-LN Transformer block。"""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))
