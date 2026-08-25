"""模型共用的 Transformer 基础模块，attention 使用 PyTorch 原生 SDPA。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _rotate_half(values: Tensor) -> Tensor:
    """把最后一维相邻的实部/虚部旋转九十度。"""

    paired = values.unflatten(-1, (-1, 2))
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_rotary(values: Tensor, positions: Tensor) -> Tensor:
    """对一段偶数维 head channel 应用一维 RoPE。"""

    channels = values.shape[-1]
    if channels == 0:
        return values
    frequencies = torch.arange(
        channels // 2,
        dtype=values.dtype,
        device=values.device,
    )
    frequencies = 1.0 / (10000.0 ** (frequencies / (channels / 2)))
    angles = positions.to(dtype=values.dtype).unsqueeze(-1) * frequencies
    # V-JEPA 2 原版把整段频率复制两次，而不是逐元素 repeat_interleave。
    # 这个排列看起来反直觉，但属于官方 checkpoint 的 RoPE 约定。
    cosine = angles.cos().repeat(1, 1, 2).unsqueeze(1)
    sine = angles.sin().repeat(1, 1, 2).unsqueeze(1)
    return values * cosine + _rotate_half(values) * sine


def _apply_3d_rope(
    query: Tensor,
    key: Tensor,
    position_ids: Tensor,
    grid_size: Tuple[int, int, int],
) -> Tuple[Tensor, Tensor]:
    """按官方 V-JEPA 2 的 channel 划分分别旋转时间、高度和宽度。"""

    _, _, _, head_dim = query.shape
    axis_dim = 2 * ((head_dim // 3) // 2)
    if axis_dim == 0:
        return query, key
    _, rows, columns = grid_size
    spatial_tokens = rows * columns
    temporal = position_ids // spatial_tokens
    spatial = position_ids % spatial_tokens
    height = spatial // columns
    width = spatial % columns
    coordinates = (temporal, height, width)
    query_parts = []
    key_parts = []
    start = 0
    for coordinate in coordinates:
        end = start + axis_dim
        query_parts.append(_apply_rotary(query[..., start:end], coordinate))
        key_parts.append(_apply_rotary(key[..., start:end], coordinate))
        start = end
    if start < head_dim:
        query_parts.append(query[..., start:])
        key_parts.append(key[..., start:])
    return torch.cat(query_parts, dim=-1), torch.cat(key_parts, dim=-1)


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
        rope_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("attention dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.rope_grid_size = rope_grid_size
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.output = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        inputs: Tensor,
        position_ids: Optional[Tensor] = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        batch, tokens, dim = inputs.shape
        # 先得到 Q/K/V，再拆成 [batch, heads, tokens, head_dim]。
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.rope_grid_size is not None:
            total_positions = math.prod(self.rope_grid_size)
            if position_ids is None:
                if tokens != total_positions:
                    raise ValueError("masked RoPE attention requires explicit position_ids")
                position_ids = torch.arange(tokens, device=inputs.device).expand(batch, -1)
            if position_ids.shape != (batch, tokens):
                raise ValueError("position_ids must have shape [B, N]")
            query, key = _apply_3d_rope(
                query,
                key,
                position_ids,
                self.rope_grid_size,
            )
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
        rope_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = Attention(
            dim,
            heads,
            dropout=dropout,
            rope_grid_size=rope_grid_size,
        )
        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        inputs = inputs + self.attention(
            self.attention_norm(inputs),
            position_ids,
        )
        return inputs + self.mlp(self.mlp_norm(inputs))
