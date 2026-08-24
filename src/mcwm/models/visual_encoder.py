"""逐帧工作的 2D ViT；M1 预训练和后续 world model 共用这个结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import TransformerBlock


@dataclass(frozen=True)
class VisualEncoderConfig:
    """Visual Encoder 的结构配置；正式配置对应 640×360 的 ViT-Base。"""

    image_height: int = 360
    image_width: int = 640
    patch_size: int = 20
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if self.dim % self.heads:
            raise ValueError("encoder dim must be divisible by heads")
        if min(self.depth, self.patch_size, self.dim) <= 0:
            raise ValueError("encoder dimensions must be positive")

    @property
    def grid_size(self) -> Tuple[int, int]:
        """patch 网格的 (行数, 列数)，正式配置为 (18, 32)。"""

        return self.image_height // self.patch_size, self.image_width // self.patch_size

    @property
    def patch_count(self) -> int:
        """单帧 patch token 数，正式配置为 576。"""

        rows, columns = self.grid_size
        return rows * columns


class VisualEncoder(nn.Module):
    """把一帧 Minecraft RGB 图像编码成 CLS latent 和 patch latents。

    M1 中 ``patch_mask=True`` 表示该 patch 对 online encoder 不可见，会被一个
    不含图像内容的可学习 mask token 替换。EMA target encoder 始终看完整图像。
    """

    def __init__(self, config: VisualEncoderConfig) -> None:
        super().__init__()
        self.config = config
        # kernel_size 和 stride 相同，所以 patch 互不重叠，也不需要额外 crop。
        self.patch_embedding = nn.Conv2d(
            3,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.position = nn.Parameter(torch.zeros(1, config.patch_count + 1, config.dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.dim,
                config.heads,
                config.mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_features(
        self,
        images: Tensor,
        patch_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W]")
        expected = (3, self.config.image_height, self.config.image_width)
        if tuple(images.shape[1:]) != expected:
            raise ValueError(f"images must have shape [B, {expected[0]}, {expected[1]}, {expected[2]}]")

        # [B, 3, H, W] -> [B, patch_count, dim]
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        if patch_mask is not None:
            if patch_mask.shape != patches.shape[:2]:
                raise ValueError("patch_mask must have shape [B, patch_count]")
            patches = torch.where(
                patch_mask.unsqueeze(-1),
                self.mask_token.to(dtype=patches.dtype).expand_as(patches),
                patches,
            )
        # CLS 放在所有 patch 前面，阶段 B 会把它作为单帧 state latent。
        cls = self.cls_token.to(dtype=patches.dtype).expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position.to(dtype=patches.dtype)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # 不保存 block 内部 activation，反向时重算，以显著降低显存。
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]

    def forward(
        self,
        images: Tensor,
        patch_mask: Optional[Tensor] = None,
        *,
        return_patch_tokens: bool = False,
    ) -> Tensor:
        pooled, patches = self.forward_features(images, patch_mask)
        return patches if return_patch_tokens else pooled
