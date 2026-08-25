"""V-JEPA 2 风格的 tubelet Video ViT visual encoder。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import TransformerBlock


@dataclass(frozen=True)
class VisualEncoderConfig:
    """Video ViT 配置；正式模型固定使用 360×640 和 20×20 spatial patch。"""

    image_height: int = 360
    image_width: int = 640
    patch_size: int = 20
    clip_frames: int = 16
    tubelet_size: int = 2
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0
    use_rope: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if self.clip_frames % self.tubelet_size:
            raise ValueError("clip_frames must be divisible by tubelet_size")
        if self.dim % self.heads:
            raise ValueError("encoder dim must be divisible by heads")
        if min(
            self.depth,
            self.patch_size,
            self.clip_frames,
            self.tubelet_size,
            self.dim,
        ) <= 0:
            raise ValueError("encoder dimensions must be positive")

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Spatial tubelet 网格的 (行数, 列数)，正式配置为 (18, 32)。"""

        return self.image_height // self.patch_size, self.image_width // self.patch_size

    @property
    def temporal_grid_size(self) -> int:
        """正式16帧、tubelet size 2 对应8个时间位置。"""

        return self.clip_frames // self.tubelet_size

    @property
    def token_grid_size(self) -> Tuple[int, int, int]:
        rows, columns = self.grid_size
        return self.temporal_grid_size, rows, columns

    @property
    def patch_count(self) -> int:
        """每个时间 tubelet 的空间 token 数，正式配置为576。"""

        rows, columns = self.grid_size
        return rows * columns

    @property
    def token_count(self) -> int:
        """整个 clip 的时空 token 数，正式配置为8×576=4608。"""

        return math.prod(self.token_grid_size)


class VisualEncoder(nn.Module):
    """用非重叠 Conv3d tubelet 和联合时空 self-attention 编码完整 clip。"""

    def __init__(self, config: VisualEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv3d(
            3,
            config.dim,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )
        rope_grid = config.token_grid_size if config.use_rope else None
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.dim,
                config.heads,
                config.mlp_dim,
                dropout=config.dropout,
                rope_grid_size=rope_grid,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.dim, eps=1e-6)
        if not config.use_rope:
            raise ValueError("the V-JEPA 2 video encoder requires 3D RoPE")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用官方 ViT 初始化并按层深缩放 residual 输出投影。"""

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for layer_index, block in enumerate(self.blocks, start=1):
            scale = math.sqrt(2.0 * layer_index)
            block.attention.output.weight.data.div_(scale)
            block.mlp[3].weight.data.div_(scale)

    def _tokenize(self, clips: Tensor) -> Tensor:
        if clips.ndim != 5:
            raise ValueError("clips must have shape [B, T, 3, H, W]")
        expected = (
            self.config.clip_frames,
            3,
            self.config.image_height,
            self.config.image_width,
        )
        if tuple(clips.shape[1:]) != expected:
            raise ValueError(
                "clips must have shape "
                f"[B, {expected[0]}, {expected[1]}, {expected[2]}, {expected[3]}]"
            )
        # Conv3d 接受 [B, C, T, H, W]，输出按 T/H/W 顺序展平成 video tokens。
        return self.patch_embedding(clips.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)

    def forward_features(
        self,
        clips: Tensor,
        token_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """编码完整 token 序列，或只编码 ``token_indices`` 指定的 context。"""

        tokens = self._tokenize(clips)
        batch = tokens.shape[0]
        if token_indices is None:
            position_ids = torch.arange(
                self.config.token_count,
                device=tokens.device,
            ).expand(batch, -1)
        else:
            if token_indices.ndim != 2 or token_indices.shape[0] != batch:
                raise ValueError("token_indices must have shape [B, K]")
            position_ids = token_indices.to(device=tokens.device, dtype=torch.long)
            tokens = tokens.gather(
                1,
                position_ids.unsqueeze(-1).expand(-1, -1, self.config.dim),
            )
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, position_ids, use_reentrant=False)
            else:
                tokens = block(tokens, position_ids)
        return self.norm(tokens)

    def forward(
        self,
        clips: Tensor,
        token_indices: Optional[Tensor] = None,
        *,
        return_patch_tokens: bool = False,
    ) -> Tensor:
        tokens = self.forward_features(clips, token_indices)
        return tokens if return_patch_tokens else tokens.mean(dim=1)
