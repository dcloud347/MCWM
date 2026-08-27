"""把视频切成时空 patch，再用 Video ViT 编码。"""

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
    """Video ViT 的输入大小和网络结构配置。

    ``clip_frames`` 是 encoder 支持的最大帧数，也是视觉预训练使用的
    clip 长度。运行时可以输入不超过该值的更短 clip，但帧数必须
    能被 ``tubelet_size`` 整除。
    """

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
        """每个时间片的 patch 行数和列数，正式配置为 (18, 32)。"""

        return self.image_height // self.patch_size, self.image_width // self.patch_size

    @property
    def temporal_grid_size(self) -> int:
        """时间方向的 token 数；16 帧每 2 帧一组时结果是 8。"""

        return self.clip_frames // self.tubelet_size

    @property
    def token_grid_size(self) -> Tuple[int, int, int]:
        """返回 token 网格的时间、行和列。"""

        rows, columns = self.grid_size
        return self.temporal_grid_size, rows, columns

    @property
    def patch_count(self) -> int:
        """每个时间位置的 patch 数，正式配置是 18×32=576。"""

        rows, columns = self.grid_size
        return rows * columns

    @property
    def token_count(self) -> int:
        """整个视频片段的 token 数，正式配置是 8×576=4608。"""

        return math.prod(self.token_grid_size)

    def runtime_token_grid_size(self, frame_count: int) -> Tuple[int, int, int]:
        """返回一个合法运行时 clip 的实际 token 网格。"""

        frames = int(frame_count)
        if frames < self.tubelet_size:
            raise ValueError(
                f"clip must contain at least {self.tubelet_size} frames"
            )
        if frames > self.clip_frames:
            raise ValueError(
                f"clip cannot exceed configured maximum of {self.clip_frames} frames"
            )
        if frames % self.tubelet_size:
            raise ValueError("runtime frame count must be divisible by tubelet_size")
        rows, columns = self.grid_size
        return frames // self.tubelet_size, rows, columns

    def runtime_token_count(self, frame_count: int) -> int:
        """返回一个合法运行时 clip 的实际 token 数。"""

        return math.prod(self.runtime_token_grid_size(frame_count))


class VisualEncoder(nn.Module):
    """切分不重叠的时空 patch，并用注意力层编码。"""

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
        """初始化模型参数，并缩小较深层的残差输出。"""

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
        frame_count, channels, height, width = clips.shape[1:]
        if (
            channels != 3
            or height != self.config.image_height
            or width != self.config.image_width
        ):
            raise ValueError(
                "clips must have shape [B, T, 3, "
                f"{self.config.image_height}, {self.config.image_width}]"
            )
        self.config.runtime_token_grid_size(frame_count)
        # Conv3d 输入顺序是 [batch, 通道, 时间, 高, 宽]。
        # 输出按时间、行、列的顺序展平，必须与 mask 的 token 编号保持一致。
        return self.patch_embedding(clips.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)

    def forward_features(
        self,
        clips: Tensor,
        token_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """编码全部 token，或只编码 ``token_indices`` 指定的可见 token。"""

        tokens = self._tokenize(clips)
        batch = tokens.shape[0]
        runtime_token_count = tokens.shape[1]
        if token_indices is None:
            position_ids = torch.arange(
                runtime_token_count,
                device=tokens.device,
            ).expand(batch, -1)
        else:
            if token_indices.ndim != 2 or token_indices.shape[0] != batch:
                raise ValueError("token_indices must have shape [B, K]")
            position_ids = token_indices.to(device=tokens.device, dtype=torch.long)
            if position_ids.numel() and (
                position_ids.min().item() < 0
                or position_ids.max().item() >= runtime_token_count
            ):
                raise ValueError("token_indices contain positions outside the runtime clip")
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
        """返回每个 patch 的特征，或返回所有 patch 的平均特征。"""

        tokens = self.forward_features(clips, token_indices)
        return tokens if return_patch_tokens else tokens.mean(dim=1)
