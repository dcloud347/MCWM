"""M1 专用的 factorized 空间—时间 latent predictor。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import TransformerBlock


@dataclass(frozen=True)
class VisualPredictorConfig:
    """predictor 配置；input_dim 同 visual encoder 的输出维度。"""

    input_dim: int = 768
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_dim: int = 1536
    max_frames: int = 16
    patch_count: int = 576
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.heads:
            raise ValueError("predictor dim must be divisible by heads")
        if min(self.depth, self.max_frames, self.patch_count) <= 0:
            raise ValueError("predictor dimensions must be positive")


class FactorizedBlock(nn.Module):
    """先在每帧内部做空间 attention，再沿同一 patch 位置做时间 attention。"""

    def __init__(self, config: VisualPredictorConfig) -> None:
        super().__init__()
        self.spatial = TransformerBlock(
            config.dim, config.heads, config.mlp_dim, dropout=config.dropout
        )
        self.temporal = TransformerBlock(
            config.dim, config.heads, config.mlp_dim, dropout=config.dropout
        )

    def forward(self, tokens: Tensor) -> Tensor:
        batch, frames, patches, dim = tokens.shape
        # 把每一帧当成独立样本，attention 长度只有 patch_count。
        spatial = tokens.reshape(batch * frames, patches, dim)
        spatial = self.spatial(spatial).reshape(batch, frames, patches, dim)
        # 再把同一空间位置的 T 个 token 放在一起，attention 长度只有 frames。
        temporal = spatial.permute(0, 2, 1, 3).reshape(batch * patches, frames, dim)
        temporal = self.temporal(temporal)
        return temporal.reshape(batch, patches, frames, dim).permute(0, 2, 1, 3)


class VisualPredictor(nn.Module):
    """根据可见 context token 预测所有被 mask 位置的 target latent。"""

    def __init__(self, config: VisualPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.dim))
        self.spatial_position = nn.Parameter(torch.zeros(1, 1, config.patch_count, config.dim))
        self.temporal_position = nn.Parameter(torch.zeros(1, config.max_frames, 1, config.dim))
        self.blocks = nn.ModuleList(FactorizedBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.dim)
        self.output_projection = nn.Linear(config.dim, config.input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)

    def forward(self, context_tokens: Tensor, target_mask: Tensor) -> Tensor:
        if context_tokens.ndim != 4:
            raise ValueError("context_tokens must have shape [B, T, N, D]")
        batch, frames, patches, _ = context_tokens.shape
        if target_mask.shape != (batch, frames, patches):
            raise ValueError("target_mask must have shape [B, T, N]")
        if frames > self.config.max_frames or patches != self.config.patch_count:
            raise ValueError("clip dimensions do not match predictor config")

        tokens = self.input_projection(context_tokens)
        # target 位置再次替换成 predictor 自己的 mask token，杜绝内容泄漏。
        tokens = torch.where(
            target_mask.unsqueeze(-1),
            self.mask_token.to(dtype=tokens.dtype).expand_as(tokens),
            tokens,
        )
        tokens = tokens + self.spatial_position.to(dtype=tokens.dtype)
        tokens = tokens + self.temporal_position[:, :frames].to(dtype=tokens.dtype)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.output_projection(self.norm(tokens))
