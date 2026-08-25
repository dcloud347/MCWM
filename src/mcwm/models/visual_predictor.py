"""V-JEPA 2 风格的联合时空 latent predictor。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import TransformerBlock


@dataclass(frozen=True)
class VisualPredictorConfig:
    """Predictor 使用较窄的维度，但保留完整的联合时空 attention。"""

    input_dim: int = 768
    dim: int = 384
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 1536
    token_grid_size: Tuple[int, int, int] = (8, 18, 32)
    num_mask_tokens: int = 2
    dropout: float = 0.0
    use_rope: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.heads:
            raise ValueError("predictor dim must be divisible by heads")
        if min(
            self.depth,
            self.input_dim,
            self.dim,
            self.mlp_dim,
            self.num_mask_tokens,
            *self.token_grid_size,
        ) <= 0:
            raise ValueError("predictor dimensions must be positive")

    @property
    def token_count(self) -> int:
        return math.prod(self.token_grid_size)


class VisualPredictor(nn.Module):
    """把可见 context 和目标位置 mask token 合并后联合建模全部时空位置。"""

    def __init__(self, config: VisualPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.dim)
        self.mask_tokens = nn.ParameterList(
            nn.Parameter(torch.zeros(1, 1, config.dim))
            for _ in range(config.num_mask_tokens)
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
        self.output_projection = nn.Linear(config.dim, config.input_dim)
        if not config.use_rope:
            raise ValueError("the V-JEPA 2 predictor requires 3D RoPE")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """官方 predictor 使用 trunc-normal 线性层和零初始化 mask token。"""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for token in self.mask_tokens:
            nn.init.zeros_(token)
        for layer_index, block in enumerate(self.blocks, start=1):
            scale = math.sqrt(2.0 * layer_index)
            block.attention.output.weight.data.div_(scale)
            block.mlp[3].weight.data.div_(scale)

    def forward(
        self,
        context_tokens: Tensor,
        context_indices: Tensor,
        target_indices: Tensor,
        *,
        mask_index: int,
    ) -> Tensor:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [B, K, D]")
        batch, context_count, input_dim = context_tokens.shape
        if input_dim != self.config.input_dim:
            raise ValueError("context token dimension does not match predictor input_dim")
        if context_indices.shape != (batch, context_count):
            raise ValueError("context_indices must have shape [B, K]")
        if target_indices.ndim != 2 or target_indices.shape[0] != batch:
            raise ValueError("target_indices must have shape [B, P]")
        target_count = target_indices.shape[1]
        if min(context_count, target_count) <= 0:
            raise ValueError("predictor requires non-empty context and target tokens")

        context = self.input_projection(context_tokens)
        token = self.mask_tokens[mask_index % len(self.mask_tokens)].to(dtype=context.dtype)
        targets = token.expand(batch, target_count, -1)
        tokens = torch.cat((context, targets), dim=1)
        position_ids = torch.cat(
            (
                context_indices.to(device=tokens.device, dtype=torch.long),
                target_indices.to(device=tokens.device, dtype=torch.long),
            ),
            dim=1,
        )
        # 官方 predictor 按原视频位置排序后做联合 attention，结束后恢复原拼接顺序。
        order = torch.argsort(position_ids, dim=1)
        position_ids = position_ids.gather(1, order)
        tokens = tokens.gather(1, order.unsqueeze(-1).expand_as(tokens))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, position_ids, use_reentrant=False)
            else:
                tokens = block(tokens, position_ids)
        tokens = self.output_projection(self.norm(tokens))
        reverse_order = torch.argsort(order, dim=1)
        tokens = tokens.gather(1, reverse_order.unsqueeze(-1).expand_as(tokens))
        return tokens[:, context_count:]
