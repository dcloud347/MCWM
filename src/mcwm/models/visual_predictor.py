"""根据可见视频 token，预测被 mask 位置的 encoder 特征。

Predictor 不直接处理像素。它接收 online encoder 输出的可见 token，在目标位置
放入可学习的 mask token，再通过多层 Transformer 推断目标位置应有的特征。
训练时，这些预测会与 EMA target encoder 在相同位置的特征计算 L1 loss。
"""

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
    """Predictor 的维度、层数和完整视频 token 网格。"""

    # Online encoder 输出的特征维度。正式配置中 encoder_dim=1024。
    input_dim: int = 768
    # Predictor 内部使用更小的维度以减少计算量。正式配置为 384。
    dim: int = 384
    # Transformer block 数量。
    depth: int = 12
    # 每个 Transformer block 的注意力头数量。
    heads: int = 12
    # Transformer 前馈网络中间层的维度。
    mlp_dim: int = 1536
    # 完整 token 网格的 (时间, 行, 列)。正式配置为 8×18×32。
    token_grid_size: Tuple[int, int, int] = (8, 18, 32)
    # 每套 mask 使用一个独立的可学习 token；正式配置有两套 mask。
    num_mask_tokens: int = 2
    # Attention 和 MLP 的 dropout 比例。
    dropout: float = 0.0
    # 是否使用时间、高度和宽度三个方向的旋转位置编码。
    use_rope: bool = True
    # 反向传播时重新计算中间结果，以计算时间换取更低显存占用。
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
        """返回完整视频片段的 token 总数，正式配置为 8×18×32=4608。"""

        return math.prod(self.token_grid_size)


class VisualPredictor(nn.Module):
    """用可见位置的 encoder 特征预测目标位置的 encoder 特征。

    输入只包含两类位置：online encoder 看见的 context 位置，以及需要预测的
    target 位置。Context 使用真实 encoder 特征，target 先使用可学习的 mask
    token 占位。二者按原视频位置排列后一起进入 Transformer，因此每个目标位置
    都能读取全部可见上下文以及其他目标位置的信息。
    """

    def __init__(self, config: VisualPredictorConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder 特征通常比 predictor 宽。先从 input_dim 投影到较小的 dim，
        # 后续 12 层 Transformer 都在较小维度中计算，以减少显存和算力开销。
        self.input_projection = nn.Linear(config.input_dim, config.dim)
        # 每套 mask 使用不同的可学习占位符。例如小块 mask 和大块 mask 可以
        # 学到不同的任务提示。形状 [1, 1, dim] 会在 forward 中扩展到整个 batch。
        self.mask_tokens = nn.ParameterList(
            nn.Parameter(torch.zeros(1, 1, config.dim))
            for _ in range(config.num_mask_tokens)
        )
        # RoPE 需要知道完整视频网格，才能把展平位置还原成时间、行和列坐标。
        rope_grid = config.token_grid_size if config.use_rope else None
        # 所有 mask 组共享同一组 Transformer 权重，只有 mask token 不同。
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
        # Transformer 输出先归一化，再投影回 encoder 特征维度，才能和 EMA
        # target encoder 的输出直接计算特征预测误差。
        self.norm = nn.LayerNorm(config.dim, eps=1e-6)
        self.output_projection = nn.Linear(config.dim, config.input_dim)
        if not config.use_rope:
            raise ValueError("the V-JEPA 2 predictor requires 3D RoPE")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """按 ViT 常用方式初始化参数，并稳定深层残差分支。"""

        # 线性层使用较小的截断正态分布；LayerNorm 初始时保持输入不变。
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
        # 越深的 residual 分支输出越小，避免多层残差相加后数值快速放大。
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
        """根据可见 token 预测目标位置的特征。

        参数形状：

        - ``context_tokens``: ``[B, K, input_dim]``，online encoder 输出的 K 个
          可见 token。
        - ``context_indices``: ``[B, K]``，每个可见 token 在完整视频网格中的位置。
        - ``target_indices``: ``[B, P]``，需要预测的 P 个位置。
        - ``mask_index``: 当前使用第几套 mask，用来选择对应的可学习 mask token。

        返回 ``[B, P, input_dim]``，顺序与 ``target_indices`` 完全一致。
        ``K + P`` 可能小于完整网格的 4608，因为组成 batch 时会把不同样本
        裁成相同长度；不参与当前任务的位置不会送入 predictor。
        """

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

        # [B, K, input_dim] -> [B, K, dim]。正式配置是 1024 -> 384。
        context = self.input_projection(context_tokens)
        # 同一 mask 组的所有 target 位置先放入相同的可学习占位符。expand 不会
        # 为每个位置复制一份参数，但这些位置会因 RoPE 和上下文不同产生不同输出。
        token = self.mask_tokens[mask_index % len(self.mask_tokens)].to(dtype=context.dtype)
        targets = token.expand(batch, target_count, -1)
        # 暂时按 [全部 context, 全部 target] 拼接，形状为 [B, K+P, dim]。
        tokens = torch.cat((context, targets), dim=1)
        # token 和位置必须使用完全相同的拼接顺序。位置编号采用展平的视频网格：
        # position = time * (rows * columns) + row * columns + column。
        position_ids = torch.cat(
            (
                context_indices.to(device=tokens.device, dtype=torch.long),
                target_indices.to(device=tokens.device, dtype=torch.long),
            ),
            dim=1,
        )
        # context_indices 和 target_indices 各自有序，但拼接后不是完整视频顺序。
        # 这里把 token 按原视频位置重新排序，例如 [0, 2, 1, 3] -> [0, 1, 2, 3]。
        # Transformer 使用非因果全局注意力，因此 context 和 target 可以互相读取；
        # 3D RoPE 则根据 position_ids 告诉模型每个 token 的时间、行和列位置。
        order = torch.argsort(position_ids, dim=1)
        position_ids = position_ids.gather(1, order)
        tokens = tokens.gather(1, order.unsqueeze(-1).expand_as(tokens))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # 不保存 block 内部激活，反向传播时重新计算，从而降低训练显存。
                tokens = checkpoint(block, tokens, position_ids, use_reentrant=False)
            else:
                tokens = block(tokens, position_ids)
        # [B, K+P, dim] -> [B, K+P, input_dim]，回到 EMA target 的特征维度。
        tokens = self.output_projection(self.norm(tokens))
        # Transformer 前做过位置排序。现在恢复最初的 [context, target] 顺序，
        # 这样最后 K 之后的 P 项就严格对应调用方传入的 target_indices。
        reverse_order = torch.argsort(order, dim=1)
        tokens = tokens.gather(1, reverse_order.unsqueeze(-1).expand_as(tokens))
        # Context 输出只参与内部推理，不计算预测 loss；只返回 P 个目标位置。
        return tokens[:, context_count:]
