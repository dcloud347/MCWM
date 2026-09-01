"""Frame/block-causal action-conditioned latent predictor。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import _apply_3d_rope


@dataclass(frozen=True)
class ACPredictorConfig:
    """Action-conditioned predictor 的结构配置。"""

    latent_dim: int = 1024
    action_dim: int = 1024
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    mlp_dim: int = 4096
    context_blocks: int = 16
    spatial_grid: Tuple[int, int] = (18, 32)
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.latent_dim,
            self.action_dim,
            self.dim,
            self.depth,
            self.heads,
            self.mlp_dim,
            self.context_blocks,
            *self.spatial_grid,
        )
        if min(dimensions) <= 0:
            raise ValueError("predictor dimensions must be positive")
        if self.dim % self.heads:
            raise ValueError("predictor dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def spatial_tokens(self) -> int:
        """返回每帧包含的 spatial token 数。"""

        rows, columns = self.spatial_grid
        return rows * columns

    @property
    def block_size(self) -> int:
        """一个 block 包含一个 action token 和全部 visual tokens。"""

        return self.spatial_tokens + 1


def block_causal_attention_mask(
    blocks: int,
    block_size: int,
    *,
    device: torch.device,
) -> Tensor:
    """返回 mask；同 block 和过去 block 可见，未来 block 不可见。"""

    if min(blocks, block_size) <= 0:
        raise ValueError("blocks and block_size must be positive")
    block_ids = torch.arange(blocks, device=device).repeat_interleave(block_size)
    # 行是 query，列是 key。True 表示这个 key 可以参与 attention。
    return block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)


class BlockCausalAttention(nn.Module):
    """使用 3D RoPE 的多头 block-causal self-attention。"""

    def __init__(self, dim: int, heads: int, *, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        inputs: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        rope_grid_size: Tuple[int, int, int],
    ) -> Tensor:
        """计算 block-causal attention。"""

        batch, tokens, dim = inputs.shape
        if position_ids.shape != (batch, tokens):
            raise ValueError("position_ids must have shape [B, N]")
        if attention_mask.shape != (tokens, tokens):
            raise ValueError("attention_mask must have shape [N, N]")

        qkv = self.qkv(inputs).reshape(
            batch,
            tokens,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_3d_rope(
            query,
            key,
            position_ids,
            rope_grid_size,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask.view(1, 1, tokens, tokens),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.output(attended)

    def forward_with_cache(
        self,
        inputs: Tensor,
        position_ids: Tensor,
        rope_grid_size: Tuple[int, int, int],
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Attend to the current block and optionally cached past blocks."""

        batch, tokens, dim = inputs.shape
        if position_ids.shape != (batch, tokens):
            raise ValueError("position_ids must have shape [B, N]")
        if past_key_value is not None and attention_mask is not None:
            raise ValueError("cached attention cannot use an attention mask")
        if attention_mask is not None and attention_mask.shape != (tokens, tokens):
            raise ValueError("attention_mask must have shape [N, N]")

        qkv = self.qkv(inputs).reshape(
            batch, tokens, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_3d_rope(
            query, key, position_ids, rope_grid_size
        )
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=(
                None
                if attention_mask is None
                else attention_mask.view(1, 1, tokens, tokens)
            ),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.output(attended), (key, value)


class BlockCausalTransformerBlock(nn.Module):
    """一个 pre-norm block-causal Transformer 层。"""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = BlockCausalAttention(dim, heads, dropout=dropout)
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
        position_ids: Tensor,
        attention_mask: Tensor,
        rope_grid_size: Tuple[int, int, int],
    ) -> Tensor:
        """依次计算 attention 和 MLP。"""

        inputs = inputs + self.attention(
            self.attention_norm(inputs),
            position_ids,
            attention_mask,
            rope_grid_size,
        )
        return inputs + self.mlp(self.mlp_norm(inputs))

    def forward_with_cache(
        self,
        inputs: Tensor,
        position_ids: Tensor,
        rope_grid_size: Tuple[int, int, int],
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Run one block while returning its attention K/V cache."""

        attended, key_value = self.attention.forward_with_cache(
            self.attention_norm(inputs),
            position_ids,
            rope_grid_size,
            past_key_value,
            attention_mask,
        )
        inputs = inputs + attended
        return inputs + self.mlp(self.mlp_norm(inputs)), key_value


class ActionConditionedPredictor(nn.Module):
    """根据当前 visual latent 和动作预测下一帧 visual latent。"""

    def __init__(self, config: ACPredictorConfig = ACPredictorConfig()) -> None:
        super().__init__()
        self.config = config
        self.visual_projection = nn.Linear(config.latent_dim, config.dim)
        self.action_projection = nn.Linear(config.action_dim, config.dim)
        self.blocks = nn.ModuleList(
            BlockCausalTransformerBlock(
                config.dim,
                config.heads,
                config.mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.dim, eps=1e-6)
        self.output_projection = nn.Linear(config.dim, config.latent_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用与仓库其他 Transformer 一致的初始化。"""

        for module in self.modules():
            if isinstance(module, nn.Linear):
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

    def _validate_teacher_inputs(self, latents: Tensor, actions: Tensor) -> None:
        if latents.ndim != 4:
            raise ValueError("latents must have shape [B, T, S, D]")
        batch, blocks, spatial_tokens, latent_dim = latents.shape
        expected_latent = (self.config.spatial_tokens, self.config.latent_dim)
        if (spatial_tokens, latent_dim) != expected_latent:
            raise ValueError(
                "latents must have shape [B, T, "
                f"{self.config.spatial_tokens}, {self.config.latent_dim}]"
            )
        if blocks <= 0 or blocks > self.config.context_blocks:
            raise ValueError(
                f"T must be in [1, {self.config.context_blocks}]"
            )
        if actions.shape != (batch, blocks, self.config.action_dim):
            raise ValueError(
                "actions must have shape [B, T, "
                f"{self.config.action_dim}]"
            )

    def _position_ids(self, batch: int, blocks: int, device: torch.device) -> Tensor:
        """给 action 和 visual token 分配所在 frame 的 3D 位置。"""

        spatial = self.config.spatial_tokens
        frame_offsets = torch.arange(blocks, device=device) * spatial
        visual = frame_offsets.unsqueeze(1) + torch.arange(spatial, device=device)
        # action 没有空间坐标，使用所在 frame 的 (row=0, column=0)。
        action = frame_offsets.unsqueeze(1)
        positions = torch.cat((action, visual), dim=1).reshape(1, -1)
        return positions.expand(batch, -1)

    def _cached_step(
        self,
        latent: Tensor,
        action: Tensor,
        caches: List[Tuple[Tensor, Tensor]],
        block_index: int,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """Predict one block using K/V from all previously processed blocks."""

        spatial_tokens = self.config.spatial_tokens
        visual_tokens = self.visual_projection(latent)
        action_token = self.action_projection(action).unsqueeze(1)
        tokens = torch.cat((action_token, visual_tokens), dim=1)
        position_ids = (
            self._position_ids(latent.shape[0], 1, latent.device)
            + block_index * self.config.spatial_tokens
        )
        rows, columns = self.config.spatial_grid
        rope_grid_size = (self.config.context_blocks, rows, columns)
        next_caches: List[Tuple[Tensor, Tensor]] = []
        max_tokens = self.config.context_blocks * self.config.block_size
        for index, block in enumerate(self.blocks):
            tokens, key_value = block.forward_with_cache(
                tokens,
                position_ids,
                rope_grid_size,
                caches[index] if caches else None,
            )
            next_caches.append(
                (
                    key_value[0][:, :, -max_tokens:],
                    key_value[1][:, :, -max_tokens:],
                )
            )
        tokens = self.norm(tokens)
        prediction = self.output_projection(
            tokens[:, 1 : spatial_tokens + 1]
        )
        return prediction, next_caches

    def _cached_context(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """Process an initial context once and retain every layer's K/V."""

        self._validate_teacher_inputs(latents, actions)
        batch, block_count, spatial_tokens, _ = latents.shape
        visual_tokens = self.visual_projection(latents)
        action_tokens = self.action_projection(actions).unsqueeze(2)
        tokens = torch.cat((action_tokens, visual_tokens), dim=2).reshape(
            batch, block_count * self.config.block_size, -1
        )
        position_ids = self._position_ids(batch, block_count, latents.device)
        attention_mask = block_causal_attention_mask(
            block_count, self.config.block_size, device=latents.device
        )
        rows, columns = self.config.spatial_grid
        rope_grid_size = (block_count, rows, columns)
        caches: List[Tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            tokens, key_value = block.forward_with_cache(
                tokens,
                position_ids,
                rope_grid_size,
                attention_mask=attention_mask,
            )
            caches.append(key_value)
        tokens = self.output_projection(
            self.norm(tokens).reshape(
                batch, block_count, self.config.block_size, self.config.dim
            )[:, :, 1 : spatial_tokens + 1]
        )
        return tokens[:, -1], caches

    def _rollout_cached(
        self,
        initial_latent: Tensor,
        actions: Tensor,
    ) -> Tensor:
        states = F.layer_norm(initial_latent, (initial_latent.shape[-1],))
        caches: List[Tuple[Tensor, Tensor]] = []
        predictions = []
        for step in range(actions.shape[1]):
            next_latent, caches = self._cached_step(
                states, actions[:, step], caches, step
            )
            predictions.append(next_latent)
            states = F.layer_norm(next_latent, (next_latent.shape[-1],))
        return torch.stack(predictions, dim=1)

    def predict_teacher_forced(self, latents: Tensor, actions: Tensor) -> Tensor:
        """并行预测每个真实当前 latent 对应的下一帧 latent。"""

        self._validate_teacher_inputs(latents, actions)
        batch, block_count, spatial_tokens, _ = latents.shape

        visual_tokens = self.visual_projection(latents)
        action_tokens = self.action_projection(actions).unsqueeze(2)
        tokens = torch.cat((action_tokens, visual_tokens), dim=2)
        tokens = tokens.reshape(batch, block_count * self.config.block_size, -1)

        position_ids = self._position_ids(batch, block_count, latents.device)
        attention_mask = block_causal_attention_mask(
            block_count,
            self.config.block_size,
            device=latents.device,
        )
        rows, columns = self.config.spatial_grid
        rope_grid_size = (block_count, rows, columns)
        for block in self.blocks:
            if (
                self.config.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # 不保存本 block 的中间激活；反向传播时重新执行一次 block。
                tokens = checkpoint(
                    block,
                    tokens,
                    position_ids,
                    attention_mask,
                    rope_grid_size,
                    use_reentrant=False,
                )
            else:
                tokens = block(
                    tokens,
                    position_ids,
                    attention_mask,
                    rope_grid_size,
                )

        tokens = self.norm(tokens)
        tokens = tokens.reshape(
            batch,
            block_count,
            self.config.block_size,
            self.config.dim,
        )
        # 每个 block 的第 0 项是 action token，只返回 visual token 的预测。
        visual_predictions = tokens[:, :, 1 : spatial_tokens + 1]
        return self.output_projection(visual_predictions)

    def forward(self, latents: Tensor, actions: Tensor) -> Tensor:
        """等价于 teacher-forced 接口。"""

        return self.predict_teacher_forced(latents, actions)

    def rollout(self, initial_latent: Tensor, actions: Tensor) -> Tensor:
        """从真实初始 latent 开始，逐步反馈归一化后的预测结果。"""

        if actions.ndim != 3 or actions.shape[2] != self.config.action_dim:
            raise ValueError(
                "actions must have shape [B, H, "
                f"{self.config.action_dim}]"
            )
        expected_initial = (
            actions.shape[0],
            self.config.spatial_tokens,
            self.config.latent_dim,
        )
        if initial_latent.shape != expected_initial:
            raise ValueError(
                "initial_latent must have shape [B, "
                f"{self.config.spatial_tokens}, {self.config.latent_dim}]"
            )

        horizon = actions.shape[1]
        if horizon == 0:
            return initial_latent.unsqueeze(1)[:, :0]

        if (
            not self.training
            and not torch.is_grad_enabled()
            and horizon <= self.config.context_blocks
        ):
            return self._rollout_cached(initial_latent, actions)

        states = [
            F.layer_norm(
                initial_latent,
                (initial_latent.shape[-1],),
            )
        ]
        predictions = []
        for step in range(horizon):
            start = max(0, len(states) - self.config.context_blocks)
            context_latents = torch.stack(states[start:], dim=1)
            context_actions = actions[:, start : step + 1]
            next_latent = self.predict_teacher_forced(
                context_latents,
                context_actions,
            )[:, -1]
            predictions.append(next_latent)
            # V-JEPA 2-AC 在每一步预测后立即归一化，再把结果回灌到下一步。
            # predictions 保留归一化前的值，供尺度退化诊断使用；loss 会对其
            # 做同样的 LayerNorm，因此训练目标与官方实现等价。
            states.append(
                F.layer_norm(
                    next_latent,
                    (next_latent.shape[-1],),
                )
            )
        return torch.stack(predictions, dim=1)

    def rollout_with_context(
        self,
        context_latents: Tensor,
        context_actions: Tensor,
        future_actions: Tensor,
    ) -> Tensor:
        """Roll out future actions while retaining observed history.

        ``context_latents`` contains C observed states and ``context_actions``
        contains the C-1 actions between them. At every predicted step the new
        candidate action is appended and the most recent ``context_blocks`` are
        passed through the same block-causal predictor used during training.
        """

        if context_latents.ndim != 4 or context_latents.shape[1] <= 0:
            raise ValueError("context_latents must have shape [B, C, S, D]")
        batch, context_length, spatial, latent_dim = context_latents.shape
        expected_latent = (self.config.spatial_tokens, self.config.latent_dim)
        if (spatial, latent_dim) != expected_latent:
            raise ValueError("context_latents have incompatible spatial/feature dimensions")
        expected_history = (batch, context_length - 1, self.config.action_dim)
        if context_actions.shape != expected_history:
            raise ValueError(
                "context_actions must have shape [B, C-1, action_dim]"
            )
        if future_actions.ndim != 3 or future_actions.shape[:1] != (batch,):
            raise ValueError("future_actions must have shape [B, H, action_dim]")
        if future_actions.shape[2] != self.config.action_dim:
            raise ValueError("future_actions have an incompatible action dimension")
        if future_actions.shape[1] == 0:
            return context_latents[:, :0]

        if (
            not self.training
            and not torch.is_grad_enabled()
            and context_length + future_actions.shape[1]
            <= self.config.context_blocks
        ):
            normalized_context = F.layer_norm(
                context_latents, (context_latents.shape[-1],)
            )
            context_start = max(
                0, context_length - self.config.context_blocks
            )
            normalized_context = normalized_context[:, context_start:]
            first_actions = torch.cat(
                (context_actions[:, context_start:], future_actions[:, :1]),
                dim=1,
            )
            next_latent, caches = self._cached_context(
                normalized_context, first_actions
            )
            predictions = [next_latent]
            state = F.layer_norm(next_latent, (next_latent.shape[-1],))
            for step in range(1, future_actions.shape[1]):
                next_latent, caches = self._cached_step(
                    state,
                    future_actions[:, step],
                    caches,
                    normalized_context.shape[1] - 1 + step,
                )
                predictions.append(next_latent)
                state = F.layer_norm(next_latent, (next_latent.shape[-1],))
            return torch.stack(predictions, dim=1)

        states = [
            F.layer_norm(value, (value.shape[-1],))
            for value in context_latents.unbind(dim=1)
        ]
        completed_actions = list(context_actions.unbind(dim=1))
        predictions = []
        for step in range(future_actions.shape[1]):
            start = max(0, len(states) - self.config.context_blocks)
            state_window = torch.stack(states[start:], dim=1)
            action_window = torch.stack(
                completed_actions[start:] + [future_actions[:, step]],
                dim=1,
            )
            next_latent = self.predict_teacher_forced(
                state_window,
                action_window,
            )[:, -1]
            predictions.append(next_latent)
            states.append(F.layer_norm(next_latent, (next_latent.shape[-1],)))
            completed_actions.append(future_actions[:, step])
        return torch.stack(predictions, dim=1)


def normalized_latent_l1_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """对最后一维做无仿射 LayerNorm 后计算 FP32 L1。"""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.ndim == 0 or prediction.shape[-1] <= 0:
        raise ValueError("latent tensors must have a feature dimension")
    prediction = F.layer_norm(
        prediction.float(),
        (prediction.shape[-1],),
    )
    target = F.layer_norm(
        target.detach().float(),
        (target.shape[-1],),
    )
    return F.l1_loss(prediction, target)


def teacher_forced_autoregressive_loss(
    predictor: ActionConditionedPredictor,
    latents: Tensor,
    actions: Tensor,
    *,
    auto_steps: int = 2,
) -> Dict[str, Tensor]:
    """计算 teacher-forced L1、autoregressive L1 及两者之和。"""

    if latents.ndim != 4 or latents.shape[1] < 2:
        raise ValueError("latents must have shape [B, T+1, S, D] with T >= 1")
    transitions = latents.shape[1] - 1
    if actions.ndim != 3 or actions.shape[:2] != (latents.shape[0], transitions):
        raise ValueError("actions must have shape [B, T, action_dim]")
    if not 1 <= auto_steps <= transitions:
        raise ValueError("auto_steps must be in [1, T]")

    # 官方 V-JEPA 2-AC 会先归一化冻结 encoder 的表示，再将其作为 predictor
    # 输入和监督目标。保持原 dtype，避免破坏外层 autocast/FSDP 混合精度。
    normalized_latents = F.layer_norm(
        latents,
        (latents.shape[-1],),
    )
    targets = normalized_latents[:, 1:].detach()
    teacher_predictions = predictor.predict_teacher_forced(
        normalized_latents[:, :-1],
        actions,
    )
    teacher_loss = normalized_latent_l1_loss(teacher_predictions, targets)

    autoregressive_predictions = predictor.rollout(
        normalized_latents[:, 0],
        actions[:, :auto_steps],
    )
    autoregressive_targets = targets[:, :auto_steps]
    autoregressive_loss = normalized_latent_l1_loss(
        autoregressive_predictions,
        autoregressive_targets,
    )
    total = teacher_loss + autoregressive_loss
    return {
        "loss": total,
        "teacher_forced_loss": teacher_loss,
        "autoregressive_loss": autoregressive_loss,
        "teacher_forced_predictions": teacher_predictions,
        "autoregressive_predictions": autoregressive_predictions,
    }


# 简短别名，便于配置和调用代码使用 AC Predictor 这一名称。
ACPredictor = ActionConditionedPredictor
