"""组合冻结视觉编码器、动作编码器和 block-causal predictor。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from torch import Tensor, nn

from .ac_predictor import (
    ActionConditionedPredictor,
    teacher_forced_autoregressive_loss,
)
from .action_encoder import MinecraftActionEncoder
from .frozen_visual_encoder import FrozenVisualEncoder


@dataclass(frozen=True)
class WorldModelConfig:
    """M2 组合层配置。"""

    auto_steps: int = 2
    encoder_frame_chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.auto_steps <= 0:
            raise ValueError("auto_steps must be positive")
        if self.encoder_frame_chunk_size is not None:
            if self.encoder_frame_chunk_size <= 0:
                raise ValueError("encoder_frame_chunk_size must be positive")


class WorldModel(nn.Module):
    """Minecraft action-conditioned latent world model。"""

    def __init__(
        self,
        visual_encoder: FrozenVisualEncoder,
        action_encoder: MinecraftActionEncoder,
        predictor: ActionConditionedPredictor,
        config: WorldModelConfig = WorldModelConfig(),
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.config = config

        if visual_encoder.config.dim != predictor.config.latent_dim:
            raise ValueError("visual encoder dim must match predictor latent_dim")
        if action_encoder.config.macro_dim != predictor.config.action_dim:
            raise ValueError("action macro_dim must match predictor action_dim")
        if visual_encoder.config.grid_size != predictor.config.spatial_grid:
            raise ValueError("visual and predictor spatial grids must match")
        self.visual_encoder.requires_grad_(False)
        self.visual_encoder.eval()

    def train(self, mode: bool = True) -> "WorldModel":
        """切换训练模式，同时确保视觉编码器始终冻结。"""

        super().train(mode)
        self.visual_encoder.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """只返回 action encoder 和 predictor 的参数。"""

        yield from self.action_encoder.parameters()
        yield from self.predictor.parameters()

    def encode_frames(self, frames: Tensor) -> Tensor:
        """逐帧产生 ``[B, T, S, D]`` frozen visual latent。"""

        return self.visual_encoder(
            frames,
            frame_chunk_size=self.config.encoder_frame_chunk_size,
        )

    def encode_actions(
        self,
        movement: Tensor,
        interaction: Tensor,
        hotbar: Tensor,
        camera: Tensor,
        cursor: Tensor,
        gui_open: Tensor,
        cursor_present: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """把 action blocks 编码成 ``[B, A, action_dim]``。"""

        return self.action_encoder(
            movement,
            interaction,
            hotbar,
            camera,
            cursor,
            gui_open,
            cursor_present,
            valid_mask,
        )

    def forward(
        self,
        frames: Tensor,
        movement: Tensor,
        interaction: Tensor,
        hotbar: Tensor,
        camera: Tensor,
        cursor: Tensor,
        gui_open: Tensor,
        cursor_present: Tensor,
        valid_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """编码 batch，并计算 teacher-forced 与 autoregressive loss。"""

        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError("frames must have shape [B, T+1, 3, H, W]")
        if movement.shape[:2] != (frames.shape[0], frames.shape[1] - 1):
            raise ValueError("actions must contain one block per frame transition")

        latents = self.encode_frames(frames)
        action_tokens = self.encode_actions(
            movement,
            interaction,
            hotbar,
            camera,
            cursor,
            gui_open,
            cursor_present,
            valid_mask,
        )
        if self.config.auto_steps > action_tokens.shape[1]:
            raise ValueError("auto_steps cannot exceed the number of transitions")
        output = teacher_forced_autoregressive_loss(
            self.predictor,
            latents,
            action_tokens,
            auto_steps=self.config.auto_steps,
        )
        output["latents"] = latents
        output["action_tokens"] = action_tokens
        output["targets"] = latents[:, 1:]
        return output
