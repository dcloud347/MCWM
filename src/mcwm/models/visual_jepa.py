"""M1 Minecraft 视觉预训练：online encoder + EMA target encoder。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from mcwm.training.ema import update_ema
from .masking import MaskConfig, SpatiotemporalMaskSampler
from .visual_encoder import VisualEncoder, VisualEncoderConfig
from .visual_predictor import VisualPredictor, VisualPredictorConfig


@dataclass(frozen=True)
class VisualJEPAConfig:
    """把 encoder、predictor、mask 和像素归一化配置组合在一起。"""

    encoder: VisualEncoderConfig
    predictor: VisualPredictorConfig
    mask: MaskConfig
    pixel_mean: tuple = (0.5, 0.5, 0.5)
    pixel_std: tuple = (0.5, 0.5, 0.5)


class VisualJEPA(nn.Module):
    """预测被 mask 的 EMA target token，梯度不会进入 target 分支。"""

    def __init__(self, config: VisualJEPAConfig) -> None:
        super().__init__()
        if config.predictor.input_dim != config.encoder.dim:
            raise ValueError("predictor input_dim must equal encoder dim")
        if config.predictor.patch_count != config.encoder.patch_count:
            raise ValueError("predictor patch_count must equal encoder patch_count")
        self.config = config
        self.online_encoder = VisualEncoder(config.encoder)
        # target 初始值必须和 online 完全相同，之后只能通过 EMA 更新。
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = VisualPredictor(config.predictor)
        self.mask_sampler = SpatiotemporalMaskSampler(config.encoder.grid_size, config.mask)
        self.register_buffer("pixel_mean", torch.tensor(config.pixel_mean).view(1, 1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(config.pixel_std).view(1, 1, 3, 1, 1))

    def train(self, mode: bool = True) -> "VisualJEPA":
        super().train(mode)
        # 即使整体进入 train 模式，target 分支也必须保持确定性的 eval 模式。
        self.target_encoder.eval()
        return self

    def normalize_frames(self, frames: Tensor) -> Tensor:
        """磁盘 uint8 或 [0,1] float 都统一归一化后再进入 encoder。"""

        if frames.dtype == torch.uint8:
            frames = frames.float().div_(255.0)
        elif not frames.is_floating_point():
            raise TypeError("frames must be uint8 or floating point")
        mean = self.pixel_mean.to(dtype=frames.dtype)
        std = self.pixel_std.to(dtype=frames.dtype)
        return (frames - mean) / std

    def forward(
        self,
        frames: Tensor,
        target_mask: Optional[Tensor] = None,
        *,
        mask_generator: Optional[torch.Generator] = None,
        online_frames: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, 3, H, W]")
        batch, clip_frames = frames.shape[:2]
        if clip_frames > self.config.predictor.max_frames:
            raise ValueError("clip has more frames than predictor max_frames")
        # target 看原图；online 可以看经过轻量颜色增强的同一批图。
        target_frames = self.normalize_frames(frames)
        context_frames = self.normalize_frames(
            frames if online_frames is None else online_frames
        )
        if target_mask is None:
            target_mask = self.mask_sampler.sample(
                batch,
                clip_frames,
                generator=mask_generator,
                device=target_frames.device,
            )
        target_mask = target_mask.to(device=target_frames.device, dtype=torch.bool)
        flat_context_frames = context_frames.flatten(0, 1)
        flat_target_frames = target_frames.flatten(0, 1)
        flat_mask = target_mask.flatten(0, 1)

        # B 和 T 合并后逐帧调用同一份 2D encoder。
        online = self.online_encoder(
            flat_context_frames,
            flat_mask,
            return_patch_tokens=True,
        ).reshape(batch, clip_frames, -1, self.config.encoder.dim)
        # no_grad + 后面的 detach 是显式 stop-gradient 双保险。
        with torch.no_grad():
            target = self.target_encoder(
                flat_target_frames,
                return_patch_tokens=True,
            ).reshape(batch, clip_frames, -1, self.config.encoder.dim)
        predicted = self.predictor(online, target_mask)
        # 只在 target_mask=True 的位置计算 L1，不惩罚可见 context token。
        selected_prediction = predicted[target_mask]
        selected_target = target.detach()[target_mask]
        loss = torch.nn.functional.l1_loss(selected_prediction, selected_target)
        return {
            "loss": loss,
            "online": online,
            "prediction": predicted,
            "target": target.detach(),
            "target_mask": target_mask,
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """optimizer step 完成后调用一次；micro-step 中不能调用。"""

        update_ema(self.online_encoder, self.target_encoder, momentum)
