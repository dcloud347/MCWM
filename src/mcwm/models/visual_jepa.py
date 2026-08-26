"""M1 Minecraft 视觉预训练：V-JEPA 2 风格 online、predictor 和 EMA target。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mcwm.training.ema import update_ema
from .masking import MaskConfig, SpatiotemporalMaskSampler
from .visual_encoder import VisualEncoder, VisualEncoderConfig
from .visual_predictor import VisualPredictor, VisualPredictorConfig


@dataclass(frozen=True)
class VisualJEPAConfig:
    """把 video encoder、predictor、mask 和像素归一化配置组合在一起。"""

    encoder: VisualEncoderConfig
    predictor: VisualPredictorConfig
    mask: MaskConfig
    pixel_mean: tuple = (0.485, 0.456, 0.406)
    pixel_std: tuple = (0.229, 0.224, 0.225)


def _indices_from_mask(mask: Tensor) -> Tuple[Tensor, Tensor]:
    """像官方 collator 一样，把 batch 内 context/target 裁到共同最短长度。"""

    if mask.ndim != 2:
        raise ValueError("flattened mask must have shape [B, N]")
    context_rows = [torch.nonzero(~row, as_tuple=False).flatten() for row in mask]
    target_rows = [torch.nonzero(row, as_tuple=False).flatten() for row in mask]
    context_keep = min(row.numel() for row in context_rows)
    target_keep = min(row.numel() for row in target_rows)
    if min(context_keep, target_keep) <= 0:
        raise ValueError("each mask sample must retain context and prediction tokens")
    return (
        torch.stack([row[:context_keep] for row in context_rows]),
        torch.stack([row[:target_keep] for row in target_rows]),
    )


class VisualJEPA(nn.Module):
    """用可见 video tokens 预测被 mask 的 EMA target tokens。"""

    def __init__(self, config: VisualJEPAConfig) -> None:
        super().__init__()
        if config.predictor.input_dim != config.encoder.dim:
            raise ValueError("predictor input_dim must equal encoder dim")
        if config.predictor.token_grid_size != config.encoder.token_grid_size:
            raise ValueError("predictor and encoder token grids must match")
        if config.predictor.num_mask_tokens < len(config.mask.generators):
            raise ValueError("predictor needs at least one mask token per mask group")
        self.config = config
        self.online_encoder = VisualEncoder(config.encoder)
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = VisualPredictor(config.predictor)
        self.mask_sampler = SpatiotemporalMaskSampler(config.encoder.grid_size, config.mask)
        self.register_buffer("pixel_mean", torch.tensor(config.pixel_mean).view(1, 1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(config.pixel_std).view(1, 1, 3, 1, 1))

    def train(self, mode: bool = True) -> "VisualJEPA":
        super().train(mode)
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
    ) -> Dict[str, object]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, 3, H, W]")
        batch, clip_frames = frames.shape[:2]
        if clip_frames != self.config.encoder.clip_frames:
            raise ValueError("clip length does not match encoder config")
        target_frames = self.normalize_frames(frames)
        context_frames = self.normalize_frames(
            frames if online_frames is None else online_frames
        )
        if target_mask is None:
            target_mask = self.mask_sampler.sample(
                batch,
                self.config.encoder.temporal_grid_size,
                generator=mask_generator,
                device=target_frames.device,
            )
        target_mask = target_mask.to(device=target_frames.device, dtype=torch.bool)
        if target_mask.ndim == 3:
            target_mask = target_mask.unsqueeze(0)
        expected_mask_shape = (
            batch,
            self.config.encoder.temporal_grid_size,
            self.config.encoder.patch_count,
        )
        if target_mask.ndim != 4 or tuple(target_mask.shape[1:]) != expected_mask_shape:
            raise ValueError(
                "target_mask must have shape [G, B, T/tubelet, N] or [B, T/tubelet, N]"
            )
        if target_mask.shape[0] > self.config.predictor.num_mask_tokens:
            raise ValueError("target_mask has more groups than predictor mask tokens")

        flat_masks = target_mask.flatten(2)
        grouped_indices = tuple(_indices_from_mask(group_mask) for group_mask in flat_masks)
        prediction_mask = torch.zeros_like(flat_masks)
        for group_index, (_, prediction_indices) in enumerate(grouped_indices):
            prediction_mask[group_index].scatter_(1, prediction_indices, True)
        prediction_mask = prediction_mask.reshape_as(target_mask)

        # 完整 clip 的 target 只编码一次；官方训练在 encoder norm 后再做 feature LN。
        with torch.no_grad():
            target_flat = self.target_encoder(
                target_frames,
                return_patch_tokens=True,
            )
            target_flat = F.layer_norm(target_flat, (target_flat.shape[-1],))
        detached_target = target_flat.detach()

        online_groups = []
        prediction_groups = []
        losses = []
        for group_index, (context_indices, prediction_indices) in enumerate(grouped_indices):
            online = self.online_encoder(
                context_frames,
                context_indices,
                return_patch_tokens=True,
            )
            prediction = self.predictor(
                online,
                context_indices,
                prediction_indices,
                mask_index=group_index,
            )
            target_values = detached_target.gather(
                1,
                prediction_indices.unsqueeze(-1).expand(-1, -1, detached_target.shape[-1]),
            )
            for batch_index in range(batch):
                losses.append(F.l1_loss(prediction[batch_index], target_values[batch_index]))
            online_groups.append(online)
            prediction_groups.append(prediction)

        loss = torch.stack(losses).mean()
        target = detached_target.reshape(
            batch,
            self.config.encoder.temporal_grid_size,
            self.config.encoder.patch_count,
            self.config.encoder.dim,
        )
        return {
            "loss": loss,
            "online": tuple(online_groups),
            "prediction": tuple(prediction_groups),
            "target": target,
            "target_mask": target_mask,
            "prediction_mask": prediction_mask,
            "context_indices": tuple(pair[0] for pair in grouped_indices),
            "prediction_indices": tuple(pair[1] for pair in grouped_indices),
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """optimizer step 完成后调用一次；micro-step 中不能调用。"""

        update_ema(self.online_encoder, self.target_encoder, momentum)
