"""用 M1 EMA 权重逐帧生成 V-JEPA 2-AC latent tokens。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .visual_encoder import VisualEncoder, VisualEncoderConfig


class FrozenVisualEncoder(nn.Module):
    """冻结的 M1 encoder；每帧复制成一个完整 tubelet 后独立编码。"""

    def __init__(
        self,
        config: VisualEncoderConfig,
        *,
        pixel_mean: tuple = (0.485, 0.456, 0.406),
        pixel_std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.encoder = VisualEncoder(config)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @property
    def config(self) -> VisualEncoderConfig:
        return self.encoder.config

    def train(self, mode: bool = True) -> "FrozenVisualEncoder":
        """允许外层模型切换模式，但 visual encoder 始终保持 eval。"""

        super().train(mode)
        self.encoder.eval()
        return self

    def normalize_frames(self, frames: Tensor) -> Tensor:
        if frames.dtype == torch.uint8:
            frames = frames.float().div_(255.0)
        elif not frames.is_floating_point():
            raise TypeError("frames must be uint8 or floating point")
        mean = self.pixel_mean.to(device=frames.device, dtype=frames.dtype)
        std = self.pixel_std.to(device=frames.device, dtype=frames.dtype)
        return (frames - mean) / std

    @torch.no_grad()
    def forward(self, frames: Tensor, *, frame_chunk_size: Optional[int] = None) -> Tensor:
        """把 ``[B,T,3,H,W]`` 编码成 ``[B,T,H'W',D]``。"""

        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, 3, H, W]")
        batch, frame_count, channels, height, width = frames.shape
        expected = (3, self.config.image_height, self.config.image_width)
        if (channels, height, width) != expected:
            raise ValueError(
                "frames must have shape [B, T, 3, "
                f"{self.config.image_height}, {self.config.image_width}]"
            )
        if frame_count <= 0:
            raise ValueError("frames must contain at least one observation")
        if frame_chunk_size is None:
            chunk_size = batch * frame_count
        else:
            chunk_size = int(frame_chunk_size)
            if chunk_size <= 0:
                raise ValueError("frame_chunk_size must be positive")

        flattened = frames.reshape(batch * frame_count, 1, channels, height, width)
        outputs = []
        for start in range(0, flattened.shape[0], chunk_size):
            chunk = flattened[start : start + chunk_size]
            tubelets = chunk.repeat(1, self.config.tubelet_size, 1, 1, 1)
            normalized = self.normalize_frames(tubelets)
            outputs.append(self.encoder(normalized, return_patch_tokens=True))
        tokens = torch.cat(outputs, dim=0)
        spatial_tokens = self.config.patch_count
        if tokens.shape[1] != spatial_tokens:
            raise RuntimeError("a repeated frame must produce exactly one spatial token grid")
        return tokens.reshape(batch, frame_count, spatial_tokens, self.config.dim)


def repeated_frame_metrics(tokens: Tensor) -> Dict[str, Tensor]:
    """返回 repeated-frame probe 使用的有限、轻量 latent 指标。"""

    if tokens.ndim != 4:
        raise ValueError("tokens must have shape [B, T, S, D]")
    flat = tokens.float().flatten(0, 2)
    centered = flat - flat.mean(dim=0, keepdim=True)
    per_dimension_std = centered.square().mean(dim=0).sqrt()
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    if normalized.shape[0] > 1:
        adjacent_cosine = (normalized[:-1] * normalized[1:]).sum(dim=-1).mean()
    else:
        adjacent_cosine = torch.ones((), device=tokens.device)
    return {
        "mean": flat.mean(),
        "average_std": per_dimension_std.mean(),
        "average_token_norm": flat.norm(dim=-1).mean(),
        "adjacent_token_cosine": adjacent_cosine,
        "finite": torch.tensor(bool(torch.isfinite(flat).all()), device=tokens.device),
    }
