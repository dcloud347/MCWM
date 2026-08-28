"""Minecraft 动作中连续分量的编码器。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def mu_law_normalize(
    values: Tensor,
    *,
    clip_value: float,
    mu: float = 255.0,
) -> Tensor:
    """裁剪连续值，并用 mu-law 将其压缩到 ``[-1, 1]``。"""

    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    if not values.is_floating_point():
        raise TypeError("values must be a floating point tensor")

    # 先除以裁剪范围，得到 [-1, 1]；mu-law 会保留正负号，并放大小动作
    # 之间的差别，同时压缩极大的动作。
    scaled = values.clamp(-clip_value, clip_value) / clip_value
    return scaled.sign() * torch.log1p(mu * scaled.abs()) / math.log1p(mu)


class CameraEncoder(nn.Module):
    """把 ``(pitch_delta, yaw_delta)`` 编码成一个特征向量。"""

    def __init__(
        self,
        output_dim: int = 64,
        *,
        hidden_dim: int = 64,
        clip_degrees: float = 180.0,
        mu: float = 255.0,
    ) -> None:
        super().__init__()
        if min(output_dim, hidden_dim) <= 0:
            raise ValueError("camera encoder dimensions must be positive")
        if clip_degrees <= 0:
            raise ValueError("clip_degrees must be positive")
        if mu <= 0:
            raise ValueError("mu must be positive")

        self.clip_degrees = float(clip_degrees)
        self.mu = float(mu)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, camera: Tensor) -> Tensor:
        """编码最后一维为 2 的 camera 张量，前面的维度保持不变。"""

        if camera.ndim == 0 or camera.shape[-1] != 2:
            raise ValueError("camera must have shape [..., 2]")
        normalized = mu_law_normalize(
            camera,
            clip_value=self.clip_degrees,
            mu=self.mu,
        )
        return self.mlp(normalized)


class CursorEncoder(nn.Module):
    """编码 GUI 光标位置；没有有效光标时输出全零。"""

    def __init__(self, output_dim: int = 64, *, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(output_dim, hidden_dim) <= 0:
            raise ValueError("cursor encoder dimensions must be positive")

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        cursor: Tensor,
        gui_open: Tensor,
        cursor_present: Tensor,
    ) -> Tensor:
        """编码 cursor，并用两个布尔状态共同屏蔽无效位置。"""

        if cursor.ndim == 0 or cursor.shape[-1] != 2:
            raise ValueError("cursor must have shape [..., 2]")
        expected_shape = cursor.shape[:-1]
        if gui_open.shape != expected_shape:
            raise ValueError("gui_open must match cursor leading dimensions")
        if cursor_present.shape != expected_shape:
            raise ValueError("cursor_present must match cursor leading dimensions")
        if gui_open.dtype != torch.bool or cursor_present.dtype != torch.bool:
            raise TypeError("gui_open and cursor_present must be boolean tensors")

        encoded = self.mlp(cursor)
        active = (gui_open & cursor_present).unsqueeze(-1)
        return encoded * active.to(dtype=encoded.dtype)
