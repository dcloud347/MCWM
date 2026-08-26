"""更新 EMA encoder，并计算每一步使用的平滑系数。"""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """用 online 参数平滑更新 target 参数。"""

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("EMA momentum must be between 0 and 1")
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("online and target parameter structures differ")
    for name, target_value in target_parameters.items():
        target_value.lerp_(online_parameters[name].detach(), 1.0 - momentum)

    # BatchNorm 等额外状态不能做参数平均，遇到时直接复制。
    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())
    if online_buffers.keys() != target_buffers.keys():
        raise ValueError("online and target buffer structures differ")
    for name, target_value in target_buffers.items():
        target_value.copy_(online_buffers[name])


def cosine_ema_momentum(step: int, total_steps: int, start: float, end: float = 1.0) -> float:
    """让 EMA 平滑系数从 start 平缓变化到 end。"""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("EMA schedule must satisfy 0 <= start <= end <= 1")
    progress = min(max(step, 0), total_steps) / total_steps
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0
