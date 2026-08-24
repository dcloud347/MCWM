"""EMA target 更新和 momentum 调度。"""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """严格执行一次 ``target = m*target + (1-m)*online``。"""

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("EMA momentum must be between 0 and 1")
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("online and target parameter structures differ")
    for name, target_value in target_parameters.items():
        target_value.lerp_(online_parameters[name].detach(), 1.0 - momentum)

    # 如果以后加入 BatchNorm 等 buffer，直接复制，不能像参数一样做平均。
    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())
    if online_buffers.keys() != target_buffers.keys():
        raise ValueError("online and target buffer structures differ")
    for name, target_value in target_buffers.items():
        target_value.copy_(online_buffers[name])


def cosine_ema_momentum(step: int, total_steps: int, start: float, end: float = 1.0) -> float:
    """让 momentum 从 start 按 cosine 逐渐接近 end。"""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("EMA schedule must satisfy 0 <= start <= end <= 1")
    progress = min(max(step, 0), total_steps) / total_steps
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0
