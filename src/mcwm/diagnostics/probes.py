"""用简单线性模型检查冻结后的 M1 特征是否有用。"""

from __future__ import annotations

from typing import Dict, Literal

import torch
from torch import Tensor


@torch.no_grad()
def ridge_linear_probe(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    task: Literal["regression", "classification"],
    ridge: float = 1e-3,
) -> Dict[str, float]:
    """拟合带正则的线性模型，全程不更新 encoder。"""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("features must have shape [samples, dimensions]")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    train_x = train_features.detach().float()
    validation_x = validation_features.detach().float()
    train_x = torch.cat((train_x, torch.ones_like(train_x[:, :1])), dim=1)
    validation_x = torch.cat((validation_x, torch.ones_like(validation_x[:, :1])), dim=1)

    if task == "classification":
        classes = int(torch.cat((train_targets, validation_targets)).max().item()) + 1
        train_y = torch.nn.functional.one_hot(train_targets.long(), classes).float()
    elif task == "regression":
        train_y = train_targets.float()
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(1)
    else:
        raise ValueError(f"unsupported probe task: {task}")

    if train_x.shape[0] < train_x.shape[1]:
        # 样本数少于特征维度时，换一种等价公式可以减少计算量。
        identity = torch.eye(train_x.shape[0], device=train_x.device, dtype=train_x.dtype)
        weights = train_x.T @ torch.linalg.solve(
            train_x @ train_x.T + ridge * identity, train_y
        )
    else:
        identity = torch.eye(train_x.shape[1], device=train_x.device, dtype=train_x.dtype)
        weights = torch.linalg.solve(train_x.T @ train_x + ridge * identity, train_x.T @ train_y)
    prediction = validation_x @ weights
    if task == "classification":
        accuracy = (prediction.argmax(dim=1) == validation_targets.long()).float().mean()
        return {"probe/accuracy": accuracy.item()}
    target = validation_targets.float()
    if target.ndim == 1:
        target = target.unsqueeze(1)
    mse = torch.nn.functional.mse_loss(prediction, target)
    variance = target.var(unbiased=False).clamp_min(1e-12)
    return {"probe/mse": mse.item(), "probe/r2": (1.0 - mse / variance).item()}
