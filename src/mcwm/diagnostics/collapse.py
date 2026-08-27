"""检查模型特征是否变得过于相似，也就是发生表征坍塌。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CollapseThresholds:
    """判断特征坍塌时使用的阈值。"""

    minimum_average_std: float = 0.35
    minimum_effective_rank: float = 2.5
    maximum_pairwise_cosine: float = 0.8


@torch.no_grad()
def collapse_metrics(latents: Tensor, *, prefix: str = "latent") -> Dict[str, float]:
    """计算一组指标，判断特征是否仍然有足够差异。

    最后一维是特征，其余维度都当作样本。即使训练使用 bf16，这里也转成
    FP32，以免诊断指标受到低精度影响。
    """

    if latents.ndim < 2:
        raise ValueError("latents must have at least a sample and feature dimension")
    values = latents.detach().float().reshape(-1, latents.shape[-1])
    if not torch.isfinite(values).all():
        return {
            f"{prefix}/mean": float("nan"),
            f"{prefix}/average_std": float("nan"),
            f"{prefix}/effective_rank": float("nan"),
            f"{prefix}/pairwise_cosine": float("nan"),
            f"{prefix}/covariance_off_diagonal": float("nan"),
        }

    mean = values.mean(dim=0)
    centered = values - mean
    standard_deviation = values.std(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(centered)
    spectrum = singular_values.square()
    probabilities = spectrum / spectrum.sum().clamp_min(torch.finfo(torch.float32).eps)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp()

    normalized = F.normalize(values, dim=-1, eps=1e-8)
    if values.shape[0] > 1:
        cosine_matrix = normalized @ normalized.transpose(0, 1)
        pairwise = (cosine_matrix.sum() - values.shape[0]) / (
            values.shape[0] * (values.shape[0] - 1)
        )
    else:
        pairwise = values.new_tensor(1.0)

    if values.shape[0] > 1:
        covariance = centered.transpose(0, 1) @ centered / (values.shape[0] - 1)
        diagonal = torch.diag_embed(torch.diagonal(covariance))
        off_diagonal = (covariance - diagonal).square().mean().sqrt()
    else:
        off_diagonal = values.new_tensor(0.0)

    return {
        f"{prefix}/mean": mean.mean().item(),
        f"{prefix}/average_std": standard_deviation.mean().item(),
        f"{prefix}/effective_rank": effective_rank.item(),
        f"{prefix}/pairwise_cosine": pairwise.item(),
        f"{prefix}/covariance_off_diagonal": off_diagonal.item(),
    }


def find_collapse_alerts(
    metrics: Dict[str, float],
    thresholds: CollapseThresholds = CollapseThresholds(),
    *,
    prefix: str = "latent",
) -> Tuple[str, ...]:
    """根据阈值把异常指标转换成容易阅读的报警信息。"""

    checks = (
        ("average_std", lambda value: value < thresholds.minimum_average_std, "std too low"),
        (
            "effective_rank",
            lambda value: value < thresholds.minimum_effective_rank,
            "effective rank too low",
        ),
        (
            "pairwise_cosine",
            lambda value: value > thresholds.maximum_pairwise_cosine,
            "pairwise cosine too high",
        ),
    )
    alerts = []
    for suffix, predicate, message in checks:
        value = metrics.get(f"{prefix}/{suffix}")
        if value is None or not torch.isfinite(torch.tensor(value)):
            alerts.append(f"{suffix} is not finite")
        elif predicate(value):
            alerts.append(message)
    return tuple(alerts)


@torch.no_grad()
def online_target_gap(online: Tensor, target: Tensor) -> float:
    """用余弦距离衡量 online encoder 和 EMA encoder 的差异。"""

    if online.shape != target.shape:
        raise ValueError("online and target latents must have matching shapes")
    return (1.0 - F.cosine_similarity(online.float(), target.float(), dim=-1)).mean().item()
