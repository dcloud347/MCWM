"""M3 multi-horizon open-loop rollout diagnostics."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


ACTION_BUCKETS = ("movement", "interaction", "camera", "hotbar", "gui")


def _action_bucket_masks(batch: Mapping[str, Tensor], horizon: int) -> Dict[str, Tensor]:
    masks = {
        "movement": batch["movement"].any(dim=-1).any(dim=-1),
        "interaction": batch["interaction"].any(dim=-1).any(dim=-1),
        "camera": batch["camera"].abs().sum(dim=-1).gt(0).any(dim=-1),
        "hotbar": batch["hotbar"].ne(0).any(dim=-1),
        "gui": batch["gui_open"].any(dim=-1),
    }
    if any(mask.shape[1] < horizon for mask in masks.values()):
        raise ValueError("action buckets do not cover the requested rollout horizon")
    return {name: mask[:, :horizon] for name, mask in masks.items()}


@torch.no_grad()
def rollout_samples(
    predictions: Tensor,
    targets: Tensor,
    batch: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    """Return compact per-transition samples for cross-batch/rank aggregation."""

    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError("rollout predictions and targets must match [B, H, S, D]")
    if predictions.shape[1] <= 0:
        raise ValueError("rollout horizon must be positive")
    prediction = F.layer_norm(predictions.float(), (predictions.shape[-1],))
    target = F.layer_norm(targets.detach().float(), (targets.shape[-1],))
    samples = {
        "l1": (prediction - target).abs().mean(dim=(-1, -2)),
        "cosine": F.cosine_similarity(prediction, target, dim=-1).mean(dim=-1),
        "norm_gap": (
            predictions.float().norm(dim=-1) - targets.float().norm(dim=-1)
        ).abs().mean(dim=-1),
    }
    samples.update(
        {
            f"bucket_{name}": mask
            for name, mask in _action_bucket_masks(
                batch,
                predictions.shape[1],
            ).items()
        }
    )
    return samples


@torch.no_grad()
def rollout_metrics(
    samples: Mapping[str, Tensor],
    *,
    horizons: Sequence[int],
    prefix: str = "m3/rollout",
) -> Dict[str, float]:
    """Aggregate exact-step, cumulative-horizon, drift, and action-bucket metrics."""

    required = {"l1", "cosine", "norm_gap"}
    missing = required - samples.keys()
    if missing:
        raise ValueError(f"rollout samples are missing: {sorted(missing)}")
    l1 = samples["l1"].detach().float()
    cosine = samples["cosine"].detach().float()
    norm_gap = samples["norm_gap"].detach().float()
    if l1.ndim != 2 or cosine.shape != l1.shape or norm_gap.shape != l1.shape:
        raise ValueError("rollout sample metrics must have matching [N, H] shapes")
    if l1.numel() == 0 or not all(
        bool(torch.isfinite(values).all()) for values in (l1, cosine, norm_gap)
    ):
        raise ValueError("rollout samples must be non-empty and finite")

    requested = tuple(int(value) for value in horizons)
    if not requested or tuple(sorted(set(requested))) != requested:
        raise ValueError("horizons must be non-empty, unique, and increasing")
    if requested[0] <= 0 or requested[-1] > l1.shape[1]:
        raise ValueError("horizons must fit within collected rollout steps")

    one_step = l1[:, 0].mean()
    metrics: Dict[str, float] = {
        f"{prefix}/sample_rollouts": float(l1.shape[0]),
        f"{prefix}/sample_transitions": float(l1.numel()),
    }
    for horizon in requested:
        step_error = l1[:, horizon - 1].mean()
        metrics[f"{prefix}/step_{horizon}_l1"] = float(step_error)
        metrics[f"{prefix}/step_{horizon}_cosine"] = float(
            cosine[:, horizon - 1].mean()
        )
        metrics[f"{prefix}/step_{horizon}_norm_gap"] = float(
            norm_gap[:, horizon - 1].mean()
        )
        metrics[f"{prefix}/horizon_{horizon}_mean_l1"] = float(
            l1[:, :horizon].mean()
        )
        metrics[f"{prefix}/step_{horizon}_relative_to_step_1"] = float(
            step_error / one_step.clamp_min(1e-8)
        )

    for start, end in ((4, 8), (8, 12), (12, 14)):
        if start in requested and end in requested:
            metrics[f"{prefix}/drift_slope_{start}_to_{end}"] = float(
                (l1[:, end - 1].mean() - l1[:, start - 1].mean())
                / (end - start)
            )

    for name in ACTION_BUCKETS:
        key = f"bucket_{name}"
        if key not in samples:
            continue
        active = samples[key].detach().bool()
        if active.shape != l1.shape:
            raise ValueError(f"{key} must match rollout sample shape")
        metrics[f"m3/action_bucket/{name}/sample_transitions"] = float(active.sum())
        if bool(active.any()):
            metrics[f"m3/action_bucket/{name}/l1"] = float(l1[active].mean())
        for horizon in requested:
            at_step = active[:, horizon - 1]
            metrics[f"m3/action_bucket/{name}/step_{horizon}_samples"] = float(
                at_step.sum()
            )
            if bool(at_step.any()):
                metrics[f"m3/action_bucket/{name}/step_{horizon}_l1"] = float(
                    l1[at_step, horizon - 1].mean()
                )
    return metrics
