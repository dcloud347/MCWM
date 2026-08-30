"""M3 violation-of-expectation diagnostics over open-loop latent rollouts."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .world_model import paired_gap_statistics


def _normalized_squared_error(predictions: Tensor, targets: Tensor) -> Tensor:
    prediction = F.layer_norm(predictions.float(), (predictions.shape[-1],))
    target = F.layer_norm(targets.detach().float(), (targets.shape[-1],))
    return (prediction - target).square().mean(dim=(-1, -2))


@torch.no_grad()
def surprise_samples(
    predictions: Tensor,
    targets: Tensor,
    *,
    perturbation_step: int,
    unrelated_targets: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Construct frame-replacement and trajectory-switch surprise curves."""

    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError("surprise predictions and targets must match [B, H, S, D]")
    if unrelated_targets is None and predictions.shape[0] < 2:
        raise ValueError("surprise evaluation requires at least two clips per batch")
    step = int(perturbation_step)
    if not 1 <= step <= predictions.shape[1]:
        raise ValueError("perturbation_step must fit within the rollout horizon")
    index = step - 1
    unrelated = (
        targets.roll(1, dims=0)
        if unrelated_targets is None
        else unrelated_targets
    )
    if unrelated.shape != targets.shape:
        raise ValueError("unrelated_targets must match targets")
    replaced = targets.clone()
    replaced[:, index] = unrelated[:, index]
    switched = targets.clone()
    switched[:, index:] = unrelated[:, index:]
    return {
        "control": _normalized_squared_error(predictions, targets),
        "frame_replaced": _normalized_squared_error(predictions, replaced),
        "trajectory_switched": _normalized_squared_error(predictions, switched),
    }


@torch.no_grad()
def surprise_metrics(
    samples: Mapping[str, Tensor],
    *,
    perturbation_step: int,
    prefix: str = "m3/surprise",
) -> Dict[str, float]:
    """Aggregate curves and paired significance at the perturbation point."""

    names = ("control", "frame_replaced", "trajectory_switched")
    missing = set(names) - samples.keys()
    if missing:
        raise ValueError(f"surprise samples are missing: {sorted(missing)}")
    curves = {name: samples[name].detach().float() for name in names}
    shapes = {tuple(values.shape) for values in curves.values()}
    if len(shapes) != 1 or curves["control"].ndim != 2:
        raise ValueError("surprise curves must have matching [N, H] shapes")
    control = curves["control"]
    if control.numel() == 0 or not all(
        bool(torch.isfinite(values).all()) for values in curves.values()
    ):
        raise ValueError("surprise samples must be non-empty and finite")
    step = int(perturbation_step)
    if not 1 <= step <= control.shape[1]:
        raise ValueError("perturbation_step must fit within surprise curves")
    index = step - 1

    metrics: Dict[str, float] = {
        f"{prefix}/sample_clips": float(control.shape[0]),
        f"{prefix}/perturbation_step": float(step),
    }
    for curve_name, values in curves.items():
        for curve_step, value in enumerate(values.mean(dim=0), start=1):
            metrics[f"{prefix}/{curve_name}_step_{curve_step}"] = float(value)

    passes = []
    for name in ("frame_replaced", "trajectory_switched"):
        perturbed = curves[name]
        ci_low, ci_high, p_value = paired_gap_statistics(
            control[:, index],
            perturbed[:, index],
        )
        delta_curve = (perturbed - control).mean(dim=0)
        peak_step = int(delta_curve.argmax()) + 1
        gap = perturbed[:, index].mean() - control[:, index].mean()
        statistically_positive = ci_low > 0 and p_value < 0.05
        passes.append(statistically_positive)
        metrics.update(
            {
                f"{prefix}/{name}_gap": float(gap),
                f"{prefix}/{name}_gap_ci95_low": ci_low,
                f"{prefix}/{name}_gap_ci95_high": ci_high,
                f"{prefix}/{name}_gap_pvalue": p_value,
                f"{prefix}/{name}_peak_step": float(peak_step),
                f"{prefix}/{name}_peak_aligned": float(
                    peak_step in {step, min(step + 1, control.shape[1])}
                ),
                f"{prefix}/{name}_pass_statistical": float(
                    statistically_positive
                ),
            }
        )
    metrics[f"{prefix}/pass_statistical"] = float(all(passes))
    return metrics
