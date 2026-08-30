"""M2 latent prediction 与动作敏感性诊断。"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from mcwm.models.ac_predictor import normalized_latent_l1_loss
from mcwm.models.world_model import WorldModel
from .collapse import collapse_metrics


ACTION_INPUT_NAMES = (
    "movement",
    "interaction",
    "hotbar",
    "camera",
    "cursor",
    "gui_open",
    "cursor_present",
    "valid_mask",
)


def _normalized_interval_error(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    target = F.layer_norm(target.float(), (target.shape[-1],))
    return (prediction - target).abs().mean(dim=(-1, -2))


@torch.no_grad()
def rollout_error_curve(prediction: Tensor, target: Tensor) -> Tensor:
    """返回每个 rollout step 的 normalized latent L1。"""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching [B, H, S, D] shapes")
    return _normalized_interval_error(prediction, target).mean(dim=0)


@torch.no_grad()
def spatial_token_error(prediction: Tensor, target: Tensor) -> Tensor:
    """返回 ``[B, T, S]`` spatial-token normalized L1，可重排成热力图。"""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching [B, T, S, D] shapes")
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    target = F.layer_norm(target.float(), (target.shape[-1],))
    return (prediction - target).abs().mean(dim=-1)


@torch.no_grad()
def spatial_error_images(
    prediction: Tensor,
    target: Tensor,
    *,
    spatial_grid: Tuple[int, int],
) -> list:
    """把固定样本各 transition 的 spatial error 转成 RGB 热力图。"""

    rows, columns = spatial_grid
    errors = spatial_token_error(prediction, target)
    if errors.shape[2] != rows * columns:
        raise ValueError("spatial_grid does not match token count")
    images = []
    for values in errors[0]:
        values = values.reshape(rows, columns)
        values = values - values.min()
        values = values / values.max().clamp_min(1e-8)
        red = values
        blue = 1.0 - values
        green = 1.0 - (2.0 * values - 1.0).abs()
        image = torch.stack((red, green, blue), dim=-1).mul(255).byte()
        images.append(image.cpu().numpy())
    return images


@torch.no_grad()
def world_model_prediction_metrics(
    output: Mapping[str, Tensor],
    *,
    prefix: str = "validation",
) -> Dict[str, float]:
    """汇总 teacher-forced、rollout 和 latent 健康指标。"""

    teacher = output["teacher_forced_predictions"]
    targets = output["targets"]
    autoregressive = output["autoregressive_predictions"]
    auto_targets = targets[:, : autoregressive.shape[1]]
    metrics = {
        f"{prefix}/loss": float(output["loss"].detach()),
        f"{prefix}/teacher_forced_l1": float(
            normalized_latent_l1_loss(teacher, targets)
        ),
        f"{prefix}/autoregressive_l1": float(
            normalized_latent_l1_loss(autoregressive, auto_targets)
        ),
        f"{prefix}/teacher_forced_cosine": float(
            F.cosine_similarity(teacher.float(), targets.float(), dim=-1).mean()
        ),
        f"{prefix}/autoregressive_cosine": float(
            F.cosine_similarity(
                autoregressive.float(),
                auto_targets.float(),
                dim=-1,
            ).mean()
        ),
        f"{prefix}/prediction_target_norm_gap": float(
            (teacher.float().norm(dim=-1) - targets.float().norm(dim=-1))
            .abs()
            .mean()
        ),
    }
    metrics.update(
        collapse_metrics(targets.mean(dim=2), prefix=f"{prefix}/target_latent")
    )
    metrics.update(
        collapse_metrics(teacher.mean(dim=2), prefix=f"{prefix}/predicted_latent")
    )
    for step, value in enumerate(
        rollout_error_curve(autoregressive, auto_targets),
        start=1,
    ):
        metrics[f"{prefix}/autoregressive_step_{step}_l1"] = float(value)
    return metrics


def _copy_action_inputs(batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    return {name: batch[name].clone() for name in ACTION_INPUT_NAMES}


def noop_action_inputs(batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """构造真实 no-op 值，同时保留原 valid_mask。"""

    result = _copy_action_inputs(batch)
    for name in ACTION_INPUT_NAMES:
        if name != "valid_mask":
            result[name].zero_()
    return result


def paired_gap_statistics(
    real: Tensor,
    baseline: Tensor,
    *,
    permutations: int = 1024,
) -> Tuple[float, float, float]:
    """返回 gap 的 95% 正态近似下界、上界和单侧置换 p-value。"""

    gaps = (baseline - real).detach().float().reshape(-1)
    mean = gaps.mean()
    if gaps.numel() > 1:
        margin = 1.96 * gaps.std(unbiased=True) / math.sqrt(gaps.numel())
    else:
        margin = gaps.new_zeros(())
    generator = torch.Generator().manual_seed(2026)
    extreme = 0
    completed = 0
    while completed < permutations:
        chunk = min(128, permutations - completed)
        signs = torch.randint(
            0,
            2,
            (chunk, gaps.numel()),
            generator=generator,
            dtype=torch.float32,
        ).mul_(2).sub_(1).to(gaps.device)
        permuted = (signs * gaps.unsqueeze(0)).mean(dim=1)
        extreme += int((permuted >= mean).sum())
        completed += chunk
    p_value = extreme / permutations
    return float(mean - margin), float(mean + margin), float(p_value)


@torch.no_grad()
def action_sensitivity_samples_from_predictions(
    real_prediction: Tensor,
    shuffled_prediction: Tensor,
    noop_prediction: Tensor,
    camera_prediction: Tensor,
    swap_prediction: Tensor,
    targets: Tensor,
    batch: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    """返回可跨 batch/rank 汇总的逐 transition 动作诊断样本。"""

    predictions = {
        "error_real": real_prediction,
        "error_shuffled": shuffled_prediction,
        "error_noop": noop_prediction,
        "error_camera_reversed": camera_prediction,
        "error_attack_use_swapped": swap_prediction,
    }
    if any(prediction.shape != targets.shape for prediction in predictions.values()):
        raise ValueError("action sensitivity predictions and targets must match")
    samples = {
        name: _normalized_interval_error(prediction, targets).reshape(-1)
        for name, prediction in predictions.items()
    }
    bucket_masks = {
        "bucket_movement": batch["movement"].any(dim=-1).any(dim=-1),
        "bucket_interaction": batch["interaction"].any(dim=-1).any(dim=-1),
        "bucket_camera": batch["camera"].abs().sum(dim=-1).gt(0).any(dim=-1),
        "bucket_hotbar": batch["hotbar"].ne(0).any(dim=-1),
        "bucket_gui": batch["gui_open"].any(dim=-1),
    }
    samples.update(
        {name: mask.reshape(-1) for name, mask in bucket_masks.items()}
    )
    return samples


@torch.no_grad()
def action_sensitivity_from_samples(
    samples: Mapping[str, Tensor],
) -> Dict[str, float]:
    """从已汇总的逐 transition 样本计算一次全局动作敏感性统计。"""

    error_names = (
        "error_real",
        "error_shuffled",
        "error_noop",
        "error_camera_reversed",
        "error_attack_use_swapped",
    )
    missing = set(error_names) - samples.keys()
    if missing:
        raise ValueError(f"action sensitivity samples are missing: {sorted(missing)}")
    errors = {
        name: samples[name].detach().float().reshape(-1)
        for name in error_names
    }
    sizes = {values.numel() for values in errors.values()}
    if sizes == {0}:
        raise ValueError("action sensitivity samples cannot be empty")
    if len(sizes) != 1:
        raise ValueError("action sensitivity error samples must have matching sizes")

    real = errors["error_real"]
    shuffled = errors["error_shuffled"]
    noop = errors["error_noop"]
    shuffled_ci_low, shuffled_ci_high, shuffled_p = paired_gap_statistics(
        real,
        shuffled,
    )
    noop_ci_low, noop_ci_high, noop_p = paired_gap_statistics(real, noop)
    real_error = real.mean()
    shuffled_error = shuffled.mean()
    noop_error = noop.mean()
    baseline = (shuffled_error + noop_error) / 2.0
    metrics = {
        "action_sensitivity/error_real": float(real_error),
        "action_sensitivity/error_shuffled": float(shuffled_error),
        "action_sensitivity/error_noop": float(noop_error),
        "action_sensitivity/gap_shuffled": float(shuffled_error - real_error),
        "action_sensitivity/gap_noop": float(noop_error - real_error),
        "action_sensitivity/ratio": float(real_error / baseline.clamp_min(1e-8)),
        "action_sensitivity/gap_shuffled_ci95_low": shuffled_ci_low,
        "action_sensitivity/gap_shuffled_ci95_high": shuffled_ci_high,
        "action_sensitivity/gap_shuffled_pvalue": shuffled_p,
        "action_sensitivity/gap_noop_ci95_low": noop_ci_low,
        "action_sensitivity/gap_noop_ci95_high": noop_ci_high,
        "action_sensitivity/gap_noop_pvalue": noop_p,
        "action_sensitivity/error_camera_reversed": float(
            errors["error_camera_reversed"].mean()
        ),
        "action_sensitivity/error_attack_use_swapped": float(
            errors["error_attack_use_swapped"].mean()
        ),
        "action_sensitivity/pass_statistical": float(
            shuffled_ci_low > 0
            and noop_ci_low > 0
            and shuffled_p < 0.05
            and noop_p < 0.05
        ),
        "action_sensitivity/sample_transitions": float(real.numel()),
    }

    for name in ("movement", "interaction", "camera", "hotbar", "gui"):
        key = f"bucket_{name}"
        if key not in samples:
            continue
        active = samples[key].detach().bool().reshape(-1)
        if active.numel() != real.numel():
            raise ValueError(f"{key} must match action sensitivity sample size")
        if bool(active.any()):
            metrics[f"action_bucket/{name}_l1"] = float(real[active].mean())
    return metrics


@torch.no_grad()
def action_sensitivity_from_predictions(
    real_prediction: Tensor,
    shuffled_prediction: Tensor,
    noop_prediction: Tensor,
    camera_prediction: Tensor,
    swap_prediction: Tensor,
    targets: Tensor,
    batch: Mapping[str, Tensor],
) -> Dict[str, float]:
    """从五组预测计算动作敏感性，允许 FSDP 通过标准 forward 产生预测。"""

    samples = action_sensitivity_samples_from_predictions(
        real_prediction,
        shuffled_prediction,
        noop_prediction,
        camera_prediction,
        swap_prediction,
        targets,
        batch,
    )
    return action_sensitivity_from_samples(samples)


@torch.no_grad()
def action_sensitivity_report(
    model: WorldModel,
    batch: Mapping[str, Tensor],
    *,
    latents: Optional[Tensor] = None,
) -> Dict[str, float]:
    """比较真实、打乱、no-op、反向 camera 和交换交互动作的误差。"""

    if latents is None:
        latents = model.encode_frames(batch["frames"])
    targets = latents[:, 1:]
    real_tokens = model.encode_actions(**_copy_action_inputs(batch))
    real_prediction = model.predictor.predict_teacher_forced(
        latents[:, :-1],
        real_tokens,
    )
    shuffled_tokens = real_tokens.roll(
        1,
        dims=0 if real_tokens.shape[0] > 1 else 1,
    )
    shuffled_prediction = model.predictor.predict_teacher_forced(
        latents[:, :-1],
        shuffled_tokens,
    )
    noop_prediction = model.predictor.predict_teacher_forced(
        latents[:, :-1],
        model.encode_actions(**noop_action_inputs(batch)),
    )

    camera_inputs = _copy_action_inputs(batch)
    camera_inputs["camera"].neg_()
    camera_prediction = model.predictor.predict_teacher_forced(
        latents[:, :-1],
        model.encode_actions(**camera_inputs),
    )

    swap_inputs = _copy_action_inputs(batch)
    attack = swap_inputs["interaction"][..., 0].clone()
    swap_inputs["interaction"][..., 0] = swap_inputs["interaction"][..., 1]
    swap_inputs["interaction"][..., 1] = attack
    swap_prediction = model.predictor.predict_teacher_forced(
        latents[:, :-1],
        model.encode_actions(**swap_inputs),
    )
    return action_sensitivity_from_predictions(
        real_prediction,
        shuffled_prediction,
        noop_prediction,
        camera_prediction,
        swap_prediction,
        targets,
        batch,
    )
