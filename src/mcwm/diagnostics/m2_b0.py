"""M2 B0 smoke-test gate。"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import Tensor

from mcwm.models.world_model import WorldModel
from .world_model import action_sensitivity_report


MODEL_INPUT_NAMES = (
    "frames",
    "movement",
    "interaction",
    "hotbar",
    "camera",
    "cursor",
    "gui_open",
    "cursor_present",
    "valid_mask",
)


def _model_inputs(batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    return {name: batch[name] for name in MODEL_INPUT_NAMES}


def run_b0_smoke_gate(
    model: WorldModel,
    batch: Mapping[str, Tensor],
    *,
    overfit_steps: int = 0,
    learning_rate: float = 1e-3,
) -> Dict[str, float]:
    """运行 shape、冻结、反向传播、固定 batch 过拟合和动作敏感性检查。"""

    frames = batch["frames"]
    if frames.ndim != 5 or frames.shape[1] != 8:
        raise ValueError("B0 batch must contain exactly 8 frames")
    if batch["valid_mask"].shape[1] != 7:
        raise ValueError("B0 batch must contain exactly 7 action blocks")
    if overfit_steps < 0:
        raise ValueError("overfit_steps must be non-negative")

    visual_parameters = tuple(model.visual_encoder.parameters())
    trainable = list(model.trainable_parameters())
    frozen_ok = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in visual_parameters
    )
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_excludes_visual = not any(
        id(parameter) in optimizer_ids for parameter in visual_parameters
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**_model_inputs(batch))
    initial_loss = float(output["loss"].detach())
    output["loss"].backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )
    visual_gradients_absent = all(
        parameter.grad is None for parameter in visual_parameters
    )

    final_loss = initial_loss
    for _ in range(overfit_steps):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = model(**_model_inputs(batch))
        output["loss"].backward()
        final_loss = float(output["loss"].detach())

    model.eval()
    with torch.no_grad():
        sensitivity = action_sensitivity_report(model, batch)
    report = {
        "b0/frozen_encoder": float(frozen_ok),
        "b0/optimizer_excludes_visual": float(optimizer_excludes_visual),
        "b0/visual_gradients_absent": float(visual_gradients_absent),
        "b0/gradients_finite": float(gradients_finite),
        "b0/initial_loss": initial_loss,
        "b0/final_loss": final_loss,
        "b0/overfit_improved": float(final_loss < initial_loss),
        "b0/action_sensitivity_pass": float(
            sensitivity["action_sensitivity/gap_shuffled"] > 0
            and sensitivity["action_sensitivity/gap_noop"] > 0
        ),
    }
    report.update(sensitivity)
    return report
