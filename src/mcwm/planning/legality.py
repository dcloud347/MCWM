"""Hard legality masks and vectorized macro expansion for planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mcwm.actions.schema import INTERACTION_NAMES, MOVEMENT_NAMES
from .macro_codebook import MacroCodebook


@dataclass(frozen=True)
class LegalityContext:
    """Online state needed to reject codes before expensive rollout."""

    gui_open: bool = False
    gui_planning: bool = False
    valid_hotbar_slots: Optional[Tuple[int, ...]] = None
    max_camera_degrees: float = 30.0

    def __post_init__(self) -> None:
        if self.max_camera_degrees <= 0:
            raise ValueError("max_camera_degrees must be positive")
        if self.valid_hotbar_slots is not None and any(
            not 1 <= slot <= 9 for slot in self.valid_hotbar_slots
        ):
            raise ValueError("valid_hotbar_slots must contain values in 1..9")


def legal_code_mask(
    codebook: MacroCodebook,
    context: LegalityContext = LegalityContext(),
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return a boolean vector over code IDs; invalid codes have zero probability."""

    allowed_slots = (
        None
        if context.valid_hotbar_slots is None
        else set(context.valid_hotbar_slots)
    )
    values = []
    for code in codebook.codes:
        legal = code.legality.v1_supported
        if not context.gui_planning and code.gui_mode != "gameplay":
            legal = False
        # The first planner has no cursor model, so GUI codes remain unsupported even
        # if callers opt in; the flag exists to make that boundary explicit.
        if context.gui_open and code.gui_mode == "gameplay" and code.name != "noop":
            legal = False
        if allowed_slots is not None and any(
            slot != 0 and slot not in allowed_slots for slot in code.hotbar
        ):
            legal = False
        if any(
            abs(value) > context.max_camera_degrees
            for tick in code.camera_mean
            for value in tick
        ):
            legal = False
        values.append(legal)
    mask = torch.tensor(values, dtype=torch.bool, device=device)
    if not bool(mask.any()):
        raise ValueError("legality context leaves no usable macro codes")
    return mask


@dataclass(frozen=True)
class ExpandedActionBatch:
    """Eight model ticks represented at the source/environment action rate."""

    movement: Tensor
    interaction: Tensor
    hotbar: Tensor
    camera: Tensor
    cursor: Tensor
    gui_open: Tensor
    cursor_present: Tensor
    valid_mask: Tensor
    camera_std: Tensor
    residual_limits: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.movement.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.movement.shape[1])

    def action_encoder_kwargs(self) -> Dict[str, Tensor]:
        return {
            "movement": self.movement,
            "interaction": self.interaction,
            "hotbar": self.hotbar,
            "camera": self.camera,
            "cursor": self.cursor,
            "gui_open": self.gui_open,
            "cursor_present": self.cursor_present,
            "valid_mask": self.valid_mask,
        }


def _codebook_tensors(codebook: MacroCodebook, device: torch.device) -> Dict[str, Tensor]:
    return {
        "movement": torch.tensor(
            [code.movement for code in codebook.codes], dtype=torch.bool, device=device
        ),
        "interaction": torch.tensor(
            [code.interaction for code in codebook.codes], dtype=torch.bool, device=device
        ),
        "hotbar": torch.tensor(
            [code.hotbar for code in codebook.codes], dtype=torch.long, device=device
        ),
        "gui_open": torch.tensor(
            [code.gui_open for code in codebook.codes], dtype=torch.bool, device=device
        ),
        "camera_mean": torch.tensor(
            [code.camera_mean for code in codebook.codes], dtype=torch.float32, device=device
        ),
        "camera_std": torch.tensor(
            [code.camera_std for code in codebook.codes], dtype=torch.float32, device=device
        ),
        "residual_limit": torch.tensor(
            [code.camera_residual_max for code in codebook.codes],
            dtype=torch.float32,
            device=device,
        ),
    }


def expand_macro_codes(
    codebook: MacroCodebook,
    code_ids: Tensor,
    camera_residuals: Tensor,
) -> ExpandedActionBatch:
    """Expand code IDs to eight model steps with ``K=action_repeat`` raw ticks."""

    if code_ids.ndim != 2 or code_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("code_ids must be a rank-two integer tensor")
    batch, macros = code_ids.shape
    horizon = macros * codebook.fit_config.macro_length
    if camera_residuals.shape != (batch, horizon, 2):
        raise ValueError("camera_residuals must have shape [B, M*2, 2]")
    if code_ids.numel() and (
        int(code_ids.min()) < 0 or int(code_ids.max()) >= len(codebook.codes)
    ):
        raise ValueError("code_ids contain an out-of-range value")

    device = code_ids.device
    tensors = _codebook_tensors(codebook, device)
    flat_ids = code_ids.long().reshape(-1)

    def gather(name: str) -> Tensor:
        value = tensors[name].index_select(0, flat_ids)
        return value.reshape(batch, macros * 2, *value.shape[2:])

    repeats = codebook.fit_config.action_repeat
    movement = gather("movement").unsqueeze(2).expand(-1, -1, repeats, -1)
    interaction = gather("interaction").unsqueeze(2).expand(-1, -1, repeats, -1)
    hotbar_event = gather("hotbar")
    hotbar = torch.zeros(batch, horizon, repeats, dtype=torch.long, device=device)
    hotbar[:, :, 0] = hotbar_event
    gui_open = gather("gui_open").unsqueeze(2).expand(-1, -1, repeats)
    camera_mean = gather("camera_mean")
    camera_std = gather("camera_std")
    camera_total = camera_mean + camera_residuals.float()
    camera = camera_total.unsqueeze(2).expand(-1, -1, repeats, -1) / repeats
    cursor = torch.zeros(batch, horizon, repeats, 2, device=device, dtype=camera.dtype)
    cursor_present = torch.zeros(
        batch, horizon, repeats, device=device, dtype=torch.bool
    )
    valid_mask = torch.ones(batch, horizon, repeats, device=device, dtype=torch.bool)
    residual_limits = tensors["residual_limit"].index_select(0, flat_ids).reshape(
        batch, macros
    )
    return ExpandedActionBatch(
        movement=movement,
        interaction=interaction,
        hotbar=hotbar,
        camera=camera,
        cursor=cursor,
        gui_open=gui_open,
        cursor_present=cursor_present,
        valid_mask=valid_mask,
        camera_std=camera_std,
        residual_limits=residual_limits,
    )
