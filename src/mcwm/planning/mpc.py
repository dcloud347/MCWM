"""Receding-horizon wrapper that exposes only the first two-tick macro."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import torch
from torch import Tensor

from mcwm.actions.schema import (
    INTERACTION_NAMES,
    MOVEMENT_NAMES,
    ActionSource,
    CanonicalActionTick,
)
from .cem import HybridCEMPlanner, PlanResult
from .legality import LegalityContext


def first_macro_actions(
    planner: HybridCEMPlanner,
    result: PlanResult,
    *,
    start_timestamp_ms: int = 0,
    tick_ms: int = 250,
) -> Tuple[CanonicalActionTick, CanonicalActionTick]:
    """Decode only the first macro; the remaining six ticks are never executed."""

    if tick_ms <= 0:
        raise ValueError("tick_ms must be positive")
    code = planner.codebook.codes[int(result.code_ids[0])]
    actions = []
    for tick in range(2):
        residual = result.camera_residuals[tick]
        camera = tuple(
            code.camera_mean[tick][dimension] + float(residual[dimension])
            for dimension in range(2)
        )
        actions.append(
            CanonicalActionTick(
                movement=code.movement[tick],
                interaction=code.interaction[tick],
                hotbar=code.hotbar[tick],
                camera=camera,  # type: ignore[arg-type]
                cursor=None,
                gui_open=code.gui_open[tick],
                valid=True,
                timestamp_ms=start_timestamp_ms + tick * tick_ms,
                source=ActionSource.MINERL,
            )
        )
    return tuple(actions)  # type: ignore[return-value]


def canonical_to_minerl_action(action: CanonicalActionTick) -> Mapping[str, object]:
    """Convert a planned canonical tick to the MineRL 1.0 action dictionary."""

    result = {
        name: int(action.movement[index])
        for index, name in enumerate(MOVEMENT_NAMES)
    }
    interaction_names = {
        "attack": "attack",
        "use": "use",
        "drop": "drop",
        "pick_item": "pickItem",
        "swap_hands": "swapHands",
        "inventory": "inventory",
        "esc": "ESC",
    }
    result.update(
        {
            interaction_names[name]: int(action.interaction[index])
            for index, name in enumerate(INTERACTION_NAMES)
        }
    )
    result["camera"] = list(action.camera)
    for slot in range(1, 10):
        result[f"hotbar.{slot}"] = int(action.hotbar == slot)
    return result


@dataclass
class RecedingHorizonMPC:
    """Stateful controller: plan eight ticks, execute two, then call again."""

    planner: HybridCEMPlanner
    planning_round: int = 0
    previous_action: Optional[CanonicalActionTick] = None
    max_context: int = 16
    context_frames: List[Tensor] = field(default_factory=list)
    context_actions: List[CanonicalActionTick] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_context < 2:
            raise ValueError("max_context must be at least two")

    def initialize_context(self, observation: Tensor, *, tick_ms: int = 250) -> None:
        """Warm up history with the first frame and valid no-op transitions."""

        if observation.ndim != 3:
            raise ValueError("observation must have shape [C, H, W]")
        frame = observation.detach()
        self.context_frames = [frame] * self.max_context
        self.context_actions = [
            CanonicalActionTick.noop(
                -(self.max_context - index - 1) * tick_ms,
                ActionSource.MINERL,
            )
            for index in range(self.max_context - 1)
        ]

    def record_transition(
        self,
        action: CanonicalActionTick,
        observation: Tensor,
    ) -> None:
        """Append the observation obtained after executing one model tick."""

        if observation.ndim != 3:
            raise ValueError("observation must have shape [C, H, W]")
        if not self.context_frames:
            raise RuntimeError("initialize_context must be called before recording")
        self.context_actions.append(action)
        self.context_frames.append(observation.detach())
        self.context_frames = self.context_frames[-self.max_context :]
        self.context_actions = self.context_actions[-(self.max_context - 1) :]

    def _encode_context_actions(self, world_model: object, device: torch.device) -> Tensor:
        actions = self.context_actions
        intervals = len(actions)
        repeats = self.planner.codebook.fit_config.action_repeat
        movement = torch.tensor(
            [[action.movement for action in actions]], dtype=torch.bool, device=device
        ).unsqueeze(2).expand(-1, -1, repeats, -1)
        interaction = torch.tensor(
            [[action.interaction for action in actions]], dtype=torch.bool, device=device
        ).unsqueeze(2).expand(-1, -1, repeats, -1)
        hotbar_event = torch.tensor(
            [[action.hotbar for action in actions]], dtype=torch.long, device=device
        )
        hotbar = torch.zeros(1, intervals, repeats, dtype=torch.long, device=device)
        hotbar[:, :, 0] = hotbar_event
        camera = torch.tensor(
            [[action.camera for action in actions]], dtype=torch.float32, device=device
        ).unsqueeze(2).expand(-1, -1, repeats, -1) / repeats
        gui_open = torch.tensor(
            [[action.gui_open for action in actions]], dtype=torch.bool, device=device
        ).unsqueeze(2).expand(-1, -1, repeats)
        cursor = torch.zeros(1, intervals, repeats, 2, device=device)
        cursor_present = torch.zeros(
            1, intervals, repeats, dtype=torch.bool, device=device
        )
        valid_mask = torch.ones(
            1, intervals, repeats, dtype=torch.bool, device=device
        )
        return world_model.encode_actions(
            movement=movement,
            interaction=interaction,
            hotbar=hotbar,
            camera=camera,
            cursor=cursor,
            gui_open=gui_open,
            cursor_present=cursor_present,
            valid_mask=valid_mask,
        )

    def plan(
        self,
        world_model: object,
        observation: Tensor,
        goal_image: Tensor,
        *,
        context: LegalityContext = LegalityContext(),
        start_timestamp_ms: Optional[int] = None,
        tick_ms: Optional[int] = None,
    ) -> Tuple[PlanResult, Tuple[CanonicalActionTick, CanonicalActionTick]]:
        resolved_tick_ms = (
            self.planner.codebook.fit_config.model_tick_ms
            if tick_ms is None
            else tick_ms
        )
        if not self.context_frames:
            self.initialize_context(observation, tick_ms=resolved_tick_ms)
        else:
            # The caller may pass the same latest observation again at the cycle
            # boundary. Intermediate tick observations enter via record_transition().
            self.context_frames[-1] = observation.detach()
        try:
            model_device = next(world_model.parameters()).device
        except (AttributeError, StopIteration):
            model_device = observation.device
        frames = torch.stack(self.context_frames).unsqueeze(0).to(model_device)
        context_latents = world_model.encode_frames(frames)
        context_action_tokens = self._encode_context_actions(
            world_model, context_latents.device
        )
        goal = goal_image
        if goal.ndim == 3:
            goal = goal.unsqueeze(0).unsqueeze(0)
        elif goal.ndim == 4:
            goal = goal.unsqueeze(1)
        if goal.ndim != 5 or goal.shape[:2] != (1, 1):
            raise ValueError("goal_image must describe one [C, H, W] frame")
        goal_latent = world_model.encode_frames(goal.to(model_device))[:, 0]
        result = self.planner.plan_latents(
            world_model,
            context_latents[:, -1],
            goal_latent,
            context=context,
            previous_action=self.previous_action,
            seed_offset=self.planning_round,
            context_latents=context_latents,
            context_action_tokens=context_action_tokens,
        )
        execute = first_macro_actions(
            self.planner,
            result,
            start_timestamp_ms=(
                self.planning_round * 2 * resolved_tick_ms
                if start_timestamp_ms is None
                else start_timestamp_ms
            ),
            tick_ms=resolved_tick_ms,
        )
        self.previous_action = execute[-1]
        self.planning_round += 1
        return result, execute
