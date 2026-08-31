"""Hybrid categorical/Gaussian CEM for goal-conditioned latent planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from mcwm.actions.schema import CanonicalActionTick
from .legality import (
    ExpandedActionBatch,
    LegalityContext,
    expand_macro_codes,
    legal_code_mask,
)
from .macro_codebook import MacroCodebook


@dataclass(frozen=True)
class HybridCEMConfig:
    """Defaults for a four-macro, eight-model-tick planning horizon."""

    macro_horizon: int = 4
    candidates: int = 64
    elites: int = 8
    iterations: int = 4
    candidate_chunk_size: Optional[int] = None
    categorical_smoothing: float = 0.25
    update_rate: float = 0.8
    initial_residual_std: float = 0.25
    minimum_residual_std: float = 0.05
    maximum_residual_std: float = 10.0
    goal_weight: float = 1.0
    goal_l1_weight: float = 1.0
    goal_cosine_weight: float = 1.0
    action_change_weight: float = 0.05
    camera_residual_weight: float = 0.01
    invalid_action_penalty: float = 1_000_000.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if min(self.macro_horizon, self.candidates, self.elites, self.iterations) <= 0:
            raise ValueError("CEM sizes must be positive")
        if self.elites > self.candidates:
            raise ValueError("elites cannot exceed candidates")
        if self.candidate_chunk_size is not None and self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive")
        if self.categorical_smoothing < 0:
            raise ValueError("categorical_smoothing cannot be negative")
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")
        if not 0 < self.minimum_residual_std <= self.initial_residual_std:
            raise ValueError("residual standard deviations are inconsistent")
        if self.maximum_residual_std < self.initial_residual_std:
            raise ValueError("maximum_residual_std is too small")
        weights = (
            self.goal_weight,
            self.goal_l1_weight,
            self.goal_cosine_weight,
            self.action_change_weight,
            self.camera_residual_weight,
            self.invalid_action_penalty,
        )
        if min(weights) < 0 or not all(math.isfinite(value) for value in weights):
            raise ValueError("objective weights cannot be negative")


@dataclass(frozen=True)
class PlanResult:
    """Best complete plan; callers execute only its first two-tick macro."""

    code_ids: Tensor
    camera_residuals: Tensor
    cost: float
    cost_terms: Dict[str, float]
    code_probabilities: Tensor
    residual_mean: Tensor
    residual_std: Tensor
    predicted_goal_costs: Tensor
    fallback_reason: Optional[str] = None


def _latent_goal_cost(
    prediction: Tensor,
    goal: Tensor,
    *,
    l1_weight: float,
    cosine_weight: float,
) -> Tensor:
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    goal = F.layer_norm(goal.float(), (goal.shape[-1],))
    l1 = (prediction - goal).abs().mean(dim=(-2, -1))
    cosine = 1.0 - F.cosine_similarity(prediction, goal, dim=-1).mean(dim=-1)
    return l1_weight * l1 + cosine_weight * cosine


def _previous_binary(
    action: Optional[CanonicalActionTick], *, device: torch.device
) -> Optional[Tensor]:
    if action is None:
        return None
    return torch.tensor(
        (*action.movement, *action.interaction), dtype=torch.bool, device=device
    )


def _action_change_cost(
    expanded: ExpandedActionBatch,
    previous_action: Optional[CanonicalActionTick],
) -> Tensor:
    binary = torch.cat((expanded.movement, expanded.interaction), dim=-1)[:, :, 0]
    changes = (binary[:, 1:] != binary[:, :-1]).float().mean(dim=-1)
    previous = _previous_binary(previous_action, device=binary.device)
    if previous is not None:
        initial = (binary[:, 0] != previous).float().mean(dim=-1, keepdim=True)
        changes = torch.cat((initial, changes), dim=1)
    binary_cost = changes.mean(dim=1) if changes.shape[1] else torch.zeros(
        binary.shape[0], device=binary.device
    )

    hotbar = expanded.hotbar[:, :, 0]
    hotbar_changes = (hotbar[:, 1:] != hotbar[:, :-1]).float()
    hotbar_cost = (
        hotbar_changes.mean(dim=1)
        if hotbar_changes.shape[1]
        else torch.zeros_like(binary_cost)
    )
    return binary_cost + 0.25 * hotbar_cost


def _camera_cost_and_invalid(
    expanded: ExpandedActionBatch,
    residuals: Tensor,
    context: LegalityContext,
) -> tuple:
    normalized = residuals / expanded.camera_std.clamp_min(1e-6)
    camera_cost = normalized.square().mean(dim=(1, 2))
    macros = expanded.residual_limits.shape[1]
    residual_norm = normalized.reshape(normalized.shape[0], macros, 2, 2).square()
    residual_norm = residual_norm.sum(dim=(2, 3)).sqrt()
    residual_invalid = (residual_norm > expanded.residual_limits).any(dim=1)
    camera_total = expanded.camera.sum(dim=2)
    camera_invalid = (camera_total.abs() > context.max_camera_degrees).any(dim=(1, 2))
    return camera_cost, residual_invalid | camera_invalid


class HybridCEMPlanner:
    """Optimize macro IDs and per-tick camera residuals without gradients."""

    def __init__(
        self,
        codebook: MacroCodebook,
        config: HybridCEMConfig = HybridCEMConfig(),
    ) -> None:
        if codebook.fit_config.macro_length != 2:
            raise ValueError("HybridCEMPlanner requires two-tick macro codes")
        self.codebook = codebook
        self.config = config

    def _score_chunk(
        self,
        world_model: object,
        current_latent: Tensor,
        goal_latent: Tensor,
        code_ids: Tensor,
        residuals: Tensor,
        context: LegalityContext,
        previous_action: Optional[CanonicalActionTick],
        context_latents: Optional[Tensor] = None,
        context_action_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        expanded = expand_macro_codes(self.codebook, code_ids, residuals)
        action_tokens = world_model.encode_actions(**expanded.action_encoder_kwargs())
        initial = current_latent.expand(code_ids.shape[0], -1, -1)
        if context_latents is None:
            predictions = world_model.predictor.rollout(initial, action_tokens)
        else:
            if context_action_tokens is None:
                raise ValueError("context_action_tokens are required with context_latents")
            history = context_latents.expand(code_ids.shape[0], -1, -1, -1)
            history_actions = context_action_tokens.expand(
                code_ids.shape[0], -1, -1
            )
            predictions = world_model.predictor.rollout_with_context(
                history,
                history_actions,
                action_tokens,
            )
        goal = goal_latent.expand(code_ids.shape[0], -1, -1)
        trajectory_goal_cost = _latent_goal_cost(
            predictions,
            goal.unsqueeze(1),
            l1_weight=self.config.goal_l1_weight,
            cosine_weight=self.config.goal_cosine_weight,
        )
        goal_cost = trajectory_goal_cost[:, -1]
        action_cost = _action_change_cost(expanded, previous_action)
        camera_cost, invalid = _camera_cost_and_invalid(expanded, residuals, context)
        nonfinite = ~(
            torch.isfinite(goal_cost)
            & torch.isfinite(action_cost)
            & torch.isfinite(camera_cost)
        )
        invalid = invalid | nonfinite
        goal_cost = torch.nan_to_num(
            goal_cost,
            nan=self.config.invalid_action_penalty,
            posinf=self.config.invalid_action_penalty,
            neginf=self.config.invalid_action_penalty,
        )
        action_cost = torch.nan_to_num(action_cost, nan=self.config.invalid_action_penalty)
        camera_cost = torch.nan_to_num(camera_cost, nan=self.config.invalid_action_penalty)
        total = (
            self.config.goal_weight * goal_cost
            + self.config.action_change_weight * action_cost
            + self.config.camera_residual_weight * camera_cost
            + self.config.invalid_action_penalty * invalid.float()
        )
        return {
            "total": total,
            "goal": goal_cost,
            "action_change": action_cost,
            "camera_residual": camera_cost,
            "invalid": invalid.float(),
            "trajectory_goal": trajectory_goal_cost,
        }

    def _fallback(
        self,
        *,
        device: torch.device,
        reason: str,
        legal: Tensor,
    ) -> PlanResult:
        noop = next(code for code in self.codebook.codes if code.name == "noop")
        codes = torch.full(
            (self.config.macro_horizon,), noop.code_id, dtype=torch.long, device=device
        )
        horizon = self.config.macro_horizon * self.codebook.fit_config.macro_length
        residuals = torch.zeros(horizon, 2, device=device)
        probabilities = torch.zeros(
            self.config.macro_horizon, len(self.codebook.codes), device=device
        )
        probabilities[:, noop.code_id] = 1.0
        # A guaranteed no-op is allowed as the emergency action even if an online GUI
        # state made every data-conditioned code ineligible.
        return PlanResult(
            code_ids=codes,
            camera_residuals=residuals,
            cost=float(self.config.invalid_action_penalty),
            cost_terms={
                "goal": 0.0,
                "action_change": 0.0,
                "camera_residual": 0.0,
                "invalid": 0.0,
            },
            code_probabilities=probabilities,
            residual_mean=residuals.clone(),
            residual_std=torch.zeros_like(residuals),
            predicted_goal_costs=torch.zeros(horizon, device=device),
            fallback_reason=reason,
        )

    @torch.no_grad()
    def plan_latents(
        self,
        world_model: object,
        current_latent: Tensor,
        goal_latent: Tensor,
        *,
        context: LegalityContext = LegalityContext(),
        previous_action: Optional[CanonicalActionTick] = None,
        seed_offset: int = 0,
        context_latents: Optional[Tensor] = None,
        context_action_tokens: Optional[Tensor] = None,
    ) -> PlanResult:
        """Plan from ``[1,S,D]`` current/goal latents and return a full horizon."""

        if current_latent.ndim == 2:
            current_latent = current_latent.unsqueeze(0)
        if goal_latent.ndim == 2:
            goal_latent = goal_latent.unsqueeze(0)
        if current_latent.ndim != 3 or goal_latent.shape != current_latent.shape:
            raise ValueError("current_latent and goal_latent must match [1, S, D]")
        if current_latent.shape[0] != 1:
            raise ValueError("online planning accepts one current state at a time")
        device = current_latent.device
        if goal_latent.device != device:
            raise ValueError("current and goal latents must share a device")
        if (context_latents is None) != (context_action_tokens is None):
            raise ValueError("context latents and actions must be provided together")
        if context_latents is not None:
            if context_latents.ndim != 4 or context_latents.shape[0] != 1:
                raise ValueError("context_latents must have shape [1, C, S, D]")
            if context_latents.shape[2:] != current_latent.shape[1:]:
                raise ValueError("context latent dimensions must match current_latent")
            if context_action_tokens.shape[:2] != (
                1,
                context_latents.shape[1] - 1,
            ):
                raise ValueError("context actions must align C-1 observed transitions")

        try:
            legal = legal_code_mask(self.codebook, context, device=device)
        except ValueError:
            legal = torch.zeros(len(self.codebook.codes), dtype=torch.bool, device=device)
            return self._fallback(device=device, reason="no_legal_codes", legal=legal)
        legal_float = legal.float()
        probabilities = legal_float.expand(self.config.macro_horizon, -1).clone()
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        horizon = self.config.macro_horizon * self.codebook.fit_config.macro_length
        residual_mean = torch.zeros(horizon, 2, device=device)
        residual_std = torch.full(
            (horizon, 2), self.config.initial_residual_std, device=device
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(self.config.seed + int(seed_offset))

        best_cost = math.inf
        best_codes = None
        best_residuals = None
        best_terms = None
        best_trajectory = None
        chunk_size = self.config.candidate_chunk_size or self.config.candidates
        for _ in range(self.config.iterations):
            sampled_codes = torch.stack(
                [
                    torch.multinomial(
                        probabilities[position],
                        self.config.candidates,
                        replacement=True,
                        generator=generator,
                    )
                    for position in range(self.config.macro_horizon)
                ],
                dim=1,
            )
            sampled_residuals = (
                residual_mean.unsqueeze(0)
                + residual_std.unsqueeze(0)
                * torch.randn(
                    self.config.candidates,
                    horizon,
                    2,
                    device=device,
                    generator=generator,
                )
            )
            term_chunks: Dict[str, list] = {}
            for start in range(0, self.config.candidates, chunk_size):
                end = min(start + chunk_size, self.config.candidates)
                try:
                    terms = self._score_chunk(
                        world_model,
                        current_latent,
                        goal_latent,
                        sampled_codes[start:end],
                        sampled_residuals[start:end],
                        context,
                        previous_action,
                        context_latents,
                        context_action_tokens,
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return self._fallback(
                        device=device,
                        reason="out_of_memory",
                        legal=legal,
                    )
                for name, value in terms.items():
                    term_chunks.setdefault(name, []).append(value)
            scored = {name: torch.cat(values) for name, values in term_chunks.items()}
            valid_indices = torch.nonzero(scored["invalid"] == 0, as_tuple=False).flatten()
            if not len(valid_indices):
                return self._fallback(
                    device=device,
                    reason="all_candidates_invalid",
                    legal=legal,
                )
            elite_count = min(self.config.elites, len(valid_indices))
            valid_costs = scored["total"].index_select(0, valid_indices)
            elite_indices = valid_indices.index_select(
                0, torch.topk(valid_costs, elite_count, largest=False).indices
            )
            iteration_best = int(elite_indices[0])
            iteration_cost = float(scored["total"][iteration_best])
            if iteration_cost < best_cost:
                best_cost = iteration_cost
                best_codes = sampled_codes[iteration_best].clone()
                best_residuals = sampled_residuals[iteration_best].clone()
                best_terms = {
                    name: float(value[iteration_best])
                    for name, value in scored.items()
                    if name not in {"total", "trajectory_goal"}
                }
                best_trajectory = scored["trajectory_goal"][iteration_best].clone()

            elite_codes = sampled_codes.index_select(0, elite_indices)
            counts = torch.zeros_like(probabilities)
            counts.scatter_add_(
                1,
                elite_codes.transpose(0, 1),
                torch.ones_like(elite_codes.transpose(0, 1), dtype=counts.dtype),
            )
            counts += self.config.categorical_smoothing * legal_float.unsqueeze(0)
            counts *= legal_float.unsqueeze(0)
            updated_probabilities = counts / counts.sum(dim=1, keepdim=True)
            probabilities = (
                (1.0 - self.config.update_rate) * probabilities
                + self.config.update_rate * updated_probabilities
            )
            probabilities *= legal_float.unsqueeze(0)
            probabilities /= probabilities.sum(dim=1, keepdim=True)

            elite_residuals = sampled_residuals.index_select(0, elite_indices)
            updated_mean = elite_residuals.mean(dim=0)
            updated_std = elite_residuals.std(dim=0, unbiased=False).clamp(
                self.config.minimum_residual_std,
                self.config.maximum_residual_std,
            )
            residual_mean = (
                (1.0 - self.config.update_rate) * residual_mean
                + self.config.update_rate * updated_mean
            )
            residual_std = (
                (1.0 - self.config.update_rate) * residual_std
                + self.config.update_rate * updated_std
            ).clamp(
                self.config.minimum_residual_std,
                self.config.maximum_residual_std,
            )

        if (
            best_codes is None
            or best_residuals is None
            or best_terms is None
            or best_trajectory is None
        ):
            raise RuntimeError("CEM did not produce a finite plan")
        return PlanResult(
            code_ids=best_codes,
            camera_residuals=best_residuals,
            cost=best_cost,
            cost_terms=best_terms,
            code_probabilities=probabilities,
            residual_mean=residual_mean,
            residual_std=residual_std,
            predicted_goal_costs=best_trajectory,
        )

    @torch.no_grad()
    def plan_observations(
        self,
        world_model: object,
        observation: Tensor,
        goal_image: Tensor,
        **kwargs: object,
    ) -> PlanResult:
        """Encode one observation and goal image, then invoke latent planning."""

        def frame(value: Tensor) -> Tensor:
            if value.ndim == 3:
                value = value.unsqueeze(0).unsqueeze(0)
            elif value.ndim == 4:
                value = value.unsqueeze(1)
            if value.ndim != 5 or value.shape[:2] != (1, 1):
                raise ValueError("observation images must describe one [C,H,W] frame")
            return value

        current = world_model.encode_frames(frame(observation))[:, 0]
        goal = world_model.encode_frames(frame(goal_image))[:, 0]
        return self.plan_latents(world_model, current, goal, **kwargs)
