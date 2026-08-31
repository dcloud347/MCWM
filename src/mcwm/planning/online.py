"""Bounded online MPC smoke loop and machine-readable M4 report."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Protocol

from torch import Tensor

from mcwm.envs.minerl1 import MineRL1EnvWrapper
from .legality import LegalityContext
from .mpc import RecedingHorizonMPC


class GoalProvider(Protocol):
    def goal(self, observation: Tensor, cycle: int) -> Tensor:
        """Return one CHW goal image for the current planning cycle."""


@dataclass(frozen=True)
class StaticGoalProvider:
    goal_image: Tensor

    def goal(self, observation: Tensor, cycle: int) -> Tensor:
        return self.goal_image


def _atomic_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def run_mpc_smoke(
    env: MineRL1EnvWrapper,
    controller: RecedingHorizonMPC,
    world_model: object,
    goal_provider: GoalProvider,
    *,
    cycles: int = 10,
    output_path: Path = Path("m4_smoke.json"),
    legality: LegalityContext = LegalityContext(),
) -> Path:
    """Run at most ``cycles`` replans, always close the env, and write a report."""

    if cycles <= 0:
        raise ValueError("cycles must be positive")
    started = time.perf_counter()
    cycle_reports = []
    macro_counts: Counter = Counter()
    fallback_count = 0
    termination_reason = "cycle_limit"
    pending_error: Optional[BaseException] = None
    try:
        _, observation, _ = env.reset()
        controller.initialize_context(observation)
        for cycle in range(cycles):
            goal = goal_provider.goal(observation, cycle)
            planning_started = time.perf_counter()
            result, actions = controller.plan(
                world_model,
                observation,
                goal,
                context=legality,
            )
            planning_seconds = time.perf_counter() - planning_started
            fallback_count += int(result.fallback_reason is not None)
            selected = [int(value) for value in result.code_ids.detach().cpu().tolist()]
            macro_counts.update(selected[:1])
            reward = 0.0
            executed = 0
            terminated = False
            truncated = False
            environment_seconds = 0.0
            for action in actions:
                tick = env.step_model_tick(action)
                executed += 1
                reward += tick.reward
                environment_seconds += tick.elapsed_seconds
                observation = tick.frame
                controller.record_transition(action, observation)
                terminated = tick.terminated
                truncated = tick.truncated
                if terminated or truncated:
                    break
            cycle_reports.append(
                {
                    "cycle": cycle,
                    "cost": result.cost,
                    "cost_terms": result.cost_terms,
                    "predicted_goal_costs": [
                        float(value)
                        for value in result.predicted_goal_costs.detach().cpu().tolist()
                    ],
                    "selected_macro_codes": selected,
                    "executed_ticks": executed,
                    "planning_seconds": planning_seconds,
                    "environment_seconds": environment_seconds,
                    "fallback_reason": result.fallback_reason,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if terminated or truncated:
                termination_reason = "terminated" if terminated else "truncated"
                break
    except KeyboardInterrupt as exc:
        termination_reason = "manual_interrupt"
        pending_error = exc
    except BaseException as exc:
        termination_reason = f"exception:{type(exc).__name__}"
        pending_error = exc
    finally:
        env.close()
        report = {
            "stage": "m4-minerl-planning-smoke",
            "requested_cycles": cycles,
            "completed_cycles": len(cycle_reports),
            "termination_reason": termination_reason,
            "fallback_count": fallback_count,
            "fallback_ratio": fallback_count / max(1, len(cycle_reports)),
            "selected_first_macro_histogram": {
                str(key): value for key, value in sorted(macro_counts.items())
            },
            "elapsed_seconds": time.perf_counter() - started,
            "cycles": cycle_reports,
        }
        _atomic_report(Path(output_path), report)
    if pending_error is not None:
        raise pending_error
    return Path(output_path)
