"""MineRL 1.0 wrapper at the world model's fixed 4 FPS control rate."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from mcwm.actions.schema import CanonicalActionTick
from mcwm.planning.mpc import canonical_to_minerl_action


def observation_to_tensor(observation: Any) -> Tensor:
    """Convert MineRL ``pov`` observations to training-compatible 3x360x640."""

    value = observation.get("pov") if isinstance(observation, Mapping) else observation
    frame = torch.as_tensor(value)
    if frame.ndim != 3:
        raise ValueError("MineRL observation must be an HWC or CHW RGB image")
    if frame.shape[-1] == 3:
        frame = frame.permute(2, 0, 1)
    elif frame.shape[0] != 3:
        raise ValueError("MineRL observation must have exactly three RGB channels")
    if frame.shape[1:] != (360, 640):
        dtype = frame.dtype
        resized = F.interpolate(
            frame.float().unsqueeze(0),
            size=(360, 640),
            mode="bilinear",
            align_corners=False,
        )[0]
        frame = (
            resized.round().clamp(0, 255).to(dtype)
            if dtype == torch.uint8
            else resized.to(dtype)
        )
    return frame.contiguous()


@dataclass(frozen=True)
class MineRLModelTick:
    observation: Any
    frame: Tensor
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    action_repeats: int
    elapsed_seconds: float


class MineRL1EnvWrapper:
    """Map one 4 FPS model tick to repeated MineRL environment actions."""

    def __init__(
        self,
        env: object,
        *,
        model_fps: float = 4.0,
        environment_fps: float = 20.0,
        action_repeat: Optional[int] = None,
    ) -> None:
        if model_fps <= 0 or environment_fps <= 0:
            raise ValueError("environment and model FPS must be positive")
        resolved_repeat = (
            int(action_repeat)
            if action_repeat is not None
            else int(round(environment_fps / model_fps))
        )
        if resolved_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        self.env = env
        self.model_fps = float(model_fps)
        self.environment_fps = float(environment_fps)
        self.action_repeat = resolved_repeat
        self.closed = False

    def reset(self, **kwargs: object) -> Tuple[Any, Tensor, Mapping[str, Any]]:
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        return observation, observation_to_tensor(observation), info

    def _environment_action(
        self,
        action: CanonicalActionTick,
        *,
        repeat_index: int,
    ) -> Mapping[str, Any]:
        planned = dict(canonical_to_minerl_action(action))
        planned["camera"] = [value / self.action_repeat for value in action.camera]
        if repeat_index > 0:
            for slot in range(1, 10):
                planned[f"hotbar.{slot}"] = 0
        action_space = getattr(self.env, "action_space", None)
        noop = getattr(action_space, "noop", None)
        if callable(noop):
            result: Dict[str, Any] = dict(noop())
            for name, value in planned.items():
                if name not in result:
                    continue
                reference = result[name]
                try:
                    import numpy as np  # type: ignore

                    array = np.asarray(value, dtype=np.asarray(reference).dtype)
                    result[name] = array.reshape(np.asarray(reference).shape)
                except (ImportError, TypeError, ValueError):
                    result[name] = value
        else:
            result = planned
        contains = getattr(action_space, "contains", None)
        if callable(contains) and not bool(contains(result)):
            raise ValueError("planned action does not belong to MineRL action_space")
        return result

    def step_model_tick(self, action: CanonicalActionTick) -> MineRLModelTick:
        if self.closed:
            raise RuntimeError("cannot step a closed MineRL environment")
        started = time.perf_counter()
        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None
        info: Mapping[str, Any] = {}
        repeats = 0
        for repeat_index in range(self.action_repeat):
            raw_action = self._environment_action(
                action,
                repeat_index=repeat_index,
            )
            result = self.env.step(raw_action)
            repeats += 1
            if not isinstance(result, tuple):
                raise ValueError("MineRL env.step must return a tuple")
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
            elif len(result) == 4:
                observation, reward, done, info = result
                terminated, truncated = bool(done), False
            else:
                raise ValueError("MineRL env.step must return four or five values")
            total_reward += float(reward)
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("MineRL action repeat produced no observation")
        return MineRLModelTick(
            observation=observation,
            frame=observation_to_tensor(observation),
            reward=total_reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            action_repeats=repeats,
            elapsed_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        if not self.closed:
            self.env.close()
            self.closed = True

    def __enter__(self) -> "MineRL1EnvWrapper":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
