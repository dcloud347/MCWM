import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import torch

    from mcwm.envs.minerl1 import MineRL1EnvWrapper
    from mcwm.planning.cem import HybridCEMConfig, HybridCEMPlanner
    from mcwm.planning.macro_codebook import (
        MacroCodebookFitConfig,
        fit_macro_codebook_from_episodes,
    )
    from mcwm.planning.mpc import RecedingHorizonMPC
    from mcwm.planning.online import StaticGoalProvider, run_mpc_smoke
except ModuleNotFoundError:
    torch = None


class _Predictor:
    def rollout(self, initial, actions):
        return initial.unsqueeze(1) + actions.cumsum(dim=1).unsqueeze(2)

    def rollout_with_context(self, context, history_actions, actions):
        return self.rollout(context[:, -1], actions)


class _WorldModel:
    def __init__(self):
        self.predictor = _Predictor()

    def encode_frames(self, frames):
        return frames.float().mean(dim=(-2, -1)).unsqueeze(2)

    def encode_actions(self, movement, interaction, hotbar, camera, **kwargs):
        return torch.stack(
            (
                movement[:, :, 0, 0].float(),
                camera.sum(dim=2)[..., 1] / 5.0,
                interaction[:, :, 0, 0].float(),
            ),
            dim=-1,
        )


class _ActionSpace:
    def contains(self, action):
        return "camera" in action


class _Environment:
    action_space = _ActionSpace()

    def __init__(self):
        self.closed = False

    def reset(self):
        return {"pov": torch.zeros(360, 640, 3, dtype=torch.uint8)}, {}

    def step(self, action):
        return {"pov": torch.zeros(360, 640, 3, dtype=torch.uint8)}, 0.0, False, False, {}

    def close(self):
        self.closed = True


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OnlineSmokeTest(unittest.TestCase):
    def test_mock_environment_completes_ten_replanning_cycles_and_closes(self):
        fit = MacroCodebookFitConfig(
            min_group_samples=1,
            min_cluster_samples=1,
            max_codes=16,
        )
        codebook = fit_macro_codebook_from_episodes((), manifest_hash="m", config=fit)
        planner = HybridCEMPlanner(
            codebook,
            HybridCEMConfig(
                candidates=8,
                elites=2,
                iterations=2,
                candidate_chunk_size=4,
                initial_residual_std=0.1,
            ),
        )
        controller = RecedingHorizonMPC(planner)
        raw_env = _Environment()
        env = MineRL1EnvWrapper(raw_env, action_repeat=1)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "m4_smoke.json"
            run_mpc_smoke(
                env,
                controller,
                _WorldModel(),
                StaticGoalProvider(torch.zeros(3, 360, 640, dtype=torch.uint8)),
                cycles=10,
                output_path=output,
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(raw_env.closed)
        self.assertEqual(report["completed_cycles"], 10)
        self.assertEqual(report["termination_reason"], "cycle_limit")
        self.assertEqual(len(report["cycles"]), 10)


if __name__ == "__main__":
    unittest.main()
