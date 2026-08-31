import unittest

try:
    import torch

    from mcwm.planning.cem import HybridCEMConfig, HybridCEMPlanner
    from mcwm.planning.macro_codebook import (
        MacroCodebookFitConfig,
        fit_macro_codebook_from_episodes,
    )
    from mcwm.planning.mpc import canonical_to_minerl_action, first_macro_actions
    from mcwm.planning.mpc import RecedingHorizonMPC
    from tests.models.test_world_model import _model as _tiny_world_model
except ModuleNotFoundError:
    torch = None


class _FakePredictor:
    def rollout(self, initial, actions):
        increments = actions.cumsum(dim=1).unsqueeze(2)
        return initial.unsqueeze(1) + increments

    def rollout_with_context(self, context, history_actions, actions):
        return self.rollout(context[:, -1], actions)


class _FakeWorldModel:
    def __init__(self):
        self.predictor = _FakePredictor()

    def encode_actions(
        self,
        movement,
        interaction,
        hotbar,
        camera,
        cursor,
        gui_open,
        cursor_present,
        valid_mask,
    ):
        forward = movement[:, :, 0, 0].float()
        yaw = camera.sum(dim=2)[..., 1] / 5.0
        attack = interaction[:, :, 0, 0].float()
        return torch.stack((forward, yaw, attack), dim=-1)

    def encode_frames(self, frames):
        return frames.float().mean(dim=(-2, -1)).unsqueeze(2)


def _planner(chunk_size):
    fit = MacroCodebookFitConfig(
        min_group_samples=1,
        min_cluster_samples=1,
        max_codes=16,
    )
    codebook = fit_macro_codebook_from_episodes((), manifest_hash="m", config=fit)
    config = HybridCEMConfig(
        candidates=24,
        elites=6,
        iterations=3,
        candidate_chunk_size=chunk_size,
        initial_residual_std=0.2,
        minimum_residual_std=0.02,
        seed=9,
    )
    return HybridCEMPlanner(codebook, config)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class HybridCEMTest(unittest.TestCase):
    def test_plan_is_deterministic_and_has_four_macros_eight_residuals(self):
        planner = _planner(5)
        current = torch.zeros(1, 1, 3)
        goal = torch.tensor([[[8.0, 0.0, 0.0]]])

        first = planner.plan_latents(_FakeWorldModel(), current, goal)
        second = planner.plan_latents(_FakeWorldModel(), current, goal)

        self.assertTrue(torch.equal(first.code_ids, second.code_ids))
        self.assertTrue(torch.equal(first.camera_residuals, second.camera_residuals))
        self.assertEqual(tuple(first.code_ids.shape), (4,))
        self.assertEqual(tuple(first.camera_residuals.shape), (8, 2))
        self.assertGreaterEqual(first.cost, 0.0)
        self.assertEqual(set(first.cost_terms), {
            "goal", "action_change", "camera_residual", "invalid"
        })

    def test_candidate_chunking_does_not_change_plan(self):
        current = torch.zeros(1, 1, 3)
        goal = torch.tensor([[[8.0, 0.0, 0.0]]])
        unchunked = _planner(None).plan_latents(_FakeWorldModel(), current, goal)
        chunked = _planner(4).plan_latents(_FakeWorldModel(), current, goal)

        self.assertTrue(torch.equal(unchunked.code_ids, chunked.code_ids))
        self.assertTrue(torch.equal(unchunked.camera_residuals, chunked.camera_residuals))
        self.assertAlmostEqual(unchunked.cost, chunked.cost, places=7)

    def test_only_first_macro_is_decoded_for_execution(self):
        planner = _planner(6)
        result = planner.plan_latents(
            _FakeWorldModel(),
            torch.zeros(1, 1, 3),
            torch.tensor([[[8.0, 0.0, 0.0]]]),
        )
        actions = first_macro_actions(planner, result, start_timestamp_ms=100)
        raw = canonical_to_minerl_action(actions[0])

        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].timestamp_ms, 100)
        self.assertEqual(actions[1].timestamp_ms, 350)
        self.assertIn("camera", raw)
        self.assertIn("hotbar.9", raw)

    def test_all_invalid_samples_return_deterministic_noop_fallback(self):
        planner = _planner(6)
        config = HybridCEMConfig(
            candidates=8,
            elites=2,
            iterations=1,
            initial_residual_std=100.0,
            minimum_residual_std=0.1,
            maximum_residual_std=100.0,
            seed=3,
        )
        planner = HybridCEMPlanner(planner.codebook, config)
        result = planner.plan_latents(
            _FakeWorldModel(),
            torch.zeros(1, 1, 3),
            torch.ones(1, 1, 3),
        )

        noop = next(code.code_id for code in planner.codebook.codes if code.name == "noop")
        self.assertEqual(result.fallback_reason, "all_candidates_invalid")
        self.assertEqual(result.code_ids.tolist(), [noop] * 4)
        self.assertTrue(torch.isfinite(torch.tensor(result.cost)))

    def test_mpc_warmup_uses_sixteen_frames_and_executes_two_ticks(self):
        planner = _planner(6)
        controller = RecedingHorizonMPC(planner)
        observation = torch.zeros(3, 2, 2)
        goal = torch.zeros(3, 2, 2)

        _, actions = controller.plan(_FakeWorldModel(), observation, goal)

        self.assertEqual(len(controller.context_frames), 16)
        self.assertEqual(len(controller.context_actions), 15)
        self.assertTrue(all(action.valid for action in controller.context_actions))
        self.assertEqual(len(actions), 2)

    def test_real_action_encoder_and_predictor_accept_eight_step_plan(self):
        base = _planner(2)
        planner = HybridCEMPlanner(
            base.codebook,
            HybridCEMConfig(
                candidates=4,
                elites=2,
                iterations=1,
                candidate_chunk_size=2,
                initial_residual_std=0.05,
                minimum_residual_std=0.01,
            ),
        )
        model = _tiny_world_model().eval()
        result = planner.plan_latents(
            model,
            torch.randn(1, 4, 12),
            torch.randn(1, 4, 12),
        )

        self.assertEqual(tuple(result.predicted_goal_costs.shape), (8,))
        self.assertTrue(torch.isfinite(torch.tensor(result.cost)))


if __name__ == "__main__":
    unittest.main()
