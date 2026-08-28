import unittest
from unittest.mock import patch

try:
    import torch
    from mcwm.models.ac_predictor import (
        ACPredictorConfig,
        ActionConditionedPredictor,
        block_causal_attention_mask,
        normalized_latent_l1_loss,
        teacher_forced_autoregressive_loss,
    )
except ModuleNotFoundError:
    torch = None


def _config():
    return ACPredictorConfig(
        latent_dim=12,
        action_dim=12,
        dim=24,
        depth=2,
        heads=4,
        mlp_dim=48,
        context_blocks=4,
        spatial_grid=(2, 2),
        dropout=0.0,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FrameBlockCausalPredictorTest(unittest.TestCase):
    def test_mask_allows_same_and_past_blocks_only(self):
        mask = block_causal_attention_mask(
            blocks=3,
            block_size=3,
            device=torch.device("cpu"),
        )

        self.assertTrue(mask[0, :3].all())
        self.assertFalse(mask[0, 3:].any())
        self.assertTrue(mask[4, :6].all())
        self.assertFalse(mask[4, 6:].any())
        self.assertTrue(mask[8].all())

    def test_teacher_forced_returns_visual_tokens_only(self):
        model = ActionConditionedPredictor(_config()).eval()
        latents = torch.randn(2, 3, 4, 12)
        actions = torch.randn(2, 3, 12)

        prediction = model.predict_teacher_forced(latents, actions)

        self.assertEqual(tuple(prediction.shape), (2, 3, 4, 12))

    def test_future_blocks_cannot_change_past_predictions(self):
        torch.manual_seed(7)
        model = ActionConditionedPredictor(_config()).eval()
        latents = torch.randn(1, 4, 4, 12)
        actions = torch.randn(1, 4, 12)

        expected = model.predict_teacher_forced(latents, actions)
        changed_latents = latents.clone()
        changed_actions = actions.clone()
        changed_latents[:, 2:] = 1000.0
        changed_actions[:, 2:] = -1000.0
        actual = model.predict_teacher_forced(changed_latents, changed_actions)

        self.assertTrue(torch.allclose(expected[:, :2], actual[:, :2], atol=1e-6))

    def test_action_communicates_with_visual_tokens_in_same_block(self):
        torch.manual_seed(9)
        model = ActionConditionedPredictor(_config()).eval()
        latents = torch.randn(1, 1, 4, 12)
        actions = torch.zeros(1, 1, 12)

        without_action = model.predict_teacher_forced(latents, actions)
        actions[:, 0] = 10.0
        with_action = model.predict_teacher_forced(latents, actions)

        self.assertFalse(torch.allclose(without_action, with_action))

    def test_rollout_feeds_prediction_back_as_next_input(self):
        model = ActionConditionedPredictor(_config()).eval()
        initial = torch.zeros(1, 4, 12)
        actions = torch.stack(
            (
                torch.ones(1, 12),
                torch.full((1, 12), 2.0),
            ),
            dim=1,
        )
        seen_latents = []

        def fake_teacher_forced(latents, context_actions):
            seen_latents.append(latents.detach().clone())
            return latents + context_actions.unsqueeze(2)

        with patch.object(
            model,
            "predict_teacher_forced",
            side_effect=fake_teacher_forced,
        ):
            prediction = model.rollout(initial, actions)

        self.assertTrue(torch.equal(prediction[:, 0], torch.ones(1, 4, 12)))
        self.assertTrue(torch.equal(prediction[:, 1], torch.full((1, 4, 12), 3.0)))
        self.assertTrue(torch.equal(seen_latents[1][:, -1], prediction[:, 0]))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TeacherForcedAutoregressiveLossTest(unittest.TestCase):
    def test_normalized_l1_ignores_shift_and_positive_scale(self):
        target = torch.randn(2, 3, 4, 12)
        prediction = target * 3.0 + 5.0

        loss = normalized_latent_l1_loss(prediction, target)

        self.assertLess(float(loss), 1e-4)

    def test_combined_loss_has_expected_paths_and_gradients(self):
        torch.manual_seed(11)
        model = ActionConditionedPredictor(_config())
        latents = torch.randn(2, 4, 4, 12)
        actions = torch.randn(2, 3, 12)

        output = teacher_forced_autoregressive_loss(
            model,
            latents,
            actions,
            auto_steps=2,
        )
        output["loss"].backward()

        self.assertEqual(
            tuple(output["teacher_forced_predictions"].shape),
            (2, 3, 4, 12),
        )
        self.assertEqual(
            tuple(output["autoregressive_predictions"].shape),
            (2, 2, 4, 12),
        )
        self.assertTrue(
            torch.allclose(
                output["loss"],
                output["teacher_forced_loss"] + output["autoregressive_loss"],
            )
        )
        gradient = model.action_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_auto_steps_cannot_exceed_available_transitions(self):
        model = ActionConditionedPredictor(_config())
        latents = torch.randn(1, 3, 4, 12)
        actions = torch.randn(1, 2, 12)

        with self.assertRaisesRegex(ValueError, "auto_steps"):
            teacher_forced_autoregressive_loss(
                model,
                latents,
                actions,
                auto_steps=3,
            )


if __name__ == "__main__":
    unittest.main()
