import unittest
from unittest.mock import patch

try:
    import torch
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    from mcwm.models.ac_predictor import (
        ACPredictorConfig,
        ActionConditionedPredictor,
        block_causal_attention_mask,
        normalized_latent_l1_loss,
        teacher_forced_autoregressive_loss,
    )
except ModuleNotFoundError:
    torch = None


def _config(*, gradient_checkpointing=False):
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
        gradient_checkpointing=gradient_checkpointing,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FrameBlockCausalPredictorTest(unittest.TestCase):
    def test_default_predictor_disables_dropout(self):
        self.assertEqual(ACPredictorConfig().dropout, 0.0)

    def test_training_checkpoints_each_transformer_block(self):
        model = ActionConditionedPredictor(
            _config(gradient_checkpointing=True)
        ).train()
        latents = torch.randn(1, 2, 4, 12)
        actions = torch.randn(1, 2, 12)

        with patch(
            "mcwm.models.ac_predictor.checkpoint",
            wraps=torch_checkpoint,
        ) as checkpoint_mock:
            prediction = model.predict_teacher_forced(latents, actions)
            prediction.sum().backward()

        self.assertEqual(checkpoint_mock.call_count, model.config.depth)

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
                torch.arange(12, dtype=torch.float32).unsqueeze(0),
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

        first = actions[:, 0].unsqueeze(1).expand(-1, 4, -1)
        normalized_first = torch.nn.functional.layer_norm(first, (12,))
        self.assertTrue(torch.equal(prediction[:, 0], first))
        self.assertTrue(torch.allclose(seen_latents[1][:, -1], normalized_first))
        self.assertTrue(
            torch.allclose(prediction[:, 1], normalized_first + 2.0)
        )

    def test_context_rollout_keeps_observed_history_and_future_action_aligned(self):
        model = ActionConditionedPredictor(_config()).eval()
        context = torch.randn(1, 4, 4, 12)
        history_actions = torch.randn(1, 3, 12)
        future_actions = torch.randn(1, 2, 12)
        seen = []

        def fake_teacher(latents, actions):
            seen.append((latents.detach().clone(), actions.detach().clone()))
            return latents + actions.unsqueeze(2)

        with patch.object(
            model,
            "predict_teacher_forced",
            side_effect=fake_teacher,
        ):
            prediction = model.rollout_with_context(
                context,
                history_actions,
                future_actions,
            )

        self.assertEqual(tuple(prediction.shape), (1, 2, 4, 12))
        self.assertEqual(tuple(seen[0][0].shape), (1, 4, 4, 12))
        self.assertEqual(tuple(seen[0][1].shape), (1, 4, 12))
        self.assertTrue(torch.equal(seen[0][1][:, -1], future_actions[:, 0]))
        self.assertEqual(tuple(seen[1][0].shape), (1, 4, 4, 12))
        self.assertTrue(torch.equal(seen[1][1][:, -1], future_actions[:, 1]))


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

    def test_combined_loss_normalizes_encoder_latents_before_prediction(self):
        model = ActionConditionedPredictor(_config()).eval()
        latents = torch.randn(1, 3, 4, 12) * 7.0 + 5.0
        actions = torch.randn(1, 2, 12)
        seen_teacher = []
        seen_initial = []

        def fake_teacher(inputs, context_actions):
            seen_teacher.append(inputs.detach().clone())
            return torch.zeros_like(inputs)

        def fake_rollout(initial, context_actions):
            seen_initial.append(initial.detach().clone())
            return torch.zeros(
                initial.shape[0],
                context_actions.shape[1],
                initial.shape[1],
                initial.shape[2],
            )

        with patch.object(model, "predict_teacher_forced", side_effect=fake_teacher):
            with patch.object(model, "rollout", side_effect=fake_rollout):
                teacher_forced_autoregressive_loss(
                    model,
                    latents,
                    actions,
                    auto_steps=2,
                )

        expected = torch.nn.functional.layer_norm(latents, (12,))
        self.assertTrue(torch.allclose(seen_teacher[0], expected[:, :-1]))
        self.assertTrue(torch.allclose(seen_initial[0], expected[:, 0]))

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
