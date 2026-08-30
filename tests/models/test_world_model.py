import unittest

try:
    import torch
    from mcwm.models.ac_predictor import ACPredictorConfig, ActionConditionedPredictor
    from mcwm.models.action_encoder import ActionEncoderConfig, MinecraftActionEncoder
    from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
    from mcwm.models.visual_encoder import VisualEncoderConfig
    from mcwm.models.world_model import WorldModel, WorldModelConfig
except ModuleNotFoundError:
    torch = None


def _model(frame_chunk_size=None):
    visual_config = VisualEncoderConfig(
        image_height=20,
        image_width=20,
        patch_size=10,
        clip_frames=16,
        tubelet_size=2,
        dim=12,
        depth=1,
        heads=3,
        mlp_dim=24,
        gradient_checkpointing=False,
    )
    action_config = ActionEncoderConfig(
        binary_embedding_dim=2,
        hotbar_embedding_dim=2,
        camera_dim=4,
        cursor_dim=4,
        component_hidden_dim=16,
        tick_dim=12,
        transformer_depth=2,
        transformer_heads=3,
        transformer_mlp_dim=24,
        macro_dim=12,
    )
    predictor_config = ACPredictorConfig(
        latent_dim=12,
        action_dim=12,
        dim=12,
        depth=2,
        heads=3,
        mlp_dim=24,
        context_blocks=8,
        spatial_grid=(2, 2),
        dropout=0.0,
    )
    return WorldModel(
        FrozenVisualEncoder(visual_config),
        MinecraftActionEncoder(action_config),
        ActionConditionedPredictor(predictor_config),
        WorldModelConfig(auto_steps=2, encoder_frame_chunk_size=frame_chunk_size),
    )


def _batch(frames=3):
    batch, ticks = 2, 2
    action_shape = (batch, frames - 1, ticks)
    return {
        "frames": torch.randint(0, 256, (batch, frames, 3, 20, 20), dtype=torch.uint8),
        "movement": torch.zeros(*action_shape, 7, dtype=torch.bool),
        "interaction": torch.zeros(*action_shape, 7, dtype=torch.bool),
        "hotbar": torch.zeros(action_shape, dtype=torch.long),
        "camera": torch.zeros(*action_shape, 2),
        "cursor": torch.zeros(*action_shape, 2),
        "gui_open": torch.zeros(action_shape, dtype=torch.bool),
        "cursor_present": torch.zeros(action_shape, dtype=torch.bool),
        "valid_mask": torch.ones(action_shape, dtype=torch.bool),
    }


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WorldModelTest(unittest.TestCase):
    def test_forward_connects_all_m2_components(self):
        model = _model(frame_chunk_size=2)
        output = model(**_batch())

        self.assertEqual(tuple(output["latents"].shape), (2, 3, 4, 12))
        self.assertEqual(tuple(output["action_tokens"].shape), (2, 2, 12))
        self.assertEqual(
            tuple(output["teacher_forced_predictions"].shape),
            (2, 2, 4, 12),
        )
        self.assertEqual(
            tuple(output["autoregressive_predictions"].shape),
            (2, 2, 4, 12),
        )

    def test_backward_never_updates_frozen_encoder(self):
        model = _model()

        output = model(**_batch())
        output["loss"].backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in model.visual_encoder.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.action_encoder.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.predictor.parameters())
        )

    def test_evaluation_can_override_trained_rollout_length(self):
        model = _model()

        output = model(**_batch(frames=7), rollout_steps=6)

        self.assertEqual(
            tuple(output["autoregressive_predictions"].shape),
            (2, 6, 4, 12),
        )

    def test_optimizer_parameters_exclude_visual_encoder(self):
        model = _model()
        trainable_ids = {id(parameter) for parameter in model.trainable_parameters()}

        self.assertFalse(
            any(id(parameter) in trainable_ids for parameter in model.visual_encoder.parameters())
        )
        self.assertTrue(
            all(id(parameter) in trainable_ids for parameter in model.predictor.parameters())
        )


if __name__ == "__main__":
    unittest.main()
