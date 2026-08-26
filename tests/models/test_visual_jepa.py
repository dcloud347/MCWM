import unittest

try:
    import torch
    from mcwm.models.masking import MaskConfig
    from mcwm.models.visual_encoder import VisualEncoderConfig
    from mcwm.models.visual_jepa import VisualJEPA, VisualJEPAConfig
    from mcwm.models.visual_predictor import VisualPredictorConfig
except ModuleNotFoundError:
    torch = None


def tiny_model():
    encoder = VisualEncoderConfig(
        image_height=20,
        image_width=30,
        patch_size=10,
        clip_frames=4,
        tubelet_size=2,
        dim=24,
        depth=2,
        heads=4,
        mlp_dim=48,
        gradient_checkpointing=False,
    )
    predictor = VisualPredictorConfig(
        input_dim=24,
        dim=16,
        depth=2,
        heads=4,
        mlp_dim=32,
        token_grid_size=encoder.token_grid_size,
        num_mask_tokens=2,
        gradient_checkpointing=False,
    )
    return VisualJEPA(
        VisualJEPAConfig(
            encoder=encoder,
            predictor=predictor,
            mask=MaskConfig(),
        )
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class VisualJEPATest(unittest.TestCase):
    def test_target_starts_equal_and_never_requires_gradient(self):
        model = tiny_model()
        for online, target in zip(
            model.online_encoder.parameters(), model.target_encoder.parameters()
        ):
            self.assertTrue(torch.equal(online, target))
            self.assertFalse(target.requires_grad)

    def test_forward_backward_uses_stop_gradient(self):
        torch.manual_seed(3)
        model = tiny_model()
        frames = torch.randint(0, 256, (2, 4, 3, 20, 30), dtype=torch.uint8)
        output = model(frames, mask_generator=torch.Generator().manual_seed(4))
        self.assertEqual(len(output["prediction"]), 2)
        self.assertEqual(tuple(output["target_mask"].shape), (2, 2, 2, 6))
        self.assertEqual(tuple(output["prediction_mask"].shape), (2, 2, 2, 6))
        self.assertEqual(tuple(output["target"].shape), (2, 2, 6, 24))
        self.assertTrue(torch.isfinite(output["loss"]))
        per_task_losses = []
        for group_index in range(2):
            flat_mask = output["target_mask"][group_index].flatten(1)
            context_indices = output["context_indices"][group_index]
            indices = output["prediction_indices"][group_index]
            self.assertFalse(flat_mask.gather(1, context_indices).any())
            self.assertTrue(flat_mask.gather(1, indices).all())
            effective_mask = output["prediction_mask"][group_index].flatten(1)
            self.assertEqual(effective_mask.sum().item(), indices.numel())
            self.assertTrue(effective_mask.gather(1, indices).all())
            targets = output["target"].flatten(1, 2).gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 24),
            )
            for batch_index in range(2):
                per_task_losses.append(
                    torch.nn.functional.l1_loss(
                        output["prediction"][group_index][batch_index],
                        targets[batch_index],
                    )
                )
        self.assertTrue(
            torch.allclose(output["loss"], torch.stack(per_task_losses).mean())
        )
        output["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in model.online_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_encoder_uses_tubelets_and_removes_masked_context_tokens(self):
        model = tiny_model()
        frames = torch.rand(2, 4, 3, 20, 30)
        indices = torch.tensor([[0, 3, 7], [1, 4, 10]])
        encoded = model.online_encoder(
            model.normalize_frames(frames),
            indices,
            return_patch_tokens=True,
        )
        self.assertEqual(tuple(encoded.shape), (2, 3, 24))
        self.assertEqual(model.config.encoder.token_grid_size, (2, 2, 3))

    def test_ema_formula_is_numerically_exact(self):
        model = tiny_model()
        online = next(model.online_encoder.parameters())
        target = next(model.target_encoder.parameters())
        before = target.detach().clone()
        with torch.no_grad():
            online.add_(2.0)
        model.update_target(0.75)
        self.assertTrue(torch.allclose(target, before + 0.5))

    def test_target_stays_in_eval_mode(self):
        model = tiny_model().train()
        self.assertTrue(model.online_encoder.training)
        self.assertFalse(model.target_encoder.training)
