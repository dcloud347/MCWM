import unittest

try:
    import torch
    from mcwm.diagnostics.m2_b0 import run_b0_smoke_gate
    from tests.models.test_world_model import _batch, _model
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WorldModelOverfitTest(unittest.TestCase):
    def test_fixed_batch_loss_improves(self):
        torch.manual_seed(17)
        model = _model()
        batch = _batch(frames=8)

        report = run_b0_smoke_gate(
            model,
            batch,
            overfit_steps=10,
            learning_rate=3e-3,
        )

        self.assertEqual(report["b0/frozen_encoder"], 1.0)
        self.assertEqual(report["b0/optimizer_excludes_visual"], 1.0)
        self.assertEqual(report["b0/visual_gradients_absent"], 1.0)
        self.assertEqual(report["b0/gradients_finite"], 1.0)
        self.assertLess(report["b0/final_loss"], report["b0/initial_loss"])


if __name__ == "__main__":
    unittest.main()
