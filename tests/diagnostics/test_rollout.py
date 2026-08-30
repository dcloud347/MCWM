import unittest

try:
    import torch

    from mcwm.diagnostics.rollout import rollout_metrics, rollout_samples
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RolloutDiagnosticsTest(unittest.TestCase):
    def test_reports_all_horizons_drift_and_action_buckets(self):
        generator = torch.Generator().manual_seed(17)
        targets = torch.randn(4, 14, 2, 6, generator=generator)
        noise = torch.linspace(0.01, 0.14, 14).view(1, 14, 1, 1)
        predictions = targets + noise * torch.randn(
            4,
            14,
            2,
            6,
            generator=generator,
        )
        batch = {
            "movement": torch.zeros(4, 14, 1, 7, dtype=torch.bool),
            "interaction": torch.zeros(4, 14, 1, 7, dtype=torch.bool),
            "hotbar": torch.zeros(4, 14, 1, dtype=torch.long),
            "camera": torch.zeros(4, 14, 1, 2),
            "gui_open": torch.zeros(4, 14, 1, dtype=torch.bool),
        }
        batch["movement"][:, :, :, 0] = True
        batch["camera"][:, 7:, :, 0] = 1.0

        metrics = rollout_metrics(
            rollout_samples(predictions, targets, batch),
            horizons=(1, 2, 4, 6, 8, 10, 12, 14),
        )

        self.assertEqual(metrics["m3/rollout/sample_rollouts"], 4.0)
        self.assertIn("m3/rollout/step_14_l1", metrics)
        self.assertIn("m3/rollout/drift_slope_12_to_14", metrics)
        self.assertEqual(
            metrics["m3/action_bucket/movement/sample_transitions"],
            56.0,
        )
        self.assertEqual(
            metrics["m3/action_bucket/camera/sample_transitions"],
            28.0,
        )

    def test_rejects_horizon_beyond_collected_steps(self):
        samples = {
            "l1": torch.ones(2, 4),
            "cosine": torch.ones(2, 4),
            "norm_gap": torch.ones(2, 4),
        }
        with self.assertRaisesRegex(ValueError, "fit within"):
            rollout_metrics(samples, horizons=(1, 5))


if __name__ == "__main__":
    unittest.main()
