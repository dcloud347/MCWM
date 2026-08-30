import unittest

try:
    import torch

    from mcwm.diagnostics.surprise import surprise_metrics, surprise_samples
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SurpriseDiagnosticsTest(unittest.TestCase):
    def test_unrelated_frame_and_trajectory_create_significant_surprise(self):
        generator = torch.Generator().manual_seed(23)
        targets = torch.randn(128, 14, 2, 8, generator=generator)
        predictions = targets.clone()

        metrics = surprise_metrics(
            surprise_samples(
                predictions,
                targets,
                perturbation_step=8,
            ),
            perturbation_step=8,
        )

        self.assertEqual(metrics["m3/surprise/pass_statistical"], 1.0)
        self.assertGreater(
            metrics["m3/surprise/frame_replaced_gap_ci95_low"],
            0.0,
        )
        self.assertGreater(
            metrics["m3/surprise/trajectory_switched_gap_ci95_low"],
            0.0,
        )
        self.assertEqual(metrics["m3/surprise/frame_replaced_peak_step"], 8.0)

    def test_requires_two_clips_for_unrelated_target(self):
        values = torch.randn(1, 4, 2, 8)
        with self.assertRaisesRegex(ValueError, "at least two"):
            surprise_samples(values, values, perturbation_step=2)


if __name__ == "__main__":
    unittest.main()
