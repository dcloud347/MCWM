import unittest

try:
    import torch

    from mcwm.diagnostics.world_model import action_sensitivity_from_samples
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WorldModelActionSensitivityTest(unittest.TestCase):
    def test_global_samples_produce_one_statistical_decision(self):
        transitions = 128
        real = torch.full((transitions,), 0.30)
        samples = {
            "error_real": real,
            "error_shuffled": real + 0.01,
            "error_noop": real + 0.02,
            "error_camera_reversed": real + 0.005,
            "error_attack_use_swapped": real + 0.004,
            "bucket_movement": torch.ones(transitions, dtype=torch.bool),
            "bucket_interaction": torch.arange(transitions).remainder(2).eq(0),
        }

        metrics = action_sensitivity_from_samples(samples)

        self.assertEqual(metrics["action_sensitivity/pass_statistical"], 1.0)
        self.assertEqual(
            metrics["action_sensitivity/sample_transitions"],
            float(transitions),
        )
        self.assertGreater(
            metrics["action_sensitivity/gap_shuffled_ci95_low"],
            0.0,
        )
        self.assertGreater(metrics["action_sensitivity/gap_noop_ci95_low"], 0.0)
        self.assertAlmostEqual(metrics["action_bucket/movement_l1"], 0.30)

    def test_rejects_mismatched_sample_sizes(self):
        samples = {
            "error_real": torch.zeros(2),
            "error_shuffled": torch.zeros(1),
            "error_noop": torch.zeros(2),
            "error_camera_reversed": torch.zeros(2),
            "error_attack_use_swapped": torch.zeros(2),
        }

        with self.assertRaisesRegex(ValueError, "matching sizes"):
            action_sensitivity_from_samples(samples)


if __name__ == "__main__":
    unittest.main()
