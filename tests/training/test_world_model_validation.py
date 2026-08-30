import unittest

try:
    import torch
    from torch import nn

    from mcwm.training.train_world_model import MODEL_INPUT_NAMES, validate
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WorldModelValidationTest(unittest.TestCase):
    def test_action_sensitivity_uses_every_requested_batch(self):
        class FixedModel(nn.Module):
            def forward(self, **batch):
                frames = batch["frames"]
                batch_size = frames.shape[0]
                transitions = frames.shape[1] - 1
                prediction = torch.zeros(batch_size, transitions, 1, 4)
                targets = torch.ones_like(prediction)
                return {
                    "loss": torch.tensor(2.0),
                    "teacher_forced_loss": torch.tensor(1.0),
                    "autoregressive_loss": torch.tensor(1.0),
                    "teacher_forced_predictions": prediction,
                    "autoregressive_predictions": prediction[:, :1],
                    "targets": targets,
                }

        def batch():
            value = {
                "frames": torch.zeros(1, 3, 3, 2, 2, dtype=torch.uint8),
                "movement": torch.zeros(1, 2, 1, 7, dtype=torch.bool),
                "interaction": torch.zeros(1, 2, 1, 7, dtype=torch.bool),
                "hotbar": torch.zeros(1, 2, 1, dtype=torch.long),
                "camera": torch.zeros(1, 2, 1, 2),
                "cursor": torch.zeros(1, 2, 1, 2),
                "gui_open": torch.zeros(1, 2, 1, dtype=torch.bool),
                "cursor_present": torch.zeros(1, 2, 1, dtype=torch.bool),
                "valid_mask": torch.ones(1, 2, 1, dtype=torch.bool),
            }
            return {name: value[name] for name in MODEL_INPUT_NAMES}

        metrics, _ = validate(
            FixedModel(),
            [batch(), batch()],
            device=torch.device("cpu"),
            precision="fp32",
            batches=2,
            spatial_grid=(1, 1),
            action_sensitivity_batches=2,
        )

        self.assertEqual(metrics["action_sensitivity/sample_transitions"], 4.0)
        self.assertIn("action_sensitivity/pass_statistical", metrics)


if __name__ == "__main__":
    unittest.main()
