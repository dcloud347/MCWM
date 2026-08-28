from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import yaml  # noqa: F401
    from mcwm.training.checkpoint import read_checkpoint
    from mcwm.training.config import load_yaml_config
    from mcwm.training.train_world_model import train
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class WorldModelCheckpointTest(unittest.TestCase):
    def test_checkpoint_saves_m2_state_and_parent_identity(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(repository / "configs" / "train_world_model_tiny.yaml")
        config["optimizer"]["iterations_per_epoch"] = 1
        config["optimizer"]["epochs"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            config["output_dir"] = temporary
            path = train(config, synthetic=True)
            payload = read_checkpoint(path)

        self.assertEqual(payload["extra"]["stage"], "m2-world-model")
        self.assertEqual(
            payload["extra"]["m1_parent_path"],
            "synthetic://random-frozen-m1",
        )
        self.assertTrue(payload["extra"]["m1_parent_sha256"])
        self.assertFalse(
            any(name.startswith("visual_encoder.") for name in payload["model"])
        )
        self.assertTrue(
            any(name.startswith("action_encoder.") for name in payload["model"])
        )
        self.assertTrue(any(name.startswith("predictor.") for name in payload["model"]))

    def test_resume_rejects_different_m1_parent_hash(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(repository / "configs" / "train_world_model_tiny.yaml")
        config["optimizer"]["iterations_per_epoch"] = 1
        config["optimizer"]["epochs"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            config["output_dir"] = temporary
            path = train(config, synthetic=True)
            payload = read_checkpoint(path)
            payload["extra"]["m1_parent_sha256"] = "different"
            torch.save(payload, path)

            config["optimizer"]["iterations_per_epoch"] = 2
            config["checkpoint"]["resume"] = str(path)
            with self.assertRaisesRegex(ValueError, "parent hash"):
                train(config, synthetic=True)


if __name__ == "__main__":
    unittest.main()
