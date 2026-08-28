from copy import deepcopy
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
class WorldModelResumeTest(unittest.TestCase):
    def test_resume_matches_uninterrupted_next_step(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(repository / "configs" / "train_world_model_tiny.yaml")
        config["optimizer"]["iterations_per_epoch"] = 2
        config["optimizer"]["epochs"] = 1
        config["checkpoint"]["every_steps"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            full_dir = Path(temporary) / "full"
            resumed_dir = Path(temporary) / "resumed"
            config["output_dir"] = str(full_dir)
            train(config, synthetic=True)

            resumed_config = deepcopy(config)
            resumed_config["output_dir"] = str(resumed_dir)
            resumed_config["checkpoint"]["resume"] = str(
                full_dir / "checkpoint-00000001.pt"
            )
            train(resumed_config, synthetic=True)

            uninterrupted = read_checkpoint(full_dir / "checkpoint-00000002.pt")
            resumed = read_checkpoint(resumed_dir / "checkpoint-00000002.pt")

        self.assertEqual(uninterrupted["optimizer_step"], resumed["optimizer_step"])
        for name, value in uninterrupted["model"].items():
            self.assertTrue(torch.equal(value, resumed["model"][name]), name)


if __name__ == "__main__":
    unittest.main()
