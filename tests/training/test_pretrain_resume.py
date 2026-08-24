from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import yaml  # noqa: F401
    from mcwm.training.checkpoint import read_checkpoint
    from mcwm.training.config import load_yaml_config
    from mcwm.training.pretrain_visual import train
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class PretrainResumeTest(unittest.TestCase):
    def test_resume_produces_same_next_step_as_uninterrupted_training(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(repository / "configs" / "pretrain_visual_tiny.yaml")
        config["optimizer"]["max_steps"] = 2
        config["checkpoint"]["every_steps"] = 1
        config["checkpoint"]["keep_last"] = 3
        config["validation"]["every_steps"] = 100
        config["wandb"]["enabled"] = False
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
