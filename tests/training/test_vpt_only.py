from pathlib import Path
import tempfile
import unittest

try:
    import torch  # noqa: F401
    import yaml  # noqa: F401
    from mcwm.data.fixtures import build_fixture_dataset
    from mcwm.training.config import load_yaml_config, validate_pretrain_config
    from mcwm.training.pretrain_visual import _make_loaders
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class VPTOnlyTrainingTest(unittest.TestCase):
    def setUp(self):
        repository = Path(__file__).resolve().parents[2]
        self.config = load_yaml_config(
            repository / "configs" / "pretrain_visual_tiny.yaml"
        )

    def test_config_does_not_expose_source_controls(self):
        self.assertNotIn("source", self.config["data"])
        self.assertNotIn("source_balanced", self.config["data"])
        validate_pretrain_config(self.config)

    def test_loader_rejects_manifest_with_non_vpt_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture_dataset(root)
            self.config["data"]["root"] = str(root)
            with self.assertRaisesRegex(ValueError, "non-VPT episodes"):
                _make_loaders(
                    self.config,
                    seed=int(self.config["seed"]),
                    synthetic=False,
                    rank=0,
                    world_size=1,
                )


if __name__ == "__main__":
    unittest.main()
