from copy import deepcopy
from pathlib import Path
import unittest

try:
    import torch
    import yaml  # noqa: F401
    from mcwm.training.config import (
        load_yaml_config,
        validate_world_model_config,
    )
    from mcwm.training.train_world_model import (
        _accumulation_steps,
        _make_loaders,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class WorldModelDistributedTest(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[2]
        self.config = load_yaml_config(
            self.repository / "configs" / "train_world_model_tiny.yaml"
        )

    def test_two_h100_config_uses_one_global_micro_step(self):
        config = load_yaml_config(
            self.repository / "configs" / "train_world_model_2xh100_sxm.yaml"
        )
        validate_world_model_config(config)
        self.assertEqual(config["distributed"]["strategy"], "fsdp")
        self.assertEqual(_accumulation_steps(config, world_size=2), 1)
        self.assertEqual(config["model"]["encoder_frame_chunk_size"], 384)

    def test_fsdp_strategy_is_valid_but_ddp_is_rejected(self):
        fsdp_config = deepcopy(self.config)
        fsdp_config["distributed"] = {"strategy": "fsdp"}
        validate_world_model_config(fsdp_config)

        ddp_config = deepcopy(self.config)
        ddp_config["distributed"] = {"strategy": "ddp"}
        with self.assertRaisesRegex(ValueError, "none or fsdp"):
            validate_world_model_config(ddp_config)

    def test_effective_batch_is_global_across_ranks(self):
        config = deepcopy(self.config)
        config["data"]["batch_size"] = 32
        config["optimizer"]["effective_batch_size"] = 64
        self.assertEqual(_accumulation_steps(config, world_size=2), 1)
        self.assertEqual(_accumulation_steps(config, world_size=1), 2)

        config["optimizer"]["effective_batch_size"] = 48
        with self.assertRaisesRegex(ValueError, "world_size"):
            _accumulation_steps(config, world_size=2)

    def test_distributed_samplers_assign_different_examples(self):
        rank_zero, _, _ = _make_loaders(
            self.config,
            synthetic=True,
            rank=0,
            world_size=2,
        )
        rank_one, _, _ = _make_loaders(
            self.config,
            synthetic=True,
            rank=1,
            world_size=2,
        )

        def sample_indices(loader):
            return {
                value[0] if isinstance(value, tuple) else value
                for value in loader.sampler
            }

        self.assertTrue(sample_indices(rank_zero).isdisjoint(sample_indices(rank_one)))


if __name__ == "__main__":
    unittest.main()
