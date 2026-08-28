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
        _adamw_parameter_groups,
        _format_duration,
        _make_loaders,
        _progress_bar,
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

    def test_progress_text_matches_m1_style(self):
        self.assertEqual(_progress_bar(5, 10, width=10), "[#####-----]")
        self.assertEqual(_format_duration(3661), "1h01m01s")

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

    def test_adamw_excludes_bias_and_one_dimensional_parameters_from_decay(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 2, bias=False),
        )
        model[2].weight.requires_grad_(False)

        decay, no_decay = _adamw_parameter_groups(model)
        decay_ids = {id(parameter) for parameter in decay["params"]}
        no_decay_ids = {id(parameter) for parameter in no_decay["params"]}

        self.assertEqual(decay_ids, {id(model[0].weight)})
        self.assertEqual(
            no_decay_ids,
            {id(model[0].bias), id(model[1].weight), id(model[1].bias)},
        )
        self.assertEqual(no_decay["weight_decay"], 0.0)
        self.assertNotIn(id(model[2].weight), decay_ids | no_decay_ids)

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
