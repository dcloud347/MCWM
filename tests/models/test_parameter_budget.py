from pathlib import Path
import unittest

try:
    import torch
    import yaml  # noqa: F401
    from mcwm.training.config import build_visual_jepa, load_yaml_config
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class M1ParameterBudgetTest(unittest.TestCase):
    def test_base_config_matches_design_budget(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(repository / "configs" / "pretrain_visual.yaml")
        # Meta tensors contain shapes but allocate no ~770 MB FP32 weights.
        with torch.device("meta"):
            model = build_visual_jepa(config)
        encoder = sum(parameter.numel() for parameter in model.target_encoder.parameters())
        predictor = sum(parameter.numel() for parameter in model.predictor.parameters())
        total = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(encoder, 86_899_968)
        self.assertEqual(predictor, 21_886_080)
        self.assertEqual(total, 195_686_016)
