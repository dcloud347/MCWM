from pathlib import Path
import unittest

try:
    import torch
    import yaml  # noqa: F401
    from mcwm.models.ac_predictor import ActionConditionedPredictor
    from mcwm.models.action_encoder import MinecraftActionEncoder
    from mcwm.training.config import build_visual_jepa, load_yaml_config
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class M1ParameterBudgetTest(unittest.TestCase):
    def test_base_config_matches_design_budget(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(repository / "configs" / "pretrain_visual.yaml")
        # Meta tensors contain shapes but allocate no ~2.5 GB FP32 weights.
        with torch.device("meta"):
            model = build_visual_jepa(config)
        encoder = sum(parameter.numel() for parameter in model.target_encoder.parameters())
        predictor = sum(parameter.numel() for parameter in model.predictor.parameters())
        total = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(encoder, 304_770_048)
        self.assertEqual(predictor, 22_082_944)
        self.assertEqual(total, 631_623_040)

    def test_m2_default_trainable_and_deploy_parameter_budget(self):
        with torch.device("meta"):
            action_encoder = MinecraftActionEncoder()
            predictor = ActionConditionedPredictor()

        action_parameters = sum(
            parameter.numel() for parameter in action_encoder.parameters()
        )
        predictor_parameters = sum(
            parameter.numel() for parameter in predictor.parameters()
        )
        self.assertEqual(action_parameters, 2_181_632)
        self.assertEqual(predictor_parameters, 305_460_224)
        self.assertEqual(
            304_770_048 + action_parameters + predictor_parameters,
            612_411_904,
        )
