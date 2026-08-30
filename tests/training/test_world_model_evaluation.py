from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import torch  # noqa: F401
    import yaml  # noqa: F401

    from mcwm.training.config import load_yaml_config
    from mcwm.training.train_world_model import evaluate, train
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class WorldModelEvaluationTest(unittest.TestCase):
    def test_evaluation_reloads_checkpoint_and_writes_gate_report(self):
        repository = Path(__file__).resolve().parents[2]
        config = load_yaml_config(
            repository / "configs" / "train_world_model_tiny.yaml"
        )
        with TemporaryDirectory() as temporary:
            config = deepcopy(config)
            config["output_dir"] = str(Path(temporary) / "train")
            config["optimizer"]["iterations_per_epoch"] = 1
            config["optimizer"]["epochs"] = 1
            config["checkpoint"]["every_steps"] = 1
            config["wandb"]["enabled"] = False
            checkpoint = train(config, synthetic=True)
            report_path = Path(temporary) / "m2_evaluation.json"

            result = evaluate(
                config,
                checkpoint,
                synthetic=True,
                output_path=report_path,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result, report_path)
        self.assertEqual(report["stage"], "m2-world-model-evaluation")
        self.assertEqual(report["optimizer_step"], 1)
        self.assertIn("global_statistical_pass", report["gate"])
        self.assertGreater(
            report["metrics"]["action_sensitivity/sample_transitions"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
