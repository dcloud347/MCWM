from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import torch  # noqa: F401
    import yaml  # noqa: F401

    from mcwm.training.config import load_yaml_config
    from mcwm.training.evaluate_m3 import _surprise_pair_indices, evaluate_m3
    from mcwm.training.train_world_model import train
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/PyYAML is not installed")
class M3EvaluationTest(unittest.TestCase):
    def test_surprise_pairs_cross_session_or_world_groups(self):
        sources, unrelated = _surprise_pair_indices(
            ["a:pts=0-1ms@4fps", "b:pts=0-1ms@4fps", "c:pts=0-1ms@4fps"],
            {
                "a": ("session-1", "world-1"),
                "b": ("session-1", "world-1"),
                "c": ("session-2", "world-2"),
            },
        )

        pairs = dict(zip(sources.tolist(), unrelated.tolist()))
        self.assertEqual(pairs[0], 2)
        self.assertEqual(pairs[1], 2)
        self.assertIn(pairs[2], {0, 1})

    def test_checkpoint_is_evaluated_at_extended_horizons(self):
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
            checkpoint = train(config, synthetic=True)
            report_path = Path(temporary) / "m3_evaluation.json"
            m3_config = {
                "checkpoint": str(checkpoint),
                "output": str(report_path),
                "data": {
                    "frames_per_sample": 8,
                    "validation_batches": 1,
                    "batch_size": 2,
                    "workers": 0,
                },
                "horizons": [1, 2, 4, 6],
                "surprise": {"perturbation_step": 4},
            }

            evaluate_m3(config, m3_config, synthetic=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["stage"], "m3-multi-horizon-evaluation")
        self.assertEqual(report["horizons"], [1, 2, 4, 6])
        self.assertEqual(len(report["sample_ids_sha256"]), 64)
        self.assertIn("m3/rollout/step_6_l1", report["metrics"])
        self.assertEqual(report["gate"]["auto_steps_1_baseline"], "pending")
        self.assertFalse(report["gate"]["m3_complete"])


if __name__ == "__main__":
    unittest.main()
