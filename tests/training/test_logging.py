import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcwm.training.logging import TrainingLogger


class LoggingTest(unittest.TestCase):
    def test_disabled_wandb_still_writes_local_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = TrainingLogger(
                Path(temporary),
                config={"seed": 1},
                wandb_config={"enabled": False, "mode": "disabled"},
            )
            logger.log({"train/loss": 1.25}, step=2)
            logger.finish()
            with (Path(temporary) / "metrics.jsonl").open(encoding="utf-8") as handle:
                record = json.loads(handle.readline())
            self.assertEqual(record["optimizer_step"], 2)
            self.assertEqual(record["train/loss"], 1.25)

    def test_resume_passes_existing_run_id_to_wandb(self):
        calls = {}

        class FakeRun:
            id = "same-run"
            name = "continued"

            def log(self, metrics, step):
                calls["log"] = (metrics, step)

            def finish(self):
                calls["finished"] = True

        class FakeWandb:
            @staticmethod
            def init(**kwargs):
                calls["init"] = kwargs
                return FakeRun()

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "sys.modules", {"wandb": FakeWandb}
        ):
            logger = TrainingLogger(
                Path(temporary),
                config={"seed": 1},
                wandb_config={"enabled": True, "mode": "offline", "project": "mcwm"},
                run_id="same-run",
            )
            logger.log({"train/loss": 0.5}, step=7)
            logger.finish()
        self.assertEqual(calls["init"]["id"], "same-run")
        self.assertEqual(calls["init"]["resume"], "allow")
        self.assertEqual(calls["log"][1], 7)
        self.assertTrue(calls["finished"])
