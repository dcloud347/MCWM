import random
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from mcwm.training.checkpoint import (
        CheckpointProvenance,
        load_checkpoint,
        save_checkpoint,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTest(unittest.TestCase):
    def test_checkpoint_restores_model_optimizer_scheduler_and_rng(self):
        torch.manual_seed(8)
        random.seed(8)
        np.random.seed(8)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        provenance = CheckpointProvenance(
            git_commit="test-commit",
            config={"model": "tiny"},
            seed=8,
            manifest_hash="manifest-a",
            parent_checkpoint=None,
            wandb_entity=None,
            wandb_project="mcwm",
            wandb_run_id="run-a",
            wandb_run_name="test",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                optimizer_step=3,
                provenance=provenance,
            )
            expected_random = torch.rand(4)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            payload = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_manifest_hash="manifest-a",
            )
            self.assertEqual(payload["optimizer_step"], 3)
            self.assertTrue(torch.equal(torch.rand(4), expected_random))
            self.assertTrue(any(parameter.abs().sum() > 0 for parameter in model.parameters()))

    def test_manifest_mismatch_is_rejected(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        provenance = CheckpointProvenance(
            git_commit="test",
            config={},
            seed=1,
            manifest_hash="first",
            parent_checkpoint=None,
            wandb_entity=None,
            wandb_project=None,
            wandb_run_id=None,
            wandb_run_name=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                optimizer_step=0,
                provenance=provenance,
            )
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path,
                    model=model,
                    expected_manifest_hash="second",
                )
