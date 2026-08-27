from pathlib import Path
import tempfile
import unittest

import torch

from mcwm.models.frozen_visual_encoder import (
    FrozenVisualEncoder,
    repeated_frame_metrics,
)
from mcwm.models.masking import MaskConfig
from mcwm.models.visual_encoder import VisualEncoderConfig
from mcwm.models.visual_jepa import VisualJEPA, VisualJEPAConfig
from mcwm.models.visual_predictor import VisualPredictorConfig
from mcwm.training.checkpoint import (
    CheckpointProvenance,
    load_frozen_m1_encoder,
    save_checkpoint,
)


def _encoder_config():
    return VisualEncoderConfig(
        image_height=20,
        image_width=30,
        patch_size=10,
        clip_frames=16,
        tubelet_size=2,
        dim=24,
        depth=1,
        heads=4,
        mlp_dim=48,
        gradient_checkpointing=False,
    )


def _m1_model():
    encoder = _encoder_config()
    predictor = VisualPredictorConfig(
        input_dim=24,
        dim=16,
        depth=1,
        heads=4,
        mlp_dim=32,
        token_grid_size=encoder.token_grid_size,
        num_mask_tokens=2,
        gradient_checkpointing=False,
    )
    return VisualJEPA(
        VisualJEPAConfig(encoder=encoder, predictor=predictor, mask=MaskConfig())
    )


def _resolved_config():
    return {
        "data": {"clip_frames": 16},
        "model": {
            "image_height": 20,
            "image_width": 30,
            "patch_size": 10,
            "tubelet_size": 2,
            "encoder_dim": 24,
            "encoder_depth": 1,
            "encoder_heads": 4,
            "encoder_mlp_dim": 48,
            "use_rope": True,
            "gradient_checkpointing": False,
        },
    }


def _save_m1_checkpoint(path, *, resolved_config=None):
    m1 = _m1_model()
    with torch.no_grad():
        for parameter in m1.online_encoder.parameters():
            parameter.zero_()
        for parameter in m1.target_encoder.parameters():
            parameter.fill_(0.125)
    optimizer = torch.optim.AdamW(m1.parameters(), lr=1e-4)
    provenance = CheckpointProvenance(
        git_commit="test",
        config=_resolved_config() if resolved_config is None else resolved_config,
        seed=8,
        manifest_hash="manifest",
        parent_checkpoint=None,
        wandb_entity=None,
        wandb_project="mcwm",
        wandb_run_id=None,
        wandb_run_name=None,
    )
    save_checkpoint(
        path,
        model=m1,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        optimizer_step=4,
        provenance=provenance,
    )


class FrozenVisualEncoderTest(unittest.TestCase):
    def test_repeated_frames_preserve_spatial_tokens_and_stay_frozen(self):
        model = FrozenVisualEncoder(_encoder_config()).train()
        frames = torch.randint(0, 256, (2, 3, 3, 20, 30), dtype=torch.uint8)

        full = model(frames)
        chunked = model(frames, frame_chunk_size=2)

        self.assertEqual(tuple(full.shape), (2, 3, 6, 24))
        self.assertTrue(torch.allclose(full, chunked, atol=1e-6, rtol=1e-5))
        self.assertFalse(full.requires_grad)
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        metrics = repeated_frame_metrics(full)
        self.assertTrue(bool(metrics["finite"]))
        self.assertGreater(float(metrics["average_token_norm"]), 0.0)

    def test_strict_loader_uses_m1_target_encoder(self):
        torch.manual_seed(8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "m1.pt"
            _save_m1_checkpoint(path)

            frozen, payload = load_frozen_m1_encoder(
                path,
                expected_manifest_hash="manifest",
            )

        self.assertEqual(payload["optimizer_step"], 4)
        loaded = next(frozen.encoder.parameters())
        self.assertTrue(torch.equal(loaded, torch.full_like(loaded, 0.125)))
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.parameters()))
        self.assertFalse(frozen.encoder.training)

    def test_loader_rejects_non_16_frame_m1_contract(self):
        config = _resolved_config()
        config["data"]["clip_frames"] = 8
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "m1.pt"
            _save_m1_checkpoint(path, resolved_config=config)

            with self.assertRaisesRegex(ValueError, "configured for 16 frames"):
                load_frozen_m1_encoder(path)


if __name__ == "__main__":
    unittest.main()
