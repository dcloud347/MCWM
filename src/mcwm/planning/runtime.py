"""Strict checkpoint and image loading helpers for M4 entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor

from mcwm.models.world_model import WorldModel
from mcwm.training.checkpoint import (
    checkpoint_sha256,
    load_frozen_m1_encoder,
    load_world_model_checkpoint,
    read_checkpoint,
)
from mcwm.training.config import build_world_model


def load_planning_world_model(
    checkpoint: Path,
    *,
    m1_checkpoint: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[WorldModel, dict]:
    """Reconstruct M1+M2, verify parent hashes, and load inference weights."""

    payload = read_checkpoint(Path(checkpoint), memory_map=True)
    extra = payload.get("extra", {})
    if extra.get("stage") != "m2-world-model":
        raise ValueError("planning requires an M2 world-model checkpoint")
    saved_parent = str(extra.get("m1_parent_path", ""))
    saved_parent_hash = str(extra.get("m1_parent_sha256", ""))
    if not saved_parent or not saved_parent_hash:
        raise ValueError("M2 checkpoint has incomplete M1 parent provenance")
    parent_path = Path(m1_checkpoint) if m1_checkpoint is not None else Path(saved_parent)
    if checkpoint_sha256(parent_path) != saved_parent_hash:
        raise ValueError("M1 parent checkpoint hash does not match M2 provenance")
    manifest_hash = str(payload["provenance"]["manifest_hash"])
    visual_encoder, _ = load_frozen_m1_encoder(
        parent_path,
        expected_manifest_hash=manifest_hash,
    )
    config = payload["provenance"]["config"]
    model = build_world_model(config, visual_encoder)
    load_world_model_checkpoint(
        Path(checkpoint),
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        expected_manifest_hash=manifest_hash,
        expected_m1_parent_path=saved_parent,
        expected_m1_parent_sha256=saved_parent_hash,
        restore_rng=False,
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA planning requested but CUDA is unavailable")
    model.to(resolved_device)
    model.eval()
    return model, payload


def load_rgb_image(path: Path) -> Tensor:
    """Load a goal/observation file as uint8 CHW at exactly 640x360."""

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("image goals require `pip install mcwm[train]`") from exc
    with Image.open(Path(path)) as image:
        image = image.convert("RGB").resize((640, 360))
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("image goals require `pip install mcwm[train]`") from exc
        array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

