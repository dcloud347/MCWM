"""验收 M1 EMA encoder 在 M2 repeated-frame 用法下的 latent。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from mcwm.data.manifest import DatasetManifest
from mcwm.data.world_model_dataset import WorldModelDataset
from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
from mcwm.training.checkpoint import load_frozen_m1_encoder
from .collapse import collapse_metrics


EVEN_FRAME_COUNTS = (2, 4, 6, 8, 10, 12, 14, 16)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _sample_feature_rows(tokens: Tensor, maximum: int = 2048) -> Tensor:
    values = tokens.detach().float().reshape(-1, tokens.shape[-1]).cpu()
    if values.shape[0] <= maximum:
        return values
    indices = torch.linspace(0, values.shape[0] - 1, maximum).round().long()
    return values.index_select(0, indices)


def _mean_cosine(left: Tensor, right: Tensor) -> Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1).mean(dim=-1)


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denominator <= torch.finfo(torch.float64).eps:
        return None
    return float((x * y).sum() / denominator)


@torch.no_grad()
def _encode_continuous_pairs(
    model: FrozenVisualEncoder,
    frames: Tensor,
    *,
    pair_chunk_size: int,
) -> Tensor:
    """按 M1 原始的连续双帧 tubelet 编码相邻画面。"""

    batch, frame_count, channels, height, width = frames.shape
    pairs = torch.stack((frames[:, :-1], frames[:, 1:]), dim=2).reshape(
        batch * (frame_count - 1), 2, channels, height, width
    )
    outputs = []
    for start in range(0, pairs.shape[0], pair_chunk_size):
        normalized = model.normalize_frames(pairs[start : start + pair_chunk_size])
        outputs.append(model.encoder(normalized, return_patch_tokens=True))
    return torch.cat(outputs).reshape(
        batch,
        frame_count - 1,
        model.config.patch_count,
        model.config.dim,
    )


def _action_transition_metrics(action_blocks: Sequence[Sequence[object]]) -> Dict[str, list]:
    camera = []
    gui_active = []
    for block in action_blocks:
        camera.append(sum(math.hypot(*action.camera) for action in block))
        gui_active.append(any(action.gui_open for action in block))
    return {"camera": camera, "gui_active": gui_active}


@torch.no_grad()
def probe_samples(
    model: FrozenVisualEncoder,
    samples: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    precision: str,
    frame_chunk_size: int,
    max_samples: int,
) -> Dict[str, object]:
    """运行真实片段 probe；只把抽样 latent 留在 CPU，控制显存和内存。"""

    health_rows = []
    continuous_health_rows = []
    repeated_continuous_cosines = []
    adjacent_distances = []
    scene_changes = []
    camera_motion = []
    gui_active = []
    deterministic_max_error = None
    samples_seen = 0
    observed_frame_count = None

    for sample in samples:
        if samples_seen >= max_samples:
            break
        frames = sample["frames"].unsqueeze(0).to(device)
        with _autocast(device, precision):
            repeated = model(frames, frame_chunk_size=frame_chunk_size)
            continuous = _encode_continuous_pairs(
                model,
                frames,
                pair_chunk_size=frame_chunk_size,
            )
        expected = (
            1,
            frames.shape[1],
            model.config.patch_count,
            model.config.dim,
        )
        if tuple(repeated.shape) != expected:
            raise RuntimeError(f"unexpected repeated latent shape: {tuple(repeated.shape)}")
        observed_frame_count = int(frames.shape[1])
        if not torch.isfinite(repeated).all() or not torch.isfinite(continuous).all():
            raise RuntimeError("encoder produced non-finite latent values")

        if deterministic_max_error is None:
            with _autocast(device, precision):
                first = model(frames[:, :1], frame_chunk_size=1)
                second = model(frames[:, :1], frame_chunk_size=1)
            deterministic_max_error = float((first.float() - second.float()).abs().max())

        repeated_continuous_cosines.extend(
            _mean_cosine(repeated[:, :-1], continuous).flatten().cpu().tolist()
        )
        transition_cosine = _mean_cosine(repeated[:, :-1], repeated[:, 1:])
        transition_distance = (1.0 - transition_cosine).flatten().cpu().tolist()
        adjacent_distances.extend(transition_distance)
        pixel_change = (
            (frames[:, 1:].float() - frames[:, :-1].float())
            .abs()
            .mean(dim=(2, 3, 4))
            .div(255.0)
            .flatten()
            .cpu()
            .tolist()
        )
        scene_changes.extend(pixel_change)
        actions = _action_transition_metrics(sample["action_blocks"])
        camera_motion.extend(actions["camera"])
        gui_active.extend(actions["gui_active"])
        health_rows.append(_sample_feature_rows(repeated, maximum=256))
        continuous_health_rows.append(_sample_feature_rows(continuous, maximum=256))
        samples_seen += 1

    if samples_seen == 0:
        raise ValueError("dataset did not produce any probe samples")
    repeated_health = collapse_metrics(torch.cat(health_rows), prefix="repeated")
    continuous_health = collapse_metrics(
        torch.cat(continuous_health_rows), prefix="continuous"
    )
    active_distances = [
        value for value, active in zip(adjacent_distances, gui_active) if active
    ]
    inactive_distances = [
        value for value, active in zip(adjacent_distances, gui_active) if not active
    ]
    return {
        "samples": samples_seen,
        "transitions": len(adjacent_distances),
        "output_shape_per_sample": [
            observed_frame_count,
            int(model.config.patch_count),
            int(model.config.dim),
        ],
        "deterministic_max_abs_error": deterministic_max_error,
        "repeated_health": repeated_health,
        "continuous_health": continuous_health,
        "distribution_shift": {
            "repeated_vs_continuous_mean_cosine": float(
                torch.tensor(repeated_continuous_cosines).mean()
            ),
            "repeated_vs_continuous_min_cosine": min(repeated_continuous_cosines),
        },
        "sensitivity": {
            "mean_adjacent_latent_cosine_distance": float(
                torch.tensor(adjacent_distances).mean()
            ),
            "scene_change_correlation": _pearson(adjacent_distances, scene_changes),
            "camera_motion_correlation": _pearson(adjacent_distances, camera_motion),
            "gui_active_mean_distance": (
                sum(active_distances) / len(active_distances) if active_distances else None
            ),
            "gui_inactive_mean_distance": (
                sum(inactive_distances) / len(inactive_distances)
                if inactive_distances
                else None
            ),
        },
    }


@torch.no_grad()
def probe_variable_lengths(
    model: FrozenVisualEncoder,
    frame: Tensor,
    *,
    device: torch.device,
    precision: str,
) -> Dict[str, object]:
    """用真实画面验证 checkpoint 在所有合法 T 上的底层 encoder 输出。"""

    results = {}
    normalized_frame = model.normalize_frames(frame.to(device))
    for frame_count in EVEN_FRAME_COUNTS:
        clip = normalized_frame.repeat(1, frame_count, 1, 1, 1)
        with _autocast(device, precision):
            tokens = model.encoder(clip, return_patch_tokens=True)
        expected_tokens = model.config.runtime_token_count(frame_count)
        results[str(frame_count)] = {
            "shape": list(tokens.shape),
            "expected_tokens": expected_tokens,
            "finite": bool(torch.isfinite(tokens).all()),
        }
        if tokens.shape[1] != expected_tokens or not results[str(frame_count)]["finite"]:
            raise RuntimeError(f"variable-T probe failed for T={frame_count}")
        del clip, tokens
    return results


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--frame-chunk-size", type=int, default=1)
    parser.add_argument("--skip-variable-t", action="store_true")
    parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        help="allow probing data other than the dataset recorded by the M1 checkpoint",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.frame_chunk_size <= 0:
        parser.error("--samples and --frame-chunk-size must be positive")

    device = _resolve_device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    if device.type == "cpu" and precision == "fp16":
        parser.error("fp16 is not supported for this CPU probe")

    probe_manifest_hash = DatasetManifest.read(
        args.data_root / "dataset_manifest.json"
    ).content_hash
    expected_manifest_hash = None if args.allow_manifest_mismatch else probe_manifest_hash
    print(f"loading checkpoint with memory mapping: {args.checkpoint}", flush=True)
    model, payload = load_frozen_m1_encoder(
        args.checkpoint,
        expected_manifest_hash=expected_manifest_hash,
    )
    provenance = payload["provenance"]
    checkpoint_report = {
        "path": str(args.checkpoint),
        "optimizer_step": int(payload["optimizer_step"]),
        "git_commit": provenance["git_commit"],
        "manifest_hash": provenance["manifest_hash"],
        "probe_manifest_hash": probe_manifest_hash,
        "probe_manifest_matches": provenance["manifest_hash"] == probe_manifest_hash,
        "external_pretrained": provenance["external_pretrained"],
        "loaded_state": "target_encoder",
    }
    del payload
    gc.collect()
    model.to(device).eval()
    frozen = all(not parameter.requires_grad for parameter in model.parameters())
    if not frozen or model.encoder.training:
        raise RuntimeError("M1 target encoder is not completely frozen in eval mode")

    dataset = WorldModelDataset(
        args.data_root,
        split=args.split,
        frames_per_sample=8,
        sample_fps=args.sample_fps,
        seed=args.seed,
    )
    print(
        f"probing {min(args.samples, len(dataset))} samples on {device} ({precision})",
        flush=True,
    )
    report = {
        "checkpoint": checkpoint_report,
        "runtime": {
            "device": str(device),
            "precision": precision,
            "frozen": frozen,
            "encoder_eval": not model.encoder.training,
        },
        "real_samples": probe_samples(
            model,
            (dataset[index] for index in range(len(dataset))),
            device=device,
            precision=precision,
            frame_chunk_size=args.frame_chunk_size,
            max_samples=args.samples,
        ),
    }
    if not args.skip_variable_t:
        first_frame = dataset[0]["frames"][:1].unsqueeze(0)
        report["variable_t"] = probe_variable_lengths(
            model,
            first_frame,
            device=device,
            precision=precision,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
