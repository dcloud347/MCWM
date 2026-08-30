"""M3 multi-horizon rollout and surprise evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from mcwm.data.manifest import DatasetManifest
from mcwm.diagnostics.collapse import collapse_metrics
from mcwm.diagnostics.rollout import rollout_metrics, rollout_samples
from mcwm.diagnostics.surprise import surprise_metrics, surprise_samples
from mcwm.training.config import (
    build_world_model,
    load_yaml_config,
    validate_world_model_config,
)
from mcwm.training.train_world_model import (
    _autocast_context,
    _distributed_context,
    _load_distributed_evaluation_checkpoint,
    _make_loaders,
    _progress_bar,
    _seed_everything,
    _to_device,
    _visual_encoder_and_parent,
    _wrap_distributed,
)


def _validate_m3_config(config: Mapping[str, Any]) -> Tuple[int, ...]:
    required = {"checkpoint", "output", "data", "horizons", "surprise"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"M3 config is missing fields: {sorted(missing)}")
    data = config["data"]
    frames = int(data.get("frames_per_sample", 0))
    batches = int(data.get("validation_batches", 0))
    batch_size = int(data.get("batch_size", 0))
    if frames < 2 or batches <= 0 or batch_size < 2:
        raise ValueError(
            "M3 frames/batches must be positive and batch_size must be at least two"
        )
    horizons = tuple(int(value) for value in config["horizons"])
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("M3 horizons must be non-empty, unique, and increasing")
    if horizons[0] <= 0 or horizons[-1] >= frames:
        raise ValueError("M3 horizons must be smaller than frames_per_sample")
    perturbation_step = int(config["surprise"].get("perturbation_step", 0))
    if not 1 <= perturbation_step <= horizons[-1]:
        raise ValueError("surprise.perturbation_step must fit within max horizon")
    return horizons


def _evaluation_config(
    m2_config: Mapping[str, Any],
    m3_config: Mapping[str, Any],
) -> Dict[str, Any]:
    config = deepcopy(dict(m2_config))
    data = m3_config["data"]
    config["data"]["frames_per_sample"] = int(data["frames_per_sample"])
    config["data"]["batch_size"] = int(data["batch_size"])
    config["data"]["workers"] = int(data.get("workers", config["data"].get("workers", 0)))
    if data.get("root") is not None:
        config["data"]["root"] = str(data["root"])
    config["validation"]["batches"] = int(data["validation_batches"])
    config["validation"]["action_sensitivity_batches"] = min(
        int(config["validation"].get("action_sensitivity_batches", 8)),
        int(data["validation_batches"]),
    )
    validate_world_model_config(config)
    return config


def _append_samples(
    destination: Dict[str, list],
    values: Mapping[str, Tensor],
) -> None:
    for name, value in values.items():
        destination.setdefault(name, []).append(value.detach().cpu())


def _episode_id(sample_id: str) -> str:
    return str(sample_id).split(":pts=", 1)[0]


def _surprise_pair_indices(
    sample_ids: Sequence[str],
    episode_groups: Mapping[str, Tuple[str, str]],
) -> Tuple[Tensor, Tensor]:
    """Pair each usable clip with a clip from another session/world group."""

    groups = [
        episode_groups.get(_episode_id(sample_id), (_episode_id(sample_id), ""))
        for sample_id in sample_ids
    ]
    sources = []
    unrelated = []
    for source, group in enumerate(groups):
        match = next(
            (
                candidate
                for candidate, candidate_group in enumerate(groups)
                if candidate != source and candidate_group != group
            ),
            None,
        )
        if match is not None:
            sources.append(source)
            unrelated.append(match)
    return torch.tensor(sources, dtype=torch.long), torch.tensor(
        unrelated,
        dtype=torch.long,
    )


def _global_samples(collected: Mapping[str, Sequence[Tensor]]) -> Dict[str, Tensor]:
    local = {name: torch.cat(list(values)) for name, values in collected.items()}
    if not torch.distributed.is_initialized():
        return local
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, local)
    names = set().union(*(rank_values.keys() for rank_values in gathered))
    return {
        name: torch.cat(
            [rank_values[name] for rank_values in gathered if name in rank_values]
        )
        for name in names
    }


def _global_scalar_averages(
    weighted_sums: Mapping[str, float],
    count: int,
) -> Dict[str, float]:
    payload = {"sums": dict(weighted_sums), "count": int(count)}
    if torch.distributed.is_initialized():
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    total_count = sum(item["count"] for item in gathered)
    if total_count <= 0:
        raise ValueError("M3 evaluation did not collect scalar diagnostics")
    names = set().union(*(item["sums"].keys() for item in gathered))
    return {
        name: sum(item["sums"].get(name, 0.0) for item in gathered) / total_count
        for name in names
    }


def _global_sample_identity(sample_ids: Sequence[str]) -> Tuple[int, str]:
    local = [str(value) for value in sample_ids]
    if torch.distributed.is_initialized():
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, local)
        values = [value for rank_values in gathered for value in rank_values]
    else:
        values = local
    digest = sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return len(values), digest


@torch.no_grad()
def evaluate_m3(
    m2_config: Mapping[str, Any],
    m3_config: Mapping[str, Any],
    *,
    synthetic: bool = False,
    checkpoint: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Evaluate one M2 checkpoint at all configured M3 horizons."""

    horizons = _validate_m3_config(m3_config)
    config = _evaluation_config(m2_config, m3_config)
    checkpoint = Path(checkpoint or str(m3_config["checkpoint"]))
    output_path = Path(output_path or str(m3_config["output"]))
    strategy = str(config.get("distributed", {}).get("strategy", "none"))
    rank, _, world_size, device = _distributed_context(
        strategy,
        str(config["device"]),
    )
    seed = int(config.get("seed", 0))
    _seed_everything(seed + rank)

    try:
        _, validation_loader, manifest_hash = _make_loaders(
            config,
            synthetic=synthetic,
            rank=rank,
            world_size=world_size,
        )
        visual_encoder, m1_parent_path, m1_parent_hash = (
            _visual_encoder_and_parent(
                config,
                synthetic=synthetic,
                manifest_hash=manifest_hash,
            )
        )
        model = _wrap_distributed(
            build_world_model(config, visual_encoder),
            device=device,
            world_size=world_size,
            precision=str(config["precision"]),
        )
        payload = _load_distributed_evaluation_checkpoint(
            checkpoint,
            model=model,
            manifest_hash=manifest_hash,
            m1_parent_path=m1_parent_path,
            m1_parent_hash=m1_parent_hash,
            world_size=world_size,
        )
        saved_model_config = payload["provenance"]["config"]["model"]
        if saved_model_config != m2_config["model"]:
            raise ValueError("M3 evaluation model does not match checkpoint config")

        model.eval()
        rollout_collected: Dict[str, list] = {}
        surprise_collected: Dict[str, list] = {}
        collapse_sums: Dict[str, float] = {}
        collapse_count = 0
        sample_ids = []
        max_horizon = horizons[-1]
        perturbation_step = int(m3_config["surprise"]["perturbation_step"])
        if synthetic:
            episode_groups: Dict[str, Tuple[str, str]] = {}
        else:
            manifest = DatasetManifest.read(
                Path(str(config["data"]["root"])) / "dataset_manifest.json"
            )
            episode_groups = {
                episode.episode_id: (episode.session_id, episode.world_id)
                for episode in manifest.episodes
            }
        requested_batches = int(m3_config["data"]["validation_batches"])
        total_batches = min(requested_batches, len(validation_loader))
        progress_every = max(1, total_batches // 10)
        if rank == 0:
            print(
                f"[M3 evaluation] {_progress_bar(0, total_batches)} "
                f"0/{total_batches}",
                flush=True,
            )

        completed_batches = 0
        for batch_index, raw_batch in enumerate(validation_loader):
            if batch_index >= requested_batches:
                break
            batch = _to_device(raw_batch, device)
            sample_ids.extend(str(value) for value in raw_batch.get("sample_id", ()))
            with _autocast_context(device, str(config["precision"])):
                output = model(**batch, rollout_steps=max_horizon)
            predictions = output["autoregressive_predictions"]
            targets = output["targets"][:, :max_horizon]
            _append_samples(
                rollout_collected,
                rollout_samples(predictions, targets, batch),
            )
            source_indices, unrelated_indices = _surprise_pair_indices(
                raw_batch.get("sample_id", ()),
                episode_groups,
            )
            if source_indices.numel() > 0:
                source_indices = source_indices.to(predictions.device)
                unrelated_indices = unrelated_indices.to(predictions.device)
                _append_samples(
                    surprise_collected,
                    surprise_samples(
                        predictions[source_indices],
                        targets[source_indices],
                        perturbation_step=perturbation_step,
                        unrelated_targets=targets[unrelated_indices],
                    ),
                )

            batch_size = int(predictions.shape[0])
            batch_collapse = {}
            batch_collapse.update(
                collapse_metrics(
                    predictions.mean(dim=2),
                    prefix="m3/predicted_latent",
                )
            )
            batch_collapse.update(
                collapse_metrics(
                    targets.mean(dim=2),
                    prefix="m3/target_latent",
                )
            )
            for name, value in batch_collapse.items():
                collapse_sums[name] = collapse_sums.get(name, 0.0) + value * batch_size
            collapse_count += batch_size
            completed_batches = batch_index + 1
            if rank == 0 and (
                completed_batches % progress_every == 0
                or completed_batches == total_batches
            ):
                print(
                    f"[M3 evaluation] "
                    f"{_progress_bar(completed_batches, total_batches)} "
                    f"{completed_batches}/{total_batches}",
                    flush=True,
                )

        if not rollout_collected:
            raise ValueError("M3 validation did not produce rollout samples")
        rollout_values = _global_samples(rollout_collected)
        surprise_values = _global_samples(surprise_collected)
        if not surprise_values:
            raise ValueError(
                "M3 validation did not find clips from different session/world groups"
            )
        metrics = rollout_metrics(rollout_values, horizons=horizons)
        metrics.update(
            surprise_metrics(
                surprise_values,
                perturbation_step=perturbation_step,
            )
        )
        metrics.update(_global_scalar_averages(collapse_sums, collapse_count))
        sample_clip_count, sample_ids_sha256 = _global_sample_identity(sample_ids)
        finite = all(
            torch.isfinite(torch.tensor(value))
            for value in metrics.values()
        )
        diagnostic_pass = bool(
            finite and metrics["m3/surprise/pass_statistical"] == 1.0
        )
        report = {
            "format_version": 1,
            "stage": "m3-multi-horizon-evaluation",
            "checkpoint": str(checkpoint),
            "optimizer_step": int(payload["optimizer_step"]),
            "checkpoint_git_commit": payload["provenance"]["git_commit"],
            "evaluation_world_size": world_size,
            "validation_batches_per_rank": completed_batches,
            "frames_per_sample": int(config["data"]["frames_per_sample"]),
            "horizons": list(horizons),
            "sample_clips": sample_clip_count,
            "sample_ids_sha256": sample_ids_sha256,
            "manifest_hash": manifest_hash,
            "m1_parent_path": m1_parent_path,
            "m1_parent_sha256": m1_parent_hash,
            "gate": {
                "diagnostics_finite": bool(finite),
                "surprise_pass": (
                    metrics["m3/surprise/pass_statistical"] == 1.0
                ),
                "diagnostic_pass": diagnostic_pass,
                "auto_steps_1_baseline": "pending",
                "m3_complete": False,
            },
            "metrics": metrics,
        }
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            temporary.replace(output_path)
        if world_size > 1:
            torch.distributed.barrier()
        return output_path
    finally:
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluate_m3.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    m3_config = load_yaml_config(args.config)
    if args.max_batches is not None:
        m3_config["data"]["validation_batches"] = int(args.max_batches)
    if args.batch_size is not None:
        m3_config["data"]["batch_size"] = int(args.batch_size)
    m2_config_value = m3_config.get("m2_config")
    if not m2_config_value:
        parser.error("M3 config requires m2_config")
    m2_config_path = Path(str(m2_config_value))
    result = evaluate_m3(
        load_yaml_config(m2_config_path),
        m3_config,
        synthetic=args.synthetic,
        checkpoint=args.checkpoint,
        output_path=args.output,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(result)


if __name__ == "__main__":
    main()
