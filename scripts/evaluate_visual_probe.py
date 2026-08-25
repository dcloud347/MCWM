#!/usr/bin/env python3
"""用冻结 linear probe 比较 M1 EMA encoder 和随机 encoder。"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from mcwm.data.visual_dataset import CanonicalVisualDataset
from mcwm.diagnostics.probes import ridge_linear_probe
from mcwm.models.visual_encoder import VisualEncoder
from mcwm.training.checkpoint import read_checkpoint
from mcwm.training.config import build_visual_jepa

VJEPA_PIXEL_MEAN = (0.485, 0.456, 0.406)
VJEPA_PIXEL_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def extract_features(
    encoder: VisualEncoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """提取冻结 encoder 特征及只供 probe 使用的标签。"""

    encoder.eval().to(device)
    features = []
    labels = {"camera_motion": [], "gui_open": [], "scene_change": []}
    mean = torch.tensor(VJEPA_PIXEL_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(VJEPA_PIXEL_STD, device=device).view(1, 1, 3, 1, 1)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        frames = batch["frames"].to(device).float().div_(255.0)
        frames = (frames - mean) / std
        batch_size = frames.shape[0]
        tokens = encoder(frames, return_patch_tokens=True).reshape(
            batch_size,
            encoder.config.temporal_grid_size,
            encoder.config.patch_count,
            encoder.config.dim,
        )
        latent = tokens.mean(dim=2)
        # 首、末 tubelet 和差值同时提供外观与运动信息。
        features.append(torch.cat((latent[:, 0], latent[:, -1], latent[:, -1] - latent[:, 0]), 1).cpu())
        for name in labels:
            labels[name].append(batch[name].cpu())
    if not features:
        raise ValueError("probe loader produced no batches")
    return torch.cat(features), {name: torch.cat(values) for name, values in labels.items()}


def evaluate(
    train_features: torch.Tensor,
    train_labels: Dict[str, torch.Tensor],
    validation_features: torch.Tensor,
    validation_labels: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """分别评估 camera、GUI 和 scene change 三类 probe。"""

    results = {}
    for name in ("camera_motion", "scene_change"):
        metrics = ridge_linear_probe(
            train_features,
            train_labels[name],
            validation_features,
            validation_labels[name],
            task="regression",
        )
        results.update({f"{name}/{key.split('/')[-1]}": value for key, value in metrics.items()})
    metrics = ridge_linear_probe(
        train_features,
        train_labels["gui_open"],
        validation_features,
        validation_labels["gui_open"],
        task="classification",
    )
    results.update({f"gui_open/{key.split('/')[-1]}": value for key, value in metrics.items()})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="append probe metrics to the W&B run stored in the checkpoint",
    )
    args = parser.parse_args()

    payload = read_checkpoint(args.checkpoint)
    config = payload["provenance"]["config"]
    clip_frames = int(config["data"]["clip_frames"])
    datasets = {
        split: CanonicalVisualDataset(
            args.data_root,
            split=split,
            clip_frames=clip_frames,
            sample_fps=int(config["data"]["sample_fps"]),
            seed=int(config.get("seed", 2026)),
            include_probe_labels=True,
        )
        for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for split, dataset in datasets.items()
    }
    device = torch.device(args.device)
    trained = build_visual_jepa(config).target_encoder
    target_state = {
        name[len("target_encoder.") :]: value
        for name, value in payload["model"].items()
        if name.startswith("target_encoder.")
    }
    trained.load_state_dict(target_state)
    trained_train = extract_features(trained, loaders["train"], device, args.max_batches)
    trained_validation = extract_features(
        trained, loaders["validation"], device, args.max_batches
    )
    trained_results = evaluate(*trained_train, *trained_validation)

    del trained
    random_encoder = build_visual_jepa(config).target_encoder
    random_train = extract_features(random_encoder, loaders["train"], device, args.max_batches)
    random_validation = extract_features(
        random_encoder, loaders["validation"], device, args.max_batches
    )
    random_results = evaluate(*random_train, *random_validation)
    report = {"trained": trained_results, "random": random_results}
    if args.log_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("--log-wandb requires `pip install mcwm[train]`") from exc
        provenance = payload["provenance"]
        run_id = provenance.get("wandb_run_id")
        if not run_id:
            raise ValueError("checkpoint has no W&B run ID")
        run = wandb.init(
            entity=provenance.get("wandb_entity"),
            project=provenance.get("wandb_project") or "mcwm",
            id=run_id,
            resume="allow",
        )
        flattened = {
            f"probe/{encoder_name}/{metric_name}": value
            for encoder_name, metrics in report.items()
            for metric_name, value in metrics.items()
        }
        run.log(flattened, step=int(payload["optimizer_step"]))
        run.finish()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
