"""M2 action-conditioned world model 训练流程。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from mcwm.data.manifest import DatasetManifest
from mcwm.data.visual_dataset import ResumableSampler
from mcwm.data.world_model_dataset import (
    WorldModelDataset,
    collate_world_model_samples,
)
from mcwm.diagnostics.world_model import (
    action_sensitivity_report,
    spatial_error_images,
    world_model_prediction_metrics,
)
from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
from mcwm.models.visual_encoder import VisualEncoderConfig
from mcwm.models.world_model import WorldModel
from .checkpoint import (
    CheckpointProvenance,
    checkpoint_sha256,
    load_frozen_m1_encoder,
    load_world_model_checkpoint,
    save_world_model_checkpoint,
)
from .config import (
    build_world_model,
    load_yaml_config,
    resolved_copy,
    validate_world_model_config,
)
from .logging import TrainingLogger


MODEL_INPUT_NAMES = (
    "frames",
    "movement",
    "interaction",
    "hotbar",
    "camera",
    "cursor",
    "gui_open",
    "cursor_present",
    "valid_mask",
)


class SyntheticWorldModelDataset(Dataset):
    """生成有动作关联的确定性小样本，只用于 M2 smoke test。"""

    def __init__(
        self,
        length: int,
        *,
        frames: int,
        height: int,
        width: int,
        seed: int,
    ) -> None:
        self.length = int(length)
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: object) -> Dict[str, object]:
        if isinstance(index, tuple):
            sample_index, sample_seed = index
        else:
            sample_index = int(index)
            sample_seed = self.seed + sample_index
        generator = torch.Generator().manual_seed(int(sample_seed))
        base = torch.randint(
            0,
            128,
            (3, self.height, self.width),
            dtype=torch.uint8,
            generator=generator,
        )
        direction = 1 if int(sample_index) % 2 == 0 else -1
        frames = torch.stack(
            [
                torch.roll(base, shifts=(0, direction * step), dims=(1, 2))
                for step in range(self.frames)
            ]
        )
        transitions = self.frames - 1
        ticks = 2
        shape = (transitions, ticks)
        movement = torch.zeros(*shape, 7, dtype=torch.bool)
        movement[..., 0 if direction > 0 else 2] = True
        interaction = torch.zeros(*shape, 7, dtype=torch.bool)
        interaction[1::2, 0, 0] = True
        camera = torch.zeros(*shape, 2)
        camera[..., 1] = float(direction)
        return {
            "frames": frames,
            "movement": movement,
            "interaction": interaction,
            "hotbar": torch.zeros(shape, dtype=torch.long),
            "camera": camera,
            "cursor": torch.zeros(*shape, 2),
            "gui_open": torch.zeros(shape, dtype=torch.bool),
            "cursor_present": torch.zeros(shape, dtype=torch.bool),
            "valid_mask": torch.ones(shape, dtype=torch.bool),
            "sample_id": f"synthetic-m2:{sample_index}",
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-worktree"


def _training_signature(config: Mapping[str, Any]) -> str:
    value = resolved_copy(config)
    value.pop("output_dir", None)
    value["checkpoint"].pop("resume", None)
    value["data"].pop("root", None)
    value["data"].pop("workers", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _synthetic_visual_encoder(config: Mapping[str, Any]) -> FrozenVisualEncoder:
    value = config["model"]["synthetic_visual_encoder"]
    encoder_config = VisualEncoderConfig(
        image_height=int(value["image_height"]),
        image_width=int(value["image_width"]),
        patch_size=int(value["patch_size"]),
        clip_frames=int(value.get("clip_frames", 16)),
        tubelet_size=int(value.get("tubelet_size", 2)),
        dim=int(value["dim"]),
        depth=int(value["depth"]),
        heads=int(value["heads"]),
        mlp_dim=int(value["mlp_dim"]),
        gradient_checkpointing=False,
    )
    return FrozenVisualEncoder(encoder_config)


def _make_loaders(
    config: Mapping[str, Any],
    *,
    synthetic: bool,
) -> Tuple[DataLoader, DataLoader, str]:
    data = config["data"]
    seed = int(config.get("seed", 0))
    batch_size = int(data["batch_size"])
    if synthetic:
        visual = config["model"]["synthetic_visual_encoder"]
        length = max(batch_size * int(config["optimizer"]["iterations_per_epoch"]), 4)
        train_dataset: Dataset = SyntheticWorldModelDataset(
            length,
            frames=int(data["frames_per_sample"]),
            height=int(visual["image_height"]),
            width=int(visual["image_width"]),
            seed=seed,
        )
        validation_dataset: Dataset = SyntheticWorldModelDataset(
            max(batch_size, 2),
            frames=int(data["frames_per_sample"]),
            height=int(visual["image_height"]),
            width=int(visual["image_width"]),
            seed=seed + 10_000,
        )
        manifest_hash = "synthetic-m2-smoke-v1"
        collate_fn = None
    else:
        root = Path(data["root"])
        manifest = DatasetManifest.read(root / "dataset_manifest.json")
        manifest_hash = manifest.content_hash
        train_dataset = WorldModelDataset(
            root,
            split=str(data.get("train_split", "train")),
            frames_per_sample=int(data["frames_per_sample"]),
            sample_fps=int(data["sample_fps"]),
            seed=seed,
            samples_per_video=int(data["samples_per_video"]),
        )
        validation_dataset = WorldModelDataset(
            root,
            split=str(data.get("validation_split", "validation")),
            frames_per_sample=int(data["frames_per_sample"]),
            sample_fps=int(data["sample_fps"]),
            seed=seed + 10_000,
            samples_per_video=int(data["samples_per_video"]),
        )
        collate_fn = collate_world_model_samples

    sampler = ResumableSampler(
        len(train_dataset),
        seed=seed,
        seed_clips=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(data.get("workers", 0)),
        drop_last=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data.get("workers", 0)),
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, validation_loader, manifest_hash


def _batch_stream(
    loader: DataLoader,
    *,
    epoch: int,
    batch_offset: int,
) -> Iterator[Tuple[Mapping[str, object], int, int]]:
    """无限产生 batch，并返回下一个 batch 对应的 sampler 位置。"""

    sampler = loader.sampler
    batch_size = int(loader.batch_size or 1)
    while True:
        sampler.set_epoch(epoch)
        sampler.set_start_index(batch_offset * batch_size)
        yielded = False
        for batch in loader:
            yielded = True
            batch_offset += 1
            yield batch, epoch, batch_offset
        if not yielded and batch_offset == 0:
            raise ValueError("training loader cannot provide a full batch")
        epoch += 1
        batch_offset = 0


def _to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in MODEL_INPUT_NAMES
    }


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _total_steps(config: Mapping[str, Any]) -> int:
    optimizer = config["optimizer"]
    return int(round(float(optimizer["epochs"]) * int(optimizer["iterations_per_epoch"])))


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.no_grad()
def validate(
    model: WorldModel,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    batches: int,
) -> Tuple[Dict[str, float], list]:
    """计算 validation prediction、collapse 与 action sensitivity 指标。"""

    model.eval()
    collected: Dict[str, list] = {}
    images = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batches:
            break
        batch = _to_device(raw_batch, device)
        with _autocast_context(device, precision):
            output = model(**batch)
        metrics = world_model_prediction_metrics(output)
        if batch_index == 0:
            metrics.update(
                action_sensitivity_report(model, batch, latents=output["latents"])
            )
            images = spatial_error_images(
                output["teacher_forced_predictions"],
                output["targets"],
                spatial_grid=model.predictor.config.spatial_grid,
            )
        for name, value in metrics.items():
            collected.setdefault(name, []).append(float(value))
    model.train()
    if not collected:
        raise ValueError("validation loader produced no batches")
    metrics = {
        name: sum(values) / len(values)
        for name, values in collected.items()
    }
    return metrics, images


def _prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    paths = sorted(output_dir.glob("checkpoint-*.pt"))
    for path in paths[: max(0, len(paths) - keep_last)]:
        path.unlink()


def train(config: Mapping[str, Any], *, synthetic: bool = False) -> Path:
    """训练 M2，并返回最终 checkpoint 路径。"""

    validate_world_model_config(config)
    seed = int(config.get("seed", 0))
    _seed_everything(seed)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA, but CUDA is unavailable")

    train_loader, validation_loader, manifest_hash = _make_loaders(
        config,
        synthetic=synthetic,
    )
    if synthetic:
        visual_encoder = _synthetic_visual_encoder(config)
        m1_parent_path = "synthetic://random-frozen-m1"
        parent_value = json.dumps(
            {
                "seed": seed,
                "config": config["model"]["synthetic_visual_encoder"],
            },
            sort_keys=True,
        ).encode("utf-8")
        m1_parent_hash = sha256(parent_value).hexdigest()
    else:
        m1_path = Path(str(config.get("m1_checkpoint")))
        visual_encoder, _ = load_frozen_m1_encoder(
            m1_path,
            expected_manifest_hash=manifest_hash,
        )
        m1_parent_path = str(m1_path)
        m1_parent_hash = checkpoint_sha256(m1_path)

    model = build_world_model(config, visual_encoder).to(device)
    trainable = list(model.trainable_parameters())
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    total_steps = _total_steps(config)
    scheduler = _scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(optimizer_config.get("warmup_steps", 0)),
    )
    use_scaler = device.type == "cuda" and config["precision"] == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    per_step_batch = int(config["data"]["batch_size"])
    effective_batch = int(optimizer_config["effective_batch_size"])
    if effective_batch % per_step_batch:
        raise ValueError("effective_batch_size must be divisible by data.batch_size")
    accumulation_steps = effective_batch // per_step_batch
    optimizer_step = 0
    sampler_epoch = 0
    batch_offset = 0
    resume_path = config["checkpoint"].get("resume")
    resume_payload = None
    if resume_path:
        resume_payload = load_world_model_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
            expected_manifest_hash=manifest_hash,
            expected_m1_parent_path=m1_parent_path,
            expected_m1_parent_sha256=m1_parent_hash,
        )
        saved_signature = _training_signature(
            resume_payload["provenance"]["config"]
        )
        if saved_signature != _training_signature(config):
            raise ValueError("resume changes M2 model/data/optimizer semantics")
        optimizer_step = int(resume_payload["optimizer_step"])
        sampler_epoch = int(resume_payload["extra"]["sampler_epoch"])
        batch_offset = int(resume_payload["extra"]["batch_offset"])

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(
        output_dir,
        config=resolved_copy(config),
        wandb_config=config["wandb"],
        run_id=(resume_payload or {}).get("provenance", {}).get("wandb_run_id"),
    )
    provenance = CheckpointProvenance(
        git_commit=_git_commit(),
        config=resolved_copy(config),
        seed=seed,
        manifest_hash=manifest_hash,
        parent_checkpoint=m1_parent_path,
        wandb_entity=config["wandb"].get("entity"),
        wandb_project=config["wandb"].get("project"),
        wandb_run_id=logger.run_id,
        wandb_run_name=logger.run_name,
    )

    batches = _batch_stream(
        train_loader,
        epoch=sampler_epoch,
        batch_offset=batch_offset,
    )
    model.train()
    last_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
    try:
        while optimizer_step < total_steps:
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {"loss": 0.0, "teacher": 0.0, "autoregressive": 0.0}
            for _ in range(accumulation_steps):
                raw_batch, sampler_epoch, batch_offset = next(batches)
                batch = _to_device(raw_batch, device)
                with _autocast_context(device, str(config["precision"])):
                    output = model(**batch)
                    loss = output["loss"] / accumulation_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated["loss"] += float(output["loss"].detach())
                accumulated["teacher"] += float(output["teacher_forced_loss"].detach())
                accumulated["autoregressive"] += float(
                    output["autoregressive_loss"].detach()
                )

            if use_scaler:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(optimizer_config.get("gradient_clip", 1.0)),
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite M2 gradient")
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer_step += 1

            if optimizer_step % int(config["wandb"].get("log_every_steps", 10)) == 0:
                step_seconds = time.perf_counter() - step_started
                logger.log(
                    {
                        "train/loss": accumulated["loss"] / accumulation_steps,
                        "train/teacher_forced_loss": accumulated["teacher"]
                        / accumulation_steps,
                        "train/autoregressive_loss": accumulated["autoregressive"]
                        / accumulation_steps,
                        "train/gradient_norm": float(gradient_norm),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "system/step_seconds": step_seconds,
                        "system/samples_per_second": effective_batch
                        / max(step_seconds, 1e-9),
                        "system/max_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                    },
                    step=optimizer_step,
                )
            if optimizer_step % int(config["validation"]["every_steps"]) == 0:
                validation_metrics, validation_images = validate(
                    model,
                    validation_loader,
                    device=device,
                    precision=str(config["precision"]),
                    batches=int(config["validation"]["batches"]),
                )
                logger.log(
                    validation_metrics,
                    step=optimizer_step,
                )
                logger.log_images(
                    "validation/spatial_token_error",
                    validation_images,
                    step=optimizer_step,
                )
            if optimizer_step % int(config["checkpoint"]["every_steps"]) == 0:
                last_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
                save_world_model_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    optimizer_step=optimizer_step,
                    provenance=provenance,
                    m1_parent_path=m1_parent_path,
                    m1_parent_sha256=m1_parent_hash,
                    sampler_epoch=sampler_epoch,
                    batch_offset=batch_offset,
                    resume_checkpoint=str(resume_path) if resume_path else None,
                )
                _prune_checkpoints(
                    output_dir,
                    int(config["checkpoint"].get("keep_last", 3)),
                )

        final_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
        if final_checkpoint != last_checkpoint or not final_checkpoint.exists():
            save_world_model_checkpoint(
                final_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                optimizer_step=optimizer_step,
                provenance=provenance,
                m1_parent_path=m1_parent_path,
                m1_parent_sha256=m1_parent_hash,
                sampler_epoch=sampler_epoch,
                batch_offset=batch_offset,
                resume_checkpoint=str(resume_path) if resume_path else None,
            )
        return final_checkpoint
    finally:
        logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_world_model.yaml"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.resume is not None:
        config["checkpoint"]["resume"] = str(args.resume)
    if args.max_steps is not None:
        config["optimizer"]["iterations_per_epoch"] = int(args.max_steps)
        config["optimizer"]["epochs"] = 1
    checkpoint = train(config, synthetic=args.synthetic)
    print(checkpoint)


if __name__ == "__main__":
    main()
