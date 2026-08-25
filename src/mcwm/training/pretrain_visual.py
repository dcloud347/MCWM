"""M1 visual JEPA 训练入口：正式训练优先使用 CUDA。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mcwm.data.manifest import DatasetManifest
from mcwm.data.visual_dataset import (
    CanonicalVisualDataset,
    DistributedResumableSampler,
    DistributedSourceBalancedSampler,
    ResumableSampler,
    source_balanced_weights,
)
from mcwm.diagnostics.collapse import (
    CollapseThresholds,
    collapse_metrics,
    find_collapse_alerts,
    online_target_gap,
)
from mcwm.diagnostics.visualization import visual_pretraining_images
from mcwm.models.visual_jepa import VisualJEPA
from .checkpoint import (
    CheckpointProvenance,
    capture_rng_state,
    load_checkpoint,
    read_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .config import build_visual_jepa, load_yaml_config, resolved_copy, validate_pretrain_config
from .ema import cosine_ema_momentum
from .logging import TrainingLogger


class SyntheticVideoDataset(Dataset):
    """可复现的移动图案，只用于 smoke test，绝不能产出正式 checkpoint。"""

    def __init__(self, length: int, frames: int, height: int, width: int, seed: int) -> None:
        self.length = length
        self.frames = frames
        self.height = height
        self.width = width
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, object]:
        generator = torch.Generator().manual_seed(self.seed + index)
        base = torch.randint(
            0,
            96,
            (3, self.height, self.width),
            dtype=torch.uint8,
            generator=generator,
        )
        clip = []
        for frame_index in range(self.frames):
            frame = torch.roll(base, shifts=(frame_index, frame_index * 2), dims=(1, 2))
            clip.append(frame)
        return {
            "frames": torch.stack(clip),
            "sample_id": f"synthetic:{index}",
            "source": "synthetic",
        }


def _git_commit() -> str:
    """记录当前 commit；未处于 Git 仓库时使用明确的占位值。"""

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
    """这些配置必须不变，才能把训练继续写进同一个 W&B run。"""

    data = dict(config["data"])
    for operational_key in ("root", "workers"):
        data.pop(operational_key, None)
    signature = {
        "seed": config.get("seed"),
        "precision": config.get("precision"),
        "data": data,
        "model": config["model"],
        "mask": config["mask"],
        "augmentation": config["augmentation"],
        "optimizer": config["optimizer"],
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _progress_bar(completed: int, total: int, *, width: int = 24) -> str:
    """Render a fixed-width progress bar suitable for redirected text logs."""

    if total <= 0:
        return "[" + "-" * width + "]"
    fraction = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_duration(seconds: float) -> str:
    """Format an elapsed duration compactly for progress logs."""

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _seed_everything(seed: int) -> None:
    """统一设置各随机数生成器；不同 rank 会传入不同 seed。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_context(strategy: str, requested_device: str) -> Tuple[int, int, int, torch.device]:
    """根据 torchrun 环境变量初始化进程组并确定当前设备。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA, but torch.cuda.is_available() is false")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        if strategy == "none":
            raise ValueError("WORLD_SIZE > 1 requires distributed.strategy=ddp or fsdp")
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    elif strategy in {"ddp", "fsdp"} and rank == 0:
        print(f"distributed.strategy={strategy} requested with one process; using an unwrapped model")
    return rank, local_rank, world_size, device


def _wrap_distributed(
    model: VisualJEPA,
    strategy: str,
    device: torch.device,
    world_size: int,
    precision: str,
) -> nn.Module:
    """单卡不包装；多卡根据配置使用 DDP 或分层 auto-wrap 的 FSDP。"""

    model.to(device)
    if world_size == 1:
        return model
    if strategy == "ddp":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        parameter_dtype = (
            torch.bfloat16
            if precision == "bf16"
            else torch.float16
            if precision == "fp16"
            else torch.float32
        )
        # 超过 1M 参数的子模块会单独 shard，避免每次 forward 同时聚合全部权重。
        return FSDP(
            model,
            use_orig_params=True,
            device_id=device,
            auto_wrap_policy=partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            mixed_precision=MixedPrecision(
                param_dtype=parameter_dtype,
                reduce_dtype=parameter_dtype,
                buffer_dtype=parameter_dtype,
            ),
        )
    raise ValueError(f"unsupported distributed strategy: {strategy}")


def _unwrapped(model: nn.Module) -> VisualJEPA:
    """从 DDP/FSDP wrapper 中取出原始 VisualJEPA。"""

    module = getattr(model, "module", model)
    if not isinstance(module, VisualJEPA):
        raise TypeError("wrapped model does not contain VisualJEPA")
    return module


def _update_target(model: nn.Module, strategy: str, world_size: int, momentum: float) -> None:
    """每个 optimizer step 后只更新一次 EMA target。"""

    if world_size > 1 and strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # EMA 按名字配对完整参数，所以 FSDP 下要临时召回完整权重。
        # 这里只在 optimizer step 后执行，不会在 accumulation micro-step 中执行。
        with FSDP.summon_full_params(model, recurse=True, writeback=True):
            _unwrapped(model).update_target(momentum)
    else:
        _unwrapped(model).update_target(momentum)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    optimizer_step: int,
    provenance: CheckpointProvenance,
    strategy: str,
    rank: int,
    world_size: int,
    training_state: Optional[Mapping[str, Any]] = None,
) -> None:
    """保存单卡/DDP/FSDP 状态；多卡还会保存每个 rank 的独立 RNG。"""

    if world_size > 1:
        rng_states = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(rng_states, capture_rng_state())
    else:
        rng_states = [capture_rng_state()]

    if world_size > 1 and strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        # 写盘的是可移植的 full state dict，而不是依赖当前 world size 的碎片。
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
        if rank == 0:
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                optimizer_step=optimizer_step,
                provenance=provenance,
                model_state_dict=model_state,
                optimizer_state_dict=optimizer_state,
                rng_state=rng_states[0],
                extra={"rng_by_rank": rng_states, **dict(training_state or {})},
            )
        torch.distributed.barrier()
        return
    if rank == 0:
        model_state = _unwrapped(model).state_dict() if world_size > 1 else model.state_dict()
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            optimizer_step=optimizer_step,
            provenance=provenance,
            rng_state=rng_states[0],
            extra={"rng_by_rank": rng_states, **dict(training_state or {})},
            model_state_dict=model_state,
        )


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    manifest_hash: str,
    strategy: str,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """加载 checkpoint，并把 full optimizer state 重新切给当前 FSDP ranks。"""

    if world_size > 1 and strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        payload = read_checkpoint(path, expected_manifest_hash=manifest_hash)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(payload["model"])
            optimizer_state = FSDP.optim_state_dict_to_load(
                model, optimizer, payload["optimizer"]
            )
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler"])
        rng_states = payload.get("extra", {}).get("rng_by_rank", [])
        restore_rng_state(rng_states[rank] if len(rng_states) == world_size else payload["rng"])
        return payload
    return load_checkpoint(
        path,
        model=_unwrapped(model) if world_size > 1 else model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_manifest_hash=manifest_hash,
    )


def _autocast_context(device: torch.device, precision: str):
    """fp32 不开 autocast；bf16/fp16 使用对应设备的 autocast。"""

    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _augment_clip(frames: Tensor, config: Mapping[str, Any]) -> Tensor:
    """只增强 online 分支；同一 clip 的所有帧使用一致颜色扰动。"""

    if not config.get("enabled", False):
        return frames
    values = frames.float().div(255.0) if frames.dtype == torch.uint8 else frames.float()
    batch = values.shape[0]
    brightness = float(config.get("brightness", 0.0))
    gamma = float(config.get("gamma", 0.0))
    color = float(config.get("color", 0.0))
    brightness_scale = 1.0 + torch.empty(
        batch, 1, 1, 1, 1, device=values.device
    ).uniform_(-brightness, brightness)
    gamma_value = 1.0 + torch.empty(
        batch, 1, 1, 1, 1, device=values.device
    ).uniform_(-gamma, gamma)
    color_scale = 1.0 + torch.empty(
        batch, 1, 3, 1, 1, device=values.device
    ).uniform_(-color, color)
    return (
        values.mul(brightness_scale)
        .clamp_(0, 1)
        .pow_(gamma_value)
        .mul_(color_scale)
        .clamp_(0, 1)
    )


def _learning_rate_lambda(step: int, *, warmup: int, total: int) -> float:
    """先线性 warmup，再做 cosine decay。"""

    if warmup > 0 and step < warmup:
        return max(step, 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _total_optimizer_steps(optimizer_config: Mapping[str, Any]) -> int:
    """Convert the configured epochs into an optimizer-step budget."""

    epochs = float(optimizer_config["epochs"])
    iterations_per_epoch = int(optimizer_config["iterations_per_epoch"])
    return math.ceil(epochs * iterations_per_epoch)


def _infinite_batches(
    loader: DataLoader,
    *,
    consumed_batches: int,
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    """无限遍历 DataLoader，并从 checkpoint 对应的 epoch/offset 接着取。"""

    sampler = loader.sampler
    full_length = int(getattr(sampler, "full_length", len(sampler)))
    batches_per_epoch = full_length // batch_size
    if batches_per_epoch <= 0:
        raise ValueError("training sampler must provide at least one full batch")
    epoch, batch_offset = divmod(consumed_batches, batches_per_epoch)
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if hasattr(sampler, "set_start_index"):
            sampler.set_start_index(batch_offset * batch_size)
        yield from loader
        epoch += 1
        batch_offset = 0


def _distributed_mean(value: float, device: torch.device) -> float:
    """把每个 rank 的标量平均成全局指标。"""

    result = torch.tensor(value, device=device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result.div_(torch.distributed.get_world_size())
    return result.item()


@torch.no_grad()
def _parameter_norm(
    parameters: list,
    *,
    device: torch.device,
    fsdp_sharded: bool,
) -> float:
    """计算全局参数 L2 norm；FSDP 下先汇总各 rank 的参数碎片。"""

    squared = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in parameters:
        squared += parameter.detach().double().square().sum()
    if fsdp_sharded:
        torch.distributed.all_reduce(squared, op=torch.distributed.ReduceOp.SUM)
    return squared.sqrt().item()


def _gather_equal_shape(tensor: Tensor) -> Tensor:
    """收集各 rank 同形状 tensor，供全局 collapse 诊断使用。"""

    if not torch.distributed.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def _grouped_online_target_gap(output: Mapping[str, Any]) -> float:
    """比较每套 mask 的可见 online token 与相同位置的完整 EMA target。"""

    target = output["target"].flatten(1, 2)
    gaps = []
    for online, indices in zip(output["online"], output["context_indices"]):
        target_values = target.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, target.shape[-1]),
        )
        gaps.append(online_target_gap(online, target_values))
    return sum(gaps) / len(gaps)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    batches: int,
    progress_label: str = "validation",
) -> Tuple[Dict[str, float], Optional[list]]:
    """计算固定 mask 的 validation loss、collapse 指标和 W&B 诊断图。"""

    model.eval()
    losses = []
    targets = []
    gaps = []
    visuals = None
    validation_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    try:
        total_batches = min(batches, len(loader))
    except TypeError:
        total_batches = batches
    progress_every = max(1, total_batches // 10)
    if validation_rank == 0:
        print(
            f"[{progress_label}] {_progress_bar(0, total_batches)} 0/{total_batches}",
            flush=True,
        )
    mask_generator = torch.Generator().manual_seed(9100 + validation_rank)
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        frames = batch["frames"].to(device, non_blocking=True)
        with _autocast_context(device, precision):
            output = model(frames, mask_generator=mask_generator)
        losses.append(output["loss"].detach().float())
        targets.append(output["target"].mean(dim=2).detach())
        gaps.append(_grouped_online_target_gap(output))
        if batch_index == 0:
            visuals = visual_pretraining_images(
                frames,
                output["target_mask"],
                output["prediction"],
                output["target"],
                prediction_indices=output["prediction_indices"],
                grid_size=_unwrapped(model).config.encoder.grid_size,
            )
        completed_batches = batch_index + 1
        if validation_rank == 0 and (
            completed_batches % progress_every == 0
            or completed_batches == total_batches
        ):
            print(
                f"[{progress_label}] "
                f"{_progress_bar(completed_batches, total_batches)} "
                f"{completed_batches}/{total_batches}",
                flush=True,
            )
    if not losses:
        raise ValueError("validation loader produced no batches")
    local_targets = torch.cat(targets)
    global_targets = _gather_equal_shape(local_targets)
    metrics = {
        "validation/loss": _distributed_mean(
            torch.stack(losses).mean().item(), device
        )
    }
    metrics.update(collapse_metrics(global_targets, prefix="validation/latent"))
    metrics["validation/online_target_cosine_gap"] = _distributed_mean(
        sum(gaps) / len(gaps), device
    )
    model.train()
    return metrics, visuals


def _prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    """只删除当前 output_dir 中超过保留数量的常规 checkpoint。"""

    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("checkpoint-*.pt"))
    for old_checkpoint in checkpoints[:-keep_last]:
        old_checkpoint.unlink()


def _make_loaders(
    config: Mapping[str, Any],
    *,
    seed: int,
    synthetic: bool,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, str]:
    """构建真实 canonical 或 synthetic smoke-test DataLoader。"""

    data = config["data"]
    model = config["model"]
    generator = torch.Generator().manual_seed(seed)
    if synthetic:
        train_dataset: Dataset = SyntheticVideoDataset(
            max(int(data["batch_size"]) * 4, 8),
            int(data["clip_frames"]),
            int(model["image_height"]),
            int(model["image_width"]),
            seed,
        )
        validation_dataset: Dataset = SyntheticVideoDataset(
            max(int(data["batch_size"]), 2),
            int(data["clip_frames"]),
            int(model["image_height"]),
            int(model["image_width"]),
            seed + 10000,
        )
        manifest_hash = "synthetic-m1-smoke-v1"
    else:
        root = Path(data["root"])
        dataset_manifest = DatasetManifest.read(root / "dataset_manifest.json")
        manifest_hash = dataset_manifest.content_hash
        train_dataset = CanonicalVisualDataset(
            root,
            split=data.get("train_split", "train"),
            clip_frames=int(data["clip_frames"]),
            sample_fps=int(data["sample_fps"]),
            seed=seed,
        )
        validation_dataset = CanonicalVisualDataset(
            root,
            split=data.get("validation_split", "validation"),
            clip_frames=int(data["clip_frames"]),
            sample_fps=int(data["sample_fps"]),
            seed=seed + 10000,
            clips_per_video=int(config["validation"].get("clips_per_video", 1)),
        )

    train_sampler = None
    validation_sampler = None
    shuffle = True
    if world_size > 1:
        if data.get("source_balanced", False) and isinstance(
            train_dataset, CanonicalVisualDataset
        ):
            train_sampler = DistributedSourceBalancedSampler(
                train_dataset, rank=rank, world_size=world_size, seed=seed
            )
        else:
            train_sampler = DistributedResumableSampler(
                len(train_dataset),
                rank=rank,
                world_size=world_size,
                seed=seed,
                seed_clips=isinstance(train_dataset, CanonicalVisualDataset),
            )
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle = False
    elif data.get("source_balanced", False) and isinstance(train_dataset, CanonicalVisualDataset):
        train_sampler = ResumableSampler(
            len(train_dataset),
            seed=seed,
            weights=source_balanced_weights(train_dataset),
            seed_clips=True,
        )
        shuffle = False
    else:
        train_sampler = ResumableSampler(
            len(train_dataset),
            seed=seed,
            seed_clips=isinstance(train_dataset, CanonicalVisualDataset),
        )
        shuffle = False

    common = {
        "batch_size": int(data["batch_size"]),
        "num_workers": int(data.get("workers", 0)),
        "pin_memory": config.get("device") == "cuda",
        "persistent_workers": int(data.get("workers", 0)) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=shuffle,
        generator=generator,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, manifest_hash


def train(config: Mapping[str, Any], *, synthetic: bool = False) -> Path:
    """运行完整 M1 训练，并返回最后一个本地 checkpoint 路径。"""

    validate_pretrain_config(config)
    resolved = resolved_copy(config)
    seed = int(config.get("seed", 2026))
    strategy = config["distributed"].get("strategy", "none")
    rank, _, world_size, device = _distributed_context(strategy, str(config["device"]))
    _seed_everything(seed + rank)
    train_loader, validation_loader, manifest_hash = _make_loaders(
        config, seed=seed, synthetic=synthetic, rank=rank, world_size=world_size
    )

    base_model = build_visual_jepa(config)
    total_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad
    )
    deploy_parameters = sum(
        parameter.numel() for parameter in base_model.target_encoder.parameters()
    )
    model = _wrap_distributed(
        base_model, strategy, device, world_size, str(config["precision"])
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    per_micro_batch = int(config["data"]["batch_size"]) * world_size
    effective_batch = int(optimizer_config["effective_batch_size"])
    if effective_batch % per_micro_batch:
        raise ValueError("effective_batch_size must be divisible by batch_size * world_size")
    accumulation_steps = effective_batch // per_micro_batch
    total_steps = _total_optimizer_steps(optimizer_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_lambda(
            step,
            warmup=int(optimizer_config["warmup_steps"]),
            total=total_steps,
        ),
    )
    use_scaler = config["precision"] == "fp16" and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # 兼容较早的受支持 PyTorch。
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    output_dir = Path(config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(resolved, handle, indent=2, sort_keys=True)
            handle.write("\n")

    resume_path = config["checkpoint"].get("resume")
    optimizer_step = 0
    collapse_bad_validations = 0
    resume_payload = None
    if resume_path:
        # 先恢复所有状态，再创建 logger，才能继续使用 checkpoint 内的 W&B run ID。
        resume_payload = _load_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
            manifest_hash=manifest_hash,
            strategy=strategy,
            rank=rank,
            world_size=world_size,
        )
        if _training_signature(resume_payload["provenance"]["config"]) != _training_signature(
            config
        ):
            raise ValueError(
                "resume changes model/data/optimizer semantics; start a new run with "
                "parent_checkpoint instead"
            )
        optimizer_step = int(resume_payload["optimizer_step"])
        collapse_bad_validations = int(
            resume_payload.get("extra", {}).get("collapse_bad_validations", 0)
        )

    previous_provenance = resume_payload["provenance"] if resume_payload else {}
    logger = TrainingLogger(
        output_dir,
        config=resolved,
        wandb_config=config["wandb"],
        run_id=previous_provenance.get("wandb_run_id"),
        rank=rank,
    )
    provenance = CheckpointProvenance(
        git_commit=_git_commit(),
        config=resolved,
        seed=seed,
        manifest_hash=manifest_hash,
        parent_checkpoint=str(resume_path) if resume_path else None,
        wandb_entity=config["wandb"].get("entity"),
        wandb_project=config["wandb"].get("project"),
        wandb_run_id=logger.run_id,
        wandb_run_name=logger.run_name,
    )
    if rank == 0:
        logger.log(
            {
                "model/total_parameters": total_parameters,
                "model/trainable_parameters": trainable_parameters,
                "model/deploy_encoder_parameters": deploy_parameters,
                "system/world_size": world_size,
                "system/effective_batch_size": effective_batch,
                "system/accumulation_steps": accumulation_steps,
                "system/total_steps": total_steps,
            },
            step=optimizer_step,
        )
    if optimizer_step == 0:
        # 保存随机初始化基线，后面可以直接判断训练是否真的带来改进。
        baseline_metrics, _ = validate(
            model,
            validation_loader,
            device=device,
            precision=str(config["precision"]),
            batches=int(config["validation"]["batches"]),
            progress_label="baseline",
        )
        logger.log(
            {
                "baseline/validation_loss": baseline_metrics["validation/loss"],
                "baseline/latent_effective_rank": baseline_metrics[
                    "validation/latent/effective_rank"
                ],
            },
            step=0,
        )

    batches = _infinite_batches(
        train_loader,
        consumed_batches=optimizer_step * accumulation_steps,
        batch_size=int(config["data"]["batch_size"]),
    )
    model.train()
    last_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
    training_started = time.perf_counter()
    starting_optimizer_step = optimizer_step
    if rank == 0:
        print(
            f"[train] {_progress_bar(optimizer_step, total_steps)} "
            f"{optimizer_step}/{total_steps}",
            flush=True,
        )
    try:
        while optimizer_step < total_steps:
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            output = None
            current_sample_ids = []
            data_loading_seconds = 0.0
            for micro_step in range(accumulation_steps):
                data_started = time.perf_counter()
                batch = next(batches)
                data_loading_seconds += time.perf_counter() - data_started
                current_sample_ids.extend(str(value) for value in batch.get("sample_id", []))
                frames = batch["frames"].to(device, non_blocking=True)
                synchronized = micro_step == accumulation_steps - 1
                # 除最后一个 micro-step 外不做梯度同步，减少多卡通信次数。
                sync_context = (
                    nullcontext()
                    if synchronized or not hasattr(model, "no_sync")
                    else model.no_sync()
                )
                with sync_context, _autocast_context(device, str(config["precision"])):
                    output = model(
                        frames,
                        online_frames=_augment_clip(frames, config["augmentation"]),
                    )
                    loss = output["loss"] / accumulation_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated_loss += output["loss"].detach().float().item()

            if use_scaler:
                scaler.unscale_(optimizer)
            if world_size > 1 and strategy == "fsdp":
                gradient_norm = model.clip_grad_norm_(float(optimizer_config["gradient_clip"]))
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, float(optimizer_config["gradient_clip"])
                )
            # 任意 rank 出现 NaN/Inf 时，所有 rank 一起保存 failure checkpoint 并退出。
            finite_step = torch.tensor(
                int(math.isfinite(accumulated_loss) and bool(torch.isfinite(gradient_norm))),
                device=device,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(finite_step, op=torch.distributed.ReduceOp.MIN)
            if not bool(finite_step.item()):
                failure_checkpoint = output_dir / f"nonfinite-{optimizer_step:08d}.pt"
                _save_checkpoint(
                    failure_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    optimizer_step=optimizer_step,
                    provenance=provenance,
                    strategy=strategy,
                    rank=rank,
                    world_size=world_size,
                    training_state={
                        "collapse_bad_validations": collapse_bad_validations,
                        "failure_sample_ids": current_sample_ids,
                    },
                )
                raise RuntimeError(
                    f"non-finite loss/gradient at optimizer step {optimizer_step}; "
                    f"saved {failure_checkpoint}"
                )
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer_step += 1
            scheduler.step()
            momentum = cosine_ema_momentum(
                optimizer_step,
                total_steps,
                float(optimizer_config["ema_start"]),
                float(optimizer_config.get("ema_end", 1.0)),
            )
            # EMA 必须放在 optimizer.step 之后，并且全局 step 只增加一次。
            _update_target(model, strategy, world_size, momentum)

            log_every = int(config["wandb"].get("log_every_steps", 10))
            diagnostics_every = int(config["wandb"].get("diagnostics_every_steps", 200))
            if optimizer_step % log_every == 0 and output is not None:
                elapsed = time.perf_counter() - step_started
                metrics: Dict[str, float] = {
                    "train/loss": accumulated_loss / accumulation_steps,
                    "train/epoch": optimizer_step
                    / int(optimizer_config["iterations_per_epoch"]),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/ema_momentum": momentum,
                    "train/gradient_norm": float(gradient_norm),
                    "system/step_seconds": elapsed,
                    "system/data_loading_seconds": data_loading_seconds,
                    "system/clips_per_second": effective_batch / max(elapsed, 1e-9),
                    "system/patch_tokens_per_second": (
                        effective_batch
                        * _unwrapped(model).config.encoder.token_count
                        / max(elapsed, 1e-9)
                    ),
                    "mask/ratio": output["target_mask"].float().mean().item(),
                    "mask/per_tubelet_ratio_std": output["target_mask"]
                    .float()
                    .mean(dim=-1)
                    .std(unbiased=False)
                    .item(),
                    "train/parameter_norm": _parameter_norm(
                        trainable,
                        device=device,
                        fsdp_sharded=world_size > 1 and strategy == "fsdp",
                    ),
                }
                for group_index, group_mask in enumerate(output["target_mask"]):
                    metrics[f"mask/group_{group_index}_ratio"] = (
                        group_mask.float().mean().item()
                    )
                if output["target_mask"].shape[2] > 1:
                    adjacent_intersection = (
                        output["target_mask"][:, :, 1:]
                        & output["target_mask"][:, :, :-1]
                    ).float().sum()
                    adjacent_union = (
                        output["target_mask"][:, :, 1:]
                        | output["target_mask"][:, :, :-1]
                    ).float().sum().clamp_min(1.0)
                    metrics["mask/adjacent_iou"] = (
                        adjacent_intersection / adjacent_union
                    ).item()
                if use_scaler:
                    metrics["train/amp_loss_scale"] = float(scaler.get_scale())
                sources = list(batch.get("source", []))
                if sources:
                    for source in ("vpt", "minerl", "synthetic"):
                        metrics[f"data/source_{source}_fraction"] = _distributed_mean(
                            sources.count(source) / len(sources), device
                        )
                if device.type == "cuda":
                    metrics["system/max_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                    metrics["system/max_reserved_memory_gib"] = (
                        torch.cuda.max_memory_reserved(device) / 2**30
                    )
                if optimizer_step % diagnostics_every == 0:
                    diagnostic_targets = _gather_equal_shape(
                        output["target"].mean(dim=2).detach()
                    )
                    metrics.update(
                        collapse_metrics(diagnostic_targets, prefix="train/latent")
                    )
                    metrics["train/online_target_cosine_gap"] = _distributed_mean(
                        _grouped_online_target_gap(output),
                        device,
                    )
                logger.log(metrics, step=optimizer_step)
                if rank == 0:
                    completed_this_run = optimizer_step - starting_optimizer_step
                    average_step_seconds = (
                        (time.perf_counter() - training_started) / completed_this_run
                    )
                    eta_seconds = (total_steps - optimizer_step) * average_step_seconds
                    percentage = 100.0 * optimizer_step / total_steps
                    print(
                        f"[train] {_progress_bar(optimizer_step, total_steps)} "
                        f"{optimizer_step}/{total_steps} ({percentage:5.1f}%) "
                        f"loss={metrics['train/loss']:.5f} "
                        f"step={metrics['system/step_seconds']:.2f}s "
                        f"clips/s={metrics['system/clips_per_second']:.2f} "
                        f"eta={_format_duration(eta_seconds)}",
                        flush=True,
                    )

            if optimizer_step % int(config["validation"]["every_steps"]) == 0:
                metrics, visuals = validate(
                    model,
                    validation_loader,
                    device=device,
                    precision=str(config["precision"]),
                    batches=int(config["validation"]["batches"]),
                    progress_label=f"validation step {optimizer_step}",
                )
                collapse_config = config["collapse"]
                alerts = (
                    find_collapse_alerts(
                        metrics,
                        CollapseThresholds(
                            minimum_average_std=float(
                                collapse_config["minimum_average_std"]
                            ),
                            minimum_effective_rank=float(
                                collapse_config["minimum_effective_rank"]
                            ),
                            maximum_pairwise_cosine=float(
                                collapse_config["maximum_pairwise_cosine"]
                            ),
                        ),
                        prefix="validation/latent",
                    )
                    if config["validation"].get("collapse_check", True)
                    else ()
                )
                collapse_bad_validations = collapse_bad_validations + 1 if alerts else 0
                metrics["validation/collapse_bad_validations"] = collapse_bad_validations
                logger.log(metrics, step=optimizer_step)
                if visuals is not None:
                    logger.log_images("validation/mask_and_error", visuals, step=optimizer_step)
                if collapse_bad_validations >= int(
                    collapse_config.get("patience_validations", 3)
                ):
                    failure_checkpoint = output_dir / f"collapse-{optimizer_step:08d}.pt"
                    _save_checkpoint(
                        failure_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if use_scaler else None,
                        optimizer_step=optimizer_step,
                        provenance=provenance,
                        strategy=strategy,
                        rank=rank,
                        world_size=world_size,
                        training_state={
                            "collapse_bad_validations": collapse_bad_validations
                        },
                    )
                    raise RuntimeError(f"latent collapse detected: {', '.join(alerts)}")

            if optimizer_step % int(config["checkpoint"]["every_steps"]) == 0:
                last_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
                _save_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    optimizer_step=optimizer_step,
                    provenance=provenance,
                    strategy=strategy,
                    rank=rank,
                    world_size=world_size,
                    training_state={"collapse_bad_validations": collapse_bad_validations},
                )
                if rank == 0:
                    _prune_checkpoints(
                        output_dir, int(config["checkpoint"].get("keep_last", 0))
                    )
        final_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
        checkpoint_exists = final_checkpoint.exists() if rank == 0 else False
        if world_size > 1:
            state = [checkpoint_exists]
            torch.distributed.broadcast_object_list(state, src=0)
            checkpoint_exists = bool(state[0])
        if not checkpoint_exists:
            _save_checkpoint(
                final_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                optimizer_step=optimizer_step,
                provenance=provenance,
                strategy=strategy,
                rank=rank,
                world_size=world_size,
                training_state={"collapse_bad_validations": collapse_bad_validations},
            )
        last_checkpoint = final_checkpoint
    finally:
        logger.finish()
        if world_size > 1:
            torch.distributed.destroy_process_group()
    return last_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_visual.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use deterministic synthetic clips for a smoke test; never for a real checkpoint",
    )
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.resume is not None:
        config["checkpoint"]["resume"] = str(args.resume)
    if args.wandb_mode is not None:
        config["wandb"]["mode"] = args.wandb_mode
        config["wandb"]["enabled"] = args.wandb_mode != "disabled"
    checkpoint = train(config, synthetic=args.synthetic)
    print(checkpoint)


if __name__ == "__main__":
    main()
