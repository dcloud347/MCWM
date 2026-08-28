"""M2 action-conditioned world model 训练流程。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mcwm.data.manifest import DatasetManifest
from mcwm.data.visual_dataset import DistributedResumableSampler, ResumableSampler
from mcwm.data.world_model_dataset import (
    WorldModelDataset,
    collate_world_model_samples,
)
from mcwm.diagnostics.world_model import (
    action_sensitivity_from_predictions,
    spatial_error_images,
    world_model_prediction_metrics,
)
from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
from mcwm.models.visual_encoder import VisualEncoderConfig
from mcwm.models.world_model import WorldModel
from .checkpoint import (
    CheckpointProvenance,
    capture_rng_state,
    checkpoint_sha256,
    load_frozen_m1_encoder,
    load_world_model_checkpoint,
    restore_rng_state,
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


def _distributed_context(
    strategy: str,
    requested_device: str,
) -> Tuple[int, int, int, torch.device]:
    """初始化 FSDP 通信，并把每个 torchrun 进程绑定到自己的 GPU。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA, but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        if strategy != "fsdp":
            raise ValueError("WORLD_SIZE > 1 requires distributed.strategy=fsdp")
        torch.distributed.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo"
        )
    elif strategy == "fsdp" and rank == 0:
        print("distributed.strategy=fsdp requested with one process; using one device")
    return rank, local_rank, world_size, device


def _wrap_distributed(
    model: WorldModel,
    *,
    device: torch.device,
    world_size: int,
    precision: str,
) -> nn.Module:
    """单进程直接返回模型，多进程时按大模块切分成 FSDP shards。"""

    model.to(device)
    if world_size == 1:
        return model
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    parameter_dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16
        if precision == "fp16"
        else torch.float32
    )
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
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=parameter_dtype,
            reduce_dtype=parameter_dtype,
            buffer_dtype=parameter_dtype,
        ),
    )


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
    value.setdefault("distributed", {"strategy": "none"})
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
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, str]:
    data = config["data"]
    seed = int(config.get("seed", 0))
    batch_size = int(data["batch_size"])
    if synthetic:
        visual = config["model"]["synthetic_visual_encoder"]
        length = max(
            batch_size * world_size * int(config["optimizer"]["iterations_per_epoch"]),
            batch_size * world_size,
            4,
        )
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

    if world_size > 1:
        sampler = DistributedResumableSampler(
            len(train_dataset),
            rank=rank,
            world_size=world_size,
            seed=seed,
            seed_clips=True,
        )
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        sampler = ResumableSampler(
            len(train_dataset),
            seed=seed,
            seed_clips=True,
        )
        validation_sampler = None
    workers = int(data.get("workers", 0))
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": config.get("device") == "cuda",
        "persistent_workers": workers > 0,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
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


def _action_variant_batches(batch: Mapping[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
    """构造 FSDP-safe 动作诊断输入；每组都通过完整 model.forward。"""

    def copied() -> Dict[str, Tensor]:
        return {name: value.clone() for name, value in batch.items()}

    action_names = tuple(name for name in MODEL_INPUT_NAMES if name != "frames")
    shuffled = copied()
    shuffle_dim = 0 if batch["frames"].shape[0] > 1 else 1
    for name in action_names:
        shuffled[name] = shuffled[name].roll(1, dims=shuffle_dim)

    noop = copied()
    for name in action_names:
        if name != "valid_mask":
            noop[name].zero_()

    camera = copied()
    camera["camera"].neg_()

    swapped = copied()
    attack = swapped["interaction"][..., 0].clone()
    swapped["interaction"][..., 0] = swapped["interaction"][..., 1]
    swapped["interaction"][..., 1] = attack
    return {
        "shuffled": shuffled,
        "noop": noop,
        "camera": camera,
        "swapped": swapped,
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


def _accumulation_steps(config: Mapping[str, Any], world_size: int) -> int:
    """按全局有效 batch 计算每张卡需要执行的梯度累积次数。"""

    per_gpu_batch = int(config["data"]["batch_size"])
    global_micro_batch = per_gpu_batch * int(world_size)
    effective_batch = int(config["optimizer"]["effective_batch_size"])
    if effective_batch % global_micro_batch:
        raise ValueError(
            "effective_batch_size must be divisible by data.batch_size * world_size"
        )
    return effective_batch // global_micro_batch


def _distributed_mean(value: float, device: torch.device) -> float:
    """把各 rank 的标量平均，供 rank 0 记录全局训练指标。"""

    result = torch.tensor(value, device=device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result.div_(torch.distributed.get_world_size())
    return float(result.item())


def _distributed_max(value: float, device: torch.device) -> float:
    """返回所有 rank 中的最大值，吞吐统计以最慢进程为准。"""

    result = torch.tensor(value, device=device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return float(result.item())


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    batches: int,
    spatial_grid: Tuple[int, int],
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
            variant_predictions = (
                {
                    name: model(**variant)["teacher_forced_predictions"]
                    for name, variant in _action_variant_batches(batch).items()
                }
                if batch_index == 0
                else {}
            )
        metrics = world_model_prediction_metrics(output)
        if batch_index == 0:
            metrics.update(
                action_sensitivity_from_predictions(
                    output["teacher_forced_predictions"],
                    variant_predictions["shuffled"],
                    variant_predictions["noop"],
                    variant_predictions["camera"],
                    variant_predictions["swapped"],
                    output["targets"],
                    batch,
                )
            )
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                images = spatial_error_images(
                    output["teacher_forced_predictions"],
                    output["targets"],
                    spatial_grid=spatial_grid,
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
    if torch.distributed.is_initialized():
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, metrics)
        names = set().union(*(rank_metrics.keys() for rank_metrics in gathered))
        metrics = {
            name: sum(rank_metrics[name] for rank_metrics in gathered if name in rank_metrics)
            / sum(name in rank_metrics for rank_metrics in gathered)
            for name in names
        }
    return metrics, images


def _prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    paths = sorted(output_dir.glob("checkpoint-*.pt"))
    for path in paths[: max(0, len(paths) - keep_last)]:
        path.unlink()


def _save_distributed_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    optimizer_step: int,
    provenance: CheckpointProvenance,
    m1_parent_path: str,
    m1_parent_hash: str,
    sampler_epoch: int,
    batch_offset: int,
    resume_path: Any,
    rank: int,
    world_size: int,
) -> None:
    """收集各 rank 的 RNG 状态，并只让 rank 0 原子保存 checkpoint。"""

    local_rng = capture_rng_state()
    if world_size > 1:
        rng_by_rank = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank = [local_rng]
    model_state = None
    optimizer_state = None
    if world_size > 1:
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        save_world_model_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            optimizer_step=optimizer_step,
            provenance=provenance,
            m1_parent_path=m1_parent_path,
            m1_parent_sha256=m1_parent_hash,
            sampler_epoch=sampler_epoch,
            batch_offset=batch_offset,
            resume_checkpoint=str(resume_path) if resume_path else None,
            rng_state=local_rng,
            rng_by_rank=rng_by_rank,
            world_size=world_size,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
        )
    if world_size > 1:
        torch.distributed.barrier()


def _load_distributed_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    manifest_hash: str,
    m1_parent_path: str,
    m1_parent_hash: str,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """加载完整 M2 checkpoint，并把 optimizer state 重新切分给 FSDP ranks。"""

    if world_size > 1:
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            payload = load_world_model_checkpoint(
                path,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                expected_manifest_hash=manifest_hash,
                expected_m1_parent_path=m1_parent_path,
                expected_m1_parent_sha256=m1_parent_hash,
                restore_rng=False,
            )
            optimizer_state = FSDP.optim_state_dict_to_load(
                model,
                optimizer,
                payload["optimizer"],
            )
        optimizer.load_state_dict(optimizer_state)
        if payload["scheduler"] is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            if payload["scaler"] is None:
                raise ValueError("checkpoint does not contain scaler state")
            scaler.load_state_dict(payload["scaler"])
    else:
        payload = load_world_model_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_manifest_hash=manifest_hash,
            expected_m1_parent_path=m1_parent_path,
            expected_m1_parent_sha256=m1_parent_hash,
            restore_rng=False,
        )

    saved_world_size = int(payload["extra"].get("world_size", 1))
    if saved_world_size != world_size:
        raise ValueError(
            "M2 FSDP resume requires the same world_size as the saved checkpoint"
        )
    rng_by_rank = payload["extra"].get("rng_by_rank")
    if isinstance(rng_by_rank, list) and len(rng_by_rank) == world_size:
        restore_rng_state(rng_by_rank[rank])
    else:
        restore_rng_state(payload["rng"])
    return payload


def train(config: Mapping[str, Any], *, synthetic: bool = False) -> Path:
    """训练 M2，并返回最终 checkpoint 路径。"""

    validate_world_model_config(config)
    seed = int(config.get("seed", 0))
    strategy = str(config.get("distributed", {}).get("strategy", "none"))
    rank, _, world_size, device = _distributed_context(
        strategy,
        str(config["device"]),
    )
    _seed_everything(seed + rank)

    train_loader, validation_loader, manifest_hash = _make_loaders(
        config,
        synthetic=synthetic,
        rank=rank,
        world_size=world_size,
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

    base_model = build_world_model(config, visual_encoder)
    model = _wrap_distributed(
        base_model,
        device=device,
        world_size=world_size,
        precision=str(config["precision"]),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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

    effective_batch = int(optimizer_config["effective_batch_size"])
    accumulation_steps = _accumulation_steps(config, world_size)
    optimizer_step = 0
    sampler_epoch = 0
    batch_offset = 0
    resume_path = config["checkpoint"].get("resume")
    resume_payload = None
    if resume_path:
        resume_payload = _load_distributed_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
            manifest_hash=manifest_hash,
            m1_parent_path=m1_parent_path,
            m1_parent_hash=m1_parent_hash,
            rank=rank,
            world_size=world_size,
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
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()
    logger = TrainingLogger(
        output_dir,
        config=resolved_copy(config),
        wandb_config=config["wandb"],
        run_id=(resume_payload or {}).get("provenance", {}).get("wandb_run_id"),
        rank=rank,
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
            for micro_step in range(accumulation_steps):
                raw_batch, sampler_epoch, batch_offset = next(batches)
                batch = _to_device(raw_batch, device)
                synchronized = micro_step == accumulation_steps - 1
                sync_context = (
                    nullcontext()
                    if synchronized or not hasattr(model, "no_sync")
                    else model.no_sync()
                )
                with sync_context, _autocast_context(device, str(config["precision"])):
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
            if world_size > 1:
                gradient_norm = model.clip_grad_norm_(
                    float(optimizer_config.get("gradient_clip", 1.0))
                )
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(optimizer_config.get("gradient_clip", 1.0)),
                )
            finite_step = torch.tensor(
                int(torch.isfinite(gradient_norm)),
                device=device,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    finite_step,
                    op=torch.distributed.ReduceOp.MIN,
                )
            if not bool(finite_step.item()):
                raise RuntimeError("non-finite M2 gradient")
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer_step += 1

            if optimizer_step % int(config["wandb"].get("log_every_steps", 10)) == 0:
                step_seconds = _distributed_max(
                    time.perf_counter() - step_started,
                    device,
                )
                train_loss = _distributed_mean(
                    accumulated["loss"] / accumulation_steps,
                    device,
                )
                teacher_loss = _distributed_mean(
                    accumulated["teacher"] / accumulation_steps,
                    device,
                )
                autoregressive_loss = _distributed_mean(
                    accumulated["autoregressive"] / accumulation_steps,
                    device,
                )
                logger.log(
                    {
                        "train/loss": train_loss,
                        "train/teacher_forced_loss": teacher_loss,
                        "train/autoregressive_loss": autoregressive_loss,
                        "train/gradient_norm": float(gradient_norm),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "system/step_seconds": step_seconds,
                        "system/samples_per_second": effective_batch
                        / max(step_seconds, 1e-9),
                        "system/world_size": world_size,
                        "system/accumulation_steps": accumulation_steps,
                        "system/max_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                    },
                    step=optimizer_step,
                )
            if optimizer_step % int(config["validation"]["every_steps"]) == 0:
                if world_size > 1:
                    torch.distributed.barrier()
                validation_metrics, validation_images = validate(
                    model,
                    validation_loader,
                    device=device,
                    precision=str(config["precision"]),
                    batches=int(config["validation"]["batches"]),
                    spatial_grid=tuple(
                        int(value)
                        for value in config["model"]["predictor"]["spatial_grid"]
                    ),
                )
                if rank == 0:
                    logger.log(
                        validation_metrics,
                        step=optimizer_step,
                    )
                    logger.log_images(
                        "validation/spatial_token_error",
                        validation_images,
                        step=optimizer_step,
                    )
                if world_size > 1:
                    torch.distributed.barrier()
            if optimizer_step % int(config["checkpoint"]["every_steps"]) == 0:
                last_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
                _save_distributed_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    optimizer_step=optimizer_step,
                    provenance=provenance,
                    m1_parent_path=m1_parent_path,
                    m1_parent_hash=m1_parent_hash,
                    sampler_epoch=sampler_epoch,
                    batch_offset=batch_offset,
                    resume_path=resume_path,
                    rank=rank,
                    world_size=world_size,
                )
                if rank == 0:
                    _prune_checkpoints(
                        output_dir,
                        int(config["checkpoint"].get("keep_last", 3)),
                    )

        final_checkpoint = output_dir / f"checkpoint-{optimizer_step:08d}.pt"
        checkpoint_exists = final_checkpoint.exists() if rank == 0 else False
        if world_size > 1:
            state = [checkpoint_exists]
            torch.distributed.broadcast_object_list(state, src=0)
            checkpoint_exists = bool(state[0])
        if not checkpoint_exists:
            _save_distributed_checkpoint(
                final_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                optimizer_step=optimizer_step,
                provenance=provenance,
                m1_parent_path=m1_parent_path,
                m1_parent_hash=m1_parent_hash,
                sampler_epoch=sampler_epoch,
                batch_offset=batch_offset,
                resume_path=resume_path,
                rank=rank,
                world_size=world_size,
            )
        return final_checkpoint
    finally:
        logger.finish()
        if world_size > 1:
            torch.distributed.destroy_process_group()


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
    if int(os.environ.get("RANK", "0")) == 0:
        print(checkpoint)


if __name__ == "__main__":
    main()
