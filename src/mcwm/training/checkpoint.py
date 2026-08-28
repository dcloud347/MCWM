"""安全地保存、读取和检查训练 checkpoint。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
from .config import visual_encoder_config


@dataclass(frozen=True)
class CheckpointProvenance:
    """记录权重来源，防止误用外部预训练权重。"""

    git_commit: str
    config: Mapping[str, Any]
    seed: int
    manifest_hash: str
    parent_checkpoint: Optional[str]
    wandb_entity: Optional[str]
    wandb_project: Optional[str]
    wandb_run_id: Optional[str]
    wandb_run_name: Optional[str]
    external_pretrained: bool = False

    def __post_init__(self) -> None:
        if self.external_pretrained:
            raise ValueError("MCWM checkpoints cannot contain external pretrained weights")
        if not self.git_commit or not self.manifest_hash:
            raise ValueError("git_commit and manifest_hash are required provenance")


def capture_rng_state() -> Dict[str, Any]:
    """保存所有随机数生成器的状态，供断点续训使用。"""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """恢复随机状态，让断点续训和不中断训练保持一致。"""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    optimizer_step: int,
    provenance: CheckpointProvenance,
    extra: Optional[Mapping[str, Any]] = None,
    model_state_dict: Optional[Mapping[str, Any]] = None,
    optimizer_state_dict: Optional[Mapping[str, Any]] = None,
    rng_state: Optional[Mapping[str, Any]] = None,
) -> None:
    """先写临时文件，全部成功后再替换正式 checkpoint。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "model": dict(model_state_dict) if model_state_dict is not None else model.state_dict(),
        "optimizer": (
            dict(optimizer_state_dict)
            if optimizer_state_dict is not None
            else optimizer.state_dict()
        ),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "optimizer_step": int(optimizer_step),
        "provenance": asdict(provenance),
        "rng": dict(rng_state) if rng_state is not None else capture_rng_state(),
        "extra": dict(extra or {}),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def read_checkpoint(
    path: Path,
    *,
    expected_manifest_hash: Optional[str] = None,
    memory_map: bool = False,
) -> Dict[str, Any]:
    """读取 checkpoint，并先检查格式版本和权重来源。"""

    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
        mmap=memory_map,
    )
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    provenance = payload.get("provenance")
    required = {
        "git_commit",
        "config",
        "seed",
        "manifest_hash",
        "parent_checkpoint",
        "wandb_entity",
        "wandb_project",
        "wandb_run_id",
        "wandb_run_name",
        "external_pretrained",
    }
    if not isinstance(provenance, dict) or not required.issubset(provenance):
        raise ValueError("checkpoint has incomplete provenance")
    if provenance["external_pretrained"] is not False:
        raise ValueError("refusing a checkpoint marked as externally pretrained")
    if expected_manifest_hash is not None and provenance["manifest_hash"] != expected_manifest_hash:
        raise ValueError("checkpoint data manifest hash differs from the current dataset")

    return payload


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    expected_manifest_hash: Optional[str] = None,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """恢复模型、优化器、调度器和随机状态。"""

    payload = read_checkpoint(path, expected_manifest_hash=expected_manifest_hash)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint does not contain scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload


def export_ema_encoder(checkpoint_path: Path, destination: Path) -> Path:
    """只导出 EMA encoder，供下一阶段训练或部署使用。"""

    try:
        from safetensors.torch import save_file  # type: ignore
    except ImportError as exc:
        raise RuntimeError("encoder export requires `pip install mcwm[train]`") from exc
    payload = read_checkpoint(checkpoint_path)
    # 最终只需要更稳定的 EMA encoder，不导出 online encoder 和 predictor。
    prefix = "target_encoder."
    encoder_state = {
        name[len(prefix) :]: value.detach().cpu().contiguous()
        for name, value in payload["model"].items()
        if name.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError("checkpoint does not contain an EMA target encoder")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = payload["provenance"]
    save_file(
        encoder_state,
        str(destination),
        metadata={
            "format": "mcwm-m1-ema-visual-encoder",
            "git_commit": str(provenance["git_commit"]),
            "manifest_hash": str(provenance["manifest_hash"]),
            "external_pretrained": "false",
        },
    )
    return destination


def load_frozen_m1_encoder(
    checkpoint_path: Path,
    *,
    expected_manifest_hash: Optional[str] = None,
) -> tuple[FrozenVisualEncoder, Dict[str, Any]]:
    """严格加载 M1 EMA target encoder，供 M2 冻结使用。"""

    payload = read_checkpoint(
        checkpoint_path,
        expected_manifest_hash=expected_manifest_hash,
        memory_map=True,
    )
    resolved_config = payload["provenance"]["config"]
    if not isinstance(resolved_config, Mapping):
        raise ValueError("M1 checkpoint resolved config must be a mapping")
    try:
        encoder_config = visual_encoder_config(resolved_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("M1 checkpoint does not contain a valid visual encoder config") from exc
    if encoder_config.tubelet_size != 2:
        raise ValueError("V-JEPA 2-AC requires an M1 encoder with tubelet_size=2")
    if encoder_config.clip_frames != 16:
        raise ValueError("V-JEPA 2-AC requires an M1 encoder configured for 16 frames")

    prefix = "target_encoder."
    target_state = {
        name[len(prefix) :]: value
        for name, value in payload["model"].items()
        if name.startswith(prefix)
    }
    if not target_state:
        raise ValueError("M1 checkpoint does not contain target_encoder weights")
    frozen = FrozenVisualEncoder(encoder_config)
    frozen.encoder.load_state_dict(target_state, strict=True)
    frozen.requires_grad_(False)
    frozen.eval()
    return frozen, payload


def checkpoint_sha256(path: Path) -> str:
    """流式计算 checkpoint SHA-256，避免把大文件一次读入内存。"""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_world_model_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    optimizer_step: int,
    provenance: CheckpointProvenance,
    m1_parent_path: str,
    m1_parent_sha256: str,
    sampler_epoch: int,
    batch_offset: int,
    resume_checkpoint: Optional[str] = None,
    rng_state: Optional[Mapping[str, Any]] = None,
    rng_by_rank: Optional[Sequence[Mapping[str, Any]]] = None,
    world_size: int = 1,
    model_state_dict: Optional[Mapping[str, Any]] = None,
    optimizer_state_dict: Optional[Mapping[str, Any]] = None,
) -> None:
    """保存 M2 权重和可验证的 M1 parent、sampler 续训状态。"""

    if not m1_parent_path or not m1_parent_sha256:
        raise ValueError("M2 checkpoint requires an M1 parent path and SHA-256")
    if provenance.parent_checkpoint != m1_parent_path:
        raise ValueError("provenance.parent_checkpoint must identify the M1 checkpoint")
    state = model_state_dict if model_state_dict is not None else model.state_dict()
    trainable_state = {
        name: value
        for name, value in state.items()
        if not name.startswith("visual_encoder.")
    }
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        optimizer_step=optimizer_step,
        provenance=provenance,
        model_state_dict=trainable_state,
        optimizer_state_dict=optimizer_state_dict,
        extra={
            "stage": "m2-world-model",
            "m1_parent_path": m1_parent_path,
            "m1_parent_sha256": m1_parent_sha256,
            "sampler_epoch": int(sampler_epoch),
            "batch_offset": int(batch_offset),
            "resume_checkpoint": resume_checkpoint,
            "rng_by_rank": list(rng_by_rank) if rng_by_rank is not None else None,
            "world_size": int(world_size),
        },
        rng_state=rng_state,
    )


def load_world_model_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    expected_manifest_hash: str,
    expected_m1_parent_path: str,
    expected_m1_parent_sha256: str,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """验证 M1 parent 和数据 manifest 后恢复 M2 训练。"""

    payload = read_checkpoint(
        path,
        expected_manifest_hash=expected_manifest_hash,
    )
    extra = payload.get("extra", {})
    if extra.get("stage") != "m2-world-model":
        raise ValueError("checkpoint is not an M2 world-model checkpoint")
    if payload["provenance"].get("parent_checkpoint") != expected_m1_parent_path:
        raise ValueError("M2 provenance identifies a different M1 parent")
    if extra.get("m1_parent_path") != expected_m1_parent_path:
        raise ValueError("M2 checkpoint uses a different M1 parent path")
    if extra.get("m1_parent_sha256") != expected_m1_parent_sha256:
        raise ValueError("M2 checkpoint uses a different M1 parent hash")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected or any(not name.startswith("visual_encoder.") for name in missing):
        raise ValueError("M2 checkpoint model state does not match the current model")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint does not contain scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload
