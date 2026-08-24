"""原子写入并严格检查 provenance 的训练 checkpoint。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointProvenance:
    """说明权重从哪里来；MCWM 永远不允许 external_pretrained=True。"""

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
    """保存 Python、NumPy、PyTorch 以及所有 CUDA 设备的随机状态。"""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """恢复随机状态，使 resume 后的下一步可以和连续训练一致。"""

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
    """先写同目录临时文件，成功后再 rename，避免留下半个 checkpoint。"""

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
) -> Dict[str, Any]:
    """读取 checkpoint，并在加载权重前检查版本和数据来源。"""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
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
    """恢复单卡或 DDP checkpoint 的全部训练状态。"""

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
    """只导出 MCWM 自己训练的 EMA encoder，供 M2 初始化或部署使用。"""

    try:
        from safetensors.torch import save_file  # type: ignore
    except ImportError as exc:
        raise RuntimeError("encoder export requires `pip install mcwm[train]`") from exc
    payload = read_checkpoint(checkpoint_path)
    # online encoder 和 M1 predictor 都不属于最终视觉预训练产物。
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
