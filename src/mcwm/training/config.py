"""读取并校验 M1 YAML，再构建完全由配置决定的模型。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

from mcwm.models.masking import MaskConfig
from mcwm.models.visual_encoder import VisualEncoderConfig
from mcwm.models.visual_jepa import VisualJEPA, VisualJEPAConfig
from mcwm.models.visual_predictor import VisualPredictorConfig


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """读取 YAML；未安装训练依赖时给出明确安装提示。"""

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("training config requires `pip install mcwm[train]`") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("training config root must be a mapping")
    return config


def validate_pretrain_config(config: Mapping[str, Any]) -> None:
    """尽早拒绝缺字段、错误精度和无效训练间隔。"""

    required_sections = {
        "data",
        "model",
        "mask",
        "augmentation",
        "optimizer",
        "validation",
        "collapse",
        "checkpoint",
        "distributed",
        "wandb",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"config is missing sections: {sorted(missing)}")
    if config.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.get("precision") not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16 or fp16")
    strategy = config["distributed"].get("strategy", "none")
    if strategy not in {"none", "ddp", "fsdp"}:
        raise ValueError("distributed.strategy must be none, ddp or fsdp")
    per_step = int(config["data"]["batch_size"])
    effective = int(config["optimizer"]["effective_batch_size"])
    if min(per_step, effective) <= 0:
        raise ValueError("batch sizes must be positive")
    positive_intervals = (
        config["wandb"].get("log_every_steps", 10),
        config["wandb"].get("diagnostics_every_steps", 200),
        config["validation"].get("every_steps", 1000),
        config["checkpoint"].get("every_steps", 1000),
    )
    if any(int(value) <= 0 for value in positive_intervals):
        raise ValueError("logging, diagnostics, validation and checkpoint intervals must be positive")


def build_visual_jepa(config: Mapping[str, Any]) -> VisualJEPA:
    """从 resolved config 构建 M1 模型，不读取任何外部权重。"""

    validate_pretrain_config(config)
    model = config["model"]
    data = config["data"]
    encoder = VisualEncoderConfig(
        image_height=int(model["image_height"]),
        image_width=int(model["image_width"]),
        patch_size=int(model["patch_size"]),
        dim=int(model["encoder_dim"]),
        depth=int(model["encoder_depth"]),
        heads=int(model["encoder_heads"]),
        mlp_dim=int(model["encoder_mlp_dim"]),
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
    )
    predictor = VisualPredictorConfig(
        input_dim=encoder.dim,
        dim=int(model["predictor_dim"]),
        depth=int(model["predictor_depth"]),
        heads=int(model["predictor_heads"]),
        mlp_dim=int(model["predictor_mlp_dim"]),
        max_frames=int(data["clip_frames"]),
        patch_count=encoder.patch_count,
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
    )
    mask = MaskConfig(
        ratio=float(config["mask"]["ratio"]),
        spatial_blocks=int(config["mask"].get("spatial_blocks", 4)),
        temporal_tubes=int(config["mask"].get("temporal_tubes", 4)),
    )
    return VisualJEPA(VisualJEPAConfig(encoder=encoder, predictor=predictor, mask=mask))


def resolved_copy(config: Mapping[str, Any]) -> Dict[str, Any]:
    """返回可安全写入 JSON、W&B 和 checkpoint 的普通深拷贝。"""

    return deepcopy(dict(config))
