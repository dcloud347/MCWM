"""读取并校验 M1 YAML，再构建完全由配置决定的模型。"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from mcwm.models.masking import MaskConfig, MaskGeneratorConfig
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


def _float_pair(values: Any, name: str) -> Tuple[float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain two numbers")
    if len(values) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return float(values[0]), float(values[1])


def _parse_mask_config(value: Any) -> MaskConfig:
    """读取与 V-JEPA 2 官方 YAML 同形状的 mask generator 列表。"""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError("mask must be a non-empty list")
    generators = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"mask[{index}] must be a mapping")
        missing = {
            "spatial_scale",
            "temporal_scale",
            "aspect_ratio",
            "num_blocks",
        } - entry.keys()
        if missing:
            raise ValueError(f"mask[{index}] is missing fields: {sorted(missing)}")
        generators.append(
            MaskGeneratorConfig(
                spatial_scale=_float_pair(
                    entry["spatial_scale"], f"mask[{index}].spatial_scale"
                ),
                temporal_scale=_float_pair(
                    entry["temporal_scale"], f"mask[{index}].temporal_scale"
                ),
                aspect_ratio=_float_pair(
                    entry["aspect_ratio"], f"mask[{index}].aspect_ratio"
                ),
                num_blocks=int(entry["num_blocks"]),
            )
        )
    return MaskConfig(generators=tuple(generators))


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
    data = config["data"]
    missing_data = {"clip_frames", "sample_fps", "clips_per_video"} - data.keys()
    if missing_data:
        raise ValueError(f"config data is missing fields: {sorted(missing_data)}")
    if int(data["clip_frames"]) < 2:
        raise ValueError("data.clip_frames must be at least two")
    if int(data["sample_fps"]) <= 0:
        raise ValueError("data.sample_fps must be positive")
    tubelet_size = int(config["model"].get("tubelet_size", 2))
    if tubelet_size <= 0 or int(data["clip_frames"]) % tubelet_size:
        raise ValueError("data.clip_frames must be divisible by model.tubelet_size")
    if int(data["clips_per_video"]) != 1:
        raise ValueError("data.clips_per_video must be one")
    _parse_mask_config(config["mask"])
    optimizer = config["optimizer"]
    epochs = optimizer.get("epochs")
    if epochs is None or not math.isfinite(float(epochs)) or float(epochs) <= 0:
        raise ValueError("optimizer.epochs must be positive and finite")
    iterations_per_epoch = optimizer.get("iterations_per_epoch")
    if iterations_per_epoch is None or int(iterations_per_epoch) <= 0:
        raise ValueError("optimizer.iterations_per_epoch must be positive")
    strategy = config["distributed"].get("strategy", "none")
    if strategy not in {"none", "ddp", "fsdp"}:
        raise ValueError("distributed.strategy must be none, ddp or fsdp")
    per_step = int(data["batch_size"])
    effective = int(optimizer["effective_batch_size"])
    if min(per_step, effective) <= 0:
        raise ValueError("batch sizes must be positive")
    positive_intervals = (
        config["wandb"].get("log_every_steps", 10),
        config["wandb"].get("diagnostics_every_steps", 200),
        config["validation"].get("every_steps", 1000),
        config["checkpoint"].get("every_steps", 1000),
    )
    if any(int(value) <= 0 for value in positive_intervals):
        raise ValueError(
            "logging, diagnostics, validation and checkpoint intervals must be positive"
        )
    if int(config["validation"].get("clips_per_video", 1)) <= 0:
        raise ValueError("validation.clips_per_video must be positive")


def build_visual_jepa(config: Mapping[str, Any]) -> VisualJEPA:
    """从 resolved config 构建 M1 模型，不读取任何外部权重。"""

    validate_pretrain_config(config)
    model = config["model"]
    data = config["data"]
    encoder = VisualEncoderConfig(
        image_height=int(model["image_height"]),
        image_width=int(model["image_width"]),
        patch_size=int(model["patch_size"]),
        clip_frames=int(data["clip_frames"]),
        tubelet_size=int(model.get("tubelet_size", 2)),
        dim=int(model["encoder_dim"]),
        depth=int(model["encoder_depth"]),
        heads=int(model["encoder_heads"]),
        mlp_dim=int(model["encoder_mlp_dim"]),
        use_rope=bool(model.get("use_rope", True)),
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
    )
    predictor = VisualPredictorConfig(
        input_dim=encoder.dim,
        dim=int(model["predictor_dim"]),
        depth=int(model["predictor_depth"]),
        heads=int(model["predictor_heads"]),
        mlp_dim=int(model["predictor_mlp_dim"]),
        token_grid_size=encoder.token_grid_size,
        num_mask_tokens=len(config["mask"]),
        use_rope=bool(model.get("use_rope", True)),
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
    )
    mask = _parse_mask_config(config["mask"])
    return VisualJEPA(VisualJEPAConfig(encoder=encoder, predictor=predictor, mask=mask))


def resolved_copy(config: Mapping[str, Any]) -> Dict[str, Any]:
    """返回可安全写入 JSON、W&B 和 checkpoint 的普通深拷贝。"""

    return deepcopy(dict(config))
