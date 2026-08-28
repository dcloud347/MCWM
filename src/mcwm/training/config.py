"""读取训练配置，检查参数并创建 M1 模型。"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from mcwm.models.ac_predictor import ACPredictorConfig, ActionConditionedPredictor
from mcwm.models.action_encoder import ActionEncoderConfig, MinecraftActionEncoder
from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
from mcwm.models.masking import MaskConfig, MaskGeneratorConfig
from mcwm.models.visual_encoder import VisualEncoderConfig
from mcwm.models.visual_jepa import VisualJEPA, VisualJEPAConfig
from mcwm.models.visual_predictor import VisualPredictorConfig
from mcwm.models.world_model import WorldModel, WorldModelConfig


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置；缺少依赖时给出安装提示。"""

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
    """把 YAML 中的 mask 列表转换成模型配置。"""

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
    """在开始训练前检查必填字段和参数范围。"""

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
    collapse = config["collapse"]
    if int(collapse.get("grace_validations", 0)) < 0:
        raise ValueError("collapse.grace_validations must be non-negative")
    if int(collapse.get("patience_validations", 3)) <= 0:
        raise ValueError("collapse.patience_validations must be positive")


def visual_encoder_config(config: Mapping[str, Any]) -> VisualEncoderConfig:
    """从 resolved M1 配置中重建 visual encoder 结构。"""

    model = config["model"]
    data = config["data"]
    return VisualEncoderConfig(
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


def build_visual_jepa(config: Mapping[str, Any]) -> VisualJEPA:
    """只根据配置创建新的 M1 模型，不加载外部权重。"""

    validate_pretrain_config(config)
    model = config["model"]
    encoder = visual_encoder_config(config)
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
    """返回可写入日志和 checkpoint 的独立配置副本。"""

    return deepcopy(dict(config))


def validate_world_model_config(config: Mapping[str, Any]) -> None:
    """检查 M2 数据、模型、优化器和 checkpoint 配置。"""

    required = {
        "data",
        "model",
        "optimizer",
        "validation",
        "checkpoint",
        "wandb",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config is missing sections: {sorted(missing)}")
    if config.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.get("precision") not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16 or fp16")

    distributed = config.get("distributed", {})
    if not isinstance(distributed, Mapping):
        raise ValueError("distributed must be a mapping")
    strategy = str(distributed.get("strategy", "none"))
    if strategy not in {"none", "fsdp"}:
        raise ValueError("M2 distributed.strategy must be none or fsdp")

    data = config["data"]
    for name in ("frames_per_sample", "sample_fps", "samples_per_video", "batch_size"):
        if int(data.get(name, 0)) <= 0:
            raise ValueError(f"data.{name} must be positive")
    if int(data["frames_per_sample"]) < 3:
        raise ValueError("data.frames_per_sample must be at least three")

    optimizer = config["optimizer"]
    for name in ("learning_rate", "iterations_per_epoch", "epochs"):
        if float(optimizer.get(name, 0)) <= 0:
            raise ValueError(f"optimizer.{name} must be positive")
    if int(optimizer.get("effective_batch_size", 0)) <= 0:
        raise ValueError("optimizer.effective_batch_size must be positive")

    model = config["model"]
    action_encoder_config(config)
    predictor = ac_predictor_config(config)
    auto_steps = int(model.get("auto_steps", 2))
    if not 1 <= auto_steps < int(data["frames_per_sample"]):
        raise ValueError("model.auto_steps must be smaller than frames_per_sample")
    if predictor.context_blocks < int(data["frames_per_sample"]) - 1:
        raise ValueError("predictor context must cover all sample transitions")
    if int(config["validation"].get("every_steps", 0)) <= 0:
        raise ValueError("validation.every_steps must be positive")
    if int(config["checkpoint"].get("every_steps", 0)) <= 0:
        raise ValueError("checkpoint.every_steps must be positive")


def action_encoder_config(config: Mapping[str, Any]) -> ActionEncoderConfig:
    """从 M2 YAML 创建 Minecraft action encoder 配置。"""

    value = config["model"]["action_encoder"]
    return ActionEncoderConfig(
        binary_embedding_dim=int(value.get("binary_embedding_dim", 16)),
        hotbar_embedding_dim=int(value.get("hotbar_embedding_dim", 32)),
        camera_dim=int(value.get("camera_dim", 64)),
        cursor_dim=int(value.get("cursor_dim", 64)),
        component_hidden_dim=int(value.get("component_hidden_dim", 512)),
        tick_dim=int(value.get("tick_dim", 256)),
        transformer_depth=int(value.get("depth", 2)),
        transformer_heads=int(value.get("heads", 8)),
        transformer_mlp_dim=int(value.get("mlp_dim", 1024)),
        macro_dim=int(value.get("macro_dim", 1024)),
        dropout=float(value.get("dropout", 0.0)),
        camera_clip_degrees=float(value.get("camera_clip_degrees", 180.0)),
        camera_mu=float(value.get("camera_mu", 255.0)),
    )


def ac_predictor_config(config: Mapping[str, Any]) -> ACPredictorConfig:
    """从 M2 YAML 创建 block-causal predictor 配置。"""

    value = config["model"]["predictor"]
    grid = value.get("spatial_grid", (18, 32))
    if not isinstance(grid, Sequence) or len(grid) != 2:
        raise ValueError("model.predictor.spatial_grid must contain two integers")
    return ACPredictorConfig(
        latent_dim=int(value.get("latent_dim", 1024)),
        action_dim=int(value.get("action_dim", 1024)),
        dim=int(value.get("dim", 1024)),
        depth=int(value.get("depth", 24)),
        heads=int(value.get("heads", 16)),
        mlp_dim=int(value.get("mlp_dim", 4096)),
        context_blocks=int(value.get("context_blocks", 16)),
        spatial_grid=(int(grid[0]), int(grid[1])),
        dropout=float(value.get("dropout", 0.1)),
        gradient_checkpointing=bool(value.get("gradient_checkpointing", True)),
    )


def build_world_model(
    config: Mapping[str, Any],
    visual_encoder: FrozenVisualEncoder,
) -> WorldModel:
    """创建随机初始化的 M2 action encoder 和 predictor。"""

    validate_world_model_config(config)
    action = MinecraftActionEncoder(action_encoder_config(config))
    predictor = ActionConditionedPredictor(ac_predictor_config(config))
    model = config["model"]
    return WorldModel(
        visual_encoder,
        action,
        predictor,
        WorldModelConfig(
            auto_steps=int(model.get("auto_steps", 2)),
            encoder_frame_chunk_size=model.get("encoder_frame_chunk_size"),
        ),
    )
