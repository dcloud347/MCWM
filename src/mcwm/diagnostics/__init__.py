"""训练中和离线使用的模型诊断。"""

from .collapse import CollapseThresholds, collapse_metrics, find_collapse_alerts

__all__ = ["CollapseThresholds", "collapse_metrics", "find_collapse_alerts"]
from .world_model import (
    action_sensitivity_from_predictions,
    action_sensitivity_from_samples,
    action_sensitivity_report,
    action_sensitivity_samples_from_predictions,
    noop_action_inputs,
    rollout_error_curve,
    spatial_error_images,
    spatial_token_error,
    world_model_prediction_metrics,
)
from .m2_b0 import run_b0_smoke_gate
from .rollout import rollout_metrics, rollout_samples
from .surprise import surprise_metrics, surprise_samples

__all__ = [
    "action_sensitivity_from_predictions",
    "action_sensitivity_from_samples",
    "action_sensitivity_report",
    "action_sensitivity_samples_from_predictions",
    "noop_action_inputs",
    "rollout_error_curve",
    "rollout_metrics",
    "rollout_samples",
    "run_b0_smoke_gate",
    "spatial_error_images",
    "spatial_token_error",
    "surprise_metrics",
    "surprise_samples",
    "world_model_prediction_metrics",
]
