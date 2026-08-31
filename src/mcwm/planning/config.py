"""Parse the small M4 planning-only YAML surface."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from mcwm.training.config import load_yaml_config
from .cem import HybridCEMConfig, HybridCEMPlanner
from .legality import LegalityContext
from .macro_codebook import MacroCodebook


def build_planner_from_config(
    codebook: MacroCodebook,
    config: Mapping[str, Any],
) -> Tuple[HybridCEMPlanner, LegalityContext]:
    if int(config.get("macro_length", 0)) != codebook.fit_config.macro_length:
        raise ValueError("planning macro_length does not match the codebook")
    cem_values: Dict[str, Any] = dict(config.get("cem", {}))
    cem_values.update(dict(config.get("objective", {})))
    cem_values["macro_horizon"] = int(config.get("macro_horizon", 4))
    cem_values["seed"] = int(config.get("seed", 2026))
    allowed = {value.name for value in fields(HybridCEMConfig)}
    unknown = set(cem_values) - allowed
    if unknown:
        raise ValueError(f"unknown CEM planning fields: {sorted(unknown)}")
    planner = HybridCEMPlanner(codebook, HybridCEMConfig(**cem_values))

    legality = dict(config.get("legality", {}))
    allowed_legality = {value.name for value in fields(LegalityContext)}
    unknown_legality = set(legality) - allowed_legality
    if unknown_legality:
        raise ValueError(f"unknown legality fields: {sorted(unknown_legality)}")
    return planner, LegalityContext(**legality)


def load_planner_config(
    path: Path,
    codebook: MacroCodebook,
) -> Tuple[HybridCEMPlanner, LegalityContext, Mapping[str, Any]]:
    config = load_yaml_config(Path(path))
    planner, legality = build_planner_from_config(codebook, config)
    return planner, legality, config

