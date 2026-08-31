#!/usr/bin/env python3
"""Run one deterministic M4 plan from observation and goal image files."""

import argparse
import json
from pathlib import Path

from mcwm.planning.config import load_planner_config
from mcwm.planning.macro_codebook import MacroCodebook
from mcwm.planning.mpc import RecedingHorizonMPC, canonical_to_minerl_action
from mcwm.planning.runtime import load_planning_world_model, load_rgb_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--m1-checkpoint", type=Path)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/plan_m4.yaml"))
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("m4_offline_plan.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    codebook = MacroCodebook.read(args.codebook)
    planner, legality, config = load_planner_config(args.config, codebook)
    model, checkpoint = load_planning_world_model(
        args.checkpoint,
        m1_checkpoint=args.m1_checkpoint,
        device=args.device,
    )
    if codebook.manifest_hash != checkpoint["provenance"]["manifest_hash"]:
        raise ValueError("codebook and checkpoint use different data manifests")
    controller = RecedingHorizonMPC(
        planner,
        max_context=int(config.get("context_frames", 16)),
    )
    result, actions = controller.plan(
        model,
        load_rgb_image(args.observation),
        load_rgb_image(args.goal),
        context=legality,
    )
    report = {
        "cost": result.cost,
        "cost_terms": result.cost_terms,
        "predicted_goal_costs": result.predicted_goal_costs.cpu().tolist(),
        "selected_macro_codes": result.code_ids.cpu().tolist(),
        "camera_residuals": result.camera_residuals.cpu().tolist(),
        "fallback_reason": result.fallback_reason,
        "first_macro_minerl_actions": [
            dict(canonical_to_minerl_action(action)) for action in actions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote offline plan to {args.output}")


if __name__ == "__main__":
    main()

