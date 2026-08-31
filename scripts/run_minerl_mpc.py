#!/usr/bin/env python3
"""Run the bounded M4 receding-horizon smoke test in MineRL 1.0."""

import argparse
from pathlib import Path

from mcwm.envs.minerl1 import MineRL1EnvWrapper
from mcwm.planning.config import load_planner_config
from mcwm.planning.macro_codebook import MacroCodebook
from mcwm.planning.mpc import RecedingHorizonMPC
from mcwm.planning.online import StaticGoalProvider, run_mpc_smoke
from mcwm.planning.runtime import load_planning_world_model, load_rgb_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--m1-checkpoint", type=Path)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/plan_m4.yaml"))
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("m4_smoke.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    try:
        import gym  # type: ignore
        import minerl  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MineRL smoke test requires `pip install -e '.[minerl]'` and JDK 8"
        ) from exc

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
    environment = MineRL1EnvWrapper(
        gym.make(args.env_id),
        model_fps=float(config.get("model_fps", 4)),
        environment_fps=float(config.get("environment_fps", 20)),
    )
    if environment.action_repeat != codebook.fit_config.action_repeat:
        raise ValueError("environment action repeat does not match codebook provenance")
    run_mpc_smoke(
        environment,
        controller,
        model,
        StaticGoalProvider(load_rgb_image(args.goal)),
        cycles=args.cycles,
        output_path=args.output,
        legality=legality,
    )
    print(f"wrote MineRL smoke report to {args.output}")


if __name__ == "__main__":
    main()
