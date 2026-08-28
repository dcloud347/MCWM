"""统计 M2 实际采样 clip 中的动作覆盖率。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from mcwm.actions.schema import CanonicalActionTick

from .visual_dataset import ResumableSampler
from .world_model_dataset import WorldModelDataset


CATEGORY_NAMES = (
    "movement",
    "interaction",
    "camera",
    "hotbar",
    "gui_open",
    "cursor",
)


class _ProgressBar:
    """不依赖第三方包的轻量终端进度条。"""

    def __init__(self, total: int, *, enabled: bool) -> None:
        self.total = max(0, int(total))
        self.enabled = enabled
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0

    def update(self, current: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if current < self.total and now - self.last_rendered_at < 0.2:
            return
        elapsed = max(now - self.started_at, 1e-9)
        fraction = min(current / self.total, 1.0) if self.total else 1.0
        filled = round(30 * fraction)
        bar = "#" * filled + "-" * (30 - filled)
        rate = current / elapsed
        print(
            f"\rScanning clips [{bar}] {fraction:6.2%} "
            f"{current}/{self.total} {rate:,.1f} clips/s",
            end="",
            file=sys.stderr,
            flush=True,
        )
        self.last_rendered_at = now

    def close(self, current: int) -> None:
        if self.enabled:
            self.update(current)
            print(file=sys.stderr, flush=True)


def _tick_categories(action: CanonicalActionTick) -> Dict[str, bool]:
    return {
        "movement": any(action.movement),
        "interaction": any(action.interaction),
        "camera": action.camera != (0.0, 0.0),
        "hotbar": action.hotbar != 0,
        "gui_open": action.gui_open,
        "cursor": action.cursor is not None,
    }


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def audit_world_model_dataset(
    dataset: WorldModelDataset,
    *,
    seed: int,
    sampling_epochs: int = 1,
    max_clips: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """按训练 sampler 的规则统计 clip、transition 和 tick 动作覆盖率。"""

    if sampling_epochs <= 0:
        raise ValueError("sampling_epochs must be positive")
    if max_clips is not None and max_clips <= 0:
        raise ValueError("max_clips must be positive")

    totals = {
        "clips": 0,
        "clips_with_action": 0,
        "transitions": 0,
        "transitions_with_action": 0,
        "ticks": 0,
        "ticks_with_action": 0,
    }
    clip_categories = {name: 0 for name in CATEGORY_NAMES}
    transition_categories = {name: 0 for name in CATEGORY_NAMES}
    tick_categories = {name: 0 for name in CATEGORY_NAMES}

    sampler = ResumableSampler(
        len(dataset),
        seed=int(seed),
        seed_clips=True,
    )
    stop = False
    completed_epochs = 0
    total_clips = len(dataset) * sampling_epochs
    if max_clips is not None:
        total_clips = min(total_clips, max_clips)
    progress = _ProgressBar(total_clips, enabled=show_progress)
    for epoch in range(sampling_epochs):
        sampler.set_epoch(epoch)
        for sample_index in sampler:
            metadata = dataset.sample_action_clip(sample_index)
            blocks = metadata["action_blocks"]
            clip_has_action = False
            clip_category_names = set()

            for block in blocks:
                transition_has_action = False
                transition_category_names = set()
                for action in block:
                    has_action = not action.is_noop
                    transition_has_action |= has_action
                    clip_has_action |= has_action
                    totals["ticks"] += 1
                    totals["ticks_with_action"] += int(has_action)
                    for name, active in _tick_categories(action).items():
                        if active:
                            tick_categories[name] += 1
                            transition_category_names.add(name)
                            clip_category_names.add(name)

                totals["transitions"] += 1
                totals["transitions_with_action"] += int(transition_has_action)
                for name in transition_category_names:
                    transition_categories[name] += 1

            totals["clips"] += 1
            totals["clips_with_action"] += int(clip_has_action)
            progress.update(totals["clips"])
            for name in clip_category_names:
                clip_categories[name] += 1

            if max_clips is not None and totals["clips"] >= max_clips:
                stop = True
                break
        completed_epochs += 1
        if stop:
            break
    progress.close(totals["clips"])

    clips = totals["clips"]
    transitions = totals["transitions"]
    ticks = totals["ticks"]
    return {
        "definition": (
            "A clip has action when any valid tick has movement, interaction, "
            "camera motion, or a hotbar switch. GUI/cursor presence is reported "
            "separately and does not by itself make a tick non-noop."
        ),
        "episode_count": len(dataset.references),
        "dataset_slots_per_epoch": len(dataset),
        "sampling_epochs_requested": sampling_epochs,
        "sampling_epochs_completed": completed_epochs,
        "sample_fps": dataset.sample_fps,
        "frames_per_clip": dataset.frames_per_sample,
        "totals": {
            **totals,
            "clips_only_noop": clips - totals["clips_with_action"],
        },
        "ratios": {
            "clips_with_action": _ratio(totals["clips_with_action"], clips),
            "clips_only_noop": _ratio(clips - totals["clips_with_action"], clips),
            "transitions_with_action": _ratio(
                totals["transitions_with_action"], transitions
            ),
            "ticks_with_action": _ratio(totals["ticks_with_action"], ticks),
        },
        "categories": {
            name: {
                "clips": clip_categories[name],
                "clip_ratio": _ratio(clip_categories[name], clips),
                "transitions": transition_categories[name],
                "transition_ratio": _ratio(
                    transition_categories[name], transitions
                ),
                "ticks": tick_categories[name],
                "tick_ratio": _ratio(tick_categories[name], ticks),
            }
            for name in CATEGORY_NAMES
        },
    }


def audit_from_config(
    config: Mapping[str, Any],
    *,
    root: Optional[Path] = None,
    split: Optional[str] = None,
    sampling_epochs: int = 1,
    max_clips: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """构建与 M2 配置一致的数据集并统计动作覆盖率。"""

    data = config["data"]
    resolved_root = Path(root) if root is not None else Path(data["root"])
    resolved_split = str(split or data.get("train_split", "train"))
    dataset = WorldModelDataset(
        resolved_root,
        split=resolved_split,
        frames_per_sample=int(data["frames_per_sample"]),
        sample_fps=int(data["sample_fps"]),
        seed=int(config.get("seed", 0)),
        samples_per_video=int(data["samples_per_video"]),
    )
    report = audit_world_model_dataset(
        dataset,
        seed=int(config.get("seed", 0)),
        sampling_epochs=sampling_epochs,
        max_clips=max_clips,
        show_progress=show_progress,
    )
    report["root"] = str(resolved_root)
    report["split"] = resolved_split
    return report


def _percent(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def render_report(report: Mapping[str, Any]) -> str:
    """把 JSON report 渲染为便于终端阅读的文本。"""

    totals = report["totals"]
    ratios = report["ratios"]
    lines = [
        "World-model clip action audit",
        f"root: {report.get('root', '<dataset>')}",
        f"split: {report.get('split', '<unknown>')}",
        (
            f"episodes: {report['episode_count']} | "
            f"sampled clips: {totals['clips']} | "
            f"sampler epochs: {report['sampling_epochs_completed']}"
        ),
        f"definition: {report['definition']}",
        "",
        (
            f"clips with action:       {totals['clips_with_action']:8d} / "
            f"{totals['clips']:8d}  {_percent(ratios['clips_with_action'])}"
        ),
        (
            f"clips only no-op:        {totals['clips_only_noop']:8d} / "
            f"{totals['clips']:8d}  {_percent(ratios['clips_only_noop'])}"
        ),
        (
            f"transitions with action: {totals['transitions_with_action']:8d} / "
            f"{totals['transitions']:8d}  {_percent(ratios['transitions_with_action'])}"
        ),
        (
            f"ticks with action:       {totals['ticks_with_action']:8d} / "
            f"{totals['ticks']:8d}  {_percent(ratios['ticks_with_action'])}"
        ),
        "",
        "category       clips       transitions       ticks",
    ]
    for name in CATEGORY_NAMES:
        values = report["categories"][name]
        lines.append(
            f"{name:<12} {_percent(values['clip_ratio'])} "
            f"     {_percent(values['transition_ratio'])} "
            f"       {_percent(values['tick_ratio'])}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """运行 M2 clip 动作覆盖率审计。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Override data.root from config")
    parser.add_argument("--split")
    parser.add_argument("--sampling-epochs", type=int, default=1)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", type=Path, help="Write the full JSON report")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    from mcwm.training.config import load_yaml_config, validate_world_model_config

    config = load_yaml_config(args.config)
    validate_world_model_config(config)
    report = audit_from_config(
        config,
        root=args.root,
        split=args.split,
        sampling_epochs=args.sampling_epochs,
        max_clips=args.max_clips,
        show_progress=not args.no_progress,
    )
    rendered_json = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered_json + "\n", encoding="utf-8")
    print(rendered_json if args.json else render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
