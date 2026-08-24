"""Data quality reports for canonical MCWM episode stores."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from mcwm.actions.schema import INTERACTION_NAMES, MOVEMENT_NAMES, CanonicalActionTick
from .alignment import align_actions_to_frames
from .episode_store import EpisodeStore, StoredEpisode
from .manifest import find_split_leakage


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _stats(values: Iterable[float]) -> Dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(materialized),
        "mean": mean(materialized),
        "p50": _percentile(materialized, 0.50),
        "p95": _percentile(materialized, 0.95),
        "max": max(materialized),
    }


def _active_names(action: CanonicalActionTick) -> List[str]:
    active = [name for name, value in zip(MOVEMENT_NAMES, action.movement) if value]
    active.extend(name for name, value in zip(INTERACTION_NAMES, action.interaction) if value)
    if action.hotbar:
        active.append(f"hotbar.{action.hotbar}")
    if action.camera != (0.0, 0.0):
        active.append("camera")
    return active


def audit_episode(episode: StoredEpisode, *, max_frame_gap_ms: int = 250) -> Dict[str, Any]:
    actions = episode.actions
    movement = Counter()
    interaction = Counter()
    combinations = Counter()
    for action in actions:
        movement.update(
            name for name, value in zip(MOVEMENT_NAMES, action.movement) if value
        )
        interaction.update(
            name for name, value in zip(INTERACTION_NAMES, action.interaction) if value
        )
        names = _active_names(action)
        combinations["+".join(names) if names else "noop"] += 1

    aligned = align_actions_to_frames(
        episode.frame_timestamps_ms,
        actions,
        max_frame_gap_ms=max_frame_gap_ms,
    )
    frame_intervals = [
        right - left
        for left, right in zip(episode.frame_timestamps_ms, episode.frame_timestamps_ms[1:])
    ]
    action_intervals = [
        right.timestamp_ms - left.timestamp_ms for left, right in zip(actions, actions[1:])
    ]
    camera_magnitude = [math.hypot(*action.camera) for action in actions]
    issues: List[str] = []
    if aligned.actions_before_first_frame:
        issues.append(f"{len(aligned.actions_before_first_frame)} actions precede the first frame")
    if aligned.actions_at_or_after_last_frame:
        issues.append(
            f"{len(aligned.actions_at_or_after_last_frame)} actions are at/after the last frame"
        )
    discontinuities = sum(not block.continuous for block in aligned.blocks)
    if discontinuities:
        issues.append(f"{discontinuities} frame gaps exceed {max_frame_gap_ms} ms")
    if any(not action.valid for action in actions):
        issues.append("stored episode contains padding ticks; padding belongs in batches only")

    return {
        "episode_id": episode.manifest.episode_id,
        "source": episode.manifest.source.value,
        "duration_ms": episode.manifest.duration_ms,
        "frame_count": episode.manifest.frame_count,
        "action_count": len(actions),
        "valid_action_count": sum(action.valid for action in actions),
        "noop_count": sum(action.is_noop for action in actions),
        "noop_ratio": (sum(action.is_noop for action in actions) / len(actions)) if actions else 0.0,
        "gui_count": sum(action.gui_open for action in actions),
        "movement_frequency": dict(sorted(movement.items())),
        "interaction_frequency": dict(sorted(interaction.items())),
        "top_action_combinations": combinations.most_common(20),
        "camera_magnitude": _stats(camera_magnitude),
        "frame_interval_ms": _stats(frame_intervals),
        "action_interval_ms": _stats(action_intervals),
        "continuous_frame_ranges": [list(value) for value in aligned.continuous_frame_ranges],
        "issues": issues,
        "ingest_audit": dict(episode.audit or {}),
    }


def audit_store(root: Path, *, max_frame_gap_ms: int = 250) -> Dict[str, Any]:
    store = EpisodeStore(root)
    manifests = store.list_manifests()
    episode_reports = [
        audit_episode(store.read_episode(manifest.episode_id), max_frame_gap_ms=max_frame_gap_ms)
        for manifest in manifests
    ]
    sources = Counter(manifest.source.value for manifest in manifests)
    total_duration_ms = sum(manifest.duration_ms for manifest in manifests)
    split_leakage = list(find_split_leakage(manifests))
    issues = list(split_leakage)
    for report in episode_reports:
        issues.extend(f"{report['episode_id']}: {issue}" for issue in report["issues"])
    return {
        "schema_version": 1,
        "episode_count": len(manifests),
        "source_episode_count": dict(sorted(sources.items())),
        "duration_hours": total_duration_ms / 3_600_000.0,
        "frame_count": sum(manifest.frame_count for manifest in manifests),
        "action_count": sum(manifest.action_count for manifest in manifests),
        "noop_count": sum(report["noop_count"] for report in episode_reports),
        "gui_count": sum(report["gui_count"] for report in episode_reports),
        "split_leakage": split_leakage,
        "issues": issues,
        "episodes": episode_reports,
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Canonical episode store root")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--max-frame-gap-ms", type=int, default=250)
    args = parser.parse_args(argv)
    report = audit_store(args.root, max_frame_gap_ms=args.max_frame_gap_ms)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 1 if report["split_leakage"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

