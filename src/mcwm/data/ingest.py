"""Dependency-free ingestion of action JSONL plus externally extracted frame PTS."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mcwm.actions.minerl_adapter import minerl_action_to_canonical
from mcwm.actions.schema import ActionSource, CanonicalActionTick
from mcwm.actions.vpt_adapter import VPTActionAdapter
from .episode_store import EpisodeStore
from .manifest import EpisodeManifest
from .video import probe_video


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def read_frame_timestamps(path: Path) -> Tuple[int, ...]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        value = value["timestamps_ms"]
    timestamps = tuple(int(item) for item in value)
    if len(timestamps) < 2 or any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("frame timestamps must contain at least two increasing values")
    return timestamps


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(
    *,
    episode_id: str,
    session_id: str,
    world_id: str,
    source: ActionSource,
    recorder_version: str,
    video_path: Path,
    action_path: Path,
    frame_timestamps_ms: Sequence[int],
    actions: Sequence[CanonicalActionTick],
    split: str = None,
) -> EpisodeManifest:
    return EpisodeManifest(
        episode_id=episode_id,
        session_id=session_id,
        world_id=world_id,
        source=source,
        recorder_version=recorder_version,
        video_path=str(video_path),
        width=640,
        height=360,
        frame_count=len(frame_timestamps_ms),
        action_count=len(actions),
        start_timestamp_ms=frame_timestamps_ms[0],
        end_timestamp_ms=frame_timestamps_ms[-1],
        split=split,
        video_sha256=file_sha256(video_path) if video_path.exists() else None,
        action_sha256=file_sha256(action_path),
    )


def ingest_vpt_episode(
    store: EpisodeStore,
    *,
    episode_id: str,
    session_id: str,
    world_id: str,
    recorder_version: str,
    video_path: Path,
    action_path: Path,
    frame_timestamps_path: Optional[Path] = None,
    split: str = None,
) -> EpisodeManifest:
    rows = read_jsonl(action_path)
    adapter = VPTActionAdapter(recorder_version=recorder_version)
    actions = adapter.adapt_many(rows)
    timestamps = (
        read_frame_timestamps(frame_timestamps_path)
        if frame_timestamps_path is not None
        else probe_video(video_path).frame_timestamps_ms
    )
    manifest = _manifest(
        episode_id=episode_id,
        session_id=session_id,
        world_id=world_id,
        source=ActionSource.VPT,
        recorder_version=recorder_version,
        video_path=video_path,
        action_path=action_path,
        frame_timestamps_ms=timestamps,
        actions=actions,
        split=split,
    )
    store.write_episode(
        manifest,
        timestamps,
        actions,
        audit={"repairs": adapter.repairs, "unknown_keys": sorted(adapter.unknown_keys)},
    )
    store.write_dataset_manifest()
    return manifest


def ingest_minerl_episode(
    store: EpisodeStore,
    *,
    episode_id: str,
    session_id: str,
    world_id: str,
    recorder_version: str,
    video_path: Path,
    action_path: Path,
    frame_timestamps_path: Optional[Path] = None,
    split: str = None,
) -> EpisodeManifest:
    rows = read_jsonl(action_path)
    actions: List[CanonicalActionTick] = []
    for row in rows:
        raw_action = row.get("action", row)
        timestamp = row.get("timestamp_ms", row.get("milli"))
        actions.append(
            minerl_action_to_canonical(
                raw_action,
                timestamp_ms=timestamp,
                gui_open=bool(row.get("gui_open", False)),
                cursor=tuple(row["cursor"]) if row.get("cursor") is not None else None,
            )
        )
    timestamps = (
        read_frame_timestamps(frame_timestamps_path)
        if frame_timestamps_path is not None
        else probe_video(video_path).frame_timestamps_ms
    )
    manifest = _manifest(
        episode_id=episode_id,
        session_id=session_id,
        world_id=world_id,
        source=ActionSource.MINERL,
        recorder_version=recorder_version,
        video_path=video_path,
        action_path=action_path,
        frame_timestamps_ms=timestamps,
        actions=actions,
        split=split,
    )
    store.write_episode(manifest, timestamps, actions)
    store.write_dataset_manifest()
    return manifest
