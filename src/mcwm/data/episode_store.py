"""Atomic, zero-dependency reference episode store for milestone M0."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mcwm.actions.codec import action_to_dict, read_actions_jsonl, write_actions_jsonl
from mcwm.actions.schema import CanonicalActionTick
from .manifest import (
    DatasetManifest,
    EpisodeManifest,
    read_episode_manifest,
    write_episode_manifest,
)


@dataclass(frozen=True)
class StoredEpisode:
    manifest: EpisodeManifest
    frame_timestamps_ms: Tuple[int, ...]
    actions: Tuple[CanonicalActionTick, ...]
    audit: Optional[Mapping[str, Any]] = None


class EpisodeStore:
    """保存 episode 元数据和动作，视频只保存路径，不重复复制大文件。

    M0 默认写 JSONL，这样不安装第三方库也能读写。安装 pyarrow 后可以通过
    ``export_parquet`` 导出正式训练管线使用的 Parquet。
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"

    def episode_dir(self, episode_id: str) -> Path:
        if not episode_id or "/" in episode_id or ".." in episode_id:
            raise ValueError("episode_id must be a safe path component")
        return self.episodes_dir / episode_id

    def write_episode(
        self,
        manifest: EpisodeManifest,
        frame_timestamps_ms: Sequence[int],
        actions: Iterable[CanonicalActionTick],
        *,
        audit: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """校验并写入一个 episode；任何计数或时间不一致都会直接报错。"""

        timestamps = tuple(int(value) for value in frame_timestamps_ms)
        action_tuple = tuple(actions)
        if len(timestamps) != manifest.frame_count:
            raise ValueError("frame timestamp count does not match manifest.frame_count")
        if len(action_tuple) != manifest.action_count:
            raise ValueError("action count does not match manifest.action_count")
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("frame timestamps must be strictly increasing")
        if timestamps[0] != manifest.start_timestamp_ms or timestamps[-1] != manifest.end_timestamp_ms:
            raise ValueError("manifest start/end timestamps must match frame timestamps")
        if any(current.timestamp_ms < previous.timestamp_ms for previous, current in zip(action_tuple, action_tuple[1:])):
            raise ValueError("actions must be sorted by timestamp")

        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        destination = self.episode_dir(manifest.episode_id)
        if destination.exists():
            raise FileExistsError(f"episode already exists: {destination}")

        # Build the complete episode beside its final location. Directory rename is
        # atomic on the same filesystem, so an interrupted ingest never publishes a
        # manifest without its actions and frame timestamps.
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.episode_id}.", dir=self.episodes_dir
        ) as temporary:
            staging = Path(temporary)
            write_episode_manifest(staging / "manifest.json", manifest)
            write_actions_jsonl(staging / "actions.jsonl", action_tuple)
            _atomic_json(staging / "frame_timestamps.json", {"timestamps_ms": timestamps})
            if audit is not None:
                _atomic_json(staging / "audit.json", dict(audit))
            staging.replace(destination)
        return destination

    def read_episode(self, episode_id: str) -> StoredEpisode:
        """读取一个 episode，并再次检查文件内容是否符合 manifest。"""

        source = self.episode_dir(episode_id)
        manifest = read_episode_manifest(source / "manifest.json")
        with (source / "frame_timestamps.json").open("r", encoding="utf-8") as handle:
            timestamps = tuple(json.load(handle)["timestamps_ms"])
        actions = tuple(read_actions_jsonl(source / "actions.jsonl"))
        audit_path = source / "audit.json"
        audit = None
        if audit_path.exists():
            with audit_path.open("r", encoding="utf-8") as handle:
                audit = json.load(handle)
        if len(timestamps) != manifest.frame_count or len(actions) != manifest.action_count:
            raise ValueError(f"stored episode {episode_id} does not match its manifest")
        return StoredEpisode(manifest, timestamps, actions, audit)

    def list_manifests(self) -> Tuple[EpisodeManifest, ...]:
        if not self.episodes_dir.exists():
            return ()
        manifests = [
            read_episode_manifest(path)
            for path in sorted(self.episodes_dir.glob("*/manifest.json"))
        ]
        return tuple(manifests)

    def write_dataset_manifest(self) -> DatasetManifest:
        manifest = DatasetManifest(self.list_manifests())
        manifest.write(self.root / "dataset_manifest.json")
        return manifest

    def export_parquet(self, episode_id: str, path: Optional[Path] = None) -> Path:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Parquet export requires `pip install mcwm[parquet]`") from exc
        episode = self.read_episode(episode_id)
        rows = [action_to_dict(action) for action in episode.actions]
        table = pa.Table.from_pylist(rows)
        destination = Path(path) if path is not None else self.episode_dir(episode_id) / "actions.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
        return destination


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
