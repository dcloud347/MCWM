"""记录数据集索引，并安全地划分训练、验证和测试集。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mcwm.actions.schema import ActionSource


MANIFEST_SCHEMA_VERSION = 1
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class EpisodeManifest:
    """一个 episode 的索引信息；视频本身仍由 ``video_path`` 指向。"""

    episode_id: str
    session_id: str
    world_id: str
    source: ActionSource
    recorder_version: str
    video_path: str
    width: int
    height: int
    frame_count: int
    action_count: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    split: Optional[str] = None
    video_sha256: Optional[str] = None
    action_sha256: Optional[str] = None
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ActionSource(self.source))
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version: {self.schema_version}")
        for name in ("episode_id", "session_id", "world_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if (self.width, self.height) != (640, 360):
            raise ValueError("MCWM episode resolution must be exactly 640x360")
        if self.frame_count < 2:
            raise ValueError("an episode must contain at least two frames")
        if self.action_count < 0:
            raise ValueError("action_count cannot be negative")
        if self.end_timestamp_ms <= self.start_timestamp_ms:
            raise ValueError("end_timestamp_ms must be greater than start_timestamp_ms")
        if self.split is not None and self.split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {self.split}")

    @property
    def duration_ms(self) -> int:
        """返回 episode 持续的毫秒数。"""

        return self.end_timestamp_ms - self.start_timestamp_ms

    def to_dict(self) -> Dict[str, Any]:
        """转换成可写入 JSON 的字典。"""

        data = asdict(self)
        data["source"] = self.source.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeManifest":
        """从字典创建 episode manifest。"""

        return cls(**dict(data))


@dataclass(frozen=True)
class DatasetManifest:
    """整个数据集的 episode 列表和格式版本。"""

    episodes: Tuple[EpisodeManifest, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset manifest version: {self.schema_version}")
        ids = [episode.episode_id for episode in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("episode_id values must be unique")

    @property
    def content_hash(self) -> str:
        """返回内容哈希，用来确认训练前后数据集没有变化。"""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """转换成可写入 JSON 的字典。"""

        return {
            "schema_version": self.schema_version,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        """从字典创建数据集 manifest。"""

        return cls(
            episodes=tuple(EpisodeManifest.from_dict(item) for item in data["episodes"]),
            schema_version=data.get("schema_version", MANIFEST_SCHEMA_VERSION),
        )

    def write(self, path: Path) -> None:
        """安全地写入数据集 manifest。"""

        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "DatasetManifest":
        """从 JSON 文件读取数据集 manifest。"""

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _atomic_json_write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _connected_groups(episodes: Sequence[EpisodeManifest]) -> List[List[EpisodeManifest]]:
    """把相同 session 或 world 的 episode 放在一组，避免数据泄漏。"""

    groups: List[List[EpisodeManifest]] = []
    for episode in episodes:
        matching = [
            index
            for index, group in enumerate(groups)
            if any(
                member.session_id == episode.session_id or member.world_id == episode.world_id
                for member in group
            )
        ]
        if not matching:
            groups.append([episode])
            continue
        merged = [episode]
        for index in reversed(matching):
            merged.extend(groups.pop(index))
        groups.append(merged)
    return groups


def assign_splits(
    episodes: Sequence[EpisodeManifest],
    *,
    train: float = 0.90,
    validation: float = 0.05,
    test: float = 0.05,
    seed: str = "mcwm-v1",
) -> Tuple[EpisodeManifest, ...]:
    """按关联组切分数据，保证相关 episode 不会出现在不同集合。"""

    ratios = (train, validation, test)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1")

    groups = _connected_groups(episodes)
    groups.sort(
        key=lambda group: sha256(
            f"{seed}:{min(item.episode_id for item in group)}".encode("utf-8")
        ).hexdigest()
    )
    count = len(groups)
    raw_counts = [count * ratio for ratio in ratios]
    split_counts = [int(value) for value in raw_counts]
    for index in sorted(
        range(3), key=lambda item: (raw_counts[item] - split_counts[item], -item), reverse=True
    )[: count - sum(split_counts)]:
        split_counts[index] += 1
    train_end = split_counts[0]
    validation_end = train_end + split_counts[1]
    assigned: List[EpisodeManifest] = []
    for index, group in enumerate(groups):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        assigned.extend(replace(episode, split=split) for episode in group)
    return tuple(sorted(assigned, key=lambda episode: episode.episode_id))


def find_split_leakage(episodes: Iterable[EpisodeManifest]) -> Tuple[str, ...]:
    """检查训练、验证和测试集是否共享 session 或 world。"""

    issues: List[str] = []
    for field in ("session_id", "world_id"):
        seen: Dict[str, set] = {}
        for episode in episodes:
            if episode.split is None:
                continue
            seen.setdefault(getattr(episode, field), set()).add(episode.split)
        for value, splits in sorted(seen.items()):
            if len(splits) > 1:
                issues.append(f"{field}={value} appears in splits {sorted(splits)}")
    return tuple(issues)


def write_episode_manifest(path: Path, manifest: EpisodeManifest) -> None:
    """安全地写入 episode manifest。"""

    _atomic_json_write(Path(path), manifest.to_dict())


def read_episode_manifest(path: Path) -> EpisodeManifest:
    """从 JSON 文件读取 episode manifest。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        return EpisodeManifest.from_dict(json.load(handle))
