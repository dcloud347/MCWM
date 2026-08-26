"""从 M0 canonical episode store 中读取时序连续的视频 clip。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .alignment import align_actions_to_frames
from .dataset import (
    eligible_clip_start_ranges,
    random_clip_frame_indices_from_ranges,
)
from .episode_store import EpisodeStore
from .manifest import DatasetManifest, EpisodeManifest

SampleIndex = Union[int, Tuple[int, int]]


@dataclass(frozen=True)
class VisualEpisodeRef:
    """一个 episode 的轻量索引；每次访问时随机选择 clip。"""

    episode_id: str
    video_path: Path
    frame_timestamps_ms: Tuple[int, ...]
    clip_start_ranges: Tuple[Tuple[int, int, int], ...]


def _resolve_video_path(root: Path, manifest: EpisodeManifest) -> Path:
    """同时兼容相对数据集根目录和相对 episode 目录的视频路径。"""

    configured = Path(manifest.video_path)
    if configured.is_absolute():
        return configured
    candidates = (
        root / configured,
        root / "episodes" / manifest.episode_id / configured,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # 都不存在时返回标准位置，让后续报错直接显示用户应该检查的路径。
    return candidates[0]


def decode_frames_at_timestamps(path: Path, timestamps_ms: Sequence[int]) -> Tensor:
    """seek 到 clip 附近并按精确 PTS 解码，绝不静默 resize。"""

    try:
        import av  # type: ignore
    except ImportError as exc:
        raise RuntimeError("visual training requires `pip install mcwm[train]`") from exc
    frames: List[Tensor] = []
    desired = tuple(int(value) for value in timestamps_ms)
    if not desired:
        raise ValueError("at least one frame timestamp is required")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path}")
        stream = container.streams.video[0]
        # PyAV seek 使用 stream time-base；先退到关键帧，再逐帧匹配精确 PTS。
        seek_offset = int((desired[0] / 1000.0) / float(stream.time_base))
        container.seek(max(seek_offset, 0), stream=stream, backward=True)
        desired_index = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"decoded frame lacks PTS/time_base in {path}")
            timestamp = int(round(float(frame.pts * frame.time_base) * 1000.0))
            if timestamp < desired[desired_index]:
                continue
            if timestamp != desired[desired_index]:
                raise ValueError(
                    f"video {path} skipped expected PTS {desired[desired_index]} ms; got {timestamp} ms"
                )
            array = frame.to_ndarray(format="rgb24")
            if array.shape != (360, 640, 3):
                raise ValueError(f"decoded frame is {array.shape}; expected (360, 640, 3)")
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
            desired_index += 1
            if desired_index == len(desired):
                break
    if len(frames) != len(desired):
        raise ValueError(f"video {path} ended before all requested frame timestamps")
    return torch.stack(frames)


class CanonicalVisualDataset(Dataset):
    """直接由 canonical 视频支持的 map-style 训练/调试数据集。

    它保留最清楚的参考行为，也适合 validation。以后加入完整 training shard
    cache 后，大规模训练可以换数据后端，但模型收到的字段保持不变。
    """

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        clip_frames: int,
        sample_fps: int,
        seed: int = 0,
        clips_per_video: int = 1,
        include_probe_labels: bool = False,
        max_frame_gap_ms: int = 250,
    ) -> None:
        self.root = Path(root)
        self.clip_frames = int(clip_frames)
        self.sample_fps = int(sample_fps)
        self.seed = int(seed)
        self.clips_per_video = int(clips_per_video)
        self.include_probe_labels = bool(include_probe_labels)
        self.max_frame_gap_ms = int(max_frame_gap_ms)
        if self.clip_frames < 2:
            raise ValueError("clip_frames must be at least two")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.clips_per_video <= 0:
            raise ValueError("clips_per_video must be positive")
        manifest = DatasetManifest.read(self.root / "dataset_manifest.json")
        store = EpisodeStore(self.root)
        references: List[VisualEpisodeRef] = []
        for episode_manifest in manifest.episodes:
            if episode_manifest.split != split:
                continue
            timestamps = store.read_frame_timestamps(episode_manifest.episode_id)
            clip_start_ranges = eligible_clip_start_ranges(
                timestamps,
                clip_frames=self.clip_frames,
                sample_fps=self.sample_fps,
                max_frame_gap_ms=self.max_frame_gap_ms,
            )
            if not clip_start_ranges:
                continue
            references.append(
                VisualEpisodeRef(
                    episode_id=episode_manifest.episode_id,
                    video_path=_resolve_video_path(self.root, episode_manifest),
                    frame_timestamps_ms=timestamps,
                    clip_start_ranges=clip_start_ranges,
                )
            )
        if not references:
            raise ValueError(f"no {split!r} videos can provide a full visual clip")
        self.references = tuple(references)

    def __len__(self) -> int:
        return len(self.references) * self.clips_per_video

    def __getitem__(self, index: SampleIndex) -> Dict[str, object]:
        if isinstance(index, tuple):
            sample_index, clip_seed = index
        else:
            sample_index = int(index)
            clip_seed = self.seed + sample_index
        episode_index = sample_index // self.clips_per_video
        reference = self.references[episode_index]
        frame_indices = random_clip_frame_indices_from_ranges(
            reference.frame_timestamps_ms,
            reference.clip_start_ranges,
            clip_frames=self.clip_frames,
            sample_fps=self.sample_fps,
            generator=random.Random(clip_seed),
        )
        timestamps_ms = tuple(
            reference.frame_timestamps_ms[value] for value in frame_indices
        )
        frames = decode_frames_at_timestamps(
            reference.video_path,
            timestamps_ms,
        )
        scene_change = (frames[-1].float() - frames[0].float()).abs().mean().div(255.0)
        result: Dict[str, object] = {
            "frames": frames,
            "sample_id": (
                f"{reference.episode_id}:{frame_indices[0]}-{frame_indices[-1] + 1}"
                f"@{self.sample_fps}fps"
            ),
            "scene_change": scene_change,
        }
        if self.include_probe_labels:
            episode = EpisodeStore(self.root).read_episode(reference.episode_id)
            aligned = align_actions_to_frames(
                episode.frame_timestamps_ms,
                episode.actions,
                max_frame_gap_ms=self.max_frame_gap_ms,
            )
            action_blocks = aligned.blocks[frame_indices[0] : frame_indices[-1]]
            result["camera_motion"] = torch.tensor(
                sum(
                    math.hypot(*action.camera)
                    for block in action_blocks
                    for action in block.actions
                    if action.valid
                ),
                dtype=torch.float32,
            )
            result["gui_open"] = torch.tensor(
                int(
                    any(
                        action.gui_open
                        for block in action_blocks
                        for action in block.actions
                        if action.valid
                    )
                ),
                dtype=torch.long,
            )
        return result


def _seeded_index(index: int, *, seed: int, epoch: int, draw: int) -> Tuple[int, int]:
    """Attach a deterministic random-clip seed to one sampled video index."""

    clip_seed = (seed * 1_000_003 + epoch * 1_000_000_007 + draw) % (2**63 - 1)
    return index, clip_seed


class ResumableSampler(Sampler[SampleIndex]):
    """每个 epoch 都可复现，并能从 epoch 内精确位置恢复的 sampler。"""

    def __init__(
        self,
        length: int,
        *,
        seed: int,
        weights: torch.Tensor = None,
        seed_clips: bool = False,
    ) -> None:
        self.full_length = int(length)
        self.seed = int(seed)
        self.weights = weights
        self.seed_clips = bool(seed_clips)
        self.epoch = 0
        self.start_index = 0

    def __len__(self) -> int:
        return self.full_length - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.full_length:
            raise ValueError("sampler start_index is out of range")
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[SampleIndex]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.weights is None:
            indices = torch.randperm(self.full_length, generator=generator)
        else:
            indices = torch.multinomial(
                self.weights,
                self.full_length,
                replacement=True,
                generator=generator,
            )
        selected = indices[self.start_index :].tolist()
        if self.seed_clips:
            selected = [
                _seeded_index(
                    int(index),
                    seed=self.seed,
                    epoch=self.epoch,
                    draw=self.start_index + offset,
                )
                for offset, index in enumerate(selected)
            ]
        return iter(selected)


class DistributedResumableSampler(Sampler[SampleIndex]):
    """支持确定性 epoch 和精确 offset resume 的多卡随机 sampler。"""

    def __init__(
        self,
        length: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        seed_clips: bool = False,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        self.dataset_length = int(length)
        self.rank = rank
        self.world_size = world_size
        self.seed = int(seed)
        self.seed_clips = bool(seed_clips)
        self.full_length = math.ceil(self.dataset_length / world_size)
        self.epoch = 0
        self.start_index = 0

    def __len__(self) -> int:
        return self.full_length - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.full_length:
            raise ValueError("sampler start_index is out of range")
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[SampleIndex]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_length, generator=generator).tolist()
        total = self.full_length * self.world_size
        # 补齐到每个 rank 样本数一致，防止某张卡提前结束造成 collective 卡死。
        if len(indices) < total:
            indices.extend(indices[: total - len(indices)])
        per_rank = indices[self.rank:total:self.world_size]
        selected = per_rank[self.start_index :]
        if self.seed_clips:
            selected = [
                _seeded_index(
                    int(index),
                    seed=self.seed,
                    epoch=self.epoch,
                    draw=(self.start_index + offset) * self.world_size + self.rank,
                )
                for offset, index in enumerate(selected)
            ]
        return iter(selected)
