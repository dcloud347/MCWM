"""从 M0 canonical episode store 中读取时序连续的视频 clip。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .dataset import iter_clip_indices
from .episode_store import EpisodeStore
from .manifest import DatasetManifest, EpisodeManifest


@dataclass(frozen=True)
class VisualClipRef:
    """一个 clip 的轻量索引；真正取样时才解码视频。"""

    episode_id: str
    video_path: Path
    start_frame: int
    end_frame: int
    frame_timestamps_ms: Tuple[int, ...]
    source: str
    camera_motion: float
    gui_open: int

    @property
    def sample_id(self) -> str:
        return f"{self.episode_id}:{self.start_frame}-{self.end_frame}"


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
        stride: int = 1,
        max_frame_gap_ms: int = 250,
    ) -> None:
        self.root = Path(root)
        self.clip_frames = int(clip_frames)
        if self.clip_frames < 2:
            raise ValueError("clip_frames must be at least two")
        manifest = DatasetManifest.read(self.root / "dataset_manifest.json")
        store = EpisodeStore(self.root)
        references: List[VisualClipRef] = []
        for episode_manifest in manifest.episodes:
            if episode_manifest.split != split:
                continue
            episode = store.read_episode(episode_manifest.episode_id)
            for clip in iter_clip_indices(
                episode,
                transitions=self.clip_frames - 1,
                stride=stride,
                max_frame_gap_ms=max_frame_gap_ms,
            ):
                references.append(
                    VisualClipRef(
                        episode_id=episode_manifest.episode_id,
                        video_path=_resolve_video_path(self.root, episode_manifest),
                        start_frame=clip.start_frame,
                        end_frame=clip.end_frame,
                        frame_timestamps_ms=episode.frame_timestamps_ms[
                            clip.start_frame : clip.end_frame
                        ],
                        source=episode_manifest.source.value,
                        camera_motion=sum(
                            math.hypot(*action.camera)
                            for block in clip.action_blocks
                            for action in block.actions
                            if action.valid
                        ),
                        gui_open=int(
                            any(
                                action.gui_open
                                for block in clip.action_blocks
                                for action in block.actions
                                if action.valid
                            )
                        ),
                    )
                )
        if not references:
            raise ValueError(f"no {split!r} visual clips found under {self.root}")
        self.references = tuple(references)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> Dict[str, object]:
        reference = self.references[index]
        frames = decode_frames_at_timestamps(
            reference.video_path,
            reference.frame_timestamps_ms,
        )
        # 这些标签只给离线 probe 使用，不会作为 visual encoder 的输入。
        scene_change = (frames[-1].float() - frames[0].float()).abs().mean().div(255.0)
        return {
            "frames": frames,
            "sample_id": reference.sample_id,
            "source": reference.source,
            "camera_motion": torch.tensor(reference.camera_motion, dtype=torch.float32),
            "gui_open": torch.tensor(reference.gui_open, dtype=torch.long),
            "scene_change": scene_change,
        }



class ResumableSampler(Sampler[int]):
    """每个 epoch 都可复现，并能从 epoch 内精确位置恢复的 sampler。"""

    def __init__(self, length: int, *, seed: int, weights: torch.Tensor = None) -> None:
        self.full_length = int(length)
        self.seed = int(seed)
        self.weights = weights
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

    def __iter__(self) -> Iterator[int]:
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
        return iter(indices[self.start_index :].tolist())


def source_balanced_weights(dataset: CanonicalVisualDataset) -> torch.Tensor:
    counts = Counter(reference.source for reference in dataset.references)
    return torch.tensor(
        [1.0 / counts[reference.source] for reference in dataset.references],
        dtype=torch.double,
    )


class DistributedSourceBalancedSampler(Sampler[int]):
    """先让两种数据源期望采样量相等，再把样本分给不同 rank。"""

    def __init__(
        self,
        dataset: CanonicalVisualDataset,
        *,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        self.weights = source_balanced_weights(dataset)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.samples_per_rank = math.ceil(len(dataset) / world_size)
        self.full_length = self.samples_per_rank
        self.start_index = 0

    def __len__(self) -> int:
        return self.full_length - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.full_length:
            raise ValueError("sampler start_index is out of range")
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.samples_per_rank * self.world_size,
            replacement=True,
            generator=generator,
        )
        per_rank = indices[self.rank :: self.world_size]
        return iter(per_rank[self.start_index :].tolist())


class DistributedResumableSampler(Sampler[int]):
    """支持确定性 epoch 和精确 offset resume 的多卡随机 sampler。"""

    def __init__(self, length: int, *, rank: int, world_size: int, seed: int) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        self.dataset_length = int(length)
        self.rank = rank
        self.world_size = world_size
        self.seed = int(seed)
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

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_length, generator=generator).tolist()
        total = self.full_length * self.world_size
        # 补齐到每个 rank 样本数一致，防止某张卡提前结束造成 collective 卡死。
        if len(indices) < total:
            indices.extend(indices[: total - len(indices)])
        per_rank = indices[self.rank:total:self.world_size]
        return iter(per_rank[self.start_index :])
