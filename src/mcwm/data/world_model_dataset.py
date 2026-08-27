"""为 V-JEPA 2-AC 读取逐帧 latent prediction 所需的数据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mcwm.actions.schema import CanonicalActionTick, ActionSource
from .alignment import ActionBlock, align_actions_to_frames
from .dataset import eligible_clip_start_ranges, random_clip_frame_indices_from_ranges
from .episode_store import EpisodeStore
from .manifest import DatasetManifest, EpisodeManifest
from .visual_dataset import decode_frames_at_timestamps


SampleIndex = Union[int, Tuple[int, int]]


@dataclass(frozen=True)
class WorldModelEpisodeRef:
    """一段可用于动作条件训练的 VPT episode 索引。"""

    episode_id: str
    video_path: Path
    frame_timestamps_ms: Tuple[int, ...]
    aligned_blocks: Tuple[ActionBlock, ...]
    clip_start_ranges: Tuple[Tuple[int, int, int], ...]


def _resolve_video_path(root: Path, manifest: EpisodeManifest) -> Path:
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
    return candidates[0]


def _actions_between_sampled_frames(
    aligned_blocks: Sequence[ActionBlock],
    frame_indices: Sequence[int],
) -> Tuple[Tuple[CanonicalActionTick, ...], ...]:
    """合并相邻采样帧之间的全部原始 action ticks。"""

    result = []
    for start, end in zip(frame_indices, frame_indices[1:]):
        if end <= start:
            raise ValueError("sampled frame indices must be strictly increasing")
        source_blocks = aligned_blocks[start:end]
        if len(source_blocks) != end - start or not all(
            block.continuous for block in source_blocks
        ):
            raise ValueError("sampled transition crosses a discontinuity")
        actions = tuple(
            action
            for block in source_blocks
            for action in block.actions
        )
        if not actions:
            raise ValueError("sampled transition has no action labels")
        if not all(action.valid for action in actions):
            raise ValueError("sampled transition contains invalid action labels")
        if not all(action.source is ActionSource.VPT for action in actions):
            raise ValueError("sampled transition contains non-VPT action labels")
        result.append(actions)
    return tuple(result)


class WorldModelDataset(Dataset):
    """返回 8 帧画面和帧间严格对齐的 7 个变长动作块。"""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        frames_per_sample: int = 8,
        sample_fps: int = 4,
        seed: int = 0,
        samples_per_video: int = 1,
        max_frame_gap_ms: int = 250,
    ) -> None:
        self.root = Path(root)
        self.frames_per_sample = int(frames_per_sample)
        self.sample_fps = int(sample_fps)
        self.seed = int(seed)
        self.samples_per_video = int(samples_per_video)
        self.max_frame_gap_ms = int(max_frame_gap_ms)
        if self.frames_per_sample < 2:
            raise ValueError("frames_per_sample must be at least two")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.samples_per_video <= 0:
            raise ValueError("samples_per_video must be positive")

        manifest = DatasetManifest.read(self.root / "dataset_manifest.json")
        selected = tuple(item for item in manifest.episodes if item.split == split)
        if any(item.source is not ActionSource.VPT for item in selected):
            raise ValueError("world-model training only accepts VPT contractor episodes")
        store = EpisodeStore(self.root)
        references: List[WorldModelEpisodeRef] = []
        for episode_manifest in selected:
            episode = store.read_episode(episode_manifest.episode_id)
            aligned = align_actions_to_frames(
                episode.frame_timestamps_ms,
                episode.actions,
                max_frame_gap_ms=self.max_frame_gap_ms,
            )
            start_ranges = eligible_clip_start_ranges(
                episode.frame_timestamps_ms,
                clip_frames=self.frames_per_sample,
                sample_fps=self.sample_fps,
                max_frame_gap_ms=self.max_frame_gap_ms,
            )
            if not start_ranges:
                continue
            references.append(
                WorldModelEpisodeRef(
                    episode_id=episode_manifest.episode_id,
                    video_path=_resolve_video_path(self.root, episode_manifest),
                    frame_timestamps_ms=episode.frame_timestamps_ms,
                    aligned_blocks=aligned.blocks,
                    clip_start_ranges=start_ranges,
                )
            )
        if not references:
            raise ValueError(f"no {split!r} VPT episodes can provide a full world-model sample")
        self.references = tuple(references)

    def __len__(self) -> int:
        return len(self.references) * self.samples_per_video

    def __getitem__(self, index: SampleIndex) -> Dict[str, object]:
        if isinstance(index, tuple):
            sample_index, sample_seed = index
        else:
            sample_index = int(index)
            sample_seed = self.seed + sample_index
        reference = self.references[sample_index // self.samples_per_video]
        frame_indices = random_clip_frame_indices_from_ranges(
            reference.frame_timestamps_ms,
            reference.clip_start_ranges,
            clip_frames=self.frames_per_sample,
            sample_fps=self.sample_fps,
            generator=random.Random(sample_seed),
        )
        timestamps_ms = tuple(reference.frame_timestamps_ms[index] for index in frame_indices)
        action_blocks = _actions_between_sampled_frames(
            reference.aligned_blocks,
            frame_indices,
        )
        frames = decode_frames_at_timestamps(reference.video_path, timestamps_ms)
        return {
            "frames": frames,
            "frame_timestamps_ms": torch.tensor(timestamps_ms, dtype=torch.int64),
            "action_blocks": action_blocks,
            "sample_id": (
                f"{reference.episode_id}:{frame_indices[0]}-{frame_indices[-1] + 1}"
                f"@{self.sample_fps}fps"
            ),
        }


def collate_world_model_samples(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """把变长 action blocks 补齐为显式带 valid mask 的规则张量。"""

    if not samples:
        raise ValueError("samples must not be empty")
    frames = torch.stack([sample["frames"] for sample in samples])
    frame_timestamps = torch.stack(
        [sample["frame_timestamps_ms"] for sample in samples]
    )
    transitions = frames.shape[1] - 1
    blocks_by_sample = [sample["action_blocks"] for sample in samples]
    if any(len(blocks) != transitions for blocks in blocks_by_sample):
        raise ValueError("each sample must contain one action block per frame transition")
    max_ticks = max(
        len(block)
        for blocks in blocks_by_sample
        for block in blocks
    )
    if max_ticks <= 0:
        raise ValueError("action blocks must contain labeled ticks")

    batch = len(samples)
    movement = torch.zeros(batch, transitions, max_ticks, 7, dtype=torch.bool)
    interaction = torch.zeros(batch, transitions, max_ticks, 7, dtype=torch.bool)
    hotbar = torch.zeros(batch, transitions, max_ticks, dtype=torch.long)
    camera = torch.zeros(batch, transitions, max_ticks, 2, dtype=torch.float32)
    cursor = torch.zeros(batch, transitions, max_ticks, 2, dtype=torch.float32)
    cursor_present = torch.zeros(batch, transitions, max_ticks, dtype=torch.bool)
    gui_open = torch.zeros(batch, transitions, max_ticks, dtype=torch.bool)
    valid_mask = torch.zeros(batch, transitions, max_ticks, dtype=torch.bool)
    action_timestamps = torch.zeros(batch, transitions, max_ticks, dtype=torch.int64)
    label_confidence = torch.zeros(batch, transitions, max_ticks, dtype=torch.float32)

    for batch_index, blocks in enumerate(blocks_by_sample):
        for transition_index, block in enumerate(blocks):
            for tick_index, action in enumerate(block):
                movement[batch_index, transition_index, tick_index] = torch.tensor(
                    action.movement, dtype=torch.bool
                )
                interaction[batch_index, transition_index, tick_index] = torch.tensor(
                    action.interaction, dtype=torch.bool
                )
                hotbar[batch_index, transition_index, tick_index] = action.hotbar
                camera[batch_index, transition_index, tick_index] = torch.tensor(
                    action.camera, dtype=torch.float32
                )
                if action.cursor is not None:
                    cursor[batch_index, transition_index, tick_index] = torch.tensor(
                        action.cursor, dtype=torch.float32
                    )
                    cursor_present[batch_index, transition_index, tick_index] = True
                gui_open[batch_index, transition_index, tick_index] = action.gui_open
                valid_mask[batch_index, transition_index, tick_index] = action.valid
                action_timestamps[batch_index, transition_index, tick_index] = (
                    action.timestamp_ms
                )
                label_confidence[batch_index, transition_index, tick_index] = (
                    action.label_confidence
                )

    return {
        "frames": frames,
        "frame_timestamps_ms": frame_timestamps,
        "movement": movement,
        "interaction": interaction,
        "hotbar": hotbar,
        "camera": camera,
        "cursor": cursor,
        "cursor_present": cursor_present,
        "gui_open": gui_open,
        "valid_mask": valid_mask,
        "action_timestamps_ms": action_timestamps,
        "label_confidence": label_confidence,
        "sample_id": [sample["sample_id"] for sample in samples],
    }
