"""Metadata-only continuous clip sampling used before video decoding is added."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

from .alignment import ActionBlock, align_actions_to_frames
from .episode_store import StoredEpisode


@dataclass(frozen=True)
class ClipIndex:
    episode_id: str
    start_frame: int
    end_frame: int
    action_blocks: Tuple[ActionBlock, ...]


def iter_clip_indices(
    episode: StoredEpisode,
    *,
    transitions: int,
    stride: int = 1,
    max_frame_gap_ms: int = 250,
) -> Iterator[ClipIndex]:
    if transitions <= 0 or stride <= 0:
        raise ValueError("transitions and stride must be positive")
    aligned = align_actions_to_frames(
        episode.frame_timestamps_ms,
        episode.actions,
        max_frame_gap_ms=max_frame_gap_ms,
    )
    for range_start, range_end in aligned.continuous_frame_ranges:
        last_start = range_end - (transitions + 1)
        for start in range(range_start, last_start + 1, stride):
            end = start + transitions + 1
            yield ClipIndex(
                episode_id=episode.manifest.episode_id,
                start_frame=start,
                end_frame=end,
                action_blocks=aligned.blocks[start : end - 1],
            )

