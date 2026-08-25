"""Metadata-only random clip sampling following the V-JEPA data contract."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def continuous_frame_ranges(
    frame_timestamps_ms: Sequence[int],
    *,
    max_frame_gap_ms: int = 250,
) -> Tuple[Tuple[int, int], ...]:
    """Return half-open frame ranges separated by timestamp discontinuities."""

    timestamps = tuple(int(value) for value in frame_timestamps_ms)
    if len(timestamps) < 2:
        raise ValueError("frame_timestamps_ms must contain at least two timestamps")
    if max_frame_gap_ms <= 0:
        raise ValueError("max_frame_gap_ms must be positive")
    ranges: List[Tuple[int, int]] = []
    range_start = 0
    for index, (previous, current) in enumerate(zip(timestamps, timestamps[1:])):
        if current <= previous:
            raise ValueError("frame_timestamps_ms must be strictly increasing")
        if current - previous > max_frame_gap_ms:
            if index + 1 - range_start >= 2:
                ranges.append((range_start, index + 1))
            range_start = index + 1
    if len(timestamps) - range_start >= 2:
        ranges.append((range_start, len(timestamps)))
    return tuple(ranges)


def eligible_clip_start_ranges(
    frame_timestamps_ms: Sequence[int],
    *,
    clip_frames: int,
    sampling_rate: int,
    max_frame_gap_ms: int = 250,
) -> Tuple[Tuple[int, int], ...]:
    """Return ``(first_start, start_count)`` for every eligible continuous range."""

    if clip_frames < 2:
        raise ValueError("clip_frames must be at least two")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    source_span = (clip_frames - 1) * sampling_rate + 1
    eligible_ranges: List[Tuple[int, int]] = []
    for range_start, range_end in continuous_frame_ranges(
        frame_timestamps_ms,
        max_frame_gap_ms=max_frame_gap_ms,
    ):
        start_count = range_end - range_start - source_span + 1
        if start_count <= 0:
            continue
        eligible_ranges.append((range_start, start_count))
    return tuple(eligible_ranges)


def random_clip_frame_indices_from_ranges(
    eligible_ranges: Sequence[Tuple[int, int]],
    *,
    clip_frames: int,
    sampling_rate: int,
    generator: random.Random,
) -> Tuple[int, ...]:
    """Uniformly sample a start from precomputed eligible ranges."""

    total_starts = sum(start_count for _, start_count in eligible_ranges)
    if total_starts <= 0:
        raise ValueError("video has no continuous range long enough for one clip")
    selected_start = generator.randrange(total_starts)
    for range_start, start_count in eligible_ranges:
        if selected_start < start_count:
            start = range_start + selected_start
            return tuple(start + offset * sampling_rate for offset in range(clip_frames))
        selected_start -= start_count
    raise AssertionError("random clip start selection fell outside eligible ranges")


def random_clip_frame_indices(
    frame_timestamps_ms: Sequence[int],
    *,
    clip_frames: int,
    sampling_rate: int,
    generator: random.Random,
    max_frame_gap_ms: int = 250,
) -> Tuple[int, ...]:
    """Uniformly sample one valid clip start, then stride through source frames."""

    eligible_ranges = eligible_clip_start_ranges(
        frame_timestamps_ms,
        clip_frames=clip_frames,
        sampling_rate=sampling_rate,
        max_frame_gap_ms=max_frame_gap_ms,
    )
    return random_clip_frame_indices_from_ranges(
        eligible_ranges,
        clip_frames=clip_frames,
        sampling_rate=sampling_rate,
        generator=generator,
    )
