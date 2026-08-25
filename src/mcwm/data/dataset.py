"""Metadata-only random clip sampling following the V-JEPA data contract."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
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
    sample_fps: int,
    max_frame_gap_ms: int = 250,
) -> Tuple[Tuple[int, int, int], ...]:
    """Return compact start ranges for timestamp-based fixed-FPS clips.

    Each entry is ``(first_start, start_count, continuous_range_end)``. A start
    is eligible when all ``clip_frames`` sample instants at ``sample_fps`` fit
    inside the same continuous range.
    """

    if clip_frames < 2:
        raise ValueError("clip_frames must be at least two")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    timestamps = tuple(int(value) for value in frame_timestamps_ms)
    eligible_ranges: List[Tuple[int, int, int]] = []
    for range_start, range_end in continuous_frame_ranges(
        timestamps,
        max_frame_gap_ms=max_frame_gap_ms,
    ):
        # Compare the requested duration against PTS rather than assuming a
        # fixed source-video frame rate.
        maximum_start_numerator = (
            timestamps[range_end - 1] * sample_fps
            - (clip_frames - 1) * 1000
        )
        maximum_start_ms = maximum_start_numerator / sample_fps
        start_end = bisect_right(
            timestamps,
            maximum_start_ms,
            lo=range_start,
            hi=range_end,
        )
        start_count = start_end - range_start
        if start_count <= 0:
            continue
        eligible_ranges.append((range_start, start_count, range_end))
    return tuple(eligible_ranges)


def _clip_frame_indices_at_fps(
    frame_timestamps_ms: Sequence[int],
    *,
    start: int,
    range_end: int,
    clip_frames: int,
    sample_fps: int,
) -> Tuple[int, ...]:
    """Select the source frame nearest each fixed-FPS sample instant."""

    timestamps = frame_timestamps_ms
    selected = [start]
    start_timestamp = int(timestamps[start])
    for offset in range(1, clip_frames):
        target_numerator = start_timestamp * sample_fps + offset * 1000
        target_ms = target_numerator / sample_fps
        lower_bound = selected[-1] + 1
        insertion = bisect_left(
            timestamps,
            target_ms,
            lo=lower_bound,
            hi=range_end,
        )
        candidates = []
        if insertion < range_end:
            candidates.append(insertion)
        if insertion - 1 >= lower_bound:
            candidates.append(insertion - 1)
        if not candidates:
            raise ValueError("continuous range cannot provide the requested sample_fps")
        index = min(
            candidates,
            key=lambda value: (
                abs(int(timestamps[value]) * sample_fps - target_numerator),
                value,
            ),
        )
        timestamp_error = abs(
            int(timestamps[index]) * sample_fps - target_numerator
        )
        if timestamp_error > 500:
            raise ValueError("source timestamps are too sparse for the requested sample_fps")
        selected.append(index)
    return tuple(selected)


def random_clip_frame_indices_from_ranges(
    frame_timestamps_ms: Sequence[int],
    eligible_ranges: Sequence[Tuple[int, int, int]],
    *,
    clip_frames: int,
    sample_fps: int,
    generator: random.Random,
) -> Tuple[int, ...]:
    """Uniformly sample a start from precomputed eligible ranges."""

    total_starts = sum(start_count for _, start_count, _ in eligible_ranges)
    if total_starts <= 0:
        raise ValueError("video has no continuous range long enough for one clip")
    selected_start = generator.randrange(total_starts)
    for range_start, start_count, range_end in eligible_ranges:
        if selected_start < start_count:
            start = range_start + selected_start
            return _clip_frame_indices_at_fps(
                frame_timestamps_ms,
                start=start,
                range_end=range_end,
                clip_frames=clip_frames,
                sample_fps=sample_fps,
            )
        selected_start -= start_count
    raise AssertionError("random clip start selection fell outside eligible ranges")


def random_clip_frame_indices(
    frame_timestamps_ms: Sequence[int],
    *,
    clip_frames: int,
    sample_fps: int,
    generator: random.Random,
    max_frame_gap_ms: int = 250,
) -> Tuple[int, ...]:
    """Uniformly sample one start, then select frames at ``sample_fps`` by PTS."""

    eligible_ranges = eligible_clip_start_ranges(
        frame_timestamps_ms,
        clip_frames=clip_frames,
        sample_fps=sample_fps,
        max_frame_gap_ms=max_frame_gap_ms,
    )
    return random_clip_frame_indices_from_ranges(
        frame_timestamps_ms,
        eligible_ranges,
        clip_frames=clip_frames,
        sample_fps=sample_fps,
        generator=generator,
    )
