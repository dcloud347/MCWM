"""Align action timestamps to adjacent video frame PTS boundaries."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from mcwm.actions.schema import CanonicalActionTick


@dataclass(frozen=True)
class ActionBlock:
    """相邻两帧之间发生的全部动作，即区间 ``[start_ms, end_ms)``。"""

    frame_index: int
    start_ms: int
    end_ms: int
    actions: Tuple[CanonicalActionTick, ...]
    continuous: bool


@dataclass(frozen=True)
class AlignmentResult:
    blocks: Tuple[ActionBlock, ...]
    actions_before_first_frame: Tuple[CanonicalActionTick, ...]
    actions_at_or_after_last_frame: Tuple[CanonicalActionTick, ...]

    @property
    def continuous_frame_ranges(self) -> Tuple[Tuple[int, int], ...]:
        """返回没有时间断点的 ``[起始帧, 结束帧)``，训练 clip 只能从中采样。"""

        if not self.blocks:
            return ()
        ranges: List[Tuple[int, int]] = []
        start_frame = 0
        for block in self.blocks:
            if not block.continuous:
                if block.frame_index + 1 - start_frame >= 2:
                    ranges.append((start_frame, block.frame_index + 1))
                start_frame = block.frame_index + 1
        final_end = len(self.blocks) + 1
        if final_end - start_frame >= 2:
            ranges.append((start_frame, final_end))
        return tuple(ranges)


def _validate_strictly_increasing(values: Sequence[int], name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    for previous, current in zip(values, values[1:]):
        if current <= previous:
            raise ValueError(f"{name} must be strictly increasing: {previous} then {current}")


def align_actions_to_frames(
    frame_timestamps_ms: Sequence[int],
    actions: Iterable[CanonicalActionTick],
    *,
    max_frame_gap_ms: int = 250,
) -> AlignmentResult:
    """按时间戳把动作分配给 ``frame[t] -> frame[t+1]``。

    动作时间如果正好等于 ``frame[t+1]``，它属于下一个区间。这个半开区间
    规则可以避免同一个动作被重复分配，也避免 VPT 和 MineRL 出现一帧偏差。
    """

    frames = tuple(int(value) for value in frame_timestamps_ms)
    _validate_strictly_increasing(frames, "frame_timestamps_ms")
    if max_frame_gap_ms <= 0:
        raise ValueError("max_frame_gap_ms must be positive")

    sorted_actions = tuple(actions)
    action_times = tuple(action.timestamp_ms for action in sorted_actions)
    if any(current < previous for previous, current in zip(action_times, action_times[1:])):
        raise ValueError("actions must be sorted by timestamp_ms")

    per_block: List[List[CanonicalActionTick]] = [[] for _ in range(len(frames) - 1)]
    before: List[CanonicalActionTick] = []
    after: List[CanonicalActionTick] = []
    for action in sorted_actions:
        if action.timestamp_ms < frames[0]:
            before.append(action)
            continue
        # bisect_right 保证落在右边界上的动作进入下一个 block。
        interval = bisect_right(frames, action.timestamp_ms) - 1
        if interval >= len(frames) - 1:
            after.append(action)
            continue
        per_block[interval].append(action)

    blocks = tuple(
        ActionBlock(
            frame_index=index,
            start_ms=frames[index],
            end_ms=frames[index + 1],
            actions=tuple(block_actions),
            continuous=(frames[index + 1] - frames[index]) <= max_frame_gap_ms,
        )
        for index, block_actions in enumerate(per_block)
    )
    return AlignmentResult(blocks, tuple(before), tuple(after))
