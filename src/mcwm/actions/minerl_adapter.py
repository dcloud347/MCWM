"""Adapter for the near-human MineRL 1.0 environment action dictionary."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from .schema import (
    INTERACTION_NAMES,
    MOVEMENT_NAMES,
    ActionSource,
    CanonicalActionTick,
)


# MineRL 使用 dict 表示动作，这张表说明每个键要写进 canonical action 的哪一组。
# 例如 raw["sneak"] == 1 会变成 movement 里的 sneak=True。
MINERL_BINARY_KEYS = {
    "forward": ("movement", "forward"),
    "back": ("movement", "back"),
    "left": ("movement", "left"),
    "right": ("movement", "right"),
    "jump": ("movement", "jump"),
    "sneak": ("movement", "sneak"),
    "sprint": ("movement", "sprint"),
    "attack": ("interaction", "attack"),
    "use": ("interaction", "use"),
    "drop": ("interaction", "drop"),
    "pickItem": ("interaction", "pick_item"),
    "swapHands": ("interaction", "swap_hands"),
    "inventory": ("interaction", "inventory"),
    "ESC": ("interaction", "esc"),
}


def _camera(values: Sequence[object]) -> Tuple[float, float]:
    # MineRL 已经给出角度增量，不需要像 VPT 原始鼠标位移那样乘缩放系数。
    if len(values) != 2:
        raise ValueError("MineRL camera must be [pitch_delta, yaw_delta]")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("MineRL camera values must be finite")
    return tuple(max(-180.0, min(180.0, value)) for value in result)  # type: ignore[return-value]


def minerl_action_to_canonical(
    raw: Mapping[str, Any],
    *,
    timestamp_ms: Optional[int] = None,
    gui_open: bool = False,
    cursor: Optional[Tuple[float, float]] = None,
    label_confidence: float = 1.0,
) -> CanonicalActionTick:
    """把一条 MineRL 1.0 action dict 转成统一格式。

    这里只处理普通 Python dict，因此做离线数据预处理时不需要启动 MineRL。
    """

    movement = {name: False for name in MOVEMENT_NAMES}
    interaction = {name: False for name in INTERACTION_NAMES}
    for raw_name, (group, canonical_name) in MINERL_BINARY_KEYS.items():
        value = bool(raw.get(raw_name, 0))
        (movement if group == "movement" else interaction)[canonical_name] = value

    # MineRL 数据可能使用 hotbar.1 ... hotbar.9，也可能直接给 hotbar 数字。
    selected_slots = [
        slot for slot in range(1, 10) if bool(raw.get(f"hotbar.{slot}", 0))
    ]
    if "hotbar" in raw and int(raw["hotbar"]) != 0:
        selected_slots.append(int(raw["hotbar"]))
    if len(set(selected_slots)) > 1:
        raise ValueError(f"multiple hotbar slots selected in one tick: {selected_slots}")
    hotbar = selected_slots[0] if selected_slots else 0

    resolved_timestamp = timestamp_ms
    if resolved_timestamp is None:
        resolved_timestamp = raw.get("timestamp_ms", raw.get("milli"))
    if resolved_timestamp is None:
        raise ValueError("MineRL action needs an explicit timestamp_ms")

    return CanonicalActionTick(
        movement=tuple(movement[name] for name in MOVEMENT_NAMES),
        interaction=tuple(interaction[name] for name in INTERACTION_NAMES),
        hotbar=hotbar,
        camera=_camera(raw.get("camera", (0.0, 0.0))),
        cursor=cursor,
        gui_open=gui_open,
        valid=True,
        timestamp_ms=int(resolved_timestamp),
        source=ActionSource.MINERL,
        label_confidence=label_confidence,
    )
