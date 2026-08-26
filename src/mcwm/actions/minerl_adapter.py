"""把 MineRL 1.0 动作字典转换成项目统一格式。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from .schema import (
    INTERACTION_NAMES,
    MOVEMENT_NAMES,
    ActionSource,
    CanonicalActionTick,
)


# 这张表说明 MineRL 的每个按键应该放进统一动作的哪个字段。
# 例如 sneak=1 会变成 movement 里的 sneak=True。
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
    # MineRL 已经使用角度，不需要再换算鼠标移动距离。
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
    """把一条 MineRL 1.0 动作转换成统一格式。

    输入只是普通字典，因此离线处理数据时不需要启动 MineRL。
    """

    movement = {name: False for name in MOVEMENT_NAMES}
    interaction = {name: False for name in INTERACTION_NAMES}
    for raw_name, (group, canonical_name) in MINERL_BINARY_KEYS.items():
        value = bool(raw.get(raw_name, 0))
        (movement if group == "movement" else interaction)[canonical_name] = value

    # 快捷栏可能由 hotbar.1 等按键表示，也可能直接给出槽位数字。
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
