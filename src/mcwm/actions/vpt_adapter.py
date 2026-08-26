"""把 VPT contractor 的 JSONL 动作转换成项目统一格式。"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from .schema import (
    INTERACTION_NAMES,
    MOVEMENT_NAMES,
    ActionSource,
    CanonicalActionTick,
)


# VPT 记录的是鼠标 dx/dy，不是角度。官方数据约定每个原始单位等于 0.15 度。
CAMERA_SCALER = 360.0 / 2400.0

# 少数录制器版本在 GUI 打开时使用不同的鼠标比例，需要单独修正。
GUI_CAMERA_VERSION_SCALERS = {
    "5.7": 0.5,
    "5.8": 0.5,
    "6.7": 2.0,
    "6.8": 2.0,
    "6.9": 2.0,
}

# VPT JSONL 中保存的是 Minecraft 完整键名，这里统一成我们的短名称。
KEYBOARD_MAPPING = {
    "key.keyboard.w": "forward",
    "key.keyboard.s": "back",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.space": "jump",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.q": "drop",
    "key.keyboard.e": "inventory",
    "key.keyboard.escape": "esc",
    "key.keyboard.f": "swap_hands",
}
HOTBAR_KEY_MAPPING = {f"key.keyboard.{slot}": slot for slot in range(1, 10)}


def _clamp_camera(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("camera delta must be finite")
    return max(-180.0, min(180.0, result))


class VPTActionAdapter:
    """按顺序把 VPT JSONL 记录转换成统一动作。

    转换器需要记住上一条记录，才能修复快捷栏和攻击键。因此同一段视频要
    复用一个实例，并在开始下一段视频前调用 ``reset()``。
    """

    def __init__(
        self,
        *,
        recorder_version: str = "unknown",
        original_width: int = 1280,
        original_height: int = 720,
        initial_hotbar_zero_based: int = 0,
    ) -> None:
        if original_width <= 0 or original_height <= 0:
            raise ValueError("original dimensions must be positive")
        self.recorder_version = recorder_version
        self.original_width = original_width
        self.original_height = original_height
        self.initial_hotbar = initial_hotbar_zero_based
        self.repairs: List[str] = []
        self.unknown_keys: Set[str] = set()
        self.reset()

    def reset(self) -> None:
        """清空上一个 episode 留下的状态和审计记录。"""

        self._step_index = 0
        self._attack_stuck = False
        self._last_hotbar = self.initial_hotbar
        self.repairs.clear()
        self.unknown_keys.clear()

    def adapt(
        self,
        raw: Mapping[str, Any],
        *,
        timestamp_ms: Optional[int] = None,
    ) -> CanonicalActionTick:
        """转换一条 VPT 记录，并更新当前 episode 的状态。"""

        keyboard = raw.get("keyboard") or {}
        mouse = raw.get("mouse") or {}
        keys = list(keyboard.get("keys") or [])
        new_buttons = list(mouse.get("newButtons") or [])
        buttons = set(mouse.get("buttons") or [])

        # 部分文件开头会错误地显示左键一直按住。修复后留下记录，方便审计。
        if self._step_index == 0 and new_buttons == [0]:
            self._attack_stuck = True
            self.repairs.append("stuck_attack_detected_at_episode_start")
        elif self._attack_stuck and 0 in new_buttons:
            self._attack_stuck = False
            self.repairs.append(f"stuck_attack_released_at_step:{self._step_index}")
        if self._attack_stuck:
            buttons.discard(0)

        movement = {name: False for name in MOVEMENT_NAMES}
        interaction = {name: False for name in INTERACTION_NAMES}
        hotbar = 0
        for key in keys:
            mapped = KEYBOARD_MAPPING.get(key)
            if mapped in movement:
                movement[mapped] = True
            elif mapped in interaction:
                interaction[mapped] = True
            elif key in HOTBAR_KEY_MAPPING:
                hotbar = HOTBAR_KEY_MAPPING[key]
            elif mapped is None:
                self.unknown_keys.add(str(key))

        interaction["attack"] = 0 in buttons
        interaction["use"] = 1 in buttons
        interaction["pick_item"] = 2 in buttons

        # 有些记录漏掉了数字键事件，但 hotbar 状态已经变化。
        # 比较前后状态可以恢复这次切换。
        current_hotbar = int(raw.get("hotbar", self._last_hotbar))
        if not 0 <= current_hotbar <= 8:
            raise ValueError(f"VPT hotbar must be zero-based in 0..8, got {current_hotbar}")
        if current_hotbar != self._last_hotbar:
            hotbar = current_hotbar + 1
            self.repairs.append(f"hotbar_change_recovered_at_step:{self._step_index}")
        self._last_hotbar = current_hotbar

        gui_open = bool(raw.get("isGuiOpen", False))
        version_scale = (
            GUI_CAMERA_VERSION_SCALERS.get(self.recorder_version, 1.0) if gui_open else 1.0
        )
        pitch = _clamp_camera(float(mouse.get("dy", 0.0)) * CAMERA_SCALER * version_scale)
        yaw = _clamp_camera(float(mouse.get("dx", 0.0)) * CAMERA_SCALER * version_scale)

        # GUI 打开时保留归一化鼠标位置；普通第一人称控制不需要 cursor。
        cursor = None
        if gui_open and "x" in mouse and "y" in mouse:
            cursor = (
                max(0.0, min(1.0, float(mouse["x"]) / self.original_width)),
                max(0.0, min(1.0, float(mouse["y"]) / self.original_height)),
            )

        resolved_timestamp = timestamp_ms
        if resolved_timestamp is None:
            resolved_timestamp = raw.get("milli")
        if resolved_timestamp is None and raw.get("tick") is not None:
            resolved_timestamp = int(raw["tick"]) * 50
        if resolved_timestamp is None:
            raise ValueError("VPT action needs milli, tick, or an explicit timestamp_ms")

        action = CanonicalActionTick(
            movement=tuple(movement[name] for name in MOVEMENT_NAMES),
            interaction=tuple(interaction[name] for name in INTERACTION_NAMES),
            hotbar=hotbar,
            camera=(pitch, yaw),
            cursor=cursor,
            gui_open=gui_open,
            valid=True,
            timestamp_ms=int(resolved_timestamp),
            source=ActionSource.VPT,
            label_confidence=1.0,
        )
        self._step_index += 1
        return action

    def adapt_many(self, rows: Iterable[Mapping[str, Any]]) -> List[CanonicalActionTick]:
        """按输入顺序转换多条 VPT 记录。"""

        return [self.adapt(row) for row in rows]
