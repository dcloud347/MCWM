"""Source-independent action contract used by MCWM datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Optional, Sequence, Tuple


# movement 中的 bool 必须严格按照这个顺序保存。
# True 表示这个 tick 正在按住对应按键，不是“刚刚按下了一次”。
MOVEMENT_NAMES: Tuple[str, ...] = (
    "forward",  # 按住 W：向前走
    "back",  # 按住 S：向后走
    "left",  # 按住 A：向左走
    "right",  # 按住 D：向右走
    "jump",  # 按住 Space：跳跃
    "sneak",  # 按住 Shift：潜行、慢走，并避免走下方块边缘
    "sprint",  # 按住 Ctrl：疾跑；通常需要同时向前走
)

# interaction 同样使用固定顺序。一个 tick 可以同时有多个 True。
INTERACTION_NAMES: Tuple[str, ...] = (
    "attack",  # 鼠标左键：攻击或挖掘方块
    "use",  # 鼠标右键：使用物品、放置方块或与方块交互
    "drop",  # Q：丢出当前物品
    "pick_item",  # 鼠标中键：选取准星指向的方块
    "swap_hands",  # F：交换主手和副手物品
    "inventory",  # E：打开或关闭背包
    "esc",  # Esc：关闭当前界面或打开暂停菜单
)


class ActionSource(str, Enum):
    """Data source after normalization to the MC 1.16.5 contract."""

    VPT = "vpt"
    MINERL = "minerl"


def _bool_tuple(values: Iterable[object], length: int, field: str) -> Tuple[bool, ...]:
    result = tuple(bool(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{field} must contain {length} values, got {len(result)}")
    return result


def _float_pair(values: Sequence[object], field: str) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{field} must contain two values")
    result = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} values must be finite")
    return result


@dataclass(frozen=True)
class CanonicalActionTick:
    """一个时间点上的完整玩家输入。

    ``movement`` 和 ``interaction`` 都是“当前是否按住”的状态。
    ``camera`` 固定为 ``(上下转动, 左右转动)``，单位是度。
    ``hotbar=0`` 表示本 tick 没有切换物品栏，1..9 表示切到对应槽位。

    没有任何输入的真实游戏 tick 是 valid no-op，仍然是有效训练数据；
    为了补齐 batch 人工添加的 padding tick 才使用 ``valid=False``。
    """

    # 7 个移动按键，顺序见 MOVEMENT_NAMES。
    movement: Tuple[bool, ...]
    # 7 个交互按键，顺序见 INTERACTION_NAMES。
    interaction: Tuple[bool, ...]
    # 0=不切换，1..9=切到对应快捷栏槽位。
    hotbar: int
    # (pitch_delta, yaw_delta)，也就是 (上下看, 左右看)，单位为度。
    camera: Tuple[float, float]
    # GUI 内鼠标位置，已归一化到 [0, 1]；不在 GUI 中时通常为 None。
    cursor: Optional[Tuple[float, float]]
    # 当前是否打开了背包、箱子、合成台等 GUI。
    gui_open: bool
    # False 只用于 padding 或损坏数据，不能用来表示真实 no-op。
    valid: bool
    # 动作发生的时间，用于和视频帧 PTS 对齐。
    timestamp_ms: int
    # 这条动作来自 VPT 还是 MineRL。
    source: ActionSource
    # 人工记录通常为 1.0；未来的伪标签可以使用更低置信度。
    label_confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "movement", _bool_tuple(self.movement, len(MOVEMENT_NAMES), "movement")
        )
        object.__setattr__(
            self,
            "interaction",
            _bool_tuple(self.interaction, len(INTERACTION_NAMES), "interaction"),
        )
        object.__setattr__(self, "camera", _float_pair(self.camera, "camera"))
        object.__setattr__(self, "source", ActionSource(self.source))
        object.__setattr__(self, "hotbar", int(self.hotbar))
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "label_confidence", float(self.label_confidence))

        if not 0 <= self.hotbar <= 9:
            raise ValueError("hotbar must be 0 (unchanged) or a slot in 1..9")
        if any(abs(value) > 180.0 for value in self.camera):
            raise ValueError("camera deltas must be within [-180, 180] degrees")
        if self.cursor is not None:
            cursor = _float_pair(self.cursor, "cursor")
            if not all(0.0 <= value <= 1.0 for value in cursor):
                raise ValueError("cursor coordinates must be normalized to [0, 1]")
            object.__setattr__(self, "cursor", cursor)
        if not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label_confidence must be within [0, 1]")
        if not self.valid and (not self.is_noop or self.label_confidence != 0.0):
            raise ValueError("padding ticks must be no-op actions with zero confidence")

    @classmethod
    def noop(
        cls,
        timestamp_ms: int,
        source: ActionSource,
        *,
        valid: bool = True,
    ) -> "CanonicalActionTick":
        """创建一个“玩家没有输入”的 tick，也可用 ``valid=False`` 创建 padding。"""

        return cls(
            movement=(False,) * len(MOVEMENT_NAMES),
            interaction=(False,) * len(INTERACTION_NAMES),
            hotbar=0,
            camera=(0.0, 0.0),
            cursor=None,
            gui_open=False,
            valid=valid,
            timestamp_ms=timestamp_ms,
            source=source,
            label_confidence=1.0 if valid else 0.0,
        )

    @property
    def is_noop(self) -> bool:
        """玩家是否没有输入；仅仅打开 GUI 不算玩家输入。"""

        return (
            not any(self.movement)
            and not any(self.interaction)
            and self.hotbar == 0
            and self.camera == (0.0, 0.0)
        )

    def movement_value(self, name: str) -> bool:
        return self.movement[MOVEMENT_NAMES.index(name)]

    def interaction_value(self, name: str) -> bool:
        return self.interaction[INTERACTION_NAMES.index(name)]
