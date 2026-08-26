"""统一 Minecraft 动作格式，并转换不同来源的动作数据。"""

from .schema import (
    INTERACTION_NAMES,
    MOVEMENT_NAMES,
    ActionSource,
    CanonicalActionTick,
)

__all__ = [
    "ActionSource",
    "CanonicalActionTick",
    "INTERACTION_NAMES",
    "MOVEMENT_NAMES",
]
