"""在统一动作对象和 JSONL 文件之间转换。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from .schema import ActionSource, CanonicalActionTick


ACTION_SCHEMA_VERSION = 1


def action_to_dict(action: CanonicalActionTick) -> Dict[str, Any]:
    """把一条统一动作转换成可写入 JSON 的字典。"""

    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "movement": list(action.movement),
        "interaction": list(action.interaction),
        "hotbar": action.hotbar,
        "camera": list(action.camera),
        "cursor": list(action.cursor) if action.cursor is not None else None,
        "gui_open": action.gui_open,
        "valid": action.valid,
        "timestamp_ms": action.timestamp_ms,
        "source": action.source.value,
        "label_confidence": action.label_confidence,
    }


def action_from_dict(data: Dict[str, Any]) -> CanonicalActionTick:
    """从字典读取一条统一动作，并检查格式版本。"""

    version = data.get("schema_version", ACTION_SCHEMA_VERSION)
    if version != ACTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported action schema version: {version}")
    return CanonicalActionTick(
        movement=tuple(data["movement"]),
        interaction=tuple(data["interaction"]),
        hotbar=data["hotbar"],
        camera=tuple(data["camera"]),
        cursor=tuple(data["cursor"]) if data.get("cursor") is not None else None,
        gui_open=data["gui_open"],
        valid=data["valid"],
        timestamp_ms=data["timestamp_ms"],
        source=ActionSource(data["source"]),
        label_confidence=data.get("label_confidence", 1.0),
    )


def write_actions_jsonl(path: Path, actions: Iterable[CanonicalActionTick]) -> None:
    """把动作逐行写入 JSONL；写完后再替换目标文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action_to_dict(action), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def read_actions_jsonl(path: Path) -> Iterator[CanonicalActionTick]:
    """逐行读取动作，遇到坏数据时指出文件和行号。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield action_from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid action at {path}:{line_number}: {exc}") from exc
