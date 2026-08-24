"""Small deterministic M0 fixture dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from mcwm.actions.minerl_adapter import minerl_action_to_canonical
from mcwm.actions.vpt_adapter import VPTActionAdapter
from mcwm.actions.schema import ActionSource
from .episode_store import EpisodeStore
from .manifest import EpisodeManifest


def _vpt_row(timestamp: int, **updates: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "milli": timestamp,
        "tick": timestamp // 50,
        "hotbar": 0,
        "isGuiOpen": False,
        "keyboard": {"keys": [], "newKeys": [], "chars": ""},
        "mouse": {
            "x": 640.0,
            "y": 360.0,
            "dx": 0.0,
            "dy": 0.0,
            "buttons": [],
            "newButtons": [],
        },
    }
    for key, value in updates.items():
        if key.startswith("mouse_"):
            row["mouse"][key[6:]] = value
        elif key.startswith("keyboard_"):
            row["keyboard"][key[9:]] = value
        else:
            row[key] = value
    return row


def build_fixture_store(root: Path) -> EpisodeStore:
    """Create two canonical episodes without requiring video codecs."""

    store = EpisodeStore(root)

    vpt_rows = [
        _vpt_row(0, mouse_buttons=[0], mouse_newButtons=[0]),
        _vpt_row(50, keyboard_keys=["key.keyboard.w"], mouse_buttons=[0]),
        _vpt_row(100, mouse_newButtons=[0]),
        _vpt_row(150, mouse_dx=4.0, mouse_dy=-2.0, mouse_buttons=[0]),
        _vpt_row(200, hotbar=2, keyboard_keys=["key.keyboard.space"]),
        _vpt_row(250, isGuiOpen=True, mouse_x=320.0, mouse_y=180.0),
    ]
    vpt_adapter = VPTActionAdapter(recorder_version="7.6")
    vpt_actions = vpt_adapter.adapt_many(vpt_rows)
    vpt_times = (0, 100, 200, 300)
    vpt_manifest = EpisodeManifest(
        episode_id="fixture-vpt",
        session_id="session-vpt",
        world_id="world-vpt",
        source=ActionSource.VPT,
        recorder_version="7.6",
        video_path="fixture://vpt-640x360.mp4",
        width=640,
        height=360,
        frame_count=len(vpt_times),
        action_count=len(vpt_actions),
        start_timestamp_ms=vpt_times[0],
        end_timestamp_ms=vpt_times[-1],
        split="train",
    )
    store.write_episode(
        vpt_manifest,
        vpt_times,
        vpt_actions,
        audit={"repairs": vpt_adapter.repairs, "unknown_keys": sorted(vpt_adapter.unknown_keys)},
    )

    minerl_rows = [
        {"forward": 1, "camera": [0.0, 0.0]},
        {"forward": 1, "jump": 1, "camera": [-1.0, 2.0]},
        {"attack": 1, "camera": [0.0, 0.0]},
        {"use": 1, "hotbar.3": 1, "camera": [0.0, 0.0]},
    ]
    minerl_actions = [
        minerl_action_to_canonical(row, timestamp_ms=1000 + index * 50)
        for index, row in enumerate(minerl_rows)
    ]
    minerl_times = (1000, 1100, 1200)
    minerl_manifest = EpisodeManifest(
        episode_id="fixture-minerl",
        session_id="session-minerl",
        world_id="world-minerl",
        source=ActionSource.MINERL,
        recorder_version="1.0",
        video_path="fixture://minerl-640x360.mp4",
        width=640,
        height=360,
        frame_count=len(minerl_times),
        action_count=len(minerl_actions),
        start_timestamp_ms=minerl_times[0],
        end_timestamp_ms=minerl_times[-1],
        split="validation",
    )
    store.write_episode(minerl_manifest, minerl_times, minerl_actions)
    store.write_dataset_manifest()
    return store

