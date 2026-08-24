import json
from pathlib import Path
import tempfile
import unittest

from mcwm.data.episode_store import EpisodeStore
from mcwm.data.ingest import ingest_vpt_episode


class IngestTest(unittest.TestCase):
    def test_vpt_jsonl_ingest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actions_path = root / "raw.jsonl"
            timestamps_path = root / "pts.json"
            rows = [
                {
                    "milli": 0,
                    "hotbar": 0,
                    "isGuiOpen": False,
                    "keyboard": {"keys": ["key.keyboard.w"]},
                    "mouse": {"dx": 0, "dy": 0, "buttons": [], "newButtons": []},
                },
                {
                    "milli": 50,
                    "hotbar": 0,
                    "isGuiOpen": False,
                    "keyboard": {"keys": []},
                    "mouse": {"dx": 1, "dy": -1, "buttons": [], "newButtons": []},
                },
            ]
            actions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            timestamps_path.write_text(json.dumps({"timestamps_ms": [0, 100]}), encoding="utf-8")
            store = EpisodeStore(root / "store")
            manifest = ingest_vpt_episode(
                store,
                episode_id="raw-vpt",
                session_id="session",
                world_id="world",
                recorder_version="7.6",
                video_path=root / "missing-but-referenced.mp4",
                action_path=actions_path,
                frame_timestamps_path=timestamps_path,
                split="train",
            )
            self.assertEqual(manifest.action_count, 2)
            loaded = store.read_episode("raw-vpt")
            self.assertTrue(loaded.actions[0].movement_value("forward"))
            self.assertEqual(loaded.actions[1].camera, (-0.15, 0.15))


if __name__ == "__main__":
    unittest.main()

