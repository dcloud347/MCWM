import json
from pathlib import Path
import tempfile
import unittest

from mcwm.data.audit import audit_store
from mcwm.data.episode_store import EpisodeStore
from mcwm.data.fixtures import build_fixture_store
from mcwm.data.manifest import DatasetManifest


class EpisodeStoreTest(unittest.TestCase):
    def test_fixture_round_trip_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = build_fixture_store(Path(temporary))
            vpt = store.read_episode("fixture-vpt")
            minerl = store.read_episode("fixture-minerl")
            self.assertEqual(vpt.manifest.width, 640)
            self.assertEqual(vpt.manifest.height, 360)
            self.assertEqual(len(vpt.actions), 6)
            self.assertTrue(any(action.is_noop for action in vpt.actions))
            self.assertEqual(len(minerl.actions), 4)

            dataset = DatasetManifest.read(Path(temporary) / "dataset_manifest.json")
            self.assertEqual(len(dataset.episodes), 2)
            self.assertEqual(len(dataset.content_hash), 64)

            report = audit_store(Path(temporary))
            self.assertEqual(report["episode_count"], 2)
            self.assertEqual(report["source_episode_count"], {"minerl": 1, "vpt": 1})
            self.assertEqual(report["split_leakage"], [])
            self.assertEqual(report["issues"], [])
            self.assertGreater(report["noop_count"], 0)

    def test_existing_episode_is_not_partially_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = build_fixture_store(Path(temporary))
            episode = store.read_episode("fixture-vpt")
            with self.assertRaises(FileExistsError):
                store.write_episode(
                    episode.manifest,
                    episode.frame_timestamps_ms,
                    episode.actions,
                    audit=episode.audit,
                )
            self.assertEqual(
                store.read_episode("fixture-vpt").frame_timestamps_ms,
                episode.frame_timestamps_ms,
            )

    def test_parquet_dependency_error_is_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = build_fixture_store(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "mcwm\[parquet\]"):
                store.export_parquet("fixture-vpt")


if __name__ == "__main__":
    unittest.main()
