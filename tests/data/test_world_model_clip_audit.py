from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcwm.actions.schema import ActionSource, CanonicalActionTick
from mcwm.data.episode_store import EpisodeStore
from mcwm.data.manifest import EpisodeManifest
from mcwm.data.world_model_clip_audit import audit_world_model_dataset
from mcwm.data.world_model_dataset import WorldModelDataset


def _write_episode(root: Path, episode_id: str, *, active: bool) -> None:
    timestamps = tuple(range(0, 3001, 50))
    actions = []
    for timestamp in timestamps[:-1]:
        if active:
            actions.append(
                CanonicalActionTick(
                    movement=(True, False, False, False, False, False, False),
                    interaction=(False,) * 7,
                    hotbar=0,
                    camera=(0.0, 0.0),
                    cursor=None,
                    gui_open=False,
                    valid=True,
                    timestamp_ms=timestamp,
                    source=ActionSource.VPT,
                )
            )
        else:
            actions.append(CanonicalActionTick.noop(timestamp, ActionSource.VPT))
    manifest = EpisodeManifest(
        episode_id=episode_id,
        session_id=f"session-{episode_id}",
        world_id=f"world-{episode_id}",
        source=ActionSource.VPT,
        recorder_version="7.6",
        video_path=f"{episode_id}.mp4",
        width=640,
        height=360,
        frame_count=len(timestamps),
        action_count=len(actions),
        start_timestamp_ms=timestamps[0],
        end_timestamp_ms=timestamps[-1],
        split="train",
    )
    EpisodeStore(root).write_episode(manifest, timestamps, actions)


class WorldModelClipAuditTest(unittest.TestCase):
    def test_reports_action_coverage_without_decoding_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root, "active", active=True)
            _write_episode(root, "noop", active=False)
            EpisodeStore(root).write_dataset_manifest()
            dataset = WorldModelDataset(
                root,
                split="train",
                frames_per_sample=4,
                sample_fps=4,
                seed=11,
                samples_per_video=3,
            )

            with patch(
                "mcwm.data.world_model_dataset.decode_frames_at_timestamps",
                side_effect=AssertionError("audit must not decode video"),
            ):
                report = audit_world_model_dataset(
                    dataset,
                    seed=11,
                    sampling_epochs=1,
                )

            self.assertEqual(report["totals"]["clips"], 6)
            self.assertEqual(report["totals"]["clips_with_action"], 3)
            self.assertEqual(report["totals"]["clips_only_noop"], 3)
            self.assertEqual(report["ratios"]["clips_with_action"], 0.5)
            self.assertEqual(report["ratios"]["transitions_with_action"], 0.5)
            self.assertEqual(report["categories"]["movement"]["clip_ratio"], 0.5)
            self.assertEqual(report["categories"]["camera"]["clip_ratio"], 0.0)

    def test_max_clips_caps_samples_across_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root, "active", active=True)
            EpisodeStore(root).write_dataset_manifest()
            dataset = WorldModelDataset(
                root,
                split="train",
                frames_per_sample=4,
                sample_fps=4,
                seed=5,
                samples_per_video=2,
            )

            report = audit_world_model_dataset(
                dataset,
                seed=5,
                sampling_epochs=5,
                max_clips=3,
            )

            self.assertEqual(report["totals"]["clips"], 3)
            self.assertEqual(report["sampling_epochs_completed"], 2)

    def test_progress_bar_reports_scanned_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_episode(root, "active", active=True)
            EpisodeStore(root).write_dataset_manifest()
            dataset = WorldModelDataset(
                root,
                split="train",
                frames_per_sample=4,
                sample_fps=4,
                seed=5,
                samples_per_video=2,
            )
            output = StringIO()

            with patch("sys.stderr", output):
                audit_world_model_dataset(dataset, seed=5, show_progress=True)

            self.assertIn("Scanning clips", output.getvalue())
            self.assertIn("2/2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
