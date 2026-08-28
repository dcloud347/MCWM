from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mcwm.actions.schema import ActionSource, CanonicalActionTick
from mcwm.data.alignment import ActionBlock
from mcwm.data.episode_store import EpisodeStore
from mcwm.data.manifest import EpisodeManifest
from mcwm.data.world_model_dataset import (
    WorldModelDataset,
    _actions_between_sampled_frames,
    collate_world_model_samples,
)


def _action(timestamp_ms, *, cursor=None):
    return CanonicalActionTick(
        movement=(True, False, False, False, False, False, False),
        interaction=(False,) * 7,
        hotbar=0,
        camera=(1.0, -2.0),
        cursor=cursor,
        gui_open=cursor is not None,
        valid=True,
        timestamp_ms=timestamp_ms,
        source=ActionSource.VPT,
    )


def _build_store(root: Path, *, source=ActionSource.VPT):
    timestamps = tuple(range(0, 2501, 50))
    actions = tuple(
        CanonicalActionTick.noop(timestamp, source)
        for timestamp in timestamps[:-1]
    )
    manifest = EpisodeManifest(
        episode_id="episode",
        session_id="session",
        world_id="world",
        source=source,
        recorder_version="7.6",
        video_path="video.mp4",
        width=640,
        height=360,
        frame_count=len(timestamps),
        action_count=len(actions),
        start_timestamp_ms=timestamps[0],
        end_timestamp_ms=timestamps[-1],
        split="train",
    )
    store = EpisodeStore(root)
    store.write_episode(manifest, timestamps, actions)
    store.write_dataset_manifest()


class WorldModelDatasetTest(unittest.TestCase):
    def test_returns_eight_frames_and_seven_half_open_action_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_store(root)
            dataset = WorldModelDataset(root, split="train", seed=4)

            def fake_decode(path, timestamps):
                return torch.zeros(len(timestamps), 3, 360, 640, dtype=torch.uint8)

            with patch(
                "mcwm.data.world_model_dataset.decode_frames_at_timestamps",
                side_effect=fake_decode,
            ):
                sample = dataset[(0, 9)]

            self.assertEqual(tuple(sample["frames"].shape), (8, 3, 360, 640))
            self.assertEqual(len(sample["action_blocks"]), 7)
            frame_times = sample["frame_timestamps_ms"].tolist()
            self.assertEqual(
                [current - previous for previous, current in zip(frame_times, frame_times[1:])],
                [250] * 7,
            )
            for index, block in enumerate(sample["action_blocks"]):
                self.assertEqual(len(block), 5)
                self.assertTrue(
                    all(
                        frame_times[index] <= action.timestamp_ms < frame_times[index + 1]
                        for action in block
                    )
                )
            self.assertEqual(
                sample["sample_id"],
                f"episode:pts={frame_times[0]}-{frame_times[-1]}ms@4fps",
            )

    def test_collate_pads_variable_ticks_without_turning_padding_into_noop(self):
        first_blocks = tuple(
            ((CanonicalActionTick.noop(index * 100, ActionSource.VPT),))
            for index in range(7)
        )
        second_blocks = tuple(
            (
                _action(index * 100),
                _action(index * 100 + 50, cursor=(0.25, 0.75)),
            )
            for index in range(7)
        )
        samples = [
            {
                "frames": torch.zeros(8, 3, 4, 4, dtype=torch.uint8),
                "frame_timestamps_ms": torch.arange(8, dtype=torch.int64) * 250,
                "action_blocks": first_blocks,
                "sample_id": "first",
            },
            {
                "frames": torch.ones(8, 3, 4, 4, dtype=torch.uint8),
                "frame_timestamps_ms": torch.arange(8, dtype=torch.int64) * 250,
                "action_blocks": second_blocks,
                "sample_id": "second",
            },
        ]

        batch = collate_world_model_samples(samples)

        self.assertEqual(tuple(batch["movement"].shape), (2, 7, 2, 7))
        self.assertFalse(batch["movement"][0].any())
        self.assertTrue(batch["valid_mask"][0, :, 0].all())
        self.assertFalse(batch["valid_mask"][0, :, 1].any())
        self.assertTrue(batch["valid_mask"][1].all())
        self.assertFalse(batch["cursor_present"][0].any())
        self.assertTrue(batch["cursor_present"][1, :, 1].all())
        self.assertEqual(batch["sample_id"], ["first", "second"])

    def test_rejects_invalid_action_instead_of_treating_it_as_padding(self):
        invalid = CanonicalActionTick.noop(0, ActionSource.VPT, valid=False)
        blocks = (
            ActionBlock(0, 0, 50, (invalid,), True),
        )

        with self.assertRaisesRegex(ValueError, "invalid action labels"):
            _actions_between_sampled_frames(blocks, (0, 1))

    def test_rejects_transition_across_discontinuity(self):
        blocks = (
            ActionBlock(0, 0, 50, (_action(0),), True),
            ActionBlock(1, 50, 500, (_action(50),), False),
        )

        with self.assertRaisesRegex(ValueError, "discontinuity"):
            _actions_between_sampled_frames(blocks, (0, 2))

    def test_rejects_non_vpt_training_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_store(root, source=ActionSource.MINERL)
            with self.assertRaisesRegex(ValueError, "only accepts VPT"):
                WorldModelDataset(root, split="train")


if __name__ == "__main__":
    unittest.main()
