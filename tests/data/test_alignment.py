import random
import unittest

from mcwm.actions.schema import ActionSource, CanonicalActionTick
from mcwm.data.alignment import align_actions_to_frames
from mcwm.data.dataset import random_clip_frame_indices
from mcwm.data.episode_store import StoredEpisode
from mcwm.data.manifest import EpisodeManifest


def noop(timestamp):
    return CanonicalActionTick.noop(timestamp, ActionSource.VPT)


class AlignmentTest(unittest.TestCase):
    def test_half_open_boundaries_and_discontinuity(self):
        frames = (0, 100, 200, 500, 600)
        actions = tuple(noop(value) for value in (-1, 0, 99, 100, 199, 200, 499, 500, 599, 600))
        result = align_actions_to_frames(frames, actions, max_frame_gap_ms=250)
        self.assertEqual([len(block.actions) for block in result.blocks], [2, 2, 2, 2])
        self.assertEqual(result.blocks[0].actions[-1].timestamp_ms, 99)
        self.assertEqual(result.blocks[1].actions[0].timestamp_ms, 100)
        self.assertFalse(result.blocks[2].continuous)
        self.assertEqual(result.continuous_frame_ranges, ((0, 3), (3, 5)))
        self.assertEqual([item.timestamp_ms for item in result.actions_before_first_frame], [-1])
        self.assertEqual([item.timestamp_ms for item in result.actions_at_or_after_last_frame], [600])

    def test_clip_windows_never_cross_gap(self):
        frames = (0, 100, 200, 500, 600)
        actions = tuple(noop(value) for value in (0, 100, 200, 500))
        manifest = EpisodeManifest(
            episode_id="e",
            session_id="s",
            world_id="w",
            source=ActionSource.VPT,
            recorder_version="7.6",
            video_path="fixture://e.mp4",
            width=640,
            height=360,
            frame_count=len(frames),
            action_count=len(actions),
            start_timestamp_ms=0,
            end_timestamp_ms=600,
        )
        episode = StoredEpisode(manifest, frames, actions)
        frame_indices = random_clip_frame_indices(
            episode.frame_timestamps_ms,
            clip_frames=3,
            sampling_rate=1,
            generator=random.Random(0),
            max_frame_gap_ms=250,
        )
        self.assertEqual(frame_indices, (0, 1, 2))

    def test_vjepa_sampling_rate_uses_random_start_and_four_source_frame_step(self):
        frames = tuple(range(0, 8001, 50))
        actions = tuple(noop(value) for value in range(0, 8000, 50))
        manifest = EpisodeManifest(
            episode_id="e",
            session_id="s",
            world_id="w",
            source=ActionSource.VPT,
            recorder_version="7.6",
            video_path="fixture://e.mp4",
            width=640,
            height=360,
            frame_count=len(frames),
            action_count=len(actions),
            start_timestamp_ms=frames[0],
            end_timestamp_ms=frames[-1],
        )
        episode = StoredEpisode(manifest, frames, actions)

        first = random_clip_frame_indices(
            episode.frame_timestamps_ms,
            clip_frames=16,
            sampling_rate=4,
            generator=random.Random(7),
        )
        repeated = random_clip_frame_indices(
            episode.frame_timestamps_ms,
            clip_frames=16,
            sampling_rate=4,
            generator=random.Random(7),
        )

        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 16)
        self.assertTrue(
            all(
                current - previous == 4
                for previous, current in zip(first, first[1:])
            )
        )
        self.assertEqual(
            [
                frames[current] - frames[previous]
                for previous, current in zip(first, first[1:])
            ],
            [200] * 15,
        )


if __name__ == "__main__":
    unittest.main()
