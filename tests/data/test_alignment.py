import unittest

from mcwm.actions.schema import ActionSource, CanonicalActionTick
from mcwm.data.alignment import align_actions_to_frames
from mcwm.data.dataset import iter_clip_indices
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
        clips = list(iter_clip_indices(episode, transitions=2, max_frame_gap_ms=250))
        self.assertEqual([(clip.start_frame, clip.end_frame) for clip in clips], [(0, 3)])


if __name__ == "__main__":
    unittest.main()

