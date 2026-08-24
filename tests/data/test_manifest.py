import unittest

from mcwm.actions.schema import ActionSource
from mcwm.data.manifest import (
    DatasetManifest,
    EpisodeManifest,
    assign_splits,
    find_split_leakage,
)


def episode(index, session=None, world=None):
    return EpisodeManifest(
        episode_id=f"episode-{index}",
        session_id=session or f"session-{index}",
        world_id=world or f"world-{index}",
        source=ActionSource.VPT,
        recorder_version="7.6",
        video_path=f"fixture://{index}.mp4",
        width=640,
        height=360,
        frame_count=3,
        action_count=2,
        start_timestamp_ms=0,
        end_timestamp_ms=100,
    )


class ManifestTest(unittest.TestCase):
    def test_split_groups_session_and_world_transitively(self):
        episodes = [episode(index) for index in range(12)]
        episodes[1] = episode(1, session=episodes[0].session_id, world="bridge-world")
        episodes[2] = episode(2, session="other-session", world="bridge-world")
        assigned = assign_splits(episodes, train=0.5, validation=0.25, test=0.25, seed="test")
        self.assertEqual(find_split_leakage(assigned), ())
        by_id = {item.episode_id: item for item in assigned}
        self.assertEqual(by_id["episode-0"].split, by_id["episode-1"].split)
        self.assertEqual(by_id["episode-1"].split, by_id["episode-2"].split)

    def test_content_hash_is_stable_and_sensitive(self):
        first = DatasetManifest((episode(0), episode(1)))
        same = DatasetManifest((episode(0), episode(1)))
        changed = DatasetManifest((episode(0), episode(2)))
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed.content_hash)

    def test_single_group_goes_to_largest_split(self):
        assigned = assign_splits([episode(0)])
        self.assertEqual(assigned[0].split, "train")

    def test_resolution_contract(self):
        with self.assertRaises(ValueError):
            EpisodeManifest(
                episode_id="bad",
                session_id="s",
                world_id="w",
                source=ActionSource.VPT,
                recorder_version="7.6",
                video_path="bad.mp4",
                width=128,
                height=128,
                frame_count=2,
                action_count=1,
                start_timestamp_ms=0,
                end_timestamp_ms=50,
            )


if __name__ == "__main__":
    unittest.main()
