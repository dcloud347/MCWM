import unittest

from mcwm.actions.minerl_adapter import minerl_action_to_canonical
from mcwm.actions.schema import ActionSource


class MineRLAdapterTest(unittest.TestCase):
    def test_near_human_action_dict(self):
        action = minerl_action_to_canonical(
            {
                "forward": 1,
                "sprint": 1,
                "attack": 1,
                "hotbar.7": 1,
                "camera": [-2.0, 3.5],
            },
            timestamp_ms=200,
        )
        self.assertEqual(action.source, ActionSource.MINERL)
        self.assertTrue(action.movement_value("forward"))
        self.assertTrue(action.movement_value("sprint"))
        self.assertTrue(action.interaction_value("attack"))
        self.assertEqual(action.hotbar, 7)
        self.assertEqual(action.camera, (-2.0, 3.5))

    def test_multiple_hotbar_slots_fail(self):
        with self.assertRaises(ValueError):
            minerl_action_to_canonical(
                {"hotbar.1": 1, "hotbar.2": 1, "camera": [0, 0]},
                timestamp_ms=0,
            )


if __name__ == "__main__":
    unittest.main()

