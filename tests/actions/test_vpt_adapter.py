import unittest

from mcwm.actions.vpt_adapter import VPTActionAdapter


def row(timestamp=0, *, keys=None, buttons=None, new_buttons=None, **values):
    result = {
        "milli": timestamp,
        "hotbar": values.pop("hotbar", 0),
        "isGuiOpen": values.pop("gui_open", False),
        "keyboard": {"keys": keys or []},
        "mouse": {
            "x": values.pop("x", 0.0),
            "y": values.pop("y", 0.0),
            "dx": values.pop("dx", 0.0),
            "dy": values.pop("dy", 0.0),
            "buttons": buttons or [],
            "newButtons": new_buttons or [],
        },
    }
    result.update(values)
    return result


class VPTAdapterTest(unittest.TestCase):
    def test_keyboard_mouse_camera_cursor_and_hotbar(self):
        adapter = VPTActionAdapter(recorder_version="7.6")
        action = adapter.adapt(
            row(
                50,
                keys=["key.keyboard.w", "key.keyboard.space", "key.keyboard.e"],
                buttons=[0, 1],
                dx=4.0,
                dy=-2.0,
                hotbar=2,
                gui_open=True,
                x=320,
                y=180,
            )
        )
        self.assertTrue(action.movement_value("forward"))
        self.assertTrue(action.movement_value("jump"))
        self.assertTrue(action.interaction_value("inventory"))
        self.assertTrue(action.interaction_value("attack"))
        self.assertTrue(action.interaction_value("use"))
        self.assertEqual(action.camera, (-0.3, 0.6))
        self.assertEqual(action.hotbar, 3)
        self.assertEqual(action.cursor, (0.25, 0.25))

    def test_stuck_attack_is_repaired_and_noop_preserved(self):
        adapter = VPTActionAdapter()
        first = adapter.adapt(row(0, buttons=[0], new_buttons=[0]))
        second = adapter.adapt(row(50, buttons=[0]))
        third = adapter.adapt(row(100, buttons=[0], new_buttons=[0]))
        noop = adapter.adapt(row(150))
        self.assertFalse(first.interaction_value("attack"))
        self.assertFalse(second.interaction_value("attack"))
        self.assertTrue(third.interaction_value("attack"))
        self.assertTrue(noop.is_noop)
        self.assertTrue(noop.valid)
        self.assertIn("stuck_attack_detected_at_episode_start", adapter.repairs)

    def test_unknown_key_is_audited_not_fatal(self):
        adapter = VPTActionAdapter()
        action = adapter.adapt(row(0, keys=["key.keyboard.f3"]));
        self.assertTrue(action.is_noop)
        self.assertEqual(adapter.unknown_keys, {"key.keyboard.f3"})


if __name__ == "__main__":
    unittest.main()

