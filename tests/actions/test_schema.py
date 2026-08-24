import unittest

from mcwm.actions.codec import action_from_dict, action_to_dict
from mcwm.actions.schema import ActionSource, CanonicalActionTick


class CanonicalActionSchemaTest(unittest.TestCase):
    def test_noop_and_padding_are_distinct(self):
        noop = CanonicalActionTick.noop(100, ActionSource.VPT)
        padding = CanonicalActionTick.noop(100, ActionSource.VPT, valid=False)
        self.assertTrue(noop.is_noop)
        self.assertTrue(noop.valid)
        self.assertTrue(padding.is_noop)
        self.assertFalse(padding.valid)
        self.assertEqual(padding.label_confidence, 0.0)

    def test_codec_round_trip(self):
        action = CanonicalActionTick(
            movement=(True, False, False, False, True, False, False),
            interaction=(True, False, False, False, False, False, False),
            hotbar=4,
            camera=(-1.25, 2.5),
            cursor=(0.25, 0.75),
            gui_open=True,
            valid=True,
            timestamp_ms=123,
            source=ActionSource.MINERL,
        )
        self.assertEqual(action_from_dict(action_to_dict(action)), action)

    def test_invalid_shapes_and_ranges_fail(self):
        with self.assertRaises(ValueError):
            CanonicalActionTick(
                movement=(False,),
                interaction=(False,) * 7,
                hotbar=0,
                camera=(0.0, 0.0),
                cursor=None,
                gui_open=False,
                valid=True,
                timestamp_ms=0,
                source=ActionSource.VPT,
            )
        with self.assertRaises(ValueError):
            CanonicalActionTick.noop(0, ActionSource.VPT).__class__(
                movement=(False,) * 7,
                interaction=(False,) * 7,
                hotbar=10,
                camera=(0.0, 0.0),
                cursor=None,
                gui_open=False,
                valid=True,
                timestamp_ms=0,
                source=ActionSource.VPT,
            )


if __name__ == "__main__":
    unittest.main()

