import unittest

try:
    import torch

    from mcwm.actions.schema import ActionSource, CanonicalActionTick
    from mcwm.envs.minerl1 import MineRL1EnvWrapper, observation_to_tensor
except ModuleNotFoundError:
    torch = None


class _ActionSpace:
    def contains(self, value):
        return "camera" in value and len(value["camera"]) == 2


class _MockEnv:
    action_space = _ActionSpace()

    def __init__(self, terminate_at=100):
        self.steps = 0
        self.terminate_at = terminate_at
        self.closed = False
        self.actions = []

    def reset(self):
        return {"pov": torch.zeros(180, 320, 3, dtype=torch.uint8)}, {"reset": True}

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        observation = {"pov": torch.zeros(360, 640, 3, dtype=torch.uint8)}
        return observation, 1.0, self.steps >= self.terminate_at, False, {}

    def close(self):
        self.closed = True


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MineRL1WrapperTest(unittest.TestCase):
    def test_observation_is_resized_to_training_contract(self):
        frame = observation_to_tensor(torch.zeros(180, 320, 3, dtype=torch.uint8))
        self.assertEqual(tuple(frame.shape), (3, 360, 640))
        self.assertEqual(frame.dtype, torch.uint8)

    def test_one_model_tick_repeats_action_and_stops_at_termination(self):
        raw = _MockEnv(terminate_at=3)
        wrapper = MineRL1EnvWrapper(raw, action_repeat=5)
        action = CanonicalActionTick.noop(0, ActionSource.MINERL)
        tick = wrapper.step_model_tick(action)

        self.assertEqual(tick.action_repeats, 3)
        self.assertEqual(tick.reward, 3.0)
        self.assertTrue(tick.terminated)
        wrapper.close()
        self.assertTrue(raw.closed)

    def test_camera_is_distributed_and_hotbar_event_is_not_repeated(self):
        raw = _MockEnv()
        wrapper = MineRL1EnvWrapper(raw, action_repeat=5)
        action = CanonicalActionTick(
            movement=(False,) * 7,
            interaction=(False,) * 7,
            hotbar=4,
            camera=(5.0, 10.0),
            cursor=None,
            gui_open=False,
            valid=True,
            timestamp_ms=0,
            source=ActionSource.MINERL,
        )
        wrapper.step_model_tick(action)

        self.assertEqual(sum(value["camera"][0] for value in raw.actions), 5.0)
        self.assertEqual(sum(value["camera"][1] for value in raw.actions), 10.0)
        self.assertEqual([value["hotbar.4"] for value in raw.actions], [1, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
