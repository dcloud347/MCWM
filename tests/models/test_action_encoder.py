import unittest

try:
    import torch
    from mcwm.models.action_encoder import (
        CameraEncoder,
        CursorEncoder,
        mu_law_normalize,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ActionComponentEncoderTest(unittest.TestCase):
    def test_mu_law_clips_and_preserves_direction(self):
        values = torch.tensor([[-20.0, -1.0, 0.0, 1.0, 20.0]])

        normalized = mu_law_normalize(values, clip_value=10.0)

        self.assertTrue(torch.equal(normalized[:, 0], normalized[:, 1] * 0 + -1))
        self.assertTrue(torch.equal(normalized[:, -1], normalized[:, -2] * 0 + 1))
        self.assertTrue(torch.allclose(normalized[:, 1], -normalized[:, 3]))
        self.assertEqual(float(normalized[:, 2]), 0.0)

    def test_camera_encoder_keeps_leading_dimensions(self):
        encoder = CameraEncoder(output_dim=8, hidden_dim=12, clip_degrees=10.0)
        camera = torch.tensor(
            [[[[1.0, -2.0], [50.0, -50.0]], [[0.0, 0.0], [2.0, 3.0]]]]
        )

        encoded = encoder(camera)
        clipped = encoder(camera.clamp(-10.0, 10.0))

        self.assertEqual(tuple(encoded.shape), (1, 2, 2, 8))
        self.assertTrue(torch.allclose(encoded, clipped))

    def test_cursor_requires_open_gui_and_present_cursor(self):
        encoder = CursorEncoder(output_dim=8, hidden_dim=12)
        cursor = torch.tensor([[[[0.2, 0.4], [0.6, 0.8], [0.5, 0.5]]]])
        gui_open = torch.tensor([[[True, False, True]]])
        cursor_present = torch.tensor([[[True, True, False]]])

        encoded = encoder(cursor, gui_open, cursor_present)

        self.assertEqual(tuple(encoded.shape), (1, 1, 3, 8))
        self.assertGreater(float(encoded[0, 0, 0].abs().sum()), 0.0)
        self.assertTrue(torch.equal(encoded[0, 0, 1], torch.zeros(8)))
        self.assertTrue(torch.equal(encoded[0, 0, 2], torch.zeros(8)))

    def test_cursor_rejects_mismatched_mask_shape(self):
        encoder = CursorEncoder(output_dim=8)
        cursor = torch.zeros(2, 3, 2)

        with self.assertRaisesRegex(ValueError, "gui_open"):
            encoder(
                cursor,
                gui_open=torch.zeros(2, dtype=torch.bool),
                cursor_present=torch.zeros(2, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
