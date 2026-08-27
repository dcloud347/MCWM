import unittest

import torch

from mcwm.models.visual_encoder import VisualEncoder, VisualEncoderConfig


def tiny_encoder():
    return VisualEncoder(
        VisualEncoderConfig(
            image_height=20,
            image_width=30,
            patch_size=10,
            clip_frames=16,
            tubelet_size=2,
            dim=24,
            depth=1,
            heads=4,
            mlp_dim=48,
            gradient_checkpointing=False,
        )
    )


class VisualEncoderTest(unittest.TestCase):
    def test_supports_every_even_frame_count_up_to_maximum(self):
        encoder = tiny_encoder().eval()
        with torch.no_grad():
            for frame_count in range(2, 17, 2):
                with self.subTest(frame_count=frame_count):
                    clips = torch.rand(1, frame_count, 3, 20, 30)
                    tokens = encoder(clips, return_patch_tokens=True)
                    expected = frame_count // 2 * 6
                    self.assertEqual(tuple(tokens.shape), (1, expected, 24))
                    self.assertEqual(
                        encoder.config.runtime_token_grid_size(frame_count),
                        (frame_count // 2, 2, 3),
                    )

    def test_rejects_odd_too_short_and_too_long_clips(self):
        encoder = tiny_encoder()
        for frame_count in (1, 3, 17, 18):
            with self.subTest(frame_count=frame_count):
                with self.assertRaises(ValueError):
                    encoder(torch.rand(1, frame_count, 3, 20, 30))

    def test_mask_indices_use_runtime_token_count(self):
        encoder = tiny_encoder().eval()
        clips = torch.rand(2, 6, 3, 20, 30)
        indices = torch.tensor([[0, 6, 17], [1, 7, 16]])
        with torch.no_grad():
            tokens = encoder(clips, indices, return_patch_tokens=True)
        self.assertEqual(tuple(tokens.shape), (2, 3, 24))
        with self.assertRaisesRegex(ValueError, "outside the runtime clip"):
            encoder(clips, torch.tensor([[0, 18], [1, 2]]), return_patch_tokens=True)

    def test_sixteen_frame_token_budget_is_unchanged(self):
        encoder = tiny_encoder().eval()
        with torch.no_grad():
            tokens = encoder(
                torch.rand(1, 16, 3, 20, 30),
                return_patch_tokens=True,
            )
        self.assertEqual(tokens.shape[1], encoder.config.token_count)


if __name__ == "__main__":
    unittest.main()
