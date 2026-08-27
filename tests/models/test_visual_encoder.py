import unittest

try:
    import torch
    from mcwm.models.visual_encoder import VisualEncoder, VisualEncoderConfig
except ModuleNotFoundError:
    torch = None


def tiny_encoder():
    config = VisualEncoderConfig(
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
    return VisualEncoder(config)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class VisualEncoderTest(unittest.TestCase):
    def test_supports_every_even_frame_count_up_to_configured_maximum(self):
        encoder = tiny_encoder().eval()
        spatial_tokens = 2 * 3

        with torch.no_grad():
            for frame_count in range(2, 17, 2):
                with self.subTest(frame_count=frame_count):
                    clips = torch.rand(1, frame_count, 3, 20, 30)
                    tokens = encoder(clips, return_patch_tokens=True)
                    pooled = encoder(clips)
                    expected_tokens = frame_count // 2 * spatial_tokens
                    self.assertEqual(tuple(tokens.shape), (1, expected_tokens, 24))
                    self.assertEqual(tuple(pooled.shape), (1, 24))
                    self.assertEqual(
                        encoder.config.runtime_token_grid_size(frame_count),
                        (frame_count // 2, 2, 3),
                    )
                    self.assertEqual(
                        encoder.config.runtime_token_count(frame_count),
                        expected_tokens,
                    )

    def test_rejects_unsupported_runtime_frame_counts(self):
        encoder = tiny_encoder()

        for frame_count in (1, 3, 17, 18):
            with self.subTest(frame_count=frame_count):
                clips = torch.rand(1, frame_count, 3, 20, 30)
                with self.assertRaises(ValueError):
                    encoder(clips)

    def test_runtime_mask_indices_use_actual_token_count(self):
        encoder = tiny_encoder().eval()
        clips = torch.rand(2, 6, 3, 20, 30)
        indices = torch.tensor([[0, 6, 17], [1, 7, 16]])

        with torch.no_grad():
            tokens = encoder(clips, indices, return_patch_tokens=True)

        self.assertEqual(tuple(tokens.shape), (2, 3, 24))
        invalid_indices = torch.tensor([[0, 18], [1, 2]])
        with self.assertRaisesRegex(ValueError, "outside the runtime clip"):
            encoder(clips, invalid_indices, return_patch_tokens=True)

    def test_sixteen_frame_behavior_keeps_configured_token_budget(self):
        encoder = tiny_encoder().eval()
        clips = torch.rand(1, 16, 3, 20, 30)

        with torch.no_grad():
            tokens = encoder(clips, return_patch_tokens=True)

        self.assertEqual(tokens.shape[1], encoder.config.token_count)
        self.assertEqual(
            encoder.config.runtime_token_grid_size(16),
            encoder.config.token_grid_size,
        )


if __name__ == "__main__":
    unittest.main()
