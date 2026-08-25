import unittest

try:
    import torch
    from mcwm.models.masking import (
        MaskConfig,
        MaskGeneratorConfig,
        SpatiotemporalMaskSampler,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskingTest(unittest.TestCase):
    def test_mask_groups_are_reproducible_and_span_full_clip(self):
        sampler = SpatiotemporalMaskSampler(
            (8, 8), MaskConfig()
        )
        first = sampler.sample(2, 4, generator=torch.Generator().manual_seed(7))
        second = sampler.sample(2, 4, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, 2, 4, 64))
        self.assertTrue(torch.equal(first[:, :, :1].expand_as(first), first))
        self.assertTrue(first.any(dim=-1).all())
        self.assertTrue((~first).any(dim=-1).all())
        self.assertTrue(first.dtype == torch.bool)

    def test_single_block_uses_configured_spatial_area(self):
        config = MaskConfig(
            generators=(
                MaskGeneratorConfig(
                    spatial_scale=(0.25, 0.25),
                    temporal_scale=(1.0, 1.0),
                    aspect_ratio=(1.0, 1.0),
                    num_blocks=1,
                ),
            )
        )
        mask = SpatiotemporalMaskSampler((4, 4), config).sample(
            1,
            3,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 16))
        self.assertEqual(mask.sum(dim=-1).tolist(), [[[4, 4, 4]]])

    def test_wide_grid_preserves_large_block_area_and_aspect_ratio(self):
        config = MaskGeneratorConfig(
            spatial_scale=(0.70, 0.70),
            temporal_scale=(1.0, 1.0),
            aspect_ratio=(0.75, 1.5),
            num_blocks=1,
        )
        sampler = SpatiotemporalMaskSampler((18, 32), MaskConfig((config,)))

        for seed in range(32):
            _, height, width = sampler._sample_block_size(
                8,
                config,
                torch.Generator().manual_seed(seed),
            )
            actual_scale = height * width / (18 * 32)
            self.assertGreaterEqual(height / width, 0.75)
            self.assertLessEqual(height / width, 1.5)
            self.assertLessEqual(abs(actual_scale - 0.70), 0.03)

    def test_wide_grid_large_mask_is_not_shrunk_by_clipping(self):
        config = MaskConfig(
            generators=(
                MaskGeneratorConfig(
                    spatial_scale=(0.70, 0.70),
                    temporal_scale=(1.0, 1.0),
                    aspect_ratio=(0.75, 1.5),
                    num_blocks=1,
                ),
            )
        )
        mask = SpatiotemporalMaskSampler((18, 32), config).sample(
            1,
            8,
            generator=torch.Generator().manual_seed(7),
        )
        frame_area = int(mask[0, 0, 0].sum().item())
        self.assertGreaterEqual(frame_area / (18 * 32), 0.67)
        self.assertLessEqual(frame_area / (18 * 32), 0.73)

    def test_invalid_spatial_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            MaskGeneratorConfig(
                spatial_scale=(0.7, 1.1),
                temporal_scale=(1.0, 1.0),
                aspect_ratio=(0.75, 1.5),
                num_blocks=2,
            )
