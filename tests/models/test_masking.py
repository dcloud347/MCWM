import unittest

try:
    import torch
    from mcwm.models.masking import MaskConfig, SpatiotemporalMaskSampler
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskingTest(unittest.TestCase):
    def test_mask_has_exact_configured_ratio_and_is_reproducible(self):
        sampler = SpatiotemporalMaskSampler(
            (4, 5), MaskConfig(ratio=0.5, spatial_blocks=1, temporal_tubes=1)
        )
        first = sampler.sample(2, 4, generator=torch.Generator().manual_seed(7))
        second = sampler.sample(2, 4, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, 4, 20))
        self.assertEqual(first.sum(dim=(1, 2)).tolist(), [40, 40])
        self.assertTrue(first.dtype == torch.bool)

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            MaskConfig(ratio=1.0)
