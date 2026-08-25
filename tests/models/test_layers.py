import unittest

try:
    import torch
    from mcwm.models.layers import _apply_rotary
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TransformerLayersTest(unittest.TestCase):
    def test_rope_uses_vjepa2_frequency_layout(self):
        values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64)
        positions = torch.tensor([[1]])

        frequency = torch.tensor([1.0, 0.01], dtype=torch.float64)
        cosine = frequency.cos().repeat(2).reshape(1, 1, 1, 4)
        sine = frequency.sin().repeat(2).reshape(1, 1, 1, 4)
        quarter_turn = torch.tensor(
            [[[[-2.0, 1.0, -4.0, 3.0]]]],
            dtype=torch.float64,
        )
        expected = values * cosine + quarter_turn * sine

        self.assertTrue(torch.allclose(_apply_rotary(values, positions), expected))


if __name__ == "__main__":
    unittest.main()
