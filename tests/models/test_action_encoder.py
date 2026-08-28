import unittest

try:
    import torch
    from mcwm.models.action_encoder import (
        ActionEncoderConfig,
        BinaryComponentEncoder,
        CameraEncoder,
        ComponentFusion,
        CursorEncoder,
        HotbarEncoder,
        MinecraftActionEncoder,
        MicroActionTransformer,
        mu_law_normalize,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ActionComponentEncoderTest(unittest.TestCase):
    def test_binary_components_use_independent_embeddings(self):
        encoder = BinaryComponentEncoder(components=14, embedding_dim=4)
        buttons = torch.zeros(2, 3, 14, dtype=torch.bool)
        buttons[0, 0, 0] = True
        buttons[0, 0, 7] = True

        encoded = encoder(buttons)

        self.assertEqual(len(encoder.embeddings), 14)
        self.assertEqual(tuple(encoded.shape), (2, 3, 56))
        self.assertFalse(torch.equal(encoded[0, 0], encoded[1, 0]))

    def test_hotbar_encoder_supports_ten_classes(self):
        encoder = HotbarEncoder(embedding_dim=6)

        encoded = encoder(torch.arange(10))

        self.assertEqual(tuple(encoded.shape), (10, 6))

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

    def test_component_fusion_produces_tick_tokens(self):
        fusion = ComponentFusion(6, 3, 4, 5, tick_dim=16, hidden_dim=24)
        leading_shape = (2, 3, 4)

        tokens = fusion(
            torch.randn(*leading_shape, 6),
            torch.randn(*leading_shape, 3),
            torch.randn(*leading_shape, 4),
            torch.randn(*leading_shape, 5),
        )

        self.assertEqual(tuple(tokens.shape), (2, 3, 4, 16))

    def test_component_fusion_rejects_mismatched_shapes(self):
        fusion = ComponentFusion(6, 3, 4, 5, tick_dim=16)

        with self.assertRaisesRegex(ValueError, "leading dimensions"):
            fusion(
                torch.randn(2, 3, 6),
                torch.randn(2, 4, 3),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 5),
            )

    def test_micro_transformer_masks_padding(self):
        transformer = MicroActionTransformer(
            dim=16,
            depth=2,
            heads=4,
            mlp_dim=32,
        ).eval()
        tokens = torch.randn(2, 3, 4, 16)
        valid_mask = torch.tensor(
            [
                [[True, True, False, False]] * 3,
                [[True, True, True, False]] * 3,
            ]
        )

        encoded = transformer(tokens, valid_mask)
        changed_padding = tokens.clone()
        changed_padding[~valid_mask] = 1000.0
        encoded_after_change = transformer(changed_padding, valid_mask)

        self.assertEqual(tuple(encoded.shape), (2, 3, 4, 16))
        self.assertTrue(
            torch.equal(
                encoded[~valid_mask],
                torch.zeros_like(encoded[~valid_mask]),
            )
        )
        self.assertTrue(
            torch.allclose(
                encoded[valid_mask],
                encoded_after_change[valid_mask],
                atol=1e-6,
            )
        )

    def test_micro_transformer_handles_empty_padded_interval(self):
        transformer = MicroActionTransformer(
            dim=16,
            depth=2,
            heads=4,
            mlp_dim=32,
        )
        tokens = torch.randn(1, 1, 3, 16)
        valid_mask = torch.zeros(1, 1, 3, dtype=torch.bool)

        encoded = transformer(tokens, valid_mask)

        self.assertTrue(torch.equal(encoded, torch.zeros_like(encoded)))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MinecraftActionEncoderTest(unittest.TestCase):
    def setUp(self):
        self.config = ActionEncoderConfig(
            binary_embedding_dim=4,
            hotbar_embedding_dim=4,
            camera_dim=8,
            cursor_dim=8,
            component_hidden_dim=32,
            tick_dim=16,
            transformer_depth=2,
            transformer_heads=4,
            transformer_mlp_dim=32,
            macro_dim=32,
        )

    @staticmethod
    def _inputs(batch=2, intervals=3, ticks=4):
        shape = (batch, intervals, ticks)
        return {
            "movement": torch.zeros(*shape, 7, dtype=torch.bool),
            "interaction": torch.zeros(*shape, 7, dtype=torch.bool),
            "hotbar": torch.zeros(shape, dtype=torch.long),
            "camera": torch.zeros(*shape, 2),
            "cursor": torch.zeros(*shape, 2),
            "gui_open": torch.zeros(shape, dtype=torch.bool),
            "cursor_present": torch.zeros(shape, dtype=torch.bool),
            "valid_mask": torch.ones(shape, dtype=torch.bool),
        }

    def test_full_encoder_outputs_one_macro_token_per_interval(self):
        model = MinecraftActionEncoder(self.config)
        inputs = self._inputs()
        inputs["movement"][..., 0] = True
        inputs["camera"].requires_grad_(True)

        output = model(**inputs)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 3, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(inputs["camera"].grad).all())

    def test_padding_values_do_not_change_macro_token(self):
        model = MinecraftActionEncoder(self.config).eval()
        inputs = self._inputs(batch=1, intervals=1, ticks=3)
        inputs["valid_mask"][..., -1] = False

        expected = model(**inputs)
        inputs["movement"][..., -1, :] = True
        inputs["interaction"][..., -1, :] = True
        inputs["hotbar"][..., -1] = 9
        inputs["camera"][..., -1, :] = 180.0
        inputs["cursor"][..., -1, :] = 1.0
        inputs["gui_open"][..., -1] = True
        inputs["cursor_present"][..., -1] = True

        actual = model(**inputs)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_tick_order_changes_macro_token(self):
        model = MinecraftActionEncoder(self.config).eval()
        inputs = self._inputs(batch=1, intervals=1, ticks=2)
        inputs["movement"][0, 0, 0, 0] = True
        inputs["interaction"][0, 0, 1, 0] = True

        forward_order = model(**inputs)
        reversed_inputs = {
            name: value.flip(2)
            for name, value in inputs.items()
        }
        reversed_order = model(**reversed_inputs)

        self.assertFalse(torch.allclose(forward_order, reversed_order))

    def test_valid_noop_is_not_padding(self):
        model = MinecraftActionEncoder(self.config).eval()
        inputs = self._inputs(batch=1, intervals=2, ticks=1)
        inputs["valid_mask"][0, 1, 0] = False

        output = model(**inputs)

        self.assertGreater(float(output[0, 0].abs().sum()), 0.0)
        self.assertTrue(torch.equal(output[0, 1], torch.zeros(32)))

    def test_default_parameter_budget(self):
        model = MinecraftActionEncoder()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertGreaterEqual(parameter_count, 1_500_000)
        self.assertLessEqual(parameter_count, 3_000_000)


if __name__ == "__main__":
    unittest.main()
