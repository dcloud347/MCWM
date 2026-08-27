import unittest

try:
    import torch
    from mcwm.actions.schema import ActionSource, CanonicalActionTick
    from mcwm.diagnostics.m2_encoder_probe import probe_samples, probe_variable_lengths
    from mcwm.models.frozen_visual_encoder import FrozenVisualEncoder
    from mcwm.models.visual_encoder import VisualEncoderConfig
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class M2EncoderProbeTest(unittest.TestCase):
    def setUp(self):
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
        self.model = FrozenVisualEncoder(config)

    def test_real_sample_report_is_finite_and_has_expected_shape(self):
        action_blocks = tuple(
            (CanonicalActionTick.noop(index * 250, ActionSource.VPT),)
            for index in range(2)
        )
        sample = {
            "frames": torch.randint(0, 256, (3, 3, 20, 30), dtype=torch.uint8),
            "action_blocks": action_blocks,
        }

        report = probe_samples(
            self.model,
            (sample,),
            device=torch.device("cpu"),
            precision="fp32",
            frame_chunk_size=1,
            max_samples=1,
        )

        self.assertEqual(report["output_shape_per_sample"], [3, 6, 24])
        self.assertEqual(report["deterministic_max_abs_error"], 0.0)
        self.assertTrue(
            torch.isfinite(
                torch.tensor(
                    report["distribution_shift"][
                        "repeated_vs_continuous_mean_cosine"
                    ]
                )
            )
        )

    def test_all_even_runtime_lengths_are_accepted(self):
        frame = torch.zeros(1, 1, 3, 20, 30, dtype=torch.uint8)

        report = probe_variable_lengths(
            self.model,
            frame,
            device=torch.device("cpu"),
            precision="fp32",
        )

        self.assertEqual(tuple(int(value) for value in report), tuple(range(2, 17, 2)))
        self.assertTrue(all(item["finite"] for item in report.values()))


if __name__ == "__main__":
    unittest.main()
