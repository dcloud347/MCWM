import unittest

try:
    import torch
    from mcwm.diagnostics.collapse import collapse_metrics, find_collapse_alerts
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CollapseTest(unittest.TestCase):
    def test_constant_latents_trigger_alerts(self):
        metrics = collapse_metrics(torch.ones(16, 8))
        alerts = find_collapse_alerts(metrics)
        self.assertIn("std too low", alerts)
        self.assertIn("effective rank too low", alerts)

    def test_diverse_latents_have_finite_metrics(self):
        metrics = collapse_metrics(torch.randn(32, 8))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
