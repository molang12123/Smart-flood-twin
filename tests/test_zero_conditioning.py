import math
import unittest

import torch

from minxiong_pinn.model import ConditionalFloodPINN


def _scales():
    return {
        "x0": 0., "y0": 0., "lx": 1., "ly": 1., "time_s": 36 * 3600.,
        "rain_m_s": 800 / 3_600_000., "duration_s": 24 * 3600.,
        "rain_log_reference_m_s": 1 / 3_600_000.,
        "duration_log_reference_s": 3600., "velocity_m_s": 20.,
        "ramp_s": 600., "wet_velocity_gate_m": 1e-5,
    }


class ZeroSafeConditioningTests(unittest.TestCase):
    def test_log1p_conditioning_maps_zero_and_domain_maximum(self):
        model = ConditionalFloodPINN(_scales(), width=4, layers=1,
                                     log1p_conditions=True)
        captured = []
        hook = model.net[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach()))
        points = torch.tensor([
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 800 / 3_600_000., 24 * 3600.],
        ], dtype=torch.float64)
        model(points); hook.remove()
        torch.testing.assert_close(captured[0][:, 3:5],
                                   torch.tensor([[0., 0.], [1., 1.]]),
                                   rtol=0, atol=1e-6)

    def test_zero_rain_is_exactly_dry_even_with_nonzero_network_output(self):
        model = ConditionalFloodPINN(_scales(), width=6, layers=2,
                                     log1p_conditions=True,
                                     wet_velocity_gate=True)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(.5)
        points = torch.tensor([
            [.2, .3, 3600., 0., 3600.],
            [.8, .9, 36 * 3600., 0., 24 * 3600.],
        ], dtype=torch.float64)
        torch.testing.assert_close(model(points), torch.zeros((2, 3)), atol=0, rtol=0)

    def test_log1p_conditioning_rejects_negative_values(self):
        model = ConditionalFloodPINN(_scales(), width=4, layers=1,
                                     log1p_conditions=True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model(torch.tensor([[0., 0., 1., -1e-8, 3600.]], dtype=torch.float64))

    def test_conditioning_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ConditionalFloodPINN(_scales(), width=4, layers=1,
                                 log_conditions=True, log1p_conditions=True)

    def test_legacy_flags_keep_previous_velocity_behavior(self):
        model = ConditionalFloodPINN(_scales(), width=4, layers=1)
        with torch.no_grad():
            model.net[-1].bias[1:] = 1.
        points = torch.tensor([[0., 0., 3600., 0., 3600.]], dtype=torch.float64)
        result = model(points)
        self.assertEqual(float(result[0, 0].detach()), 0.)
        self.assertGreater(
            float(torch.linalg.vector_norm(result[0, 1:]).detach()), 0.)


if __name__ == "__main__":
    unittest.main()
