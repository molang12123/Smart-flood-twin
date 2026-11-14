import unittest

import numpy as np
import torch

from minxiong_pinn.model import (
    ConditionalFloodPINN, ENHANCED_TIME_FEATURE_NAMES, TERRAIN_FEATURE_NAMES,
)


class _FlatTerrain:
    def __init__(self):
        self.z = np.zeros((2, 2)); self.mask = np.ones((2, 2), dtype=bool)
        self.zx = np.zeros((2, 2)); self.zy = np.zeros((2, 2))
        self.active_rows, self.active_cols = np.nonzero(self.mask)
        self.active_count = 4; self.bounds = (0., 2., 0., 2.)
        self.dx = self.dy = 1.; self.width = self.height = 2


def _scales():
    return {"x0": 0., "y0": 0., "lx": 2., "ly": 2.,
            "time_s": 36 * 3600., "rain_m_s": 800 / 3_600_000.,
            "duration_s": 24 * 3600., "velocity_m_s": 4., "ramp_s": 600.,
            "time_log_reference_s": 600.}


class EnhancedTimeFeatureTests(unittest.TestCase):
    def test_legacy_and_terrain_only_input_widths_are_unchanged(self):
        self.assertEqual(ConditionalFloodPINN(_scales(), 4, 1).net[0].in_features, 5)
        model = ConditionalFloodPINN(_scales(), 4, 1, terrain_features=True,
                                     terrain=_FlatTerrain())
        self.assertEqual(model.net[0].in_features, 5 + len(TERRAIN_FEATURE_NAMES))

    def test_log_time_and_rain_phases_have_fixed_order(self):
        model = ConditionalFloodPINN(_scales(), 4, 1, enhanced_time_features=True)
        captured = []
        hook = model.net[0].register_forward_pre_hook(
            lambda _m, args: captured.append(args[0].detach()))
        points = torch.tensor([
            [1., 1., 0., 10 / 3_600_000., 3600.],
            [1., 1., 1800., 10 / 3_600_000., 3600.],
            [1., 1., 7200., 10 / 3_600_000., 3600.],
        ], dtype=torch.float64)
        model(points); hook.remove()
        temporal = captured[0][:, -len(ENHANCED_TIME_FEATURE_NAMES):]
        self.assertEqual(model.net[0].in_features, 8)
        self.assertEqual(float(temporal[0, 0]), 0.)
        torch.testing.assert_close(temporal[:, 1], torch.tensor([0., .5, 1.]),
                                   atol=1e-6, rtol=0)
        self.assertEqual(float(temporal[0, 2]), 0.)
        self.assertEqual(float(temporal[1, 2]), 0.)
        self.assertGreater(float(temporal[2, 2]), 0.)

    def test_time_features_follow_terrain_features(self):
        model = ConditionalFloodPINN(_scales(), 4, 1, terrain_features=True,
                                     terrain=_FlatTerrain(),
                                     enhanced_time_features=True)
        self.assertEqual(model.net[0].in_features,
                         5 + len(TERRAIN_FEATURE_NAMES) + len(ENHANCED_TIME_FEATURE_NAMES))

    def test_invalid_time_or_duration_is_rejected(self):
        model = ConditionalFloodPINN(_scales(), 4, 1, enhanced_time_features=True)
        with self.assertRaisesRegex(ValueError, "nonnegative time"):
            model(torch.tensor([[1., 1., -1., 1e-6, 3600.]], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "positive duration"):
            model(torch.tensor([[1., 1., 1., 1e-6, 0.]], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
