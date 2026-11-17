from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from minxiong_pinn.continuous_audit import (
    evaluate_continuous_domain, latin_hypercube_events,
)
from minxiong_pinn.overlap_audit import build_approved_manifest


class _Terrain:
    def __init__(self):
        self.active_rows = np.array([0, 0, 1, 1])
        self.active_cols = np.array([0, 1, 0, 1])
        self.active_count = 4
        self.bounds = (0., 2., 0., 2.)
        self.dx = self.dy = 1.
        self.fingerprint = "continuous-audit-terrain"
        self._xy = np.array([[.5, 1.5], [1.5, 1.5],
                             [.5, .5], [1.5, .5]])

    def sample_points(self, rng, count, *, jitter=True, slope_fraction=0.):
        indexes = rng.integers(0, self.active_count, count)
        return self._xy[indexes]

    def sample_boundary(self, rng, count):
        indexes = rng.integers(0, self.active_count, count)
        normals = np.array([[-1., 0.], [1., 0.],
                            [0., -1.], [0., 1.]])[indexes]
        return self._xy[indexes], normals

    def query(self, x, y):
        x = np.asarray(x, dtype=float)
        return {"z": np.zeros_like(x), "zx": np.zeros_like(x),
                "zy": np.zeros_like(x), "n": np.full_like(x, .03)}


class _UniformRain(nn.Module):
    def __init__(self, factor=1.):
        super().__init__()
        self.factor = float(factor)
        self.scales = {"velocity_m_s": 1., "time_s": 7200.}

    def forward(self, points):
        h = (points[:, 3] * torch.minimum(points[:, 2], points[:, 4]) *
             self.factor)
        return torch.stack((h, torch.zeros_like(h), torch.zeros_like(h)), dim=1)


def _master(*, loose=False):
    limit = 1e9 if loose else 1e-10
    return {
        "durations_hr": [1., 2.],
        "routing": {"low_max_mm_hr": 40.},
        "continuous_domain_audit": {"enabled": True},
        "experts": [
            {"name": "low", "intensities_mm_hr": [0., 50.]},
            {"name": "medium", "intensities_mm_hr": [30., 300.]},
        ],
        "training": {
            "time_end_s": 7200., "physics_form": "strong",
            "interior_points": 8, "boundary_points": 4, "mass_points": 4,
            "h_regularization_m": .001, "slope_sampling_fraction": 0.,
            "continuity_scale_m_s": 1e-8,
            "momentum_scale_m2_s2": .005, "momentum_slope_scale": .1,
            "loss_weights": {"pde": 1., "boundary": 1., "mass": 1.,
                             "interface": 0.},
            "success_thresholds": {
                "test_mean_pde_loss": limit,
                "test_mean_continuity_loss": limit,
                "test_mean_mass_loss": limit,
                "worst_all_event_pde_loss": limit,
                "worst_all_event_continuity_loss": limit,
                "worst_all_event_mass_loss": limit,
                "worst_all_event_boundary_rms_normal_velocity_m_s": limit,
            },
        },
    }


def _checkpoints():
    return {name: {"scales": {}}
            for name in ("low", "medium")}


def _settings(**changes):
    result = {
        "conditions_per_expert": 4,
        "full_domain_conditions_per_expert": 2,
        "interior_points": 8,
        "repeats": 1,
        "batch_size": 4,
        "seed": 17,
        "max_full_domain_mass_relative_error": 1e-10,
    }
    result.update(changes)
    return result


class ContinuousDomainAuditTests(unittest.TestCase):
    def test_latin_hypercube_is_deterministic_off_grid_and_in_primary_routes(self):
        master = _master()
        first = latin_hypercube_events(master, 8, seed=123)
        second = latin_hypercube_events(master, 8, seed=123)
        self.assertEqual(first, second)
        fixed_rain = {0., 30., 40., 50., 300.}
        fixed_duration = {3600., 7200.}
        for name, bounds in {"low": (0., 40.),
                             "medium": (40., 300.)}.items():
            self.assertEqual(len(first[name]), 8)
            for event in first[name]:
                self.assertGreater(event["mean_intensity_mm_hr"], bounds[0])
                self.assertLess(event["mean_intensity_mm_hr"], bounds[1])
                self.assertNotIn(event["mean_intensity_mm_hr"], fixed_rain)
                self.assertNotIn(event["duration_s"], fixed_duration)

    def test_exact_uniform_rain_passes_physics_and_full_active_dem_mass(self):
        experts = {"low": _UniformRain(), "medium": _UniformRain()}
        report = evaluate_continuous_domain(
            experts, _checkpoints(), _Terrain(), _master(), _settings())
        self.assertTrue(report["passed"])
        self.assertEqual(report["full_domain_active_cells"], 4)
        for expert in report["experts"]:
            self.assertTrue(expert["physics"]["passed"])
            self.assertTrue(expert["full_domain_mass_passed"])
            self.assertLessEqual(
                expert["worst_full_domain_mass_relative_error"], 1e-10)
            self.assertTrue(expert["full_domain_mass_cases"])

    def test_full_domain_storage_bias_fails_even_with_loose_pointwise_gates(self):
        experts = {"low": _UniformRain(1.2), "medium": _UniformRain(1.2)}
        report = evaluate_continuous_domain(
            experts, _checkpoints(), _Terrain(), _master(loose=True),
            _settings(max_full_domain_mass_relative_error=.05))
        self.assertFalse(report["passed"])
        self.assertTrue(all(item["physics"]["passed"]
                            for item in report["experts"]))
        self.assertTrue(all(not item["full_domain_mass_passed"]
                            for item in report["experts"]))
        self.assertTrue(all(
            item["worst_full_domain_mass_relative_error"] > .19
            for item in report["experts"]))

    def test_failed_required_continuous_audit_prevents_manifest(self):
        master = _master()
        checkpoints = {
            "low": {"training_rain_domain": {
                "rainfall_mm_hr": [0., 50.], "duration_s": [3600., 7200.]}},
            "medium": {"training_rain_domain": {
                "rainfall_mm_hr": [30., 300.], "duration_s": [3600., 7200.]}},
        }
        audit = {
            "experts": [{"physics_training_passed": True}] * 2,
            "overlap": {"passed": True},
            "continuous_domain": {"passed": False},
        }
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "did not pass"):
                build_approved_manifest(
                    master, SimpleNamespace(fingerprint="x"), checkpoints,
                    audit, Path(folder))


if __name__ == "__main__":
    unittest.main()
