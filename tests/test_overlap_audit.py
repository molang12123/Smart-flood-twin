from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from minxiong_pinn.overlap_audit import (
    build_approved_manifest,
    evaluate_overlap_consistency,
)


class _Terrain:
    def __init__(self):
        self.active_rows = np.array([0, 0, 1, 1])
        self.active_cols = np.array([0, 1, 0, 1])
        self.active_count = 4
        self.bounds = (0., 2., 0., 2.)
        self.dx = self.dy = 1.
        self.fingerprint = "fixed-terrain"

    def sample_boundary(self, rng, count):
        indexes = rng.integers(0, 4, count)
        xy = np.array([[.5, .5], [1.5, .5], [.5, 1.5], [1.5, 1.5]])[indexes]
        normals = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])[indexes]
        return xy, normals


class _Expert(nn.Module):
    def __init__(self, depth_factor=1., velocity=0.):
        super().__init__()
        self.depth_factor = depth_factor
        self.velocity = velocity

    def forward(self, points):
        h = (points[:, 3] * torch.minimum(points[:, 2], points[:, 4]) *
             self.depth_factor)
        u = torch.full_like(h, self.velocity)
        return torch.stack((h, u, torch.zeros_like(h)), dim=1)


class _BiasedProjectionExpert(_Expert):
    mass_conserving_projection = True

    def __init__(self):
        nn.Module.__init__(self)

    def depth_factor(self, points):
        return torch.full((len(points), 1), 1.2, dtype=points.dtype)

    def _anchor_factor_means(self, conditions):
        return torch.ones(len(conditions), dtype=conditions.dtype)

    def forward(self, points, *, projection_factor_mean=None):
        h = points[:, 3] * torch.minimum(points[:, 2], points[:, 4])
        if projection_factor_mean is not None:
            h = h / projection_factor_mean
        return torch.stack((h, torch.zeros_like(h), torch.zeros_like(h)), dim=1)


class OverlapAuditTests(unittest.TestCase):
    def settings(self):
        return dict(
            terrain=_Terrain(),
            overlaps=[{"left": "low", "right": "medium",
                       "rainfall_mm_hr": [30., 50.]}],
            durations_hr=[1., 6.],
            time_end_s=36 * 3600.,
            thresholds={"overlap_mean_storage_difference": .02,
                        "overlap_relative_l2_depth": .1},
            spatial_samples=4,
            boundary_samples=8,
            batch_size=16,
            seed=1,
        )

    def test_identical_experts_pass_and_record_boundary_diagnostics(self):
        experts = {"low": _Expert(), "medium": _Expert()}
        result = evaluate_overlap_consistency(experts=experts, **self.settings())
        self.assertTrue(result["passed"])
        self.assertEqual(result["case_count"], 18)
        self.assertEqual(result["worst_depth_relative_l2_difference"], 0.)
        self.assertIn("boundary_relative_l2_discharge_difference", result["cases"][0])

    def test_storage_or_depth_disagreement_fails(self):
        experts = {"low": _Expert(), "medium": _Expert(depth_factor=1.2)}
        result = evaluate_overlap_consistency(experts=experts, **self.settings())
        self.assertFalse(result["passed"])
        self.assertGreater(result["worst_mean_storage_relative_difference"], .02)

    def test_absolute_boundary_velocity_gate_can_reject_matching_experts(self):
        settings = self.settings()
        settings["thresholds"]["overlap_boundary_rms_normal_velocity_m_s"] = .01
        experts = {"low": _Expert(velocity=.1), "medium": _Expert(velocity=.1)}
        result = evaluate_overlap_consistency(experts=experts, **settings)
        self.assertFalse(result["passed"])
        self.assertGreater(result["worst_boundary_rms_normal_velocity_m_s"], .01)

    def test_full_domain_projection_bias_is_a_deployment_gate(self):
        settings = self.settings()
        settings["thresholds"]["mass_projection_anchor_relative_error"] = .02
        experts = {"low": _BiasedProjectionExpert(),
                   "medium": _BiasedProjectionExpert()}
        result = evaluate_overlap_consistency(experts=experts, **settings)
        self.assertFalse(result["passed"])
        self.assertGreater(result["worst_mass_projection_anchor_relative_error"], .02)

    def test_manifest_cannot_be_built_from_failed_gate(self):
        master = {
            "durations_hr": [1., 24.],
            "routing": {"low_max_mm_hr": 40., "medium_max_mm_hr": 250.},
            "training": {"time_end_s": 36 * 3600.},
            "experts": [
                {"name": "low", "intensities_mm_hr": [0., 50.]},
                {"name": "medium", "intensities_mm_hr": [30., 300.]},
                {"name": "high", "intensities_mm_hr": [200., 800.]},
            ],
        }
        checkpoints = {name: {"training_rain_domain": {
            "rainfall_mm_hr": ranges, "duration_s": [3600., 86400.]}}
            for name, ranges in {"low": [0., 50.], "medium": [30., 300.],
                                 "high": [200., 800.]}.items()}
        failed = {"experts": [{"physics_training_passed": True}] * 3,
                  "overlap": {"passed": False}}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "did not pass"):
                build_approved_manifest(master, SimpleNamespace(fingerprint="x"),
                                        checkpoints, failed, Path(folder))
            self.assertFalse((Path(folder) / "manifest.json").exists())

    def test_two_expert_manifest_covers_zero_to_300_without_high_route(self):
        master = {
            "durations_hr": [1., 24.],
            "routing": {"low_max_mm_hr": 40.},
            "training": {"time_end_s": 36 * 3600.},
            "experts": [
                {"name": "low", "intensities_mm_hr": [0., 50.]},
                {"name": "medium", "intensities_mm_hr": [30., 300.]},
            ],
        }
        checkpoints = {
            "low": {"training_rain_domain": {
                "rainfall_mm_hr": [0., 50.],
                "duration_s": [3600., 86400.]}},
            "medium": {"training_rain_domain": {
                "rainfall_mm_hr": [30., 300.],
                "duration_s": [3600., 86400.]}},
        }
        passed = {
            "experts": [{"physics_training_passed": True}] * 2,
            "overlap": {"passed": True},
        }
        with tempfile.TemporaryDirectory() as folder:
            manifest = build_approved_manifest(
                master, SimpleNamespace(fingerprint="x"), checkpoints,
                passed, Path(folder))
        self.assertEqual(
            manifest["training_rain_domain"]["rainfall_mm_hr"], [0., 300.])
        self.assertEqual(manifest["routing"], {"low_max_mm_hr": 40.})
        self.assertEqual(
            [item["model_id"] for item in manifest["experts"]],
            ["low", "medium"])


if __name__ == "__main__":
    unittest.main()
