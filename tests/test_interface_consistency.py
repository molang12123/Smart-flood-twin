from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import (
    _interface_consistency_loss, _interface_thresholds_pass,
    _scheduled_interface_weight,
)


def _terrain(folder, values):
    values = np.asarray(values, dtype=np.float32)
    path = Path(folder) / "terrain.tif"
    with rasterio.open(
            path, "w", driver="GTiff", width=values.shape[1],
            height=values.shape[0], count=1, dtype="float32",
            crs="EPSG:3826", transform=from_origin(
                0., 20. * values.shape[0], 20., 20.), nodata=-9999.) as dst:
        dst.write(values, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


def _case():
    return {"mean_intensity_mm_hr": 5., "duration_s": 3600.,
            "total_mm": 5.}


def _config(**updates):
    result = {"time_end_s": 7200., "h_regularization_m": .001,
              "interface_points": 1, "high_relief_fraction": 1.,
              "h_wet": .001}
    result.update(updates)
    return result


class _StepLake(torch.nn.Module):
    def __init__(self, face_x=20., level=2.):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.face_x = face_x
        self.level = level

    def forward(self, points):
        right = points[:, 0] >= self.face_x
        bed = torch.where(right, points.new_tensor(1.), points.new_tensor(0.))
        h = self.level - bed + self.marker * 0.
        return torch.stack((h, torch.zeros_like(h), torch.zeros_like(h)), dim=1)


class _Discontinuous(torch.nn.Module):
    def __init__(self, face_x=20.):
        super().__init__()
        self.jump = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
        self.face_x = face_x

    def forward(self, points):
        right = points[:, 0] >= self.face_x
        h = torch.where(right, 2. + self.jump, 1. + self.jump * 0.)
        u = torch.where(right, 3. * self.jump, self.jump)
        v = torch.where(right, 2. * self.jump, .5 * self.jump)
        return torch.stack((h, u, v), dim=1)


class _WetDry(torch.nn.Module):
    def __init__(self, face_x=20., *, both_dry=False):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.face_x = face_x
        self.both_dry = both_dry

    def forward(self, points):
        right = points[:, 0] >= self.face_x
        left_depth = 0. if self.both_dry else 1.
        h = torch.where(right, points.new_zeros(()),
                        points.new_full((), left_depth))
        u = torch.ones_like(h) * self.marker
        return torch.stack((h, u, -u), dim=1)


class InterfaceConsistencyTests(unittest.TestCase):
    def test_bed_step_lake_at_rest_is_exactly_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0., 1.]])
            parts = _interface_consistency_loss(
                _StepLake(), terrain, _case(), np.random.default_rng(1),
                _config())
            for name, value in parts.items():
                with self.subTest(name=name):
                    self.assertEqual(float(value.detach()), 0.)

    def test_discontinuous_wet_traces_are_positive_and_differentiable(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0., 0.]])
            model = _Discontinuous()
            parts = _interface_consistency_loss(
                model, terrain, _case(), np.random.default_rng(2), _config())
            self.assertGreater(float(parts["interface_depth"].detach()), 0.)
            self.assertGreater(float(parts["interface_normal_flux"].detach()), 0.)
            self.assertGreater(float(parts["interface_tangential_flux"].detach()), 0.)
            parts["interface"].backward()
            self.assertIsNotNone(model.jump.grad)
            self.assertTrue(torch.isfinite(model.jump.grad))
            self.assertNotEqual(float(model.jump.grad), 0.)

    def test_wet_dry_and_both_dry_faces_are_stable_and_unpenalised(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0., 0.]])
            for both_dry in (False, True):
                with self.subTest(both_dry=both_dry):
                    model = _WetDry(both_dry=both_dry)
                    parts = _interface_consistency_loss(
                        model, terrain, _case(), np.random.default_rng(3),
                        _config())
                    self.assertEqual(float(parts["interface"].detach()), 0.)
                    parts["interface"].backward()
                    self.assertTrue(torch.isfinite(model.marker.grad))
                    self.assertEqual(float(model.marker.grad), 0.)

    def test_invalid_settings_and_nodata_separation_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0., 0.]])
            for changes in ({"interface_points": 0}, {"h_wet": 0.}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    _interface_consistency_loss(
                        _StepLake(), terrain, _case(), np.random.default_rng(4),
                        _config(**changes))
        with tempfile.TemporaryDirectory() as folder:
            disconnected = _terrain(folder, [[0., -9999., 1.]])
            with self.assertRaisesRegex(ValueError, "no active-active"):
                _interface_consistency_loss(
                    _StepLake(), disconnected, _case(),
                    np.random.default_rng(5), _config())

    def test_weight_curriculum_defaults_off_and_ramps_to_target(self):
        self.assertEqual(_scheduled_interface_weight({"loss_weights": {}}), 0.)
        cfg = {"loss_weights": {"interface": .2},
               "interface_start_step": 100, "interface_ramp_steps": 300}
        self.assertEqual(_scheduled_interface_weight(cfg, 100), 0.)
        self.assertAlmostEqual(_scheduled_interface_weight(cfg, 250), .1)
        self.assertEqual(_scheduled_interface_weight(cfg, 400), .2)
        for updates in (
                {"interface_loss_weight": -1.},
                {"interface_start_step": -1},
                {"interface_ramp_steps": True}):
            invalid = {**cfg, **updates}
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                _scheduled_interface_weight(invalid, 1)

    def test_interface_gates_are_optional_and_enforced_when_present(self):
        test = {
            "mean_interface_loss": .04,
            "mean_interface_depth_loss": .01,
            "mean_interface_normal_flux_loss": .03,
            "mean_interface_tangential_flux_loss": .02,
        }
        worst = {"interface": .06, "depth": .02,
                 "normal_flux": .04, "tangential_flux": .03}
        self.assertTrue(_interface_thresholds_pass({}, test, worst))
        self.assertTrue(_interface_thresholds_pass(
            {"test_mean_interface_depth_loss": .02}, test, worst))
        self.assertFalse(_interface_thresholds_pass(
            {"worst_all_event_interface_loss": .05}, test, worst))


if __name__ == "__main__":
    unittest.main()
