from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import _fv_residuals, _hll_flux, _hydrostatic_face


def _terrain(folder, elevations):
    path = Path(folder) / "dem.tif"
    elevations = np.asarray(elevations, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=elevations.shape[1],
                       height=elevations.shape[0], count=1, dtype="float32",
                       crs="EPSG:3826",
                       transform=from_origin(0, elevations.shape[0] * 20, 20, 20),
                       nodata=-9999.) as dst:
        dst.write(elevations, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


class _LakeAtRest(torch.nn.Module):
    def __init__(self, terrain, eta):
        super().__init__(); self.terrain, self.eta = terrain, float(eta)

    def forward(self, points):
        xy = points[:, :2].detach().cpu().numpy()
        z = self.terrain.query(xy[:, 0], xy[:, 1])["z"]
        h = torch.clamp(points.new_tensor(self.eta - z), min=0.)
        return torch.stack((h, torch.zeros_like(h), torch.zeros_like(h)), dim=1)


class _UniformRain(torch.nn.Module):
    def forward(self, points):
        h = points[:, 3] * torch.minimum(points[:, 2], points[:, 4])
        return torch.stack((h, torch.zeros_like(h), torch.zeros_like(h)), dim=1)


class _ScriptedRng:
    def integers(self, low, high=None, size=None): return np.zeros(size, dtype=int)
    def uniform(self, low=0., high=1., size=None): return np.full(size, 3300.)


class WellBalancedFiniteVolumeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"interior_points": 24, "weak_dt_s": 600.,
                    "time_end_s": 7200., "h_regularization_m": 1e-3}

    def test_hll_equal_state_is_physical_flux_and_dry_is_zero(self):
        state = torch.tensor([[2., 3., -.5], [.7, -1.2, .4]], dtype=torch.float64)
        g = 9.81
        expected_x = torch.stack((state[:, 0] * state[:, 1],
                                  state[:, 0] * state[:, 1].square() + .5*g*state[:, 0].square(),
                                  state[:, 0] * state[:, 1] * state[:, 2]), dim=1)
        expected_y = torch.stack((state[:, 0] * state[:, 2],
                                  state[:, 0] * state[:, 1] * state[:, 2],
                                  state[:, 0] * state[:, 2].square() + .5*g*state[:, 0].square()), dim=1)
        torch.testing.assert_close(_hll_flux(state, state, axis=0), expected_x)
        torch.testing.assert_close(_hll_flux(state, state, axis=1), expected_y)
        dry = torch.zeros((3, 3), dtype=torch.float64)
        torch.testing.assert_close(_hll_flux(dry, dry, axis=0), dry)

    def test_hll_shared_face_reversal_is_conservative(self):
        left = torch.tensor([[1.2, .8, -.2]], dtype=torch.float64)
        right = torch.tensor([[.7, -.4, .3]], dtype=torch.float64)
        flux = _hll_flux(left, right, axis=0)
        reflected_left = right.clone(); reflected_left[:, 1] *= -1
        reflected_right = left.clone(); reflected_right[:, 1] *= -1
        reflected = _hll_flux(reflected_left, reflected_right, axis=0)
        expected = flux.clone(); expected[:, (0, 2)] *= -1
        torch.testing.assert_close(reflected, expected, rtol=1e-12, atol=1e-12)

    def test_hydrostatic_correction_recovers_each_cell_pressure(self):
        g, eta = 9.81, 5.
        zl = torch.tensor([1.], dtype=torch.float64); zr = torch.tensor([3.], dtype=torch.float64)
        left = torch.tensor([[4., 0., 0.]], dtype=torch.float64)
        right = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
        fl = _hydrostatic_face(left, right, zl, zr, axis=0, centre_side="left")
        fr = _hydrostatic_face(left, right, zl, zr, axis=0, centre_side="right")
        self.assertAlmostEqual(fl[0, 1].item(), .5*g*left[0, 0].item()**2)
        self.assertAlmostEqual(fr[0, 1].item(), .5*g*right[0, 0].item()**2)

    def test_fully_wet_irregular_lake_at_rest(self):
        z = np.array([[0,.2,.6,.4,.1],[.1,.4,.8,.5,.3],[.3,.7,1.1,.9,.6],
                      [.2,.5,.9,.6,.4],[0,.3,.5,.3,.1]])
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, z)
            raw, _, _ = _fv_residuals(_LakeAtRest(terrain, 2.), terrain,
                                      {"mean_intensity_mm_hr":0., "duration_s":3600.},
                                      np.random.default_rng(11), self.cfg)
            torch.testing.assert_close(raw, torch.zeros_like(raw), rtol=0, atol=2e-14)

    def test_wet_dry_shoreline_lake_at_rest(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.tile(np.arange(5, dtype=float), (5,1)))
            cfg = {**self.cfg, "interior_points":128}
            raw, _, _ = _fv_residuals(_LakeAtRest(terrain, 2.5), terrain,
                                      {"mean_intensity_mm_hr":0., "duration_s":3600.},
                                      np.random.default_rng(12), cfg)
            torch.testing.assert_close(raw, torch.zeros_like(raw), rtol=0, atol=2e-14)

    def test_flat_uniform_rain_is_exact_across_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.zeros((3,3)))
            raw, _, _ = _fv_residuals(_UniformRain(), terrain,
                                      {"mean_intensity_mm_hr":10., "duration_s":3600.},
                                      _ScriptedRng(), self.cfg)
            torch.testing.assert_close(raw, torch.zeros_like(raw), rtol=0, atol=2e-14)

    def test_all_dry_zero_rain_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(25, dtype=float).reshape(5,5))
            raw, _, _ = _fv_residuals(_LakeAtRest(terrain, -10.), terrain,
                                      {"mean_intensity_mm_hr":0., "duration_s":3600.},
                                      np.random.default_rng(13), self.cfg)
            torch.testing.assert_close(raw, torch.zeros_like(raw), rtol=0, atol=0)


if __name__ == "__main__": unittest.main()
