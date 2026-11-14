from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.model import ConditionalFloodPINN
from minxiong_pinn.predict import predict
from minxiong_pinn.terrain import MaskedRasterTerrain


def _terrain(folder):
    path = Path(folder) / "terrain.tif"
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1,
                       dtype="float32", crs="EPSG:3826",
                       transform=from_origin(0, 80, 20, 20), nodata=-9999.) as dst:
        dst.write(values, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


def _scales():
    return {"x0": 0., "y0": 0., "lx": 80., "ly": 80.,
            "time_s": 36 * 3600., "rain_m_s": 800 / 3_600_000.,
            "duration_s": 24 * 3600., "velocity_m_s": 4., "ramp_s": 600.}


def _model(terrain, anchors=16):
    model = ConditionalFloodPINN(
        _scales(), width=4, layers=1, mass_conserving_projection=True,
        mass_projection_anchor_points=anchors, mass_projection_strata=min(4, anchors),
        mass_projection_seed=7, terrain=terrain)
    with torch.no_grad():
        model.net[0].weight.zero_(); model.net[0].bias.zero_()
        model.net[0].weight[0, 0] = 4.
        model.net[-1].weight.zero_(); model.net[-1].bias.zero_()
        model.net[-1].weight[0, 0] = 3.
    return model


def _centres(terrain, *, time_s=3600., rain_mm_hr=10., duration_s=3600.):
    west, _, _, north = terrain.bounds
    xy = np.column_stack((west + (terrain.active_cols + .5) * terrain.dx,
                          north - (terrain.active_rows + .5) * terrain.dy))
    points = np.empty((len(xy), 5), dtype=np.float64)
    points[:, :2] = xy; points[:, 2] = time_s
    points[:, 3] = rain_mm_hr / 3_600_000.; points[:, 4] = duration_s
    return torch.as_tensor(points, dtype=torch.float64)


class HardMassProjectionTests(unittest.TestCase):
    def test_anchor_projection_is_batch_order_and_mixed_condition_invariant(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder); model = _model(terrain, anchors=8)
            first = _centres(terrain)[:3]
            later = first.clone(); later[:, 2] = 7200.
            mixed = torch.cat((later[1:], first, later[:1]), dim=0)
            expected_first, expected_later = model(first), model(later)
            actual = model(mixed)
            torch.testing.assert_close(actual[2:5], expected_first, rtol=0, atol=0)
            torch.testing.assert_close(actual[:2], expected_later[1:], rtol=0, atol=0)
            torch.testing.assert_close(actual[5:], expected_later[:1], rtol=0, atol=0)
            for index in range(3):
                torch.testing.assert_close(model(first[index:index + 1]),
                                           expected_first[index:index + 1], rtol=0, atol=0)

    def test_all_cell_anchors_enforce_exact_area_mean_and_finite_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder); model = _model(terrain, terrain.active_count)
            result = model(_centres(terrain))
            torch.testing.assert_close(result[:, 0].mean(), torch.tensor(.01),
                                       rtol=2e-6, atol=1e-9)
            result[:, 0].square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_external_full_domain_mean_is_chunk_invariant(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder); model = _model(terrain, anchors=4)
            points = _centres(terrain); factor_mean = model.depth_factor(points).mean()
            whole = model(points, projection_factor_mean=factor_mean)
            chunks = torch.cat([model(points[i:i + 3], projection_factor_mean=factor_mean)
                                for i in range(0, len(points), 3)])
            torch.testing.assert_close(chunks, whole, rtol=0, atol=0)
            torch.testing.assert_close(whole[:, 0].mean(), torch.tensor(.01),
                                       rtol=2e-6, atol=1e-9)

    def test_zero_rain_and_zero_time_are_dry_without_division_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder); model = _model(terrain, anchors=4)
            points = _centres(terrain)[:2]; points[0, 3] = 0.; points[1, 2] = 0.
            result = model(points)
            torch.testing.assert_close(result[:, 0], torch.zeros(2), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result).all())

    def test_prediction_two_pass_is_exact_and_batch_size_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder); model = _model(terrain, anchors=4)
            checkpoint = {"training_rain_domain": {
                "rainfall_mm_hr": [0., 800.], "duration_s": [3600., 86400.]},
                "model_id": "projection-test"}
            one = Path(folder) / "one"; seven = Path(folder) / "seven"
            predict(model, checkpoint, terrain, 10., 3600., [3600.], one, batch_size=1)
            predict(model, checkpoint, terrain, 10., 3600., [3600.], seven, batch_size=7)
            left = np.load(one / "frame_000.npz")["h"]
            right = np.load(seven / "frame_000.npz")["h"]
            np.testing.assert_array_equal(left, right)
            self.assertAlmostEqual(float(left.mean()), .01, places=7)


if __name__ == "__main__":
    unittest.main()
