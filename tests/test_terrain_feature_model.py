from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.model import (ConditionalFloodPINN, TERRAIN_FEATURE_NAMES,
                                 TERRAIN_LOCAL_COORDINATE_NAMES)
from minxiong_pinn.predict import load_model
from minxiong_pinn.terrain import MaskedRasterTerrain


def _terrain(folder, values, *, nodata=-9999.0):
    path = Path(folder) / "dem.tif"
    values = np.asarray(values, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="float32", crs="EPSG:3826",
        transform=from_origin(0, values.shape[0] * 20, 20, 20), nodata=nodata,
    ) as dst:
        dst.write(values, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


def _scales(terrain):
    return {
        "x0": terrain.bounds[0], "y0": terrain.bounds[2],
        "lx": terrain.bounds[1] - terrain.bounds[0],
        "ly": terrain.bounds[3] - terrain.bounds[2],
        "time_s": 36 * 3600., "rain_m_s": 800 / 3_600_000.,
        "duration_s": 24 * 3600., "depth_m": 19.2,
        "velocity_m_s": 20., "ramp_s": 600.,
    }


class TerrainFeatureModelTests(unittest.TestCase):
    def test_local_cell_coordinates_have_physical_spatial_derivatives(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
            model = ConditionalFloodPINN(
                _scales(terrain), width=4, layers=1, terrain=terrain,
                terrain_features=True, terrain_local_coordinates=True)
            points = torch.tensor(
                [[12., 47., 3600., 10 / 3_600_000., 3600.]],
                dtype=torch.float64, requires_grad=True)
            features = model._network_features(points)
            self.assertEqual(features.shape[1],
                             5 + len(TERRAIN_FEATURE_NAMES) +
                             len(TERRAIN_LOCAL_COORDINATE_NAMES))
            torch.testing.assert_close(
                features[0, -2:], features.new_tensor([.1, -.15]))
            dx = torch.autograd.grad(features[0, -2], points,
                                     retain_graph=True)[0][0]
            dy = torch.autograd.grad(features[0, -1], points)[0][0]
            self.assertAlmostEqual(float(dx[0]), 1 / terrain.dx)
            self.assertAlmostEqual(float(dy[1]), 1 / terrain.dy)

    def test_adjacent_sharp_cells_have_explicit_distinct_features(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0, 30, 1], [2, 3, 4], [5, 6, 7]])
            model = ConditionalFloodPINN(_scales(terrain), width=4, layers=1,
                                         terrain_features=True, terrain=terrain)
            captured = []
            hook = model.net[0].register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach().clone()))
            points = torch.tensor([
                [10., 50., 3600., 10 / 3_600_000., 3600.],
                [30., 50., 3600., 10 / 3_600_000., 3600.],
            ], dtype=torch.float64)
            model(points); hook.remove()
            self.assertEqual(captured[0].shape[1], 5 + len(TERRAIN_FEATURE_NAMES))
            self.assertFalse(torch.equal(captured[0][0, 5:], captured[0][1, 5:]))
            self.assertTrue(torch.isfinite(captured[0]).all())

    def test_nodata_is_rejected_and_never_used_as_a_feature_value(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[-2, -9999, 2], [3, 4, 5], [6, 7, 8]])
            model = ConditionalFloodPINN(_scales(terrain), width=4, layers=1,
                                         terrain_features=True, terrain=terrain)
            self.assertGreater(model.terrain_feature_metadata["z_centre_m"], -10.)
            self.assertTrue(torch.isfinite(model._terrain_feature_grid).all())
            model(torch.tensor([[10., 50., 1., 0., 3600.]], dtype=torch.float64))
            with self.assertRaisesRegex(ValueError, "NoData"):
                model(torch.tensor([[30., 50., 1., 0., 3600.]], dtype=torch.float64))

    def test_feature_checkpoint_rebuilds_grid_and_legacy_shape_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
            scales = _scales(terrain)
            legacy = ConditionalFloodPINN(scales, width=6, layers=2)
            self.assertEqual(legacy.net[0].in_features, 5)
            self.assertFalse(any("terrain_feature" in key for key in legacy.state_dict()))
            legacy_copy = ConditionalFloodPINN(scales, width=6, layers=2)
            legacy_copy.load_state_dict(legacy.state_dict())

            model = ConditionalFloodPINN(
                scales, width=6, layers=2, terrain_features=True,
                terrain=terrain, terrain_local_coordinates=True)
            checkpoint = {
                "model_state": model.state_dict(), "scales": scales,
                "width": 6, "layers": 2, "terrain_features": True,
                "terrain_local_coordinates": True,
                "terrain_feature_version": model.terrain_feature_metadata["version"],
                "terrain_fingerprint": terrain.fingerprint,
                "training_rain_domain": {
                    "rainfall_mm_hr": [0., 800.], "duration_s": [3600., 86400.],
                },
            }
            path = Path(folder) / "model.pt"; torch.save(checkpoint, path)
            loaded, loaded_checkpoint = load_model(path, terrain)
            points = torch.tensor([[30., 30., 3600., 20 / 3_600_000., 3600.]],
                                  dtype=torch.float64)
            torch.testing.assert_close(loaded(points), model(points))
            self.assertTrue(loaded_checkpoint["terrain_features"])
            self.assertTrue(loaded.terrain_local_coordinates)

    def test_terrain_is_required_only_for_feature_models(self):
        scales = {"x0": 0., "y0": 0., "lx": 1., "ly": 1., "time_s": 1.,
                  "rain_m_s": 1., "duration_s": 1., "velocity_m_s": 1.,
                  "ramp_s": 1.}
        ConditionalFloodPINN(scales, width=4, layers=1)
        with self.assertRaisesRegex(ValueError, "terrain is required"):
            ConditionalFloodPINN(scales, width=4, layers=1, terrain_features=True)


if __name__ == "__main__":
    unittest.main()
