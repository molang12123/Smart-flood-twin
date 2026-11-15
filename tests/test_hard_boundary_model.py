from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.model import ConditionalFloodPINN
from minxiong_pinn.predict import load_model
from minxiong_pinn.terrain import MaskedRasterTerrain


def _terrain(folder):
    path = Path(folder) / "masked.tif"
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    values[1, 1] = -9999.
    with rasterio.open(path, "w", driver="GTiff", width=3, height=3, count=1,
                       dtype="float32", crs="EPSG:3826",
                       transform=from_origin(0, 60, 20, 20), nodata=-9999.) as dst:
        dst.write(values, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


def _scales():
    return {"x0": 0., "y0": 0., "lx": 60., "ly": 60.,
            "time_s": 7200., "rain_m_s": 50 / 3_600_000.,
            "duration_s": 3600., "velocity_m_s": 4., "ramp_s": 60.}


def _points(xy):
    result = np.empty((len(xy), 5), dtype=np.float64)
    result[:, :2] = xy
    result[:, 2] = 3600.
    result[:, 3] = 10 / 3_600_000.
    result[:, 4] = 3600.
    return torch.as_tensor(result, dtype=torch.float64)


class HardBoundaryModelTests(unittest.TestCase):
    def test_mask_faces_zero_only_the_normal_velocity_at_the_wall(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder)
            model = ConditionalFloodPINN(
                _scales(), width=4, layers=1, terrain=terrain,
                separate_output_networks=True, hard_mask_boundary=True)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.velocity_net[-1].bias.fill_(1.)
            epsilon = 1e-8
            points = _points(np.array([
                [0., 30.], [60., 30.], [10., 60.], [10., 0.],
                [20. - epsilon, 30.], [10., 30.],
            ]))
            result = model(points)
            self.assertEqual(float(result[0, 1].detach()), 0.)
            self.assertEqual(float(result[1, 1].detach()), 0.)
            self.assertEqual(float(result[2, 2].detach()), 0.)
            self.assertEqual(float(result[3, 2].detach()), 0.)
            self.assertLess(abs(float(result[4, 1].detach())), 1e-6)
            self.assertGreater(float(result[0, 2].detach()), 0.)
            self.assertGreater(float(result[2, 1].detach()), 0.)
            self.assertGreater(float(result[5, 1].detach()), 0.)
            self.assertGreater(float(result[5, 2].detach()), 0.)

    def test_separate_depth_network_can_be_frozen_without_velocity_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder)
            model = ConditionalFloodPINN(
                _scales(), width=4, layers=1, terrain=terrain,
                separate_output_networks=True, hard_mask_boundary=True)
            model.zero_output_layers()
            result = model(_points(np.array([[10., 30.]])))
            result[:, 0].sum().backward()
            model.scale_depth_gradients(0.)
            self.assertTrue(all(parameter.grad is None or
                                torch.count_nonzero(parameter.grad) == 0
                                for parameter in model.depth_net.parameters()))
            self.assertTrue(all(parameter.grad is None or
                                torch.count_nonzero(parameter.grad) == 0
                                for parameter in model.velocity_net.parameters()))

    def test_hard_boundary_requires_a_terrain(self):
        with self.assertRaisesRegex(ValueError, "terrain is required"):
            ConditionalFloodPINN(_scales(), width=4, layers=1,
                                 hard_mask_boundary=True)

    def test_hydrostatic_baseline_has_exact_full_domain_mass_and_level(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder)
            model = ConditionalFloodPINN(
                _scales(), width=4, layers=1, terrain=terrain,
                separate_output_networks=True, mass_conserving_projection=True,
                mass_projection_anchor_points=terrain.active_count,
                hydrostatic_depth_baseline=True,
                hydrostatic_learned_blend=0.)
            model.zero_output_layers()
            west, _, _, north = terrain.bounds
            xy = np.column_stack((
                west + (terrain.active_cols + .5) * terrain.dx,
                north - (terrain.active_rows + .5) * terrain.dy,
            ))
            points = _points(xy)
            points[:, 3] = 1000 / 3_600_000.  # one metre over one hour
            depths = model(points)[:, 0].detach().numpy()
            self.assertAlmostEqual(float(depths.mean()), 1., places=6)
            elevations = terrain.z[terrain.active_rows, terrain.active_cols]
            wet_surface = elevations[depths > 0] + depths[depths > 0]
            np.testing.assert_allclose(wet_surface, wet_surface[0], rtol=0, atol=1e-6)
            self.assertTrue(model.exact_hydrostatic_mass)
            self.assertFalse(model.uses_external_mass_projection)

    def test_hydrostatic_blend_range_is_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder)
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                ConditionalFloodPINN(
                    _scales(), width=4, layers=1, terrain=terrain,
                    hydrostatic_depth_baseline=True,
                    hydrostatic_learned_blend=1.1)

    def test_new_architecture_round_trips_through_checkpoint_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder)
            model = ConditionalFloodPINN(
                _scales(), width=4, layers=1, terrain=terrain,
                separate_output_networks=True, hard_mask_boundary=True,
                mass_conserving_projection=True,
                mass_projection_anchor_points=terrain.active_count,
                hydrostatic_depth_baseline=True,
                hydrostatic_learned_blend=0.)
            checkpoint = {
                "model_state": model.state_dict(), "scales": _scales(),
                "width": 4, "layers": 1,
                "terrain_fingerprint": terrain.fingerprint,
                "training_rain_domain": {
                    "rainfall_mm_hr": [0., 50.], "duration_s": [3600., 86400.]},
                "separate_output_networks": True, "hard_mask_boundary": True,
                "mass_conserving_projection": True,
                "mass_projection_anchor_points": terrain.active_count,
                "mass_projection_strata": 32, "mass_projection_seed": 20261113,
                "mass_projection_anchor_fingerprint":
                    model.mass_projection_anchor_fingerprint,
                "hydrostatic_depth_baseline": True,
                "hydrostatic_learned_blend": 0.,
            }
            path = Path(folder) / "model.pt"
            torch.save(checkpoint, path)
            loaded, _ = load_model(path, terrain)
            points = _points(np.array([[10., 30.], [0., 30.]]))
            torch.testing.assert_close(loaded(points), model(points))


if __name__ == "__main__":
    unittest.main()
