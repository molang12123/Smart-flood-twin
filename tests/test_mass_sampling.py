from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from minxiong_pinn.terrain import MaskedRasterTerrain


def _terrain(folder, values, *, nodata=-9999.):
    values = np.asarray(values, dtype=np.float32)
    path = Path(folder) / "terrain.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="float32", crs="EPSG:3826",
        transform=from_origin(0, values.shape[0] * 20, 20, 20), nodata=nodata,
    ) as dst:
        dst.write(values, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


class ElevationStratifiedMassSamplingTests(unittest.TestCase):
    def test_all_elevation_strata_are_covered_and_weighted_mean_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(16).reshape(4, 4))
            xy, weights = terrain.sample_mass_points(
                np.random.default_rng(4), terrain.active_count,
                strata=4, jitter=False)
            elevations = terrain.query(xy[:, 0], xy[:, 1])["z"]
            self.assertEqual(len(np.unique(xy, axis=0)), terrain.active_count)
            self.assertTrue(all(np.any((elevations >= low) & (elevations < high))
                                for low, high in ((0, 4), (4, 8), (8, 12), (12, 16))))
            self.assertAlmostEqual(float(weights.sum()), 1.)
            self.assertAlmostEqual(float(np.sum(elevations * weights)), 7.5)

    def test_unequal_stratum_sizes_use_unbiased_area_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(10).reshape(2, 5))
            xy, weights = terrain.sample_mass_points(
                np.random.default_rng(5), 3, strata=3, jitter=False)
            elevations = terrain.query(xy[:, 0], xy[:, 1])["z"]
            expected = np.where(elevations < 4, .4, .3)
            np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)
            self.assertAlmostEqual(float(weights.sum()), 1.)

    def test_nodata_cells_are_never_sampled(self):
        with tempfile.TemporaryDirectory() as folder:
            values = np.array([[-8., -9999., -2.], [0., 1., 2.], [3., 4., 5.]])
            terrain = _terrain(folder, values)
            xy, weights = terrain.sample_mass_points(
                np.random.default_rng(6), 100, strata=4, jitter=False)
            elevations = terrain.query(xy[:, 0], xy[:, 1])["z"]
            self.assertFalse(np.any(elevations == -9999.))
            self.assertIn(-8., elevations)
            self.assertAlmostEqual(float(weights.sum()), 1.)

    def test_invalid_sample_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[1., 2.], [3., 4.]])
            rng = np.random.default_rng(7)
            for count in (0, -1, 1.5, True, float("nan"), float("inf"), "bad"):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    terrain.sample_mass_points(rng, count)
            for strata in (0, -1, 1.5, True, float("nan"), float("inf"), "bad"):
                with self.subTest(strata=strata), self.assertRaises(ValueError):
                    terrain.sample_mass_points(rng, 2, strata=strata)

    def test_full_domain_water_level_preserves_requested_mean_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(25).reshape(5, 5))
            for mean_depth in (0., .1, 2., 20.):
                level = terrain.water_level_for_mean_depth(mean_depth)
                depths = np.maximum(level - terrain.z[terrain.mask], 0.)
                self.assertAlmostEqual(float(depths.mean()), mean_depth, places=11)

    def test_shoreline_control_volume_sampling_targets_nearby_elevations(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(25).reshape(5, 5))
            xy = terrain.sample_control_volumes_near_levels(
                np.random.default_rng(8), np.full(100, 12.), window_cells=1)
            elevations = terrain.query(xy[:, 0], xy[:, 1])["z"]
            self.assertTrue(set(elevations) <= {11., 12., 13.})
            with self.assertRaises(ValueError):
                terrain.sample_control_volumes_near_levels(
                    np.random.default_rng(8), [12.], window_cells=0)


if __name__ == "__main__":
    unittest.main()
