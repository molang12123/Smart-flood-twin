from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from minxiong_pinn.terrain import MaskedRasterTerrain


def _terrain(folder, values):
    values = np.asarray(values, dtype=np.float32)
    path = Path(folder) / "terrain.tif"
    with rasterio.open(
            path, "w", driver="GTiff", width=values.shape[1],
            height=values.shape[0], count=1, dtype="float32",
            crs="EPSG:3826", transform=from_origin(
                100., 100. + 20. * values.shape[0], 20., 20.),
            nodata=-9999.) as destination:
        destination.write(values, 1)
    return MaskedRasterTerrain(path, manning_n=.03)


class InteriorFaceSamplingTests(unittest.TestCase):
    def test_faces_are_unique_active_pairs_with_fixed_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [
                [1., 2., 3.],
                [4., -9999., 6.],
                [7., 8., 12.],
            ])
            self.assertEqual(len(terrain.interior_face_xy), 8)
            signatures = np.column_stack((
                terrain.interior_face_xy, terrain.interior_face_normals))
            self.assertEqual(len(np.unique(signatures, axis=0)), len(signatures))
            expected_normals = {(1., 0.), (0., -1.)}
            self.assertEqual(
                {tuple(value) for value in terrain.interior_face_normals},
                expected_normals)

            normals = terrain.interior_face_normals
            half_spacing = np.where(
                normals[:, 0] != 0., terrain.dx / 2., terrain.dy / 2.)
            left_xy = terrain.interior_face_xy - normals * half_spacing[:, None]
            right_xy = terrain.interior_face_xy + normals * half_spacing[:, None]
            left_query = terrain.query(left_xy[:, 0], left_xy[:, 1])
            right_query = terrain.query(right_xy[:, 0], right_xy[:, 1])
            np.testing.assert_array_equal(
                terrain.interior_face_left_z, left_query["z"])
            np.testing.assert_array_equal(
                terrain.interior_face_right_z, right_query["z"])

            west, east, south, north = terrain.bounds
            self.assertTrue(np.all(
                (terrain.interior_face_xy[:, 0] > west) &
                (terrain.interior_face_xy[:, 0] < east) &
                (terrain.interior_face_xy[:, 1] > south) &
                (terrain.interior_face_xy[:, 1] < north)))

    def test_sampling_is_reproducible_and_without_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(25).reshape(5, 5))
            first = terrain.sample_interior_faces(
                np.random.default_rng(17), 20, high_relief_fraction=.4)
            second = terrain.sample_interior_faces(
                np.random.default_rng(17), 20, high_relief_fraction=.4)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(len(np.unique(first["index"])), 20)
            np.testing.assert_array_equal(
                first["xy"], terrain.interior_face_xy[first["index"]])
            np.testing.assert_array_equal(
                first["normal"], terrain.interior_face_normals[first["index"]])

    def test_high_relief_fraction_biases_toward_larger_bed_jumps(self):
        with tempfile.TemporaryDirectory() as folder:
            columns = np.arange(40, dtype=np.float32)
            terrain = _terrain(folder, [np.square(columns)])
            uniform_relief, weighted_relief = [], []
            for seed in range(400):
                uniform = terrain.sample_interior_faces(
                    np.random.default_rng(seed), 1)
                weighted = terrain.sample_interior_faces(
                    np.random.default_rng(seed), 1,
                    high_relief_fraction=1.)
                uniform_relief.append(abs(
                    uniform["right_z"][0] - uniform["left_z"][0]))
                weighted_relief.append(abs(
                    weighted["right_z"][0] - weighted["left_z"][0]))
            self.assertGreater(
                np.mean(weighted_relief), 1.1 * np.mean(uniform_relief))

    def test_count_and_relief_fraction_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, np.arange(9).reshape(3, 3))
            invalid_counts = (0, -1, 1.5, True, np.nan, None, "bad",
                              len(terrain.interior_face_xy) + 1)
            for value in invalid_counts:
                with self.subTest(count=value), self.assertRaises(ValueError):
                    terrain.sample_interior_faces(
                        np.random.default_rng(1), value)
            for value in (-.1, 1.1, True, np.nan, None, "bad"):
                with self.subTest(fraction=value), self.assertRaises(ValueError):
                    terrain.sample_interior_faces(
                        np.random.default_rng(1), 1,
                        high_relief_fraction=value)

    def test_no_active_pair_is_reported_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = _terrain(folder, [[1., -9999.], [-9999., 2.]])
            self.assertEqual(terrain.interior_face_xy.shape, (0, 2))
            with self.assertRaisesRegex(ValueError, "no active-active"):
                terrain.sample_interior_faces(np.random.default_rng(1), 1)


if __name__ == "__main__":
    unittest.main()
