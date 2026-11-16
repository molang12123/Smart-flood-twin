from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class MaskedRasterTerrain:
    """Read a north-up projected GeoTIFF without filling its NoData cells.

    Elevation can be negative. Validity comes from the raster mask, never from
    an elevation comparison such as ``z <= 0``.
    """

    def __init__(self, path: str | Path, *, manning_n: float):
        import rasterio

        self.path = Path(path).resolve()
        with rasterio.open(self.path) as ds:
            if ds.count != 1 or ds.crs is None or not ds.crs.is_projected:
                raise ValueError("DEM must be a single-band projected raster")
            transform = ds.transform
            if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
                raise ValueError("DEM must be an unrotated north-up raster")
            unit = (ds.crs.linear_units or "").lower()
            if unit not in {"metre", "meter", "metres", "meters"}:
                raise ValueError("DEM horizontal CRS must use metres")
            raw = ds.read(1).astype(np.float64)
            mask = ds.read_masks(1) > 0
            self.crs = ds.crs.to_string()
            self.transform = transform
            self.width, self.height = ds.width, ds.height
            self.bounds = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
            self.dx, self.dy = float(transform.a), float(-transform.e)
            self.nodata = ds.nodata
        if not np.any(mask) or not np.isfinite(raw[mask]).all():
            raise ValueError("DEM contains no valid finite elevation cells")
        if not np.isfinite(manning_n) or manning_n <= 0:
            raise ValueError("manning_n must be finite and positive")
        self.z = raw
        self.mask = mask
        self.manning_n = float(manning_n)
        self.active_rows, self.active_cols = np.nonzero(mask)
        self.active_count = int(mask.sum())
        self.active_area_m2 = self.active_count * self.dx * self.dy
        # Stable ordering makes equal-population elevation strata reproducible
        # without ever consulting NoData values.
        self._elevation_order = np.argsort(
            self.z[self.active_rows, self.active_cols], kind="stable")
        self._sorted_active_elevation = self.z[
            self.active_rows[self._elevation_order],
            self.active_cols[self._elevation_order]]
        self._elevation_prefix_sum = np.concatenate((
            [0.], np.cumsum(self._sorted_active_elevation, dtype=np.float64)))
        wet_count = np.arange(1, self.active_count, dtype=np.float64)
        breaks = (wet_count * self._sorted_active_elevation[1:] -
                  self._elevation_prefix_sum[1:-1])
        self._elevation_volume_breaks = np.maximum.accumulate(
            np.maximum(breaks, 0.))
        self.zx, self.zy = self._local_slopes()
        active_slope = np.hypot(self.zx[mask], self.zy[mask])
        cap = max(float(np.percentile(active_slope, 99)), 1e-12)
        weights = 1. + np.clip(active_slope / cap, 0., 1.) * 4.
        self._slope_probabilities = weights / weights.sum()
        (self.interior_face_xy, self.interior_face_left_z,
         self.interior_face_right_z,
         self.interior_face_normals) = self._interior_faces()
        face_relief = np.abs(
            self.interior_face_right_z - self.interior_face_left_z)
        if len(face_relief):
            relief_cap = max(float(np.percentile(face_relief, 99)), 1e-12)
            face_weights = 1. + 4. * np.clip(face_relief / relief_cap, 0., 1.)
            self._interior_face_relief_probabilities = (
                face_weights / face_weights.sum())
        else:
            self._interior_face_relief_probabilities = np.empty(
                0, dtype=np.float64)
        interior = mask.copy()
        interior[0, :] = interior[-1, :] = False
        interior[:, 0] = interior[:, -1] = False
        interior[1:-1, 1:-1] &= (mask[1:-1, :-2] & mask[1:-1, 2:] &
                                 mask[:-2, 1:-1] & mask[2:, 1:-1])
        self.interior_rows, self.interior_cols = np.nonzero(interior)
        interior_order = np.argsort(
            self.z[self.interior_rows, self.interior_cols], kind="stable")
        self._sorted_interior_rows = self.interior_rows[interior_order]
        self._sorted_interior_cols = self.interior_cols[interior_order]
        self._sorted_interior_elevation = self.z[
            self._sorted_interior_rows, self._sorted_interior_cols]
        self.boundary_xy, self.boundary_normals, self.boundary_lengths = self._boundary_faces()
        digest = hashlib.sha256(self.path.read_bytes())
        digest.update(np.asarray([self.manning_n], dtype="<f8").tobytes())
        self.fingerprint = digest.hexdigest()

    def _local_slopes(self):
        zx = np.zeros_like(self.z)
        zy = np.zeros_like(self.z)
        rows, cols = self.active_rows, self.active_cols
        # A local least-squares plane uses only valid neighbours and never lets
        # NoData sentinel values contaminate the slope.
        for r, c in zip(rows, cols):
            r0, r1 = max(0, r - 1), min(self.height, r + 2)
            c0, c1 = max(0, c - 1), min(self.width, c + 2)
            rr, cc = np.nonzero(self.mask[r0:r1, c0:c1])
            rr, cc = rr + r0, cc + c0
            if len(rr) < 3:
                continue
            x = (cc - c) * self.dx
            y = (r - rr) * self.dy
            design = np.column_stack((x, y, np.ones_like(x)))
            coef, *_ = np.linalg.lstsq(design, self.z[rr, cc], rcond=None)
            zx[r, c], zy[r, c] = coef[:2]
        return zx, zy

    def _boundary_faces(self):
        xy, normals, lengths = [], [], []
        west, _, _, north = self.bounds
        directions = ((0, -1, -1., 0., self.dy), (0, 1, 1., 0., self.dy),
                      (1, 0, 0., -1., self.dx), (-1, 0, 0., 1., self.dx))
        for r, c in zip(self.active_rows, self.active_cols):
            cx = west + (c + .5) * self.dx
            cy = north - (r + .5) * self.dy
            for dr, dc, nx, ny, length in directions:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width or not self.mask[nr, nc]:
                    xy.append((cx + nx * self.dx / 2, cy + ny * self.dy / 2))
                    normals.append((nx, ny)); lengths.append(length)
        return (np.asarray(xy, dtype=np.float64), np.asarray(normals, dtype=np.float64),
                np.asarray(lengths, dtype=np.float64))

    def _interior_faces(self):
        """Build each active-active raster face once with a fixed orientation.

        East faces are oriented west-to-east with normal ``(+1, 0)``. South
        faces are oriented north-to-south with normal ``(0, -1)``. Restricting
        construction to those two directions avoids duplicate shared faces.
        """
        west, _, _, north = self.bounds
        east_rows, east_cols = np.nonzero(
            self.mask[:, :-1] & self.mask[:, 1:])
        south_rows, south_cols = np.nonzero(
            self.mask[:-1, :] & self.mask[1:, :])

        east_xy = np.column_stack((
            west + (east_cols + 1.) * self.dx,
            north - (east_rows + .5) * self.dy,
        ))
        south_xy = np.column_stack((
            west + (south_cols + .5) * self.dx,
            north - (south_rows + 1.) * self.dy,
        ))
        xy = np.concatenate((east_xy, south_xy), axis=0).astype(
            np.float64, copy=False)
        left_z = np.concatenate((
            self.z[east_rows, east_cols],
            self.z[south_rows, south_cols],
        )).astype(np.float64, copy=False)
        right_z = np.concatenate((
            self.z[east_rows, east_cols + 1],
            self.z[south_rows + 1, south_cols],
        )).astype(np.float64, copy=False)
        normals = np.concatenate((
            np.tile(np.asarray((1., 0.)), (len(east_rows), 1)),
            np.tile(np.asarray((0., -1.)), (len(south_rows), 1)),
        ), axis=0).astype(np.float64, copy=False)
        return xy, left_z, right_z, normals

    def sample_points(self, rng: np.random.Generator, count: int, *, jitter: bool = True,
                      slope_fraction: float = 0.):
        if not 0 <= slope_fraction <= 1:
            raise ValueError("slope_fraction must be between zero and one")
        weighted = int(round(count * slope_fraction))
        indexes = np.concatenate((
            rng.integers(0, self.active_count, size=count - weighted),
            rng.choice(self.active_count, size=weighted, replace=True, p=self._slope_probabilities),
        ))
        rng.shuffle(indexes)
        rows, cols = self.active_rows[indexes], self.active_cols[indexes]
        west, _, _, north = self.bounds
        jx = rng.uniform(-.499, .499, count) if jitter else np.zeros(count)
        jy = rng.uniform(-.499, .499, count) if jitter else np.zeros(count)
        x = west + (cols + .5 + jx) * self.dx
        y = north - (rows + .5 + jy) * self.dy
        return np.column_stack((x, y))

    def sample_mass_points(self, rng: np.random.Generator, count: int, *,
                           strata: int = 16, jitter: bool = True):
        """Sample all elevation quantiles and return area-mean weights.

        Active cells are divided into nearly equal-population elevation
        strata. At least one point is drawn from each retained stratum, so
        low-lying wet cells cannot disappear from a volume audit. The returned
        Horvitz--Thompson weights account for stratum population and sample
        allocation and always sum to one. Sampling is without replacement
        whenever a stratum has enough cells; oversized requests fall back to
        replacement only within the affected stratum.
        """
        try:
            count_value = float(count)
        except (TypeError, ValueError):
            count_value = float("nan")
        try:
            strata_value = float(strata)
        except (TypeError, ValueError):
            strata_value = float("nan")
        if (isinstance(count, (bool, np.bool_)) or not np.isfinite(count_value) or
                not count_value.is_integer() or count_value <= 0):
            raise ValueError("mass sample count must be a positive integer")
        if (isinstance(strata, (bool, np.bool_)) or not np.isfinite(strata_value) or
                not strata_value.is_integer() or strata_value <= 0):
            raise ValueError("mass sampling strata must be a positive integer")
        count = int(count_value)
        strata = min(int(strata_value), count, self.active_count)
        groups = np.array_split(self._elevation_order, strata)
        sizes = np.asarray([len(group) for group in groups], dtype=np.int64)

        # Reserve one draw per stratum, then distribute remaining draws in
        # proportion to the exact stratum populations using largest remainder.
        allocation = np.ones(strata, dtype=np.int64)
        remaining = count - strata
        if remaining:
            ideal = remaining * sizes / sizes.sum()
            extra = np.floor(ideal).astype(np.int64)
            allocation += extra
            leftover = remaining - int(extra.sum())
            if leftover:
                remainders = ideal - extra
                winners = np.argsort(-remainders, kind="stable")[:leftover]
                allocation[winners] += 1

        active_indexes, weights = [], []
        for group, population, samples in zip(groups, sizes, allocation):
            selected = group[rng.choice(population, size=samples,
                                        replace=samples > population)]
            active_indexes.append(selected)
            weights.append(np.full(samples, population / (self.active_count * samples),
                                   dtype=np.float64))
        indexes = np.concatenate(active_indexes)
        area_weights = np.concatenate(weights)
        order = rng.permutation(count)
        indexes, area_weights = indexes[order], area_weights[order]

        rows, cols = self.active_rows[indexes], self.active_cols[indexes]
        west, _, _, north = self.bounds
        jx = rng.uniform(-.499, .499, count) if jitter else np.zeros(count)
        jy = rng.uniform(-.499, .499, count) if jitter else np.zeros(count)
        xy = np.column_stack((west + (cols + .5 + jx) * self.dx,
                              north - (rows + .5 + jy) * self.dy))
        return xy, area_weights

    def sample_boundary(self, rng: np.random.Generator, count: int):
        indexes = rng.integers(0, len(self.boundary_xy), size=count)
        normals = self.boundary_normals[indexes]
        # Evaluate the wall condition infinitesimally inside the active cell;
        # the exact face may map to its NoData neighbour under floor indexing.
        epsilon = min(self.dx, self.dy) * 1e-6
        return self.boundary_xy[indexes] - normals * epsilon, normals

    def sample_interior_faces(self, rng: np.random.Generator, count: int, *,
                              high_relief_fraction: float = 0.):
        """Sample unique active-active faces, optionally favouring bed jumps.

        The returned mapping contains the stable face indexes, face centres,
        left/right cell-centre elevations, and the fixed left-to-right normal.
        Sampling is always without replacement, including across the weighted
        and uniform portions of a mixed request.
        """
        try:
            count_value = float(count)
        except (TypeError, ValueError):
            count_value = float("nan")
        if (isinstance(count, (bool, np.bool_)) or
                not np.isfinite(count_value) or not count_value.is_integer() or
                count_value <= 0):
            raise ValueError("interior face sample count must be a positive integer")
        count = int(count_value)
        available_count = len(self.interior_face_xy)
        if not available_count:
            raise ValueError("terrain has no active-active interior faces")
        if count > available_count:
            raise ValueError(
                "interior face sample count cannot exceed available faces")

        try:
            relief_fraction = float(high_relief_fraction)
        except (TypeError, ValueError):
            relief_fraction = float("nan")
        if (isinstance(high_relief_fraction, (bool, np.bool_)) or
                not np.isfinite(relief_fraction) or
                not 0. <= relief_fraction <= 1.):
            raise ValueError("high_relief_fraction must be between zero and one")

        weighted_count = int(round(count * relief_fraction))
        if weighted_count:
            weighted = rng.choice(
                available_count, size=weighted_count, replace=False,
                p=self._interior_face_relief_probabilities)
        else:
            weighted = np.empty(0, dtype=np.int64)
        uniform_count = count - weighted_count
        if uniform_count:
            # Interface batches are tiny compared with the full DEM. Rejection
            # sampling avoids sorting or allocating a full-face permutation on
            # every training step, while retaining a dense fallback for large
            # requests.
            excluded = set(map(int, weighted))
            if count > available_count // 3:
                available = np.ones(available_count, dtype=bool)
                available[weighted] = False
                remaining = np.flatnonzero(available)
                uniform = rng.choice(
                    remaining, size=uniform_count, replace=False)
            else:
                selected = []
                while len(selected) < uniform_count:
                    needed = uniform_count - len(selected)
                    candidates = rng.choice(
                        available_count,
                        size=min(available_count, max(32, 2 * needed)),
                        replace=False)
                    for candidate in candidates:
                        candidate = int(candidate)
                        if candidate not in excluded:
                            excluded.add(candidate)
                            selected.append(candidate)
                            if len(selected) == uniform_count:
                                break
                uniform = np.asarray(selected, dtype=np.int64)
        else:
            uniform = np.empty(0, dtype=np.int64)
        indexes = np.concatenate((weighted, uniform))
        rng.shuffle(indexes)
        return {
            "index": indexes,
            "xy": self.interior_face_xy[indexes],
            "left_z": self.interior_face_left_z[indexes],
            "right_z": self.interior_face_right_z[indexes],
            "normal": self.interior_face_normals[indexes],
        }

    def sample_control_volumes(self, rng: np.random.Generator, count: int):
        if len(self.interior_rows) == 0:
            raise ValueError("terrain has no active cells with four active neighbours")
        indexes = rng.integers(0, len(self.interior_rows), size=count)
        rows, cols = self.interior_rows[indexes], self.interior_cols[indexes]
        west, _, _, north = self.bounds
        return np.column_stack((west + (cols + .5) * self.dx,
                                north - (rows + .5) * self.dy))

    def water_level_for_mean_depth(self, accumulated_depth_m):
        """Solve the equal-area lake level for a full-domain mean depth."""
        depth = np.asarray(accumulated_depth_m, dtype=np.float64)
        if np.any(~np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError("accumulated depth must be finite and nonnegative")
        volume_sum = depth * self.active_count
        wet_count = (np.searchsorted(self._elevation_volume_breaks, volume_sum,
                                     side="right") + 1)
        wet_count = np.minimum(wet_count, self.active_count)
        level = ((volume_sum + self._elevation_prefix_sum[wet_count]) /
                 wet_count)
        return float(level) if level.ndim == 0 else level

    def sample_control_volumes_near_levels(self, rng: np.random.Generator,
                                           levels, *, window_cells=256):
        """Sample interior cells around one or more wet/dry water levels."""
        levels = np.asarray(levels, dtype=np.float64).reshape(-1)
        if not len(levels) or np.any(~np.isfinite(levels)):
            raise ValueError("shoreline levels must be a nonempty finite sequence")
        if (isinstance(window_cells, (bool, np.bool_)) or
                int(window_cells) != window_cells or window_cells <= 0):
            raise ValueError("shoreline window_cells must be a positive integer")
        available = len(self._sorted_interior_elevation)
        if not available:
            raise ValueError("terrain has no active cells with four active neighbours")
        window = min(int(window_cells), available)
        centre = np.searchsorted(self._sorted_interior_elevation, levels,
                                 side="left")
        low = np.maximum(0, centre - window)
        high = np.minimum(available, centre + window + 1)
        low = np.minimum(low, high - 1)
        ranks = rng.integers(low, high)
        rows = self._sorted_interior_rows[ranks]
        cols = self._sorted_interior_cols[ranks]
        west, _, _, north = self.bounds
        return np.column_stack((west + (cols + .5) * self.dx,
                                north - (rows + .5) * self.dy))

    def query(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        west, east, south, north = self.bounds
        col = np.floor((x - west) / self.dx).astype(int)
        row = np.floor((north - y) / self.dy).astype(int)
        col = np.where(x == east, self.width - 1, col)
        row = np.where(y == south, self.height - 1, row)
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        safe_r, safe_c = np.clip(row, 0, self.height - 1), np.clip(col, 0, self.width - 1)
        if not np.all(inside & self.mask[safe_r, safe_c]):
            raise ValueError("terrain query entered a NoData or out-of-domain cell")
        return {"z": self.z[safe_r, safe_c], "zx": self.zx[safe_r, safe_c],
                "zy": self.zy[safe_r, safe_c],
                "n": np.full(x.shape, self.manning_n, dtype=float)}

    def metadata(self):
        return {"source": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "fingerprint": self.fingerprint, "crs": self.crs, "width": self.width,
                "height": self.height, "dx_m": self.dx, "dy_m": self.dy,
                "bounds": list(self.bounds), "active_cells": self.active_count,
                "active_area_m2": self.active_area_m2, "nodata": self.nodata,
                "valid_elevation_min": float(self.z[self.mask].min()),
                "valid_elevation_max": float(self.z[self.mask].max()),
                "manning_n": self.manning_n}
