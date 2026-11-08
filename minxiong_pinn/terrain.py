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
        self.zx, self.zy = self._local_slopes()
        active_slope = np.hypot(self.zx[mask], self.zy[mask])
        cap = max(float(np.percentile(active_slope, 99)), 1e-12)
        weights = 1. + np.clip(active_slope / cap, 0., 1.) * 4.
        self._slope_probabilities = weights / weights.sum()
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

    def sample_boundary(self, rng: np.random.Generator, count: int):
        indexes = rng.integers(0, len(self.boundary_xy), size=count)
        normals = self.boundary_normals[indexes]
        # Evaluate the wall condition infinitesimally inside the active cell;
        # the exact face may map to its NoData neighbour under floor indexing.
        epsilon = min(self.dx, self.dy) * 1e-6
        return self.boundary_xy[indexes] - normals * epsilon, normals

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
