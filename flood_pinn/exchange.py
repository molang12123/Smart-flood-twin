"""Unique rectangular face topology and raw PINN field flux integration.

Shared faces provide antisymmetric accounting, not a discretely conservative
PINN solution. All primary fluxes use the raw physical field.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Faces:
    ids: np.ndarray
    axis: np.ndarray
    minus: np.ndarray
    plus: np.ndarray
    x: np.ndarray
    y: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    length: np.ndarray
    boundary: np.ndarray


def build_faces(terrain):
    nx, ny = terrain.nx, terrain.ny
    row, k = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing='ij')
    j, col = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing='ij')
    xm = np.where(k > 0, row * nx + k - 1, -1).ravel()
    xp = np.where(k < nx, row * nx + k, -1).ravel()
    ym = np.where(j > 0, (j - 1) * nx + col, -1).ravel()
    yp = np.where(j < ny, j * nx + col, -1).ravel()
    xcount = ny * (nx + 1)
    count = xcount + (ny + 1) * nx
    boundary_x = np.where(k == 0, 'west', np.where(k == nx, 'east', '')).ravel()
    boundary_y = np.where(j == 0, 'south', np.where(j == ny, 'north', '')).ravel()
    return Faces(np.arange(count), np.r_[np.full(xcount, 'x'), np.full(count-xcount, 'y')],
                 np.r_[xm, ym], np.r_[xp, yp],
                 np.r_[(terrain.x_origin + k * terrain.dx).ravel(),
                       (terrain.x_origin + (col + .5) * terrain.dx).ravel()],
                 np.r_[(terrain.y_origin + (row + .5) * terrain.dy).ravel(),
                       (terrain.y_origin + j * terrain.dy).ravel()],
                 np.r_[np.ones(xcount), np.zeros(count-xcount)],
                 np.r_[np.zeros(xcount), np.ones(count-xcount)],
                 np.r_[np.full(xcount, terrain.dy), np.full(count-xcount, terrain.dx)],
                 np.r_[boundary_x, boundary_y])


def evaluate_points(evaluate, points, batch_size=4096):
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    values = []
    for start in range(0, len(points), batch_size):
        value = np.asarray(evaluate(points[start:start+batch_size]), dtype=float)
        if value.shape != (len(points[start:start+batch_size]), 3):
            raise ValueError('evaluate must return an N by 3 physical field')
        if not np.isfinite(value).all():
            raise FloatingPointError('NONFINITE_PREDICTION: raw h/u/v')
        if (value[:, 0] < 0).any():
            raise FloatingPointError('negative predicted depth')
        values.append(value)
    return np.concatenate(values, axis=0) if values else np.empty((0, 3))


def physical_points(x, y, t, rainfall):
    x, y = np.broadcast_arrays(x, y)
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, t),
                            np.full(x.size, rainfall.intensity_mm_hr / 3600000.),
                            np.full(x.size, rainfall.duration_s)])


def face_fluxes(terrain, evaluate, t, rainfall, batch_size=4096, faces=None):
    """Two-point Gauss line quadrature on each unique face (m³/s)."""
    faces = build_faces(terrain) if faces is None else faces
    offset = faces.length[:, None] * np.array([-.5/np.sqrt(3), .5/np.sqrt(3)])
    x = faces.x[:, None] + offset * faces.ny[:, None]
    y = faces.y[:, None] + offset * faces.nx[:, None]
    field = evaluate_points(evaluate, physical_points(x, y, t, rainfall), batch_size).reshape(-1, 2, 3)
    q = field[:, :, 0] * (field[:, :, 1] * faces.nx[:, None] + field[:, :, 2] * faces.ny[:, None])
    flux = q.mean(axis=1) * faces.length
    if not np.isfinite(flux).all():
        raise FloatingPointError('NONFINITE_PREDICTION: face flux')
    return flux


def _linear_parts(a, b, dt):
    """Integrate positive/negative parts of a linearly reconstructed Q."""
    positive = dt * (np.maximum(a, 0) + np.maximum(b, 0)) / 2
    negative = dt * (np.maximum(-a, 0) + np.maximum(-b, 0)) / 2
    crossing = a * b < 0
    difference = np.abs(b-a)
    correction = np.zeros_like(a)
    # Remove the trapezoid triangles outside the actual sign interval.
    correction[crossing] = dt * np.abs(a[crossing]*b[crossing]) / (2*difference[crossing])
    return positive-correction, negative-correction


def integrate_exchange(terrain, evaluate, rainfall, t0, t1, *, batch_size=4096,
                       time_substeps=2, faces=None):
    """Composite time quadrature, split at rain stop, fine/coarse estimate.

    Error is a refinement difference estimate rather than a guaranteed bound.
    A linear reconstruction resolves endpoint sign changes within each substep.
    """
    if not np.isfinite([t0, t1]).all() or not 0 <= t0 < t1:
        raise ValueError('exchange requires 0 <= t0 < t1')
    if isinstance(time_substeps, bool) or not isinstance(time_substeps, (int, np.integer)) or time_substeps < 1:
        raise ValueError('time_substeps must be a positive integer')
    faces = build_faces(terrain) if faces is None else faces
    breaks = [t0] + ([rainfall.duration_s] if t0 < rainfall.duration_s < t1 else []) + [t1]
    cache = {}
    def q(t):
        key = float(t)
        if key not in cache:
            cache[key] = face_fluxes(terrain, evaluate, key, rainfall, batch_size, faces)
        return cache[key]
    def integrate(n):
        vp = np.zeros(len(faces.ids)); vn = np.zeros_like(vp)
        for left, right in zip(breaks[:-1], breaks[1:]):
            times = np.linspace(left, right, n+1)
            for a, b in zip(times[:-1], times[1:]):
                plus, minus = _linear_parts(q(a), q(b), b-a)
                vp += plus; vn += minus
        return vp, vn
    cp, cn = integrate(time_substeps)
    fp, fn = integrate(2*time_substeps)
    return {'Q_start': q(t0), 'Q_end': q(t1), 'V_forward': fp, 'V_reverse': fn,
            'V_signed': fp-fn, 'error_est': np.abs(fp-cp)+np.abs(fn-cn)}


def cell_sides(terrain):
    row, col = np.meshgrid(np.arange(terrain.ny), np.arange(terrain.nx), indexing='ij')
    offset = terrain.ny * (terrain.nx+1)
    return {'west': ((row*(terrain.nx+1)+col).ravel(), -1),
            'east': ((row*(terrain.nx+1)+col+1).ravel(), 1),
            'south': ((offset+row*terrain.nx+col).ravel(), -1),
            'north': ((offset+(row+1)*terrain.nx+col).ravel(), 1)}
