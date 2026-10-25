"""Mathematical handoff only; no GIS transformation or Unreal scene code."""
import numpy as np


def _mapping(anchor, axis_map):
    anchor, matrix = np.asarray(anchor, float), np.asarray(axis_map, float)
    if anchor.shape != (3,) or matrix.shape != (3,3) or not np.isfinite(anchor).all() or not np.isfinite(matrix).all():
        raise ValueError("Explicit finite 3D anchor and 3x3 axis map are required")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("Axis map must be orthogonal (reflection allowed for handedness)")
    return anchor, matrix


def to_ue(points_m, anchor, axis_map):
    """Subtract double-precision GIS anchor, apply agreed axis map, metres→cm."""
    origin, matrix = _mapping(anchor, axis_map)
    points = np.asarray(points_m, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite (...,3) points")
    return 100 * ((points - origin) @ matrix.T)


def from_ue(points_cm, anchor, axis_map):
    origin, matrix = _mapping(anchor, axis_map)
    points = np.asarray(points_cm, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite (...,3) points")
    return (points/100) @ matrix + origin


def velocity_to_ue(velocity_m_s, axis_map):
    _, matrix = _mapping([0,0,0], axis_map)
    vector = np.asarray(velocity_m_s, dtype=np.float64)
    if vector.shape[-1:] != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite (...,3) velocity")
    return 100*(vector @ matrix.T)
