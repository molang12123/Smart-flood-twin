"""Conservative two-dimensional SWE residuals in physical SI units.

Terrain queries are fixed data, never trainable fields. Rain intensity and
duration are conditional inputs, while PDE derivatives are only x/y/t.
"""
from __future__ import annotations

import math

import torch


def derivative(field: torch.Tensor, points: torch.Tensor, column: int) -> torch.Tensor:
    """Pointwise derivative, including exactly constant manufactured fields.

    The model must act independently on each sample (as the supplied MLP does).
    Batch-coupled layers would require a different Jacobian calculation.
    """
    if column not in (0, 1, 2):
        raise ValueError("PDE derivatives are defined for x, y, t only")
    if not field.requires_grad:
        return points[:, column:column + 1] * 0.0
    gradient = torch.autograd.grad(
        field, points, grad_outputs=torch.ones_like(field),
        create_graph=True, retain_graph=True, allow_unused=True,
    )[0]
    if gradient is None:
        return points[:, column:column + 1] * 0.0
    return gradient[:, column:column + 1]


def manning_source(h, u, v, n, *, h_reg=1e-4, gravity=9.81, velocity_epsilon=1e-6):
    """Return conservative momentum sinks, each in m²/s².

    Multiplication by h is essential: the h^(-4/3) expression without h
    belongs to the primitive velocity equation, not conserved momentum.
    """
    for name, value in [("h_reg", h_reg), ("gravity", gravity),
                        ("velocity_epsilon", velocity_epsilon)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    speed = torch.sqrt(u.square() + v.square() + velocity_epsilon ** 2)
    factor = gravity * n.square() * h * speed / h.clamp_min(h_reg).pow(4.0 / 3.0)
    return factor * u, factor * v


def _terrain_column(value, points):
    result = torch.as_tensor(value, dtype=points.dtype, device=points.device)
    if result.numel() == 1:
        result = result.reshape(1, 1).expand(points.shape[0], 1)
    else:
        result = result.reshape(-1, 1)
    if result.shape[0] != points.shape[0] or not torch.isfinite(result).all():
        raise ValueError("terrain query produced invalid shape or nonfinite values")
    return result


def residuals(model, points, terrain, h_reg=1e-4, *, gravity=9.81, velocity_epsilon=1e-6):
    """Return [continuity, x momentum, y momentum] for physical Nx5 points.

    Input columns: x m, y m, t s, rainfall m/s, rain duration s. Returned
    continuity has units m/s; momentum residuals have units m²/s².
    Strong residuals at the exact rain-stop event should be excluded by the
    sampler; check that event through time-integrated continuity instead.
    """
    if points.ndim != 2 or points.shape[1] != 5 or points.shape[0] == 0:
        raise ValueError("points must be a nonempty Nx5 tensor")
    if not torch.isfinite(points).all():
        raise ValueError("nonfinite physical coordinates or rain parameters")
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)
    values = model(points)
    if values.shape != (points.shape[0], 3):
        raise ValueError("model must return an Nx3 tensor")
    if not torch.isfinite(values).all():
        raise FloatingPointError("model produced nonfinite physical fields")
    h, u, v = values.split(1, dim=1)
    data = terrain.query(points[:, 0].detach().cpu().numpy(),
                         points[:, 1].detach().cpu().numpy())
    zx, zy, n = (_terrain_column(data[key], points) for key in ("zx", "zy", "n"))
    sx, sy = manning_source(h, u, v, n, h_reg=h_reg, gravity=gravity,
                           velocity_epsilon=velocity_epsilon)
    hu, hv = h * u, h * v
    rain = torch.where((points[:, 2:3] >= 0) &
                       (points[:, 2:3] < points[:, 4:5]),
                       points[:, 3:4], torch.zeros_like(h))
    continuity = derivative(h, points, 2) + derivative(hu, points, 0) + derivative(hv, points, 1) - rain
    momentum_x = (derivative(hu, points, 2) +
                  derivative(h * u.square() + 0.5 * gravity * h.square(), points, 0) +
                  derivative(h * u * v, points, 1) + gravity * h * zx + sx)
    momentum_y = (derivative(hv, points, 2) +
                  derivative(h * u * v, points, 0) +
                  derivative(h * v.square() + 0.5 * gravity * h.square(), points, 1) +
                  gravity * h * zy + sy)
    return torch.cat((continuity, momentum_x, momentum_y), dim=1)
