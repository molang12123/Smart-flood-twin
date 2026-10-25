"""Corrected legacy Case A FV benchmark, not a real-terrain reference solver.

The original split flux/source method is retained. In particular, it is not
hydrostatically reconstructed or established as well-balanced at wet/dry shores.
"""
import operator
import numpy as np

Lx = Ly = 1.0


def hll_x(hL, huL, hvL, hR, huR, hvR, *, g=9.81, h_wet=1e-4):
    """HLL flux for the x direction (legacy dry-depth regularization)."""
    hL = np.maximum(hL, h_wet)
    hR = np.maximum(hR, h_wet)
    uL, uR = huL / hL, huR / hR
    cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)
    sL, sR = np.minimum(uL-cL, uR-cR), np.maximum(uL+cL, uR+cR)
    denominator = np.maximum(sR-sL, 1e-12)

    def flux(FL, FR, UL, UR):
        return np.where(sL >= 0, FL, np.where(sR <= 0, FR,
            (sR*FL - sL*FR + sL*sR*(UR-UL)) / denominator))
    return (flux(huL, huR, hL, hR),
            flux(huL*uL + .5*g*hL*hL, huR*uR + .5*g*hR*hR, huL, huR),
            flux(hvL*uL, hvR*uR, hvL, hvR))


def hll_y(hB, huB, hvB, hT, huT, hvT, *, g=9.81, h_wet=1e-4):
    """Rotationally consistent HLL flux in y, returned as (mass, hu, hv)."""
    mass, normal, tangent = hll_x(hB, hvB, huB, hT, hvT, huT, g=g, h_wet=h_wet)
    return mass, tangent, normal


def _grid_size(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('grid dimensions must be integers >= 2')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError('grid dimensions must be integers >= 2') from exc
    if result < 2:
        raise ValueError('grid dimensions must be integers >= 2')
    return result


def run_fv_A(t_targets, nx=40, ny=40, *, slope=0.05, rainfall_mm_hr=50.,
             rain_duration_s=180., manning_n=.02, cfl=.30, max_dt=.5,
             max_steps=200000, return_diagnostics=False):
    """Return {exact_target_seconds: depth_array}, optionally with diagnostics.

    All walls reflect normal momentum; the domain is a one-square-metre plane.
    Rainfall is integrated exactly over each step. A stopping budget prevents
    a numerically collapsed time step from creating an unbounded computation.
    """
    nx, ny = _grid_size(nx), _grid_size(ny)
    try:
        targets = sorted(set(float(t) for t in t_targets))
    except (TypeError, ValueError) as exc:
        raise ValueError('target times must be finite and nonnegative') from exc
    if not targets or any(not np.isfinite(t) or t < 0 for t in targets):
        raise ValueError('target times must be finite and nonnegative')
    try:
        params = np.asarray([slope, rainfall_mm_hr, rain_duration_s, manning_n, cfl, max_dt], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('physical and numerical parameters must be numeric') from exc
    if not np.isfinite(params).all():
        raise ValueError('physical and numerical parameters must be finite')
    slope, rain, duration, manning_n, cfl, max_dt = params
    if rain < 0 or duration < 0 or manning_n < 0 or not 0 < cfl <= .5 or max_dt <= 0:
        raise ValueError('invalid physical or numerical parameters')
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')

    g, h_wet = 9.81, 1e-4
    rate = rain / 3_600_000.
    dx, dy = Lx/nx, Ly/ny
    area = dx*dy
    h = np.zeros((ny, nx))
    hu, hv = np.zeros_like(h), np.zeros_like(h)
    time = 0.
    snaps = {}
    diag = dict(steps=0, depth_clip_count=0, depth_clip_added_volume_m3=0.,
                momentum_clip_count=0, dry_momentum_reset_count=0,
                rain_volume_m3=0., snapshot_times_s=[],
                method='legacy_split_hll_corrected', externally_validated=False,
                limitations=['not well-balanced for general terrain',
                             'dry-depth flux regularization',
                             'positivity clipping and momentum clipping retained'])

    def positive_depth(raw):
        if not np.isfinite(raw).all():
            raise FloatingPointError('nonfinite FV depth before positivity correction')
        mask = raw < 0
        diag['depth_clip_count'] += int(mask.sum())
        diag['depth_clip_added_volume_m3'] += float(-raw[mask].sum()*area)
        return np.maximum(raw, 0.)

    def sweep_x(step):
        nonlocal h, hu, hv
        hg = np.pad(h, ((0, 0), (1, 1)), mode='edge')
        hug = np.pad(hu, ((0, 0), (1, 1)), mode='edge')
        hvg = np.pad(hv, ((0, 0), (1, 1)), mode='edge')
        hug[:, 0], hug[:, -1] = -hug[:, 1], -hug[:, -2]
        f0, f1, f2 = hll_x(hg[:, :-1], hug[:, :-1], hvg[:, :-1], hg[:, 1:], hug[:, 1:], hvg[:, 1:])
        h = positive_depth(h - step/dx*(f0[:, 1:]-f0[:, :-1]))
        hu -= step/dx*(f1[:, 1:]-f1[:, :-1])
        hv -= step/dx*(f2[:, 1:]-f2[:, :-1])

    for target in targets:
        while time < target:
            if diag['steps'] >= max_steps:
                raise RuntimeError('FV max_steps exceeded; no completed reference returned')
            dry = h < h_wet
            diag['dry_momentum_reset_count'] += int(np.count_nonzero(dry & ((hu != 0) | (hv != 0))))
            hu[dry], hv[dry] = 0., 0.
            wet = ~dry
            dt = min(max_dt, target-time)
            if time < duration:
                dt = min(dt, duration-time)
            if wet.any():
                c = np.sqrt(g*h[wet])
                dt = min(dt, cfl*dx/(np.max(np.abs(hu[wet]/h[wet])+c)+1e-10),
                         cfl*dy/(np.max(np.abs(hv[wet]/h[wet])+c)+1e-10))
            if not np.isfinite(dt) or dt <= 0 or time+dt <= time:
                raise FloatingPointError('FV time step failed to advance')
            next_time = time+dt
            # Exact assignment avoids event drift and missing subnanosecond targets.
            if dt == target-time:
                next_time = target
            elif time < duration and dt == duration-time:
                next_time = duration
            sweep_x(dt/2)
            hg = np.vstack([h[:1], h, h[-1:]])
            hug = np.vstack([hu[:1], hu, hu[-1:]])
            hvg = np.vstack([-hv[:1], hv, -hv[-1:]])
            f0, f1, f2 = hll_y(hg[:-1], hug[:-1], hvg[:-1], hg[1:], hug[1:], hvg[1:])
            h = positive_depth(h-dt/dy*(f0[1:]-f0[:-1]))
            hu -= dt/dy*(f1[1:]-f1[:-1])
            hv -= dt/dy*(f2[1:]-f2[:-1])
            sweep_x(dt/2)

            wet = h >= h_wet
            hs = np.maximum(h, h_wet)
            u, v = np.where(wet, hu/hs, 0.), np.where(wet, hv/hs, 0.)
            denominator = 1+dt*g*manning_n**2*np.sqrt(u*u+v*v)/hs**(4/3)
            hu = np.where(wet, (hu+dt*g*hs*slope)/denominator, 0.)
            hv = np.where(wet, hv/denominator, 0.)
            rain_depth = rate*(min(next_time, duration)-min(time, duration))
            h = positive_depth(h+rain_depth)
            diag['rain_volume_m3'] += rain_depth*Lx*Ly
            if not all(np.isfinite(a).all() for a in (h, hu, hv)):
                raise FloatingPointError('nonfinite FV state before momentum clipping')
            diag['momentum_clip_count'] += int(np.count_nonzero((np.abs(hu) > 5) | (np.abs(hv) > 5)))
            hu, hv = np.clip(hu, -5., 5.), np.clip(hv, -5., 5.)
            if not all(np.isfinite(a).all() for a in (h, hu, hv)):
                raise FloatingPointError('nonfinite FV state')
            time = next_time
            diag['steps'] += 1
        snaps[target] = h.copy()
        diag['snapshot_times_s'].append(target)
    diag['final_storage_m3'] = float(h.sum()*area)
    diag['mass_balance_error_m3'] = diag['final_storage_m3']-diag['rain_volume_m3']
    return (snaps, diag) if return_diagnostics else snaps
