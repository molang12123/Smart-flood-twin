from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from torch import nn


TERRAIN_FEATURE_VERSION = 1
TERRAIN_FEATURE_NAMES = ("z", "slope_x", "slope_y", "dz_east", "dz_west",
                         "dz_north", "dz_south")
ENHANCED_TIME_FEATURE_NAMES = ("log_time", "rain_progress", "log_post_rain")


def _terrain_feature_array(terrain):
    """Build robustly normalised per-cell features without using NoData."""
    z = np.asarray(terrain.z, dtype=np.float64)
    mask = np.asarray(terrain.mask, dtype=bool)
    active_z = z[mask]
    z_centre = float(np.median(active_z))
    z_low, z_high = np.percentile(active_z, (5., 95.))
    z_scale = max(float(z_high - z_low), 1e-6)
    slope_magnitude = np.hypot(terrain.zx[mask], terrain.zy[mask])
    slope_scale = max(float(np.percentile(slope_magnitude, 99.)), 1e-6)

    def delta(dr, dc):
        shifted_z = np.zeros_like(z)
        shifted_mask = np.zeros_like(mask)
        if dr >= 0:
            source_r, target_r = slice(0, z.shape[0] - dr), slice(dr, z.shape[0])
        else:
            source_r, target_r = slice(-dr, z.shape[0]), slice(0, z.shape[0] + dr)
        if dc >= 0:
            source_c, target_c = slice(0, z.shape[1] - dc), slice(dc, z.shape[1])
        else:
            source_c, target_c = slice(-dc, z.shape[1]), slice(0, z.shape[1] + dc)
        shifted_z[target_r, target_c] = z[source_r, source_c]
        shifted_mask[target_r, target_c] = mask[source_r, source_c]
        pair = mask & shifted_mask
        result = np.zeros_like(z)
        result[pair] = shifted_z[pair] - z[pair]
        return result, pair

    east, east_pair = delta(0, -1)
    west, west_pair = delta(0, 1)
    north, north_pair = delta(1, 0)
    south, south_pair = delta(-1, 0)
    neighbour_values = np.concatenate((
        np.abs(east[east_pair]), np.abs(west[west_pair]),
        np.abs(north[north_pair]), np.abs(south[south_pair]),
    ))
    relief_scale = (max(float(np.percentile(neighbour_values, 99.)), 1e-6)
                    if neighbour_values.size else 1.)

    features = np.zeros((*z.shape, len(TERRAIN_FEATURE_NAMES)), dtype=np.float32)
    features[..., 0][mask] = ((z[mask] - z_centre) / z_scale).astype(np.float32)
    features[..., 1][mask] = (terrain.zx[mask] / slope_scale).astype(np.float32)
    features[..., 2][mask] = (terrain.zy[mask] / slope_scale).astype(np.float32)
    for index, values in enumerate((east, west, north, south), start=3):
        features[..., index][mask] = (values[mask] / relief_scale).astype(np.float32)
    metadata = {
        "version": TERRAIN_FEATURE_VERSION,
        "names": list(TERRAIN_FEATURE_NAMES),
        "z_centre_m": z_centre,
        "z_scale_m": z_scale,
        "slope_scale": slope_scale,
        "relief_scale_m": relief_scale,
    }
    return features, metadata


def _mass_projection_anchors(terrain, count, strata, seed):
    """Return fixed elevation-stratified cell centres and area weights."""
    if isinstance(count, bool) or int(count) != count or count <= 0:
        raise ValueError("mass projection anchor count must be a positive integer")
    if isinstance(strata, bool) or int(strata) != strata or strata <= 0:
        raise ValueError("mass projection strata must be a positive integer")
    count = min(int(count), terrain.active_count)
    strata = min(int(strata), count, terrain.active_count)
    order = np.argsort(
        terrain.z[terrain.active_rows, terrain.active_cols], kind="stable")
    groups = np.array_split(order, strata)
    sizes = np.asarray([len(group) for group in groups], dtype=np.int64)

    allocation = np.ones(strata, dtype=np.int64)
    remaining = count - strata
    if remaining:
        capacity = sizes - 1
        ideal = remaining * sizes / sizes.sum()
        extra = np.minimum(np.floor(ideal).astype(np.int64), capacity)
        allocation += extra
        leftover = remaining - int(extra.sum())
        remainders = ideal - np.floor(ideal)
        while leftover:
            eligible = np.flatnonzero(allocation < sizes)
            if not len(eligible):
                raise RuntimeError("cannot allocate mass projection anchors")
            ranking = eligible[np.argsort(-remainders[eligible], kind="stable")]
            take = min(leftover, len(ranking))
            allocation[ranking[:take]] += 1
            remainders[ranking[:take]] = 0.
            leftover -= take

    rng = np.random.default_rng(int(seed))
    indexes, weights = [], []
    for group, population, samples in zip(groups, sizes, allocation):
        selected = rng.choice(group, size=int(samples), replace=False)
        indexes.append(selected)
        weights.append(np.full(int(samples), population / (terrain.active_count * samples),
                               dtype=np.float64))
    indexes = np.concatenate(indexes)
    area_weights = np.concatenate(weights)
    permutation = rng.permutation(len(indexes))
    indexes, area_weights = indexes[permutation], area_weights[permutation]
    rows, cols = terrain.active_rows[indexes], terrain.active_cols[indexes]
    west, _, _, north = terrain.bounds
    xy = np.column_stack((west + (cols + .5) * terrain.dx,
                          north - (rows + .5) * terrain.dy))
    return xy.astype(np.float64), area_weights


class ConditionalFloodPINN(nn.Module):
    """(x,y,t,mean rain intensity,duration) -> (water depth,u,v)."""

    def __init__(self, scales: dict, width: int = 96, layers: int = 5,
                 log_conditions: bool = False, *, log1p_conditions: bool = False,
                 wet_velocity_gate: bool = False, terrain_features: bool = False,
                 terrain=None, enhanced_time_features: bool = False,
                 mass_conserving_projection: bool = False,
                 mass_projection_anchor_points: int = 512,
                 mass_projection_strata: int = 32,
                 mass_projection_seed: int = 20261113):
        super().__init__()
        self.scales = {k: float(v) for k, v in scales.items()}
        self.log_conditions = bool(log_conditions)
        self.log1p_conditions = bool(log1p_conditions)
        self.wet_velocity_gate = bool(wet_velocity_gate)
        self.terrain_features = bool(terrain_features)
        self.enhanced_time_features = bool(enhanced_time_features)
        self.mass_conserving_projection = bool(mass_conserving_projection)
        self.mass_projection_anchor_points = int(mass_projection_anchor_points)
        self.mass_projection_strata = int(mass_projection_strata)
        self.mass_projection_seed = int(mass_projection_seed)
        if self.log_conditions and self.log1p_conditions:
            raise ValueError("log_conditions and log1p_conditions are mutually exclusive")
        if self.terrain_features:
            if terrain is None:
                raise ValueError("terrain is required when terrain_features is enabled")
            feature_array, metadata = _terrain_feature_array(terrain)
            self.register_buffer("_terrain_feature_grid", torch.from_numpy(feature_array),
                                 persistent=False)
            self.register_buffer("_terrain_valid_mask", torch.from_numpy(terrain.mask.copy()),
                                 persistent=False)
            west, east, south, north = terrain.bounds
            self._terrain_geometry = (float(west), float(east), float(south), float(north),
                                      float(terrain.dx), float(terrain.dy),
                                      int(terrain.width), int(terrain.height))
            self.terrain_feature_metadata = metadata
        else:
            self.terrain_feature_metadata = None
        if self.mass_conserving_projection:
            if terrain is None:
                raise ValueError("terrain is required for mass-conserving projection")
            anchor_xy, anchor_weights = _mass_projection_anchors(
                terrain, self.mass_projection_anchor_points,
                self.mass_projection_strata, self.mass_projection_seed)
            self.mass_projection_anchor_points = len(anchor_xy)
            self.mass_projection_strata = min(
                self.mass_projection_strata, self.mass_projection_anchor_points,
                terrain.active_count)
            self.register_buffer("_mass_anchor_xy", torch.from_numpy(anchor_xy),
                                 persistent=False)
            self.register_buffer("_mass_anchor_weights", torch.from_numpy(anchor_weights),
                                 persistent=False)
            digest = hashlib.sha256(anchor_xy.astype("<f8", copy=False).tobytes())
            digest.update(anchor_weights.astype("<f8", copy=False).tobytes())
            self.mass_projection_anchor_fingerprint = digest.hexdigest()
        else:
            self.mass_projection_anchor_fingerprint = None
        input_features = (5 +
                          (len(TERRAIN_FEATURE_NAMES) if self.terrain_features else 0) +
                          (len(ENHANCED_TIME_FEATURE_NAMES)
                           if self.enhanced_time_features else 0))
        blocks = [nn.Linear(input_features, width), nn.Tanh()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(width, width), nn.Tanh()]
        blocks.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*blocks)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _raw_outputs(self, points):
        if points.ndim != 2 or points.shape[1] != 5:
            raise ValueError("points must have shape (N, 5)")
        s = self.scales
        p = points.to(torch.float64)
        offset = p.new_tensor([s["x0"], s["y0"], 0., 0., 0.])
        scale = p.new_tensor([s["lx"], s["ly"], s["time_s"],
                              s["rain_m_s"], s["duration_s"]])
        q = (p - offset) / scale
        if self.log_conditions:
            rain_min = s["rain_min_m_s"]
            duration_min = s["duration_min_s"]
            if torch.any(p[:, 3] <= 0) or torch.any(p[:, 4] <= 0):
                raise ValueError("log-conditioned rain intensity and duration must be positive")
            q[:, 3] = ((torch.log(p[:, 3]) - math.log(rain_min)) /
                       (math.log(s["rain_m_s"]) - math.log(rain_min)))
            q[:, 4] = ((torch.log(p[:, 4]) - math.log(duration_min)) /
                       (math.log(s["duration_s"]) - math.log(duration_min)))
        elif self.log1p_conditions:
            if torch.any(p[:, 3] < 0) or torch.any(p[:, 4] < 0):
                raise ValueError("log1p-conditioned rain intensity and duration must be nonnegative")
            rain_reference = s.get("rain_log_reference_m_s", 1. / 3_600_000.)
            duration_reference = s.get("duration_log_reference_s", 3600.)
            if rain_reference <= 0 or duration_reference <= 0:
                raise ValueError("log1p reference scales must be positive")
            q[:, 3] = (torch.log1p(p[:, 3] / rain_reference) /
                       math.log1p(s["rain_m_s"] / rain_reference))
            q[:, 4] = (torch.log1p(p[:, 4] / duration_reference) /
                       math.log1p(s["duration_s"] / duration_reference))
        if self.terrain_features:
            west, east, south, north, dx, dy, width, height = self._terrain_geometry
            col = torch.floor((p[:, 0] - west) / dx).to(torch.long)
            row = torch.floor((north - p[:, 1]) / dy).to(torch.long)
            col = torch.where(p[:, 0] == east, col.new_full((), width - 1), col)
            row = torch.where(p[:, 1] == south, row.new_full((), height - 1), row)
            inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            safe_row = row.clamp(0, height - 1)
            safe_col = col.clamp(0, width - 1)
            valid = inside & self._terrain_valid_mask[safe_row, safe_col]
            if not bool(torch.all(valid).item()):
                raise ValueError("terrain-feature lookup entered a NoData or out-of-domain cell")
            terrain_q = self._terrain_feature_grid[safe_row, safe_col].to(q.dtype)
            q = torch.cat((q, terrain_q), dim=1)
        if self.enhanced_time_features:
            time_reference = s.get("time_log_reference_s", 600.)
            if time_reference <= 0 or s["time_s"] <= 0:
                raise ValueError("enhanced time reference and time horizon must be positive")
            if torch.any(p[:, 2] < 0) or torch.any(p[:, 4] <= 0):
                raise ValueError("enhanced time features require nonnegative time and positive duration")
            denominator = math.log1p(s["time_s"] / time_reference)
            log_time = torch.log1p(p[:, 2] / time_reference) / denominator
            rain_progress = torch.minimum(p[:, 2], p[:, 4]) / p[:, 4]
            post_rain = torch.relu(p[:, 2] - p[:, 4])
            log_post_rain = torch.log1p(post_rain / time_reference) / denominator
            temporal_q = torch.stack((log_time, rain_progress, log_post_rain), dim=1)
            q = torch.cat((q, temporal_q), dim=1)
        q = q.to(next(self.parameters()).dtype)
        return self.net(q)

    def depth_factor(self, points):
        """Return the positive, unnormalised spatial depth factor."""
        raw = self._raw_outputs(points)
        return torch.nn.functional.softplus(raw[:, :1]) / math.log(2.)

    def _anchor_factor_means(self, conditions):
        """Compute one fixed-domain factor mean for each (t,R,D) row."""
        groups = len(conditions)
        if not groups:
            return conditions.new_empty((0,))
        anchors = self._mass_anchor_xy.to(device=conditions.device,
                                          dtype=conditions.dtype)
        anchor_count = len(anchors)
        xy = anchors.repeat(groups, 1)
        repeated_conditions = conditions.repeat_interleave(anchor_count, dim=0)
        anchor_points = torch.cat((xy, repeated_conditions), dim=1)
        factors = self.depth_factor(anchor_points).reshape(groups, anchor_count)
        weights = self._mass_anchor_weights.to(device=factors.device,
                                                dtype=factors.dtype)
        return (factors * weights[None, :]).sum(dim=1)

    def forward(self, points, *, projection_factor_mean=None):
        raw = self._raw_outputs(points)
        s = self.scales
        ramp = 1. - torch.exp(-points[:, 2:3].to(raw.dtype) / s["ramp_s"])
        precise = points.to(raw.dtype)
        accumulated_rain = precise[:, 3:4] * torch.minimum(precise[:, 2:3],
                                                            precise[:, 4:5])
        depth_factor = torch.nn.functional.softplus(raw[:, :1]) / math.log(2.)
        if projection_factor_mean is not None and not self.mass_conserving_projection:
            raise ValueError("projection_factor_mean requires mass-conserving projection")
        if self.mass_conserving_projection and len(points):
            conditions = precise[:, 2:5]
            unique_conditions, inverse = torch.unique(
                conditions, dim=0, sorted=True, return_inverse=True)
            if projection_factor_mean is None:
                accumulated_by_group = (unique_conditions[:, 1] *
                                        torch.minimum(unique_conditions[:, 0],
                                                      unique_conditions[:, 2]))
                positive = accumulated_by_group > 0
                factor_means = raw.new_ones(len(unique_conditions))
                if torch.any(positive):
                    positive_means = self._anchor_factor_means(unique_conditions[positive])
                    factor_means = factor_means.index_copy(
                        0, torch.nonzero(positive, as_tuple=False)[:, 0], positive_means)
            else:
                factor_means = torch.as_tensor(
                    projection_factor_mean, dtype=raw.dtype, device=raw.device)
                if factor_means.ndim == 0:
                    factor_means = factor_means.expand(len(unique_conditions))
                if factor_means.shape != (len(unique_conditions),):
                    raise ValueError("projection_factor_mean must be scalar or one value per condition")
            denominator = factor_means[inverse, None].clamp_min(torch.finfo(raw.dtype).tiny)
            h = accumulated_rain * depth_factor / denominator
        else:
            h = accumulated_rain * depth_factor
        uv = ramp * torch.tanh(raw[:, 1:]) * s["velocity_m_s"]
        if self.wet_velocity_gate:
            wet_gate_depth = s.get("wet_velocity_gate_m", 1e-5)
            if wet_gate_depth <= 0:
                raise ValueError("wet_velocity_gate_m must be positive")
            wet_gate = accumulated_rain / (accumulated_rain + wet_gate_depth)
            uv = uv * wet_gate
        return torch.cat((h, uv), dim=1)
