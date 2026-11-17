"""Independent off-grid physics and full-domain mass audit for experts."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .training import evaluate


SUPPORTED_EXPERT_NAMES = ("low", "medium", "high")


def _positive_integer(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or
            int(value) <= 0):
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _primary_rain_ranges(master):
    experts = master.get("experts")
    if not isinstance(experts, list) or len(experts) not in (2, 3):
        raise ValueError("continuous audit requires two or three experts")
    names = tuple(item.get("name") for item in experts if isinstance(item, dict))
    if names != SUPPORTED_EXPERT_NAMES[:len(experts)]:
        raise ValueError("continuous audit experts must be ordered low, medium[, high]")
    rain_low = min(float(item["intensities_mm_hr"][0]) for item in experts)
    rain_high = max(float(item["intensities_mm_hr"][-1]) for item in experts)
    routing = master.get("routing", {})
    low_max = float(routing.get("low_max_mm_hr", math.nan))
    if len(experts) == 2:
        edges = (rain_low, low_max, rain_high)
    else:
        medium_max = float(routing.get("medium_max_mm_hr", math.nan))
        edges = (rain_low, low_max, medium_max, rain_high)
    if (not all(math.isfinite(value) for value in edges) or
            any(right <= left for left, right in zip(edges, edges[1:]))):
        raise ValueError("continuous audit routing domain is invalid")
    return dict(zip(names, zip(edges, edges[1:])))


def latin_hypercube_events(master, count, *, seed=20261119):
    """Return deterministic off-grid conditions inside each primary route."""
    count = _positive_integer(count, "continuous audit condition count")
    durations = np.asarray(master.get("durations_hr", ()), dtype=np.float64)
    if (durations.size == 0 or not np.isfinite(durations).all() or
            np.any(durations <= 0)):
        raise ValueError("continuous audit durations are invalid")
    duration_low, duration_high = float(durations.min()), float(durations.max())
    if duration_high <= duration_low:
        raise ValueError("continuous audit requires a duration interval")
    rng = np.random.default_rng(int(seed))
    result = {}
    for expert_index, (name, rain_range) in enumerate(
            _primary_rain_ranges(master).items()):
        rain_q = (np.arange(count) + rng.random(count)) / count
        duration_q = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(rain_q)
        rng.shuffle(duration_q)
        rain = rain_range[0] + rain_q * (rain_range[1] - rain_range[0])
        duration_hr = duration_low + duration_q * (duration_high - duration_low)
        result[name] = [{
            "event_id": f"continuous_audit_{name}_{index:03d}",
            "split": "test",
            "mean_intensity_mm_hr": float(rain[index]),
            "duration_s": float(duration_hr[index] * 3600.),
            "total_mm": float(rain[index] * duration_hr[index]),
            "continuous_domain_audit": True,
            "expert_index": expert_index,
        } for index in range(count)]
    return result


def _full_xy(terrain):
    west, _, _, north = terrain.bounds
    return np.column_stack((
        west + (terrain.active_cols + .5) * terrain.dx,
        north - (terrain.active_rows + .5) * terrain.dy,
    )).astype(np.float64, copy=False)


def _points(xy, time_s, rainfall_mm_hr, duration_s):
    points = np.empty((len(xy), 5), dtype=np.float64)
    points[:, :2] = xy
    points[:, 2] = time_s
    points[:, 3] = rainfall_mm_hr / 3_600_000.
    points[:, 4] = duration_s
    return points


def _projection_factor_mean(model, xy, event, time_s, batch_size):
    if not getattr(model, "uses_external_mass_projection", False):
        return None
    factor_sum = 0.
    factor_method = getattr(model, "projection_factor", model.depth_factor)
    with torch.no_grad():
        for start in range(0, len(xy), batch_size):
            points = _points(
                xy[start:start + batch_size], time_s,
                event["mean_intensity_mm_hr"], event["duration_s"])
            factor = factor_method(torch.as_tensor(points, dtype=torch.float64))
            factor_sum += float(factor.to(torch.float64).sum())
    result = factor_sum / len(xy)
    if not math.isfinite(result) or result <= 0.:
        raise ValueError("continuous audit found an invalid projection factor")
    return result


def _full_domain_mean_depth(model, xy, event, time_s, batch_size):
    factor_mean = _projection_factor_mean(model, xy, event, time_s, batch_size)
    depth_sum = 0.
    with torch.no_grad():
        for start in range(0, len(xy), batch_size):
            points = _points(
                xy[start:start + batch_size], time_s,
                event["mean_intensity_mm_hr"], event["duration_s"])
            tensor = torch.as_tensor(points, dtype=torch.float64)
            values = (model(tensor) if factor_mean is None else
                      model(tensor, projection_factor_mean=factor_mean))
            if values.shape != (len(tensor), 3) or not torch.isfinite(values).all():
                raise ValueError("continuous audit model prediction is invalid")
            depth_sum += float(values[:, 0].to(torch.float64).sum())
    return depth_sum / len(xy)


def _physics_summary(rows, model, thresholds):
    velocity_scale = float(model.scales["velocity_m_s"])
    mean_pde = float(np.mean([row["pde_loss"] for row in rows]))
    mean_continuity = float(np.mean([
        row["pde_continuity_loss"] for row in rows]))
    mean_mass = float(np.mean([row["mass_loss"] for row in rows]))
    worst_pde = max(float(row["pde_loss"]) for row in rows)
    worst_continuity = max(float(row["pde_continuity_loss"]) for row in rows)
    worst_mass = max(float(row["mass_loss"]) for row in rows)
    mean_boundary = velocity_scale * math.sqrt(float(np.mean([
        row["boundary_loss"] for row in rows])))
    worst_boundary = velocity_scale * math.sqrt(max(
        float(row["boundary_loss"]) for row in rows))
    passed = bool(
        mean_pde <= float(thresholds["test_mean_pde_loss"]) and
        mean_continuity <= float(thresholds.get(
            "test_mean_continuity_loss", math.inf)) and
        mean_mass <= float(thresholds["test_mean_mass_loss"]) and
        worst_pde <= float(thresholds["worst_all_event_pde_loss"]) and
        worst_continuity <= float(thresholds.get(
            "worst_all_event_continuity_loss", math.inf)) and
        worst_mass <= float(thresholds["worst_all_event_mass_loss"]) and
        mean_boundary <= float(thresholds.get(
            "test_boundary_rms_normal_velocity_m_s", math.inf)) and
        worst_boundary <= float(thresholds.get(
            "worst_all_event_boundary_rms_normal_velocity_m_s", math.inf)))
    return {
        "passed": passed,
        "mean_pde_loss": mean_pde,
        "mean_continuity_loss": mean_continuity,
        "mean_mass_loss": mean_mass,
        "worst_pde_loss": worst_pde,
        "worst_continuity_loss": worst_continuity,
        "worst_mass_loss": worst_mass,
        "mean_boundary_rms_normal_velocity_m_s": mean_boundary,
        "worst_boundary_rms_normal_velocity_m_s": worst_boundary,
    }


def evaluate_continuous_domain(experts, checkpoints, terrain, master, settings):
    """Audit off-grid conditions and exact active-DEM storage."""
    condition_count = _positive_integer(
        settings.get("conditions_per_expert", 8),
        "continuous audit conditions_per_expert")
    full_count = _positive_integer(
        settings.get("full_domain_conditions_per_expert", 2),
        "continuous audit full_domain_conditions_per_expert")
    full_count = min(full_count, condition_count)
    repeats = _positive_integer(settings.get("repeats", 3),
                                "continuous audit repeats")
    interior_points = _positive_integer(
        settings.get("interior_points", 1024),
        "continuous audit interior_points")
    batch_size = _positive_integer(settings.get("batch_size", 8192),
                                   "continuous audit batch_size")
    seed = int(settings.get("seed", 20261119))
    mass_limit = float(settings.get(
        "max_full_domain_mass_relative_error", .05))
    if not math.isfinite(mass_limit) or mass_limit < 0.:
        raise ValueError(
            "continuous audit full-domain mass threshold must be finite and nonnegative")
    events = latin_hypercube_events(master, condition_count, seed=seed)
    thresholds = master["training"]["success_thresholds"]
    xy = _full_xy(terrain)
    expert_reports = []
    for expert_index, (name, model) in enumerate(experts.items()):
        if name not in events or name not in checkpoints:
            raise ValueError("continuous audit expert/checkpoint set is incomplete")
        cfg = copy.deepcopy(master["training"])
        cfg["scales"] = dict(checkpoints[name]["scales"])
        rows = evaluate(
            model, terrain, events[name], cfg,
            seed=seed + 10_000 * (expert_index + 1), repeats=repeats,
            interior_points=interior_points)
        physics = _physics_summary(rows, model, thresholds)
        full_mass_rows = []
        for event in events[name][:full_count]:
            duration_s = float(event["duration_s"])
            time_end_s = float(master["training"]["time_end_s"])
            times = sorted({min(.5 * duration_s, time_end_s),
                            min(duration_s, time_end_s), time_end_s})
            for time_s in times:
                mean_depth = _full_domain_mean_depth(
                    model, xy, event, time_s, batch_size)
                expected = (event["mean_intensity_mm_hr"] / 3_600_000. *
                            min(time_s, duration_s))
                relative_error = abs(mean_depth - expected) / max(expected, 1e-9)
                full_mass_rows.append({
                    "event_id": event["event_id"],
                    "rainfall_mm_hr": event["mean_intensity_mm_hr"],
                    "duration_hr": duration_s / 3600.,
                    "time_hr": time_s / 3600.,
                    "mean_depth_m": mean_depth,
                    "expected_mean_depth_m": expected,
                    "relative_error": relative_error,
                })
        worst_full_mass = max(row["relative_error"] for row in full_mass_rows)
        mass_passed = worst_full_mass <= mass_limit
        expert_reports.append({
            "model_id": name,
            "passed": bool(physics["passed"] and mass_passed),
            "physics": physics,
            "full_domain_mass_passed": bool(mass_passed),
            "worst_full_domain_mass_relative_error": worst_full_mass,
            "conditions": events[name],
            "physics_cases": rows,
            "full_domain_mass_cases": full_mass_rows,
        })
    return {
        "enabled": True,
        "passed": bool(all(item["passed"] for item in expert_reports)),
        "seed": seed,
        "conditions_per_expert": condition_count,
        "full_domain_conditions_per_expert": full_count,
        "interior_points": interior_points,
        "repeats": repeats,
        "full_domain_active_cells": int(terrain.active_count),
        "max_full_domain_mass_relative_error": mass_limit,
        "experts": expert_reports,
    }
