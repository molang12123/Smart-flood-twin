"""Deterministic consistency audit for overlapping rainfall experts.

Individual physics gates are necessary but not sufficient for deployment: two
adjacent experts can both pass and still disagree at a routing threshold.  This
module compares them on identical terrain cells, event phases, and mask-boundary
points.  It emits an activatable bundle manifest only after every individual
and overlap gate passes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .predict import load_model
from .terrain import MaskedRasterTerrain


EXPERT_NAMES = ("low", "medium", "high")


def _finite_positive(value, label):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _cell_centres(terrain, count, rng):
    count = min(int(count), terrain.active_count)
    if count <= 0:
        raise ValueError("overlap audit spatial_samples must be positive")
    indexes = rng.choice(terrain.active_count, size=count, replace=False)
    rows = terrain.active_rows[indexes]
    cols = terrain.active_cols[indexes]
    west, _, _, north = terrain.bounds
    return np.column_stack((west + (cols + .5) * terrain.dx,
                            north - (rows + .5) * terrain.dy))


def _points(xy, time_s, rainfall_mm_hr, duration_s):
    result = np.empty((len(xy), 5), dtype=np.float64)
    result[:, :2] = xy
    result[:, 2] = time_s
    result[:, 3] = rainfall_mm_hr / 3_600_000.
    result[:, 4] = duration_s
    return result


def _infer(model, points, batch_size, *, projection_factor_mean=None):
    values = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.as_tensor(points[start:start + batch_size], dtype=torch.float64)
            if projection_factor_mean is None:
                result = model(batch)
            else:
                result = model(batch, projection_factor_mean=projection_factor_mean)
            if result.ndim != 2 or result.shape != (len(batch), 3):
                raise ValueError("expert returned an invalid prediction shape during overlap audit")
            values.append(result.detach().cpu().numpy())
    result = np.concatenate(values, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("expert returned nonfinite predictions during overlap audit")
    return result


def _full_domain_projection(model, full_xy, time_s, rainfall_mm_hr, duration_s,
                            batch_size):
    """Return deployment-equivalent depth-factor mean and anchor error."""
    if not getattr(model, "mass_conserving_projection", False):
        return None, 0.
    condition = torch.tensor(
        [[time_s, rainfall_mm_hr / 3_600_000., duration_s]],
        dtype=torch.float64)
    with torch.no_grad():
        anchor_mean = float(model._anchor_factor_means(condition)[0])
    factor_sum = 0.
    with torch.no_grad():
        for start in range(0, len(full_xy), batch_size):
            xy = full_xy[start:start + batch_size]
            points = _points(xy, time_s, rainfall_mm_hr, duration_s)
            factors = model.depth_factor(torch.as_tensor(points, dtype=torch.float64))
            factor_sum += float(factors.to(torch.float64).sum())
    full_mean = factor_sum / len(full_xy)
    if not math.isfinite(full_mean) or full_mean <= 0:
        raise ValueError("expert returned an invalid full-domain projection factor")
    relative_error = abs(anchor_mean - full_mean) / full_mean
    return full_mean, relative_error


def _symmetric_relative_scalar(left, right, floor=1e-12):
    return abs(left - right) / max(.5 * (abs(left) + abs(right)), floor)


def _symmetric_relative_l2(left, right, floor=1e-12):
    numerator = float(np.linalg.norm(left - right))
    denominator = max(.5 * float(np.linalg.norm(left) + np.linalg.norm(right)), floor)
    return numerator / denominator


def evaluate_overlap_consistency(experts, terrain, overlaps, durations_hr, time_end_s,
                                 thresholds, *, spatial_samples=4096,
                                 boundary_samples=1024, batch_size=8192,
                                 seed=20261113):
    """Evaluate adjacent experts on a deterministic, shared audit matrix.

    Each overlap is sampled at its two endpoints and midpoint.  Every configured
    duration is checked at rain midpoint, rain stop, and the model horizon.
    """
    time_end_s = _finite_positive(time_end_s, "time_end_s")
    batch_size = int(_finite_positive(batch_size, "batch_size"))
    boundary_samples = int(_finite_positive(boundary_samples, "boundary_samples"))
    durations = sorted({_finite_positive(value, "duration_hr") for value in durations_hr})
    if not durations:
        raise ValueError("overlap audit requires at least one duration")
    storage_limit = _finite_positive(
        thresholds["overlap_mean_storage_difference"],
        "overlap_mean_storage_difference")
    depth_limit = _finite_positive(
        thresholds["overlap_relative_l2_depth"],
        "overlap_relative_l2_depth")
    boundary_limit = thresholds.get("overlap_boundary_rms_normal_velocity_m_s")
    if boundary_limit is not None:
        boundary_limit = _finite_positive(
            boundary_limit, "overlap_boundary_rms_normal_velocity_m_s")
    projection_limit = thresholds.get("mass_projection_anchor_relative_error")
    if projection_limit is not None:
        projection_limit = _finite_positive(
            projection_limit, "mass_projection_anchor_relative_error")

    rng = np.random.default_rng(int(seed))
    xy = _cell_centres(terrain, spatial_samples, rng)
    west, _, _, north = terrain.bounds
    full_xy = np.column_stack((west + (terrain.active_cols + .5) * terrain.dx,
                               north - (terrain.active_rows + .5) * terrain.dy))
    boundary_xy, boundary_normals = terrain.sample_boundary(rng, boundary_samples)
    projection_cache = {}
    rows = []
    for overlap in overlaps:
        left_name, right_name = overlap["left"], overlap["right"]
        if left_name not in experts or right_name not in experts:
            raise ValueError("overlap audit references an unknown expert")
        rain_range = overlap["rainfall_mm_hr"]
        if not isinstance(rain_range, (list, tuple)) or len(rain_range) != 2:
            raise ValueError("overlap rainfall range must contain two values")
        rain_low, rain_high = map(float, rain_range)
        if (not 0 <= rain_low < rain_high or
                not all(map(math.isfinite, (rain_low, rain_high)))):
            raise ValueError("overlap rainfall range is invalid")
        rainfalls = (rain_low, .5 * (rain_low + rain_high), rain_high)
        for rainfall in rainfalls:
            for duration_hr in durations:
                duration_s = duration_hr * 3600.
                times = sorted({min(.5 * duration_s, time_end_s),
                                min(duration_s, time_end_s), time_end_s})
                for time_s in times:
                    projection = {}
                    projection_errors = {}
                    for model_name in (left_name, right_name):
                        key = (model_name, float(time_s), float(rainfall), float(duration_s))
                        if key not in projection_cache:
                            projection_cache[key] = _full_domain_projection(
                                experts[model_name], full_xy, time_s, rainfall,
                                duration_s, batch_size)
                        projection[model_name], projection_errors[model_name] = (
                            projection_cache[key])
                    interior_points = _points(xy, time_s, rainfall, duration_s)
                    left = _infer(experts[left_name], interior_points, batch_size,
                                  projection_factor_mean=projection[left_name])
                    right = _infer(experts[right_name], interior_points, batch_size,
                                   projection_factor_mean=projection[right_name])
                    mean_left = float(np.mean(left[:, 0]))
                    mean_right = float(np.mean(right[:, 0]))

                    boundary_points = _points(boundary_xy, time_s, rainfall, duration_s)
                    left_boundary = _infer(
                        experts[left_name], boundary_points, batch_size,
                        projection_factor_mean=projection[left_name])
                    right_boundary = _infer(
                        experts[right_name], boundary_points, batch_size,
                        projection_factor_mean=projection[right_name])
                    left_normal = np.sum(left_boundary[:, 1:] * boundary_normals, axis=1)
                    right_normal = np.sum(right_boundary[:, 1:] * boundary_normals, axis=1)
                    left_discharge = left_boundary[:, 0] * left_normal
                    right_discharge = right_boundary[:, 0] * right_normal
                    rows.append({
                        "experts": [left_name, right_name],
                        "rainfall_mm_hr": rainfall,
                        "duration_hr": duration_hr,
                        "time_hr": time_s / 3600.,
                        "mean_depth_m": [mean_left, mean_right],
                        "mean_storage_relative_difference": _symmetric_relative_scalar(
                            mean_left, mean_right),
                        "depth_relative_l2_difference": _symmetric_relative_l2(
                            left[:, 0], right[:, 0]),
                        "velocity_relative_l2_difference": _symmetric_relative_l2(
                            left[:, 1:], right[:, 1:]),
                        "boundary_rms_normal_velocity_m_s": [
                            float(np.sqrt(np.mean(left_normal ** 2))),
                            float(np.sqrt(np.mean(right_normal ** 2))),
                        ],
                        "boundary_relative_l2_discharge_difference": _symmetric_relative_l2(
                            left_discharge, right_discharge),
                        "mass_projection_anchor_relative_error": [
                            projection_errors[left_name], projection_errors[right_name]],
                    })

    if not rows:
        raise ValueError("overlap audit matrix is empty")
    worst_storage = max(row["mean_storage_relative_difference"] for row in rows)
    worst_depth = max(row["depth_relative_l2_difference"] for row in rows)
    worst_boundary = max(max(row["boundary_rms_normal_velocity_m_s"]) for row in rows)
    worst_projection = max(max(row["mass_projection_anchor_relative_error"])
                           for row in rows)
    passed = worst_storage <= storage_limit and worst_depth <= depth_limit
    if boundary_limit is not None:
        passed = passed and worst_boundary <= boundary_limit
    if projection_limit is not None:
        passed = passed and worst_projection <= projection_limit
    return {
        "passed": bool(passed),
        "sample_count": len(xy),
        "boundary_sample_count": len(boundary_xy),
        "case_count": len(rows),
        "thresholds": {
            "overlap_mean_storage_difference": storage_limit,
            "overlap_relative_l2_depth": depth_limit,
            "overlap_boundary_rms_normal_velocity_m_s": boundary_limit,
            "mass_projection_anchor_relative_error": projection_limit,
        },
        "worst_mean_storage_relative_difference": worst_storage,
        "worst_depth_relative_l2_difference": worst_depth,
        "worst_boundary_rms_normal_velocity_m_s": worst_boundary,
        "worst_mass_projection_anchor_relative_error": worst_projection,
        "cases": rows,
    }


def _configured_overlaps(master):
    ranges = {item["name"]: (float(item["intensities_mm_hr"][0]),
                              float(item["intensities_mm_hr"][-1]))
              for item in master["experts"]}
    result = []
    for left, right in zip(EXPERT_NAMES, EXPERT_NAMES[1:]):
        low = max(ranges[left][0], ranges[right][0])
        high = min(ranges[left][1], ranges[right][1])
        if not low < high:
            raise ValueError(f"experts {left}/{right} have no audit overlap")
        result.append({"left": left, "right": right,
                       "rainfall_mm_hr": [low, high]})
    return result


def build_approved_manifest(master, terrain, checkpoints, audit_report, output_root):
    """Build an activation manifest only after every deployment gate passes."""
    individual_passed = all(item["physics_training_passed"]
                            for item in audit_report["experts"])
    overlap_passed = bool(audit_report["overlap"]["passed"])
    if not individual_passed or not overlap_passed:
        raise ValueError("cannot approve an expert bundle whose audit gates did not pass")
    rain_min = min(float(item["intensities_mm_hr"][0]) for item in master["experts"])
    rain_max = max(float(item["intensities_mm_hr"][-1]) for item in master["experts"])
    duration_min = min(map(float, master["durations_hr"])) * 3600.
    duration_max = max(map(float, master["durations_hr"])) * 3600.
    experts = []
    for name in EXPERT_NAMES:
        domain = checkpoints[name]["training_rain_domain"]
        model_path = output_root / name / "model.pt"
        experts.append({
            "model_id": name,
            "path": model_path.relative_to(output_root).as_posix(),
            "trained_rainfall_mm_hr": list(map(float, domain["rainfall_mm_hr"])),
        })
    routing = master["routing"]
    return {
        "schema_version": 1,
        "bundle_id": f"minxiong-{output_root.name}",
        "activation_approved": True,
        "physics_training_passed": True,
        "overlap_consistency_passed": True,
        "terrain_fingerprint": terrain.fingerprint,
        "time_s": float(master["training"]["time_end_s"]),
        "training_rain_domain": {
            "rainfall_mm_hr": [rain_min, rain_max],
            "duration_s": [duration_min, duration_max],
        },
        "routing": {
            "low_max_mm_hr": float(routing["low_max_mm_hr"]),
            "medium_max_mm_hr": float(routing["medium_max_mm_hr"]),
        },
        "audit_file": "overlap_audit.json",
        "experts": experts,
    }


def audit_and_publish(config_path, *, model_loader=load_model):
    """Audit trained checkpoints and conditionally write ``manifest.json``."""
    config_path = Path(config_path).resolve()
    master = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = config_path.parent
    resolve = lambda value: (base / value).resolve()
    output_root = resolve(master["output_root"])
    if not output_root.is_dir():
        raise FileNotFoundError(f"expert output directory does not exist: {output_root}")
    terrain = MaskedRasterTerrain(resolve(master["terrain_tif"]),
                                  manning_n=master["manning_n"])
    experts, checkpoints, individual = {}, {}, []
    for name in EXPERT_NAMES:
        folder = output_root / name
        report_path = folder / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model, checkpoint = model_loader(folder / "model.pt", terrain)
        if checkpoint.get("terrain_fingerprint") != terrain.fingerprint:
            raise ValueError(f"expert {name} terrain fingerprint mismatch")
        experts[name] = model
        checkpoints[name] = checkpoint
        individual.append({
            "model_id": name,
            "physics_training_passed": report.get("physics_training_passed") is True,
            "completed_steps": report.get("completed_steps"),
            "best_validation_loss": report.get("best_validation_loss"),
            "report": str(report_path),
        })

    settings = master.get("overlap_audit", {})
    overlap = evaluate_overlap_consistency(
        experts, terrain, _configured_overlaps(master), master["durations_hr"],
        master["training"]["time_end_s"], master["training"]["success_thresholds"],
        spatial_samples=settings.get("spatial_samples", 4096),
        boundary_samples=settings.get("boundary_samples", 1024),
        batch_size=settings.get("batch_size", 8192),
        seed=settings.get("seed", 20261113))
    all_individual = all(item["physics_training_passed"] for item in individual)
    audit_report = {
        "physics_training_passed": all_individual,
        "overlap_consistency_passed": overlap["passed"],
        "activation_approved": bool(all_individual and overlap["passed"]),
        "terrain_fingerprint": terrain.fingerprint,
        "experts": individual,
        "overlap": overlap,
    }
    audit_path = output_root / "overlap_audit.json"
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    if audit_report["activation_approved"]:
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
        manifest = build_approved_manifest(master, terrain, checkpoints,
                                           audit_report, output_root)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        audit_report["manifest"] = str(manifest_path)
    return audit_report
