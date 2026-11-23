from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .model import ConditionalFloodPINN


FRAME_CSV_FIELDS = (
    "time_s", "row", "col", "x_m", "y_m", "depth_m", "u_m_s", "v_m_s",
    "speed_m_s",
)
SUMMARY_CSV_FIELDS = (
    "model_id", "primary_model_id", "bundle_id", "routing_policy",
    "expert_weights", "rainfall_mm_hr", "duration_s", "time_s", "file",
    "npz_file", "depth_tif_file", "max_depth_m", "mean_depth_m",
    "max_speed_m_s", "mass_projection",
)


def _write_frame_csv(path, *, time_s, terrain, xy, result):
    """Write one complete active-domain prediction frame with stable columns."""
    if (xy.shape != (terrain.active_count, 2) or
            result.shape != (terrain.active_count, 3)):
        raise ValueError("prediction CSV arrays do not match the active terrain domain")
    if not np.isfinite(xy).all() or not np.isfinite(result).all():
        raise FloatingPointError("prediction CSV contains nonfinite values")
    speed = np.linalg.norm(result[:, 1:].astype(np.float64), axis=1)
    table = np.column_stack((
        np.full(terrain.active_count, float(time_s), dtype=np.float64),
        terrain.active_rows, terrain.active_cols, xy[:, 0], xy[:, 1],
        result[:, 0], result[:, 1], result[:, 2], speed,
    ))
    np.savetxt(
        path, table, delimiter=",", header=",".join(FRAME_CSV_FIELDS), comments="",
        fmt=("%.17g", "%d", "%d", "%.17g", "%.17g", "%.9g", "%.9g",
             "%.9g", "%.9g"),
    )


def _write_summary_csv(path, *, summary):
    """Write a one-row-per-frame index suitable for Unreal runtime import."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for frame in summary["frames"]:
            writer.writerow({
                "model_id": summary.get(
                    "routing_label", summary["model_id"]),
                "primary_model_id": summary["model_id"],
                "bundle_id": summary.get("bundle_id", ""),
                "routing_policy": summary.get("routing_policy", ""),
                "expert_weights": (
                    json.dumps(summary["expert_weights"], sort_keys=True,
                               separators=(",", ":"))
                    if "expert_weights" in summary else ""),
                "rainfall_mm_hr": summary["rainfall_mm_hr"],
                "duration_s": summary["duration_s"],
                "time_s": frame["time_s"],
                "file": frame["csv_file"],
                "npz_file": frame["file"],
                "depth_tif_file": frame["depth_tif_file"],
                "max_depth_m": frame["max_depth_m"],
                "mean_depth_m": frame["mean_depth_m"],
                "max_speed_m_s": frame["max_speed_m_s"],
                "mass_projection": frame["mass_projection"],
            })


def load_model(path, terrain):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["terrain_fingerprint"] != terrain.fingerprint:
        raise ValueError("checkpoint terrain fingerprint does not match this DEM/Manning setting")
    model = ConditionalFloodPINN(
        checkpoint["scales"], checkpoint["width"], checkpoint["layers"],
        checkpoint.get("log_conditions", False),
        log1p_conditions=checkpoint.get("log1p_conditions", False),
        wet_velocity_gate=checkpoint.get("wet_velocity_gate", False),
        terrain_features=checkpoint.get("terrain_features", False),
        terrain=terrain,
        enhanced_time_features=checkpoint.get("enhanced_time_features", False),
        mass_conserving_projection=checkpoint.get("mass_conserving_projection", False),
        mass_projection_anchor_points=checkpoint.get("mass_projection_anchor_points", 512),
        mass_projection_strata=checkpoint.get("mass_projection_strata", 32),
        mass_projection_seed=checkpoint.get("mass_projection_seed", 20261113),
        separate_output_networks=checkpoint.get("separate_output_networks", False),
        hard_mask_boundary=checkpoint.get("hard_mask_boundary", False),
        hydrostatic_depth_baseline=checkpoint.get("hydrostatic_depth_baseline", False),
        hydrostatic_learned_blend=checkpoint.get("hydrostatic_learned_blend", 1.0),
        hydrostatic_correction_mode=checkpoint.get("hydrostatic_correction_mode", "convex"),
        hydrostatic_uniform_floor=checkpoint.get("hydrostatic_uniform_floor", 0.0),
        terrain_local_coordinates=checkpoint.get("terrain_local_coordinates", False),
        terrain_local_plane_feature=checkpoint.get(
            "terrain_local_plane_feature", False),
        conditioned_velocity_scale=checkpoint.get(
            "conditioned_velocity_scale", False),
        conditioned_velocity_scale_coefficient=checkpoint.get(
            "conditioned_velocity_scale_coefficient", 1.0),
        conditioned_velocity_scale_epsilon_m_s=checkpoint.get(
            "conditioned_velocity_scale_epsilon_m_s", 1e-3),
        network_architecture=checkpoint.get("network_architecture", "plain"),
    )
    if checkpoint.get("terrain_features", False):
        expected_version = checkpoint.get("terrain_feature_version")
        actual_version = model.terrain_feature_metadata["version"]
        if expected_version is not None and int(expected_version) != actual_version:
            raise ValueError("checkpoint terrain-feature version is unsupported")
    if checkpoint.get("terrain_local_plane_feature", False):
        expected_version = checkpoint.get("terrain_local_plane_feature_version")
        actual_version = model.terrain_local_plane_feature_metadata["version"]
        if expected_version is not None and int(expected_version) != actual_version:
            raise ValueError("checkpoint local-plane terrain-feature version is unsupported")
    expected_anchors = checkpoint.get("mass_projection_anchor_fingerprint")
    actual_anchors = getattr(model, "mass_projection_anchor_fingerprint", None)
    if expected_anchors is not None and expected_anchors != actual_anchors:
        raise ValueError("checkpoint mass-projection anchors cannot be reproduced")
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, checkpoint


def predict(model, checkpoint, terrain, rainfall_mm_hr, duration_s, times_s, output_dir, *,
            batch_size=8192, progress_callback=None):
    times_s = tuple(float(value) for value in times_s)
    if not times_s or any(not np.isfinite(value) for value in times_s):
        raise ValueError("prediction times must be a nonempty finite sequence")
    if any(right <= left for left, right in zip(times_s, times_s[1:])):
        raise ValueError("prediction times must be strictly increasing without duplicates")
    rain_domain = checkpoint["training_rain_domain"]
    if not rain_domain["rainfall_mm_hr"][0] <= rainfall_mm_hr <= rain_domain["rainfall_mm_hr"][1]:
        raise ValueError("rainfall intensity is outside the trained event range")
    if not rain_domain["duration_s"][0] <= duration_s <= rain_domain["duration_s"][1]:
        raise ValueError("rain duration is outside the trained event range")
    if any(not np.isfinite(t) or t < 0 or t > model.scales["time_s"] for t in times_s):
        raise ValueError("prediction time is outside the trained model horizon")
    model_id = (model.model_id_for_rainfall(rainfall_mm_hr)
                if hasattr(model, "model_id_for_rainfall")
                else checkpoint.get("model_id", "single"))
    summary = {"rainfall_mm_hr": rainfall_mm_hr, "duration_s": duration_s,
               "times_s": list(map(float, times_s)), "model_id": model_id,
               "frames": []}
    expert_weights = None
    if hasattr(model, "expert_weights_for_rainfall"):
        expert_weights = model.expert_weights_for_rainfall(rainfall_mm_hr)
        summary["expert_weights"] = expert_weights
        summary["routing_policy"] = getattr(
            model, "routing_policy", "single_expert")
        summary["routing_label"] = model.routing_label_for_rainfall(
            rainfall_mm_hr)
    if checkpoint.get("bundle_id"):
        summary["bundle_id"] = checkpoint["bundle_id"]
    prediction_model = model
    if hasattr(model, "experts") and model_id in model.experts:
        prediction_model = model.experts[model_id]
    active_blend = ([name for name, weight in expert_weights.items() if weight > 0.]
                    if expert_weights is not None else [])
    if len(active_blend) > 1 and any(
            getattr(model.experts[name], "uses_external_mass_projection", False)
            for name in active_blend):
        raise ValueError(
            "full-domain prediction cannot blend externally projected experts")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    xy = terrain.sample_points(np.random.default_rng(0), terrain.active_count, jitter=False)
    # sample_points uses replacement; use exact raster order for complete output.
    west, _, _, north = terrain.bounds
    xy[:, 0] = west + (terrain.active_cols + .5) * terrain.dx
    xy[:, 1] = north - (terrain.active_rows + .5) * terrain.dy
    for index, t in enumerate(times_s):
        result = np.empty((terrain.active_count, 3), dtype=np.float32)
        factor_mean = None
        if getattr(prediction_model, "uses_external_mass_projection", False):
            # Exact whole-domain first pass. The second pass shares this one
            # scalar across chunks, making the result batch/order invariant.
            factor_sum = 0.
            if rainfall_mm_hr > 0 and t > 0:
                for start in range(0, len(xy), batch_size):
                    stop = min(len(xy), start + batch_size)
                    p = np.empty((stop - start, 5)); p[:, :2] = xy[start:stop]
                    p[:, 2] = t; p[:, 3] = rainfall_mm_hr / 3_600_000.; p[:, 4] = duration_s
                    with torch.no_grad():
                        factor_method = getattr(prediction_model, "projection_factor",
                                                prediction_model.depth_factor)
                        factors = factor_method(
                            torch.as_tensor(p, dtype=torch.float64))
                    factor_sum += float(factors.to(torch.float64).sum())
                factor_mean = factor_sum / terrain.active_count
            else:
                factor_mean = 1.
        for start in range(0, len(xy), batch_size):
            stop = min(len(xy), start + batch_size)
            p = np.empty((stop - start, 5)); p[:, :2] = xy[start:stop]
            p[:, 2] = t; p[:, 3] = rainfall_mm_hr / 3_600_000.; p[:, 4] = duration_s
            with torch.no_grad():
                point_tensor = torch.as_tensor(p, dtype=torch.float64)
                if factor_mean is None:
                    values = model(point_tensor)
                else:
                    values = prediction_model(
                        point_tensor, projection_factor_mean=factor_mean)
                result[start:stop] = values.numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("model produced nonfinite prediction values")
        frame = output / f"frame_{index:03d}.npz"
        frame_csv = output / f"frame_{index:03d}.csv"
        tif = output / f"depth_{index:03d}.tif"
        np.savez_compressed(frame, row=terrain.active_rows, col=terrain.active_cols,
                            x=xy[:, 0], y=xy[:, 1], h=result[:, 0], u=result[:, 1], v=result[:, 2])
        _write_frame_csv(frame_csv, time_s=t, terrain=terrain, xy=xy, result=result)
        summary["frames"].append({"time_s": float(t), "file": frame.name,
                                  "csv_file": frame_csv.name,
                                  "depth_tif_file": tif.name,
                                  "max_depth_m": float(result[:, 0].max()),
                                  "mean_depth_m": float(result[:, 0].mean()),
                                  "max_speed_m_s": float(np.linalg.norm(result[:, 1:], axis=1).max()),
                                  "mass_projection": ("full_active_domain"
                                                      if factor_mean is not None else "none")})
        raster = np.full((terrain.height, terrain.width), np.nan, dtype=np.float32)
        raster[terrain.active_rows, terrain.active_cols] = result[:, 0]
        import rasterio
        with rasterio.open(tif, "w", driver="GTiff", width=terrain.width, height=terrain.height,
                           count=1, dtype="float32", crs=terrain.crs, transform=terrain.transform,
                           nodata=np.nan, compress="deflate") as dst:
            dst.write(raster, 1)
        if progress_callback is not None:
            progress_callback(index + 1, len(times_s), float(t), summary["frames"][-1])
    _write_summary_csv(output / "prediction_summary.csv", summary=summary)
    (output / "prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
