from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import ConditionalFloodPINN


def load_model(path, terrain):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["terrain_fingerprint"] != terrain.fingerprint:
        raise ValueError("checkpoint terrain fingerprint does not match this DEM/Manning setting")
    model = ConditionalFloodPINN(checkpoint["scales"], checkpoint["width"], checkpoint["layers"])
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, checkpoint


def predict(model, checkpoint, terrain, rainfall_mm_hr, duration_s, times_s, output_dir, *, batch_size=8192):
    rain_domain = checkpoint["training_rain_domain"]
    if not rain_domain["rainfall_mm_hr"][0] <= rainfall_mm_hr <= rain_domain["rainfall_mm_hr"][1]:
        raise ValueError("rainfall intensity is outside the trained event range")
    if not rain_domain["duration_s"][0] <= duration_s <= rain_domain["duration_s"][1]:
        raise ValueError("rain duration is outside the trained event range")
    if any(not np.isfinite(t) or t < 0 or t > model.scales["time_s"] for t in times_s):
        raise ValueError("prediction time is outside the trained model horizon")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    xy = terrain.sample_points(np.random.default_rng(0), terrain.active_count, jitter=False)
    # sample_points uses replacement; use exact raster order for complete output.
    west, _, _, north = terrain.bounds
    xy[:, 0] = west + (terrain.active_cols + .5) * terrain.dx
    xy[:, 1] = north - (terrain.active_rows + .5) * terrain.dy
    summary = {"rainfall_mm_hr": rainfall_mm_hr, "duration_s": duration_s,
               "times_s": list(map(float, times_s)), "frames": []}
    for index, t in enumerate(times_s):
        result = np.empty((terrain.active_count, 3), dtype=np.float32)
        for start in range(0, len(xy), batch_size):
            stop = min(len(xy), start + batch_size)
            p = np.empty((stop - start, 5)); p[:, :2] = xy[start:stop]
            p[:, 2] = t; p[:, 3] = rainfall_mm_hr / 3_600_000.; p[:, 4] = duration_s
            with torch.no_grad(): result[start:stop] = model(torch.as_tensor(p, dtype=torch.float64)).numpy()
        frame = output / f"frame_{index:03d}.npz"
        np.savez_compressed(frame, row=terrain.active_rows, col=terrain.active_cols,
                            x=xy[:, 0], y=xy[:, 1], h=result[:, 0], u=result[:, 1], v=result[:, 2])
        summary["frames"].append({"time_s": float(t), "file": frame.name,
                                  "max_depth_m": float(result[:, 0].max()),
                                  "mean_depth_m": float(result[:, 0].mean()),
                                  "max_speed_m_s": float(np.linalg.norm(result[:, 1:], axis=1).max())})
        raster = np.full((terrain.height, terrain.width), np.nan, dtype=np.float32)
        raster[terrain.active_rows, terrain.active_cols] = result[:, 0]
        import rasterio
        tif = output / f"depth_{index:03d}.tif"
        with rasterio.open(tif, "w", driver="GTiff", width=terrain.width, height=terrain.height,
                           count=1, dtype="float32", crs=terrain.crs, transform=terrain.transform,
                           nodata=np.nan, compress="deflate") as dst:
            dst.write(raster, 1)
    (output / "prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
