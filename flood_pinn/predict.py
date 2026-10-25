"""Independent, bounded, streaming inference. This module never imports training."""
from pathlib import Path
from time import perf_counter
import math
import uuid
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .contracts import PredictionError, write_json
from .rainfall import Rainfall, output_times


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PredictionError("INVALID_ARGUMENT", f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PredictionError("INVALID_ARGUMENT", f"{name} must be finite and nonnegative")
    return result


def _within_domain(domain, rain, duration):
    if not domain:
        return False
    if domain["mode"] == "discrete":
        return any(math.isclose(rain, s["rainfall_mm_hr"], abs_tol=1e-10, rel_tol=1e-12)
                   and math.isclose(duration, s["duration_s"], abs_tol=1e-10, rel_tol=1e-12)
                   for s in domain["scenarios"])
    return any(r["rainfall_min_mm_hr"] <= rain <= r["rainfall_max_mm_hr"]
               and r["duration_min_s"] <= duration <= r["duration_max_s"]
               for r in domain["regions"])


class PredictionResult:
    """Consume frames_iterator, then finalize; summary is pending until completion."""
    def __init__(self, predictor, rainfall, times, coverage, output_dir, batch_size, validated):
        self.predictor, self.rainfall, self.times = predictor, rainfall, times
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.batch_size, self.coverage = batch_size, coverage
        self.run_id = uuid.uuid4().hex
        self.model_id = predictor.metadata["model_id"]
        self.grid_id = predictor.terrain.metadata["grid_id"]
        self.validation_status = "validated" if validated else "research_unvalidated"
        self.status = "pending"
        self.summary = {"status": "pending"}
        self.artifacts = {}
        self.actual_output_times_s = list(times)
        self._started = perf_counter()
        self._max_h = self._max_speed = self._max_area = 0.
        self._max_h_t = self._max_speed_t = self._max_area_t = float(times[0])
        self.frames_iterator = self._frames()

    def _frames(self):
        grid = self.predictor.terrain
        try:
            for frame_id, t in enumerate(self.times):
                raw = self.predictor.evaluate(self._points(t), batch_size=self.batch_size)
                h, ur, vr = raw.T
                dry = h <= self.predictor.dry_depth
                u, v = np.where(dry, 0., ur), np.where(dry, 0., vr)
                speed = np.hypot(u, v)
                direction = np.full(h.shape, np.nan)
                moving = (~dry) & (speed > 1e-10)
                direction[moving] = np.degrees(np.arctan2(v[moving], u[moving])) % 360
                flooded = h >= self.predictor.flood_depth
                if h.max() > self._max_h:
                    self._max_h, self._max_h_t = float(h.max()), float(t)
                if speed.max() > self._max_speed:
                    self._max_speed, self._max_speed_t = float(speed.max()), float(t)
                area = float(flooded.sum()*grid.dx*grid.dy)
                if area > self._max_area:
                    self._max_area, self._max_area_t = area, float(t)
                # Python lists with None, not NaN, so API callers can JSON-encode safely.
                yield {"frame_id": frame_id, "t": float(t), "cell_id": list(range(len(h))),
                       "x": grid.x.tolist(), "y": grid.y.tolist(), "z": grid.z.ravel().tolist(),
                       "h": h.tolist(), "water_surface": (grid.z.ravel()+h).tolist(),
                       "u": u.tolist(), "v": v.tolist(), "speed": speed.tolist(),
                       "direction": [None if np.isnan(d) else float(d) for d in direction],
                       "flooded": flooded.astype(int).tolist(), "is_dry": dry.astype(int).tolist(),
                       "u_raw": ur.tolist(), "v_raw": vr.tolist()}
            self.status = "frames_complete"
        except Exception as exc:
            self.status = "failed"
            self.summary = {"status": "failed", "error": str(exc)}
            if self.output_dir is not None:
                write_json(self.output_dir / "run_status.json", self.summary)
            raise

    def _points(self, t):
        grid = self.predictor.terrain
        n = len(grid.x)
        return np.column_stack((grid.x, grid.y, np.full(n, t),
                                np.full(n, self.rainfall.intensity_mm_hr/3600000),
                                np.full(n, self.rainfall.duration_s)))

    def finalize(self):
        if self.status == "complete" or self.status == "failed":
            return self.summary
        if self.status != "frames_complete":
            return {"status": "pending", "message": "Consume frames_iterator before finalizing"}
        metadata = {"run_id": self.run_id, "model_id": self.model_id,
                    "end_s": self.predictor.metadata["config"]["time"]["end_s"],
                    "coverage": self.coverage, "validation_status": self.validation_status,
                    "reference_validation_status": self.predictor.metadata["training_report"].get(
                        "reference_validation_status", "not_completed")}
        try:
            if self.output_dir is not None:
                from .export import export_simulation
                summary = export_simulation(
                    self.predictor.terrain, self.predictor.evaluate, self.times, self.rainfall,
                    self.output_dir, run_metadata=metadata,
                    dry_depth=self.predictor.dry_depth, flood_depth=self.predictor.flood_depth,
                    batch_size=self.batch_size)
                self.artifacts = {p.name: str(p) for p in sorted(self.output_dir.iterdir()) if p.is_file()}
            else:
                summary = {**metadata, "max_depth_m": self._max_h, "max_depth_time_s": self._max_h_t,
                           "max_speed_m_s": self._max_speed, "max_speed_time_s": self._max_speed_t,
                           "max_flooded_area_m2": self._max_area, "max_flooded_area_time_s": self._max_area_t,
                           "water_balance_status": "not_evaluated_without_export"}
            summary.update(status="complete", inference_and_export_elapsed_s=perf_counter()-self._started)
            self.summary, self.status = summary, "complete"
            if self.output_dir is not None:
                write_json(self.output_dir / "simulation_summary.json", self.summary)
                write_json(self.output_dir / "run_status.json", {"status": self.status})
                from .export import refresh_manifest
                refresh_manifest(self.output_dir)
            return summary
        except Exception as exc:
            self.status = "failed"
            self.summary = {**metadata, "status": "failed", "error": str(exc)}
            if self.output_dir is not None:
                write_json(self.output_dir / "run_status.json", self.summary)
            raise


class Predictor:
    def __init__(self, model, metadata, terrain):
        self.model, self.metadata, self.terrain = model, metadata, terrain
        thresholds = metadata["config"].get("thresholds", {})
        self.dry_depth = float(thresholds.get("dry_depth_m", 1e-4))
        self.flood_depth = float(thresholds.get("flood_depth_m", .01))
        if not 0 <= self.dry_depth < self.flood_depth or not math.isfinite(self.flood_depth):
            raise PredictionError("MODEL_INCOMPATIBLE", "Invalid dry/flood thresholds")

    @classmethod
    def from_checkpoint(cls, model_dir, terrain, *, conditions=None, device="cpu"):
        model, metadata = load_checkpoint(model_dir, terrain, conditions=conditions, device=device)
        try:
            return cls(model, metadata, terrain)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise PredictionError("MODEL_INCOMPATIBLE", str(exc)) from exc

    def evaluate(self, points, *, batch_size=4096):
        """Low-level research/face quadrature evaluator; API clients must use predict.

        Caller supplies already-approved physical points. This is not a public
        production scenario entry point and deliberately does not infer approval.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 5 or not np.isfinite(points).all():
            raise PredictionError("INVALID_ARGUMENT", "Expected finite Nx5 physical points")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise PredictionError("INVALID_ARGUMENT", "batch_size must be a positive integer")
        parts = []
        parameter = next(self.model.parameters())
        with torch.inference_mode():
            for start in range(0, len(points), batch_size):
                batch = torch.as_tensor(points[start:start+batch_size], dtype=torch.float64,
                                        device=parameter.device)
                output = self.model(batch).detach().cpu().numpy().astype(float)
                if output.shape != (len(batch), 3) or not np.isfinite(output).all() or (output[:,0] < 0).any():
                    raise PredictionError("NONFINITE_PREDICTION", "Nonfinite or negative raw prediction")
                parts.append(output)
        return np.concatenate(parts) if parts else np.empty((0, 3))

    def predict(self, rainfall_mm_hr, duration_min, output_times_s, *, output_mode="complete",
                batch_size=4096, output_dir=None, allow_unvalidated=False):
        if not isinstance(allow_unvalidated, bool):
            raise PredictionError("INVALID_ARGUMENT", "allow_unvalidated must be boolean")
        rain = _number(rainfall_mm_hr, "rainfall_mm_hr")
        duration = 60*_number(duration_min, "duration_min")
        end = float(self.metadata["config"]["time"]["end_s"])
        if duration > end:
            raise PredictionError("TIME_OUT_OF_RANGE", "Rain duration exceeds the trained time domain")
        if not self.metadata["architecture"]["parameterized"]:
            train = [s for s in self.metadata["config"]["scenarios"] if s["split"] == "train"]
            if len(train) != 1 or not (math.isclose(rain, train[0]["rainfall_mm_hr"])
                                      and math.isclose(duration, train[0]["duration_s"])):
                raise PredictionError("UNVALIDATED_SCENARIO", "Single-scenario model cannot change rainfall")
        validated = _within_domain(self.metadata.get("validated_domain"), rain, duration)
        if not validated and not allow_unvalidated:
            raise PredictionError("UNVALIDATED_SCENARIO", "This rainfall has no passed validation evidence")
        # Research override permits unvalidated interpolation, never unchecked extrapolation.
        scenarios = self.metadata["config"]["scenarios"]
        training = [s for s in scenarios if s["split"] == "train"]
        if not validated and not (min(s["rainfall_mm_hr"] for s in training) <= rain <= max(s["rainfall_mm_hr"] for s in training)
                                  and min(s["duration_s"] for s in training) <= duration <= max(s["duration_s"] for s in training)):
            raise PredictionError("UNVALIDATED_SCENARIO", "Research inference is also restricted to training bounds")
        try:
            requested = [_number(t, "output time") for t in output_times_s]
            if not requested or max(requested) > end:
                raise PredictionError("TIME_OUT_OF_RANGE", "Output times must lie inside the trained time domain")
            times = output_times(end, self.metadata["config"]["time"]["output_interval_s"],
                                 duration, requested=requested, mode=output_mode)
        except PredictionError:
            raise
        except (ValueError, TypeError) as exc:
            raise PredictionError("INVALID_ARGUMENT", str(exc)) from exc
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise PredictionError("INVALID_ARGUMENT", "batch_size must be a positive integer")
        if output_dir is not None:
            folder = Path(output_dir)
            try:
                if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
                    raise PredictionError("OUTPUT_IO_ERROR", "Output directory must be empty to preserve previous runs")
            except OSError as exc:
                raise PredictionError("OUTPUT_IO_ERROR", str(exc)) from exc
        return PredictionResult(self, Rainfall(rain, duration), times,
                                "complete" if output_mode == "complete" else "partial",
                                output_dir, batch_size, validated)
