"""Local HTTP bridge between Unreal Engine and the Minxiong predictor.

The server deliberately uses only Python's standard library.  Unreal can call
it with the built-in HTTP module (or an HTTP Blueprint plugin), while the
existing Python prediction pipeline remains the single source of validation
and model inference.
"""
from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import uuid

from .expert_bundle import load_preferred_model
from .predict import predict
from .terrain import MaskedRasterTerrain
from .user_input import build_prediction_request


MAX_REQUEST_BYTES = 64 * 1024
OUTPUT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class PredictionBusyError(RuntimeError):
    """Raised when a prediction is already using the shared loaded model."""


def _finite_number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _safe_output_name(value):
    if value in (None, ""):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"unreal_{stamp}_{uuid.uuid4().hex[:8]}"
    if not isinstance(value, str) or not OUTPUT_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "output_name must be 1-64 ASCII letters, digits, dot, dash, or underscore")
    if value in {".", ".."} or ".." in value:
        raise ValueError("output_name must not contain '..'")
    return value


class UnrealFloodService:
    """Validate Unreal inputs and run one prediction at a time."""

    def __init__(self, terrain, model, checkpoint, output_root, *,
                 predictor=predict, max_frames=500):
        self.terrain = terrain
        self.model = model
        self.checkpoint = dict(checkpoint)
        self.output_root = Path(output_root).resolve()
        self.predictor = predictor
        self.max_frames = int(max_frames)
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self._prediction_lock = threading.Lock()

    @classmethod
    def from_paths(cls, *, terrain_path, fallback_model_path,
                   expert_manifest_path, output_root, manning_n=.03,
                   predictor=predict, max_frames=500):
        terrain = MaskedRasterTerrain(terrain_path, manning_n=manning_n)
        model, checkpoint = load_preferred_model(
            expert_manifest_path, fallback_model_path, terrain)
        return cls(terrain, model, checkpoint, output_root,
                   predictor=predictor, max_frames=max_frames)

    def _domain(self):
        domain = self.checkpoint.get("training_rain_domain", {})
        rain = domain.get("rainfall_mm_hr", (None, None))
        duration = domain.get("duration_s", (None, None))
        return {
            "rainfall_mm_hr": list(rain),
            "duration_hours": [float(duration[0]) / 3600.,
                               float(duration[1]) / 3600.],
            "simulation_hours_max": float(self.model.scales["time_s"]) / 3600.,
            "max_frames": self.max_frames,
        }

    def health(self):
        fallback_reason = self.checkpoint.get("bundle_fallback_reason")
        return {
            "status": "ready",
            "service": "minxiong-flood-pinn-unreal",
            "expert_bundle_active": bool(self.checkpoint.get("expert_bundle")),
            "bundle_id": self.checkpoint.get("bundle_id"),
            "fallback_active": not bool(self.checkpoint.get("expert_bundle")),
            "fallback_reason": fallback_reason,
            "active_model_domain": self._domain(),
            "terrain_active_cells": int(self.terrain.active_count),
        }

    def schema(self):
        return {
            "endpoint": "/predict",
            "method": "POST",
            "content_type": "application/json",
            "active_model_domain": self._domain(),
            "inputs": {
                "rainfall_mm_hr": {"type": "number", "minimum": 0},
                "duration_hr": {"type": "number", "exclusive_minimum": 0},
                "simulation_hours": {"type": "number", "exclusive_minimum": 0},
                "interval_minutes": {"type": "number", "exclusive_minimum": 0},
                "output_name": {
                    "type": "string", "required": False,
                    "pattern": "ASCII letters/digits plus ._-; no path separators",
                },
            },
            "outputs": {
                "csv": "CSV files written directly by the predictor",
                "geotiff": "water-depth GeoTIFF files",
                "npz": "lossless NumPy frame files retained for compatibility",
            },
            "notes": [
                "The active checkpoint range is authoritative.",
                "An unapproved expert bundle never expands the fallback model range.",
                "The call is synchronous; Unreal should send it from a worker/asynchronous HTTP task.",
            ],
        }

    def _request(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        required = ("rainfall_mm_hr", "duration_hr", "simulation_hours",
                    "interval_minutes")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError("missing required fields: " + ", ".join(missing))
        rainfall = _finite_number(payload["rainfall_mm_hr"], "rainfall_mm_hr")
        duration = _finite_number(payload["duration_hr"], "duration_hr")
        simulation = _finite_number(payload["simulation_hours"], "simulation_hours")
        interval = _finite_number(payload["interval_minutes"], "interval_minutes")
        request = build_prediction_request(
            rainfall_mm_hr=rainfall,
            rain_duration_value=duration, rain_duration_unit="小时",
            simulation_value=simulation, simulation_unit="小时",
            interval_value=interval, interval_unit="分钟",
            checkpoint=self.checkpoint,
            model_time_s=float(self.model.scales["time_s"]),
            active_cells=int(self.terrain.active_count),
            max_frames=self.max_frames,
        )
        return request, _safe_output_name(payload.get("output_name"))

    def run_prediction(self, payload):
        request, output_name = self._request(payload)
        if not self._prediction_lock.acquire(blocking=False):
            raise PredictionBusyError("another prediction is already running")
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            target = (self.output_root / output_name).resolve()
            if target.parent != self.output_root:
                raise ValueError("output_name escapes the configured output directory")
            if target.exists():
                raise FileExistsError(f"output directory already exists: {output_name}")
            summary = self.predictor(
                self.model, self.checkpoint, self.terrain,
                request.rainfall_mm_hr, request.rain_duration_s,
                request.output_times_s, target)
            csv_files = sorted(str(path.resolve()) for path in target.glob("*.csv"))
            tif_files = sorted(str(path.resolve()) for path in target.glob("*.tif"))
            npz_files = sorted(str(path.resolve()) for path in target.glob("*.npz"))
            return {
                "status": "complete",
                "request": {
                    "rainfall_mm_hr": request.rainfall_mm_hr,
                    "duration_hr": request.rain_duration_s / 3600.,
                    "simulation_hours": request.simulation_s / 3600.,
                    "interval_minutes": request.interval_s / 60.,
                    "frame_count": request.frame_count,
                    "output_times_s": list(request.output_times_s),
                },
                "active_model": {
                    "expert_bundle": bool(self.checkpoint.get("expert_bundle")),
                    "bundle_id": self.checkpoint.get("bundle_id"),
                    "model_id": summary.get("model_id", "single"),
                },
                "output_directory": str(target),
                "files": {"csv": csv_files, "geotiff": tif_files,
                          "npz": npz_files},
                "summary": summary,
            }
        finally:
            self._prediction_lock.release()


def _handler_class(service):
    class UnrealRequestHandler(BaseHTTPRequestHandler):
        server_version = "MinxiongFloodPINN/1"
        protocol_version = "HTTP/1.1"

        def _send_json(self, status, value):
            body = json.dumps(
                value, ensure_ascii=False, allow_nan=False,
                separators=(",", ":")).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, code, message):
            self._send_json(status, {
                "status": "error", "error": code, "message": str(message)})

        def do_OPTIONS(self):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, service.health())
            elif self.path == "/schema":
                self._send_json(HTTPStatus.OK, service.schema())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")

        def do_POST(self):
            if self.path != "/predict":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type",
                            "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length < 0:
                self._error(HTTPStatus.LENGTH_REQUIRED, "content_length",
                            "a valid Content-Length header is required")
                return
            if length > MAX_REQUEST_BYTES:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large",
                            f"request body exceeds {MAX_REQUEST_BYTES} bytes")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = service.run_prediction(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_json", exc)
            except PredictionBusyError as exc:
                self._error(HTTPStatus.CONFLICT, "prediction_busy", exc)
            except FileExistsError as exc:
                self._error(HTTPStatus.CONFLICT, "output_exists", exc)
            except (KeyError, TypeError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", exc)
            except Exception as exc:  # keep internal failures structured for Unreal
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            "prediction_failed", f"{type(exc).__name__}: {exc}")
            else:
                self._send_json(HTTPStatus.OK, result)

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            # Avoid dumping Unreal polling traffic; callers may subclass if
            # access logging is wanted.
            return

    return UnrealRequestHandler


def make_server(service, host="127.0.0.1", port=8765):
    """Return a configured threaded HTTP server (port 0 selects a test port)."""
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a nonempty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    return ThreadingHTTPServer((host, port), _handler_class(service))

