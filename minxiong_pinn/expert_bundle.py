"""Strict loading and routing for an optional approved rainfall-expert bundle.

The existing single checkpoint remains the production fallback.  Merely
placing experimental model files in the repository never activates them: the
manifest must explicitly record individual physics and overlap acceptance.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .predict import load_model


SCHEMA_VERSION = 1
SUPPORTED_MODEL_IDS = ("low", "medium", "high")


class ExpertBundle(nn.Module):
    """Route each condition to one independently trained rainfall expert."""

    def __init__(self, experts: dict[str, nn.Module], manifest: dict):
        super().__init__()
        self.experts = nn.ModuleDict(experts)
        self.model_ids = tuple(experts)
        self.bundle_id = manifest["bundle_id"]
        self.low_max_mm_hr = float(manifest["routing"]["low_max_mm_hr"])
        medium_max = manifest["routing"].get("medium_max_mm_hr")
        self.medium_max_mm_hr = (None if medium_max is None else float(medium_max))
        self.rainfall_domain = tuple(
            float(value) for value in manifest["training_rain_domain"]["rainfall_mm_hr"])
        first = experts[self.model_ids[0]]
        self.scales = dict(first.scales)
        self.scales["time_s"] = float(manifest["time_s"])

    def model_id_for_rainfall(self, rainfall_mm_hr: float) -> str:
        rain = float(rainfall_mm_hr)
        if (not math.isfinite(rain) or
                not self.rainfall_domain[0] <= rain <= self.rainfall_domain[1]):
            raise ValueError("rainfall intensity is outside the expert bundle range")
        if rain <= self.low_max_mm_hr:
            return "low"
        if len(self.model_ids) == 2:
            return "medium"
        if rain <= self.medium_max_mm_hr:
            return "medium"
        return "high"

    def forward(self, points):
        if points.ndim != 2 or points.shape[1] != 5:
            raise ValueError("expert bundle expects points shaped (N, 5)")
        rain = points[:, 3] * 3_600_000.0
        if torch.any(~torch.isfinite(rain)):
            raise ValueError("rainfall intensity must be finite")
        lo, hi = self.rainfall_domain
        # Account for float32 round-trips at exact decimal mm/hr endpoints,
        # without accepting a materially out-of-range user value.
        dtype_epsilon = torch.finfo(rain.dtype).eps if rain.dtype.is_floating_point else 0.
        tolerance = max(1e-6, abs(hi) * 1e-6, abs(hi) * float(dtype_epsilon) * 8)
        if torch.any(rain < lo - tolerance) or torch.any(rain > hi + tolerance):
            raise ValueError("rainfall intensity is outside the expert bundle range")
        rain = rain.clamp(min=lo, max=hi)
        masks = {"low": rain <= self.low_max_mm_hr + tolerance}
        if len(self.model_ids) == 2:
            masks["medium"] = rain > self.low_max_mm_hr + tolerance
        else:
            masks.update({
                "medium": ((rain > self.low_max_mm_hr + tolerance) &
                           (rain <= self.medium_max_mm_hr + tolerance)),
                "high": rain > self.medium_max_mm_hr + tolerance,
            })
        result = None
        for model_id in self.model_ids:
            mask = masks[model_id]
            if not torch.any(mask):
                continue
            values = self.experts[model_id](points[mask])
            if values.ndim != 2 or values.shape != (int(mask.sum()), 3):
                raise ValueError(f"expert {model_id!r} returned an invalid prediction shape")
            if result is None:
                result = values.new_empty((len(points), 3))
            result[mask] = values
        return points.new_empty((0, 3)) if result is None else result


def _pair(value, label, *, nonnegative=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expert bundle {label} must contain two values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result) or result[0] > result[1]:
        raise ValueError(f"expert bundle {label} is invalid")
    if (result[0] < 0 if nonnegative else result[0] <= 0):
        raise ValueError(f"expert bundle {label} has an invalid lower bound")
    return result


def _manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read expert bundle manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("expert bundle manifest must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported expert bundle schema: {value.get('schema_version')!r}")
    for field in ("activation_approved", "physics_training_passed",
                  "overlap_consistency_passed",
                  "continuous_domain_audit_passed"):
        if value.get(field) is not True:
            raise ValueError(f"expert bundle is not approved: {field} is not true")
    if not isinstance(value.get("bundle_id"), str) or not value["bundle_id"].strip():
        raise ValueError("expert bundle_id must be a nonempty string")
    if not isinstance(value.get("terrain_fingerprint"), str) or not value["terrain_fingerprint"]:
        raise ValueError("expert bundle terrain_fingerprint is missing")
    time_s = float(value.get("time_s", 0))
    if not math.isfinite(time_s) or time_s <= 0:
        raise ValueError("expert bundle time_s must be positive and finite")
    domain = value.get("training_rain_domain")
    if not isinstance(domain, dict):
        raise ValueError("expert bundle training_rain_domain is missing")
    rain_domain = _pair(domain.get("rainfall_mm_hr"), "rainfall range", nonnegative=True)
    duration_domain = _pair(domain.get("duration_s"), "duration range")
    routing = value.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("expert bundle routing is missing")
    low_max = float(routing.get("low_max_mm_hr", math.nan))
    experts = value.get("experts")
    if not isinstance(experts, list) or len(experts) not in (2, 3):
        raise ValueError("expert bundle must contain two or three experts")
    ids = [item.get("model_id") for item in experts if isinstance(item, dict)]
    expected_ids = SUPPORTED_MODEL_IDS[:len(experts)]
    if tuple(ids) != expected_ids:
        raise ValueError(
            "expert bundle models must be ordered low, medium[, high]")
    if len(experts) == 2:
        if not rain_domain[0] <= low_max < rain_domain[1]:
            raise ValueError("expert bundle routing threshold does not cover the rainfall domain")
        normalized_routing = {"low_max_mm_hr": low_max}
    else:
        medium_max = float(routing.get("medium_max_mm_hr", math.nan))
        if not rain_domain[0] <= low_max < medium_max < rain_domain[1]:
            raise ValueError("expert bundle routing thresholds do not cover the rainfall domain")
        normalized_routing = {
            "low_max_mm_hr": low_max, "medium_max_mm_hr": medium_max}
    value["time_s"] = time_s
    value["training_rain_domain"] = {
        "rainfall_mm_hr": list(rain_domain), "duration_s": list(duration_domain)}
    value["routing"] = normalized_routing
    return value


def _contained_bundle_file(manifest_path: Path, relative_path, label) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"expert bundle {label} path must be a nonempty relative path")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"expert bundle {label} path must be relative to the bundle")
    bundle_root = manifest_path.parent.resolve()
    target = (bundle_root / relative).resolve()
    if target == bundle_root or bundle_root not in target.parents:
        raise ValueError(f"expert bundle {label} path escapes the bundle directory")
    return target


def _contained_checkpoint(manifest_path: Path, relative_path) -> Path:
    return _contained_bundle_file(manifest_path, relative_path, "checkpoint")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _audit_evidence(manifest_path, manifest):
    audit_path = _contained_bundle_file(
        manifest_path, manifest.get("audit_file"), "audit file")
    expected_audit_hash = manifest.get("audit_sha256")
    if not _valid_sha256(expected_audit_hash):
        raise ValueError("expert bundle audit file sha256 is invalid")
    try:
        actual_audit_hash = _sha256(audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read expert bundle audit evidence: {exc}") from exc
    if not hmac.compare_digest(actual_audit_hash, expected_audit_hash):
        raise ValueError("expert bundle audit file sha256 mismatch")
    if not isinstance(audit, dict):
        raise ValueError("expert bundle audit evidence must be a JSON object")
    for field in ("activation_approved", "physics_training_passed",
                  "overlap_consistency_passed", "continuous_domain_passed"):
        if audit.get(field) is not True:
            raise ValueError(
                f"expert bundle audit evidence is not approved: {field} is not true")
    if audit.get("terrain_fingerprint") != manifest["terrain_fingerprint"]:
        raise ValueError("expert bundle audit terrain fingerprint differs from the manifest")
    overlap = audit.get("overlap")
    continuous = audit.get("continuous_domain")
    if not isinstance(overlap, dict) or overlap.get("passed") is not True:
        raise ValueError("expert bundle overlap audit evidence did not pass")
    if not isinstance(continuous, dict) or continuous.get("passed") is not True:
        raise ValueError("expert bundle continuous-domain audit evidence did not pass")
    audit_experts = audit.get("experts")
    if not isinstance(audit_experts, list):
        raise ValueError("expert bundle audit expert evidence is missing")
    if len(audit_experts) != len(manifest["experts"]):
        raise ValueError("expert bundle audit expert evidence differs from the manifest")
    for audited, declared in zip(audit_experts, manifest["experts"]):
        if (not isinstance(audited, dict) or
                audited.get("model_id") != declared["model_id"] or
                audited.get("physics_training_passed") is not True or
                audited.get("checkpoint_sha256") != declared.get("sha256")):
            raise ValueError(
                "expert bundle audit expert evidence differs from the manifest")
    return audit


def load_expert_bundle(
    manifest_path, terrain, *,
    model_loader: Callable[[Path, object], tuple[nn.Module, dict]] = load_model,
):
    """Load and fully validate an approved expert bundle."""
    path = Path(manifest_path)
    manifest = _manifest(path)
    if manifest["terrain_fingerprint"] != terrain.fingerprint:
        raise ValueError("expert bundle terrain fingerprint does not match this DEM/Manning setting")
    _audit_evidence(path, manifest)
    loaded: dict[str, nn.Module] = {}
    checkpoint_domains = {}
    duration_domain = tuple(manifest["training_rain_domain"]["duration_s"])
    for entry in manifest["experts"]:
        model_id = entry["model_id"]
        checkpoint_path = _contained_checkpoint(path, entry.get("path"))
        expected_hash = entry.get("sha256")
        if not _valid_sha256(expected_hash):
            raise ValueError(f"expert {model_id!r} checkpoint sha256 is invalid")
        try:
            actual_hash = _sha256(checkpoint_path)
        except OSError as exc:
            raise ValueError(
                f"cannot read expert {model_id!r} checkpoint for sha256: {exc}") from exc
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(f"expert {model_id!r} checkpoint sha256 mismatch")
        model, checkpoint = model_loader(checkpoint_path, terrain)
        if checkpoint.get("terrain_fingerprint") != manifest["terrain_fingerprint"]:
            raise ValueError(f"expert {model_id!r} terrain fingerprint differs from the manifest")
        domain = checkpoint.get("training_rain_domain", {})
        trained_rain = _pair(domain.get("rainfall_mm_hr"),
                             f"{model_id} rainfall range", nonnegative=True)
        trained_duration = _pair(domain.get("duration_s"), f"{model_id} duration range")
        declared_rain = _pair(entry.get("trained_rainfall_mm_hr"),
                              f"{model_id} declared rainfall range", nonnegative=True)
        if trained_rain != declared_rain:
            raise ValueError(f"expert {model_id!r} rainfall range differs from the manifest")
        if trained_duration[0] > duration_domain[0] or trained_duration[1] < duration_domain[1]:
            raise ValueError(f"expert {model_id!r} does not cover the bundle duration range")
        if not math.isclose(float(model.scales["time_s"]), manifest["time_s"],
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"expert {model_id!r} time horizon differs from the manifest")
        loaded[model_id] = model
        checkpoint_domains[model_id] = trained_rain

    model_ids = tuple(entry["model_id"] for entry in manifest["experts"])
    low_max = manifest["routing"]["low_max_mm_hr"]
    rain_lo, rain_hi = manifest["training_rain_domain"]["rainfall_mm_hr"]
    if len(model_ids) == 2:
        required_primary = {
            "low": (rain_lo, low_max), "medium": (low_max, rain_hi)}
    else:
        medium_max = manifest["routing"]["medium_max_mm_hr"]
        required_primary = {
            "low": (rain_lo, low_max),
            "medium": (low_max, medium_max),
            "high": (medium_max, rain_hi),
        }
    for model_id, required in required_primary.items():
        trained = checkpoint_domains[model_id]
        if trained[0] > required[0] or trained[1] < required[1]:
            raise ValueError(f"expert {model_id!r} does not cover its primary routing range")

    model = ExpertBundle(loaded, manifest)
    model.eval()
    checkpoint = {
        "expert_bundle": True,
        "bundle_id": manifest["bundle_id"],
        "terrain_fingerprint": manifest["terrain_fingerprint"],
        "training_rain_domain": manifest["training_rain_domain"],
        "routing": manifest["routing"],
        "expert_domains": {key: list(value) for key, value in checkpoint_domains.items()},
        "activation_approved": True,
        "physics_training_passed": True,
        "overlap_consistency_passed": True,
        "continuous_domain_audit_passed": True,
    }
    return model, checkpoint


def load_preferred_model(bundle_manifest, fallback_model, terrain):
    """Prefer an approved bundle; otherwise retain the official checkpoint."""
    bundle_path = Path(bundle_manifest)
    if bundle_path.is_file():
        try:
            return load_expert_bundle(bundle_path, terrain)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            model, checkpoint = load_model(fallback_model, terrain)
            checkpoint = dict(checkpoint)
            checkpoint["bundle_fallback_reason"] = str(exc)
            return model, checkpoint
    return load_model(fallback_model, terrain)
