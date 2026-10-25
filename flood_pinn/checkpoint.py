"""State-dict-only persistence, fixed-condition identity, explicit validity evidence."""
import hashlib
import json
from pathlib import Path
import torch

from .contracts import (SCHEMA_VERSION, PHYSICS_VERSION, PredictionError,
                        canonical_hash, write_json)
from .model import ParametricPINN


def _validate_domain(domain, report, config=None):
    if not domain:
        return
    if not (report.get("reference_validation_status") == "passed"
            and report.get("held_out_scenario_status") == "passed"
            and report.get("physics_checks_passed") is True):
        raise ValueError("A validated domain requires passed physics, reference and held-out reports")
    if domain.get("mode") not in ("discrete", "regions"):
        raise ValueError("Unknown validated-domain mode")
    if not domain.get("evidence_id"):
        raise ValueError("Validated domain needs a traceable evidence_id")
    import math
    entries = domain.get("scenarios" if domain["mode"] == "discrete" else "regions", [])
    if not entries:
        raise ValueError("Validated domain cannot be empty")
    for entry in entries:
        keys = ("rainfall_mm_hr", "duration_s") if domain["mode"] == "discrete" else (
            "rainfall_min_mm_hr", "rainfall_max_mm_hr", "duration_min_s", "duration_max_s")
        if any(not isinstance(entry.get(k), (int, float)) or isinstance(entry[k], bool)
               or not math.isfinite(entry[k]) or entry[k] < 0 for k in keys):
            raise ValueError("Invalid validated-domain bounds")
        if domain["mode"] == "regions" and (
                entry[keys[0]] > entry[keys[1]] or entry[keys[2]] > entry[keys[3]]):
            raise ValueError("Reversed validated-domain bounds")
        if config is not None:
            cases = [s for s in config["scenarios"] if s["split"] == "train"]
            rain_lo, rain_hi = min(s["rainfall_mm_hr"] for s in cases), max(s["rainfall_mm_hr"] for s in cases)
            dur_lo, dur_hi = min(s["duration_s"] for s in cases), max(s["duration_s"] for s in cases)
            rain_values = [entry[k] for k in keys if k.startswith("rainfall")]
            dur_values = [entry[k] for k in keys if k.startswith("duration")]
            if not all(rain_lo <= v <= rain_hi for v in rain_values) or not all(
                    dur_lo <= v <= min(dur_hi, config["time"]["end_s"]) for v in dur_values):
                raise ValueError("Validated domain must lie within training and time support")


def save_checkpoint(model, terrain, config, out_dir, report, validated_domain=None):
    """Save research models too, but never promote them automatically to validated."""
    _validate_domain(validated_domain, report, config)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "weights.pt"
    temporary = folder / "weights.pt.tmp"
    for tensor in model.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("Refusing nonfinite model weights")
    torch.save(model.state_dict(), temporary)
    temporary.replace(weights)
    architecture = {"width": model.width, "layers": model.layers,
                    "parameterized": model.parameterized}
    identity = {"terrain_fingerprint": terrain.fingerprint,
                "conditions": config["conditions"], "scales": model.scales,
                "architecture": architecture, "physics_version": PHYSICS_VERSION}
    metadata = {
        "schema_version": SCHEMA_VERSION, "physics_version": PHYSICS_VERSION,
        "model_id": canonical_hash(identity)[:16] + "-" + hashlib.sha256(weights.read_bytes()).hexdigest()[:12],
        "terrain_fingerprint": terrain.fingerprint,
        "condition_fingerprint": canonical_hash(config["conditions"]),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "architecture": architecture, "scales": model.scales,
        "config": config, "training_report": report,
        "validated_domain": validated_domain,
        "torch_version": str(torch.__version__),
    }
    metadata["metadata_sha256"] = canonical_hash(metadata)
    write_json(folder / "model_metadata.json", metadata)
    return metadata


def load_checkpoint(folder, terrain, *, conditions=None, device="cpu"):
    folder = Path(folder)
    try:
        metadata = json.loads((folder / "model_metadata.json").read_text())
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON object")
        integrity = {k: v for k, v in metadata.items() if k != "metadata_sha256"}
        if canonical_hash(integrity) != metadata.get("metadata_sha256"):
            raise ValueError("Checkpoint metadata checksum mismatch")
        if metadata["schema_version"] != SCHEMA_VERSION or metadata["physics_version"] != PHYSICS_VERSION:
            raise ValueError("Unsupported checkpoint schema/physics")
        if metadata["terrain_fingerprint"] != terrain.fingerprint:
            raise PredictionError("TERRAIN_VERSION_MISMATCH", "Terrain or roughness has changed")
        expected_conditions = metadata["config"]["conditions"] if conditions is None else conditions
        if canonical_hash(expected_conditions) != metadata["condition_fingerprint"]:
            raise PredictionError("CONDITION_VERSION_MISMATCH", "Initial/boundary conditions have changed")
        if any(metadata["scales"][k] != metadata["config"]["scales"][k] for k in metadata["scales"]):
            raise ValueError("Saved scales disagree with training configuration")
        if any(metadata["config"].get("model", {}).get(k, v) != v for k, v in metadata["architecture"].items()):
            raise ValueError("Saved network disagrees with training configuration")
        _validate_domain(metadata.get("validated_domain"), metadata["training_report"], metadata["config"])
        weights = folder / "weights.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["weights_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        model = ParametricPINN(metadata["scales"], **metadata["architecture"]).to(device)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite weights")
        model.eval()
        return model, metadata
    except PredictionError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise PredictionError("MODEL_INCOMPATIBLE", str(exc)) from exc
