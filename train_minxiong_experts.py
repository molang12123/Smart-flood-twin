from __future__ import annotations

import copy
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from minxiong_pinn.rainfall import add_design_events, extract_events
from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import _training_identity, train


ROOT = Path(__file__).resolve().parent
EXPERT_OPERATIONAL_OVERRIDE_KEYS = frozenset({
    "adam_steps", "max_seconds", "checkpoint_interval",
})


def _apply_expert_operational_overrides(cfg, expert):
    """Apply only resume-safe, per-expert run limits.

    These keys are deliberately excluded from ``_training_identity``.  Model,
    loss, sampling, seed, and physics settings cannot be changed here, so an
    expert may be extended without silently changing its optimization problem.
    """
    overrides = expert.get("training_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("expert training_overrides must be an object")
    unknown = set(overrides) - EXPERT_OPERATIONAL_OVERRIDE_KEYS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(
            f"expert training_overrides contains non-operational keys: {names}")
    result = copy.deepcopy(cfg)
    result.update(overrides)
    return result


def _artifact_values_equal(left, right):
    """Compare nested checkpoint values without tensor truth ambiguity."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and
                left.dtype == right.dtype and left.shape == right.shape and
                torch.equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping) and
                left.keys() == right.keys() and
                all(_artifact_values_equal(left[key], right[key])
                    for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_artifact_values_equal(a, b)
                    for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _load_completed_expert_report(terrain, events, cfg, folder):
    """Return a reusable passed report only when every saved artifact agrees.

    A completed optimizer state alone is not enough: the final evaluation may
    have failed, or ``model.pt`` may belong to a different terrain.  Failure of
    any check deliberately falls back to :func:`train`, whose normal resume
    validation either repairs missing final artifacts or rejects an
    incompatible training state.
    """
    if not bool(cfg.get("resume_training", False)):
        return None

    folder = Path(folder)
    state_path = folder / "training_state.pt"
    report_path = folder / "training_report.json"
    model_path = folder / "model.pt"
    if not all(path.is_file() for path in (state_path, report_path, model_path)):
        return None

    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        # The canonical trainer will either regenerate final artifacts from a
        # valid state or surface the malformed state as an explicit error.
        return None
    if not all(isinstance(value, dict) for value in (state, checkpoint, report)):
        return None
    if state.get("schema_version") != 2:
        return None

    saved_identity = state.get("identity")
    if not isinstance(saved_identity, dict):
        return None
    saved_identity = dict(saved_identity)
    missing = object()
    saved_anchor = saved_identity.pop(
        "mass_projection_anchor_fingerprint", missing)
    if saved_anchor is missing or saved_identity != _training_identity(
            terrain, events, cfg):
        return None
    if ("mass_projection_anchor_fingerprint" not in checkpoint or
            checkpoint["mass_projection_anchor_fingerprint"] != saved_anchor):
        return None
    if ("best_state" not in state or "model_state" not in checkpoint or
            not _artifact_values_equal(
                state["best_state"], checkpoint["model_state"])):
        return None

    state_best = state.get("best")
    report_best = report.get("best_validation_loss")
    if (isinstance(state_best, bool) or isinstance(report_best, bool) or
            not isinstance(state_best, (int, float)) or
            not isinstance(report_best, (int, float)) or
            not math.isfinite(float(state_best)) or
            not math.isfinite(float(report_best)) or
            not math.isclose(float(state_best), float(report_best),
                             rel_tol=1e-12, abs_tol=1e-15)):
        return None
    saved_config = saved_identity.get("config")
    if not isinstance(saved_config, dict):
        return None
    for field in ("success_thresholds", "assumptions"):
        if field not in saved_config or report.get(field) != saved_config[field]:
            return None

    requested = cfg.get("adam_steps")
    state_step = state.get("step")
    report_requested = report.get("requested_steps")
    report_completed = report.get("completed_steps")
    whole_numbers = (requested, state_step, report_requested, report_completed)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in whole_numbers):
        return None
    if (requested <= 0 or state_step < requested or
            report_requested != requested or report_completed != state_step or
            report_completed < report_requested):
        return None
    if report.get("physics_training_passed") is not True:
        return None

    terrain_fingerprint = terrain.fingerprint
    report_terrain = report.get("terrain")
    if (checkpoint.get("terrain_fingerprint") != terrain_fingerprint or
            not isinstance(report_terrain, dict) or
            report_terrain.get("fingerprint") != terrain_fingerprint):
        return None
    return report


def run_expert_config(master, base):
    """Train an already loaded expert configuration relative to ``base``."""
    base = Path(base).resolve()
    resolve = lambda value: (base / value).resolve()
    terrain = MaskedRasterTerrain(resolve(master["terrain_tif"]), manning_n=master["manning_n"])
    observed = extract_events(resolve(master["rainfall_zip"]))
    output_root = resolve(master["output_root"])
    resume_training = bool(master["training"].get("resume_training", False))
    output_root.mkdir(parents=True, exist_ok=resume_training)
    summaries = []
    duration_min_hr, duration_max_hr = min(master["durations_hr"]), max(master["durations_hr"])
    markers = master.get("design_split_markers", {})
    for index, expert in enumerate(master["experts"]):
        rates = expert["intensities_mm_hr"]
        selected_observed = [
            event for event in observed
            if (rates[0] <= event["mean_intensity_mm_hr"] <= rates[-1]
                and duration_min_hr * 3600 <= event["duration_s"] <= duration_max_hr * 3600)
        ]
        events = add_design_events(
            selected_observed, rates, master["durations_hr"],
            validation_markers=markers.get("validation", (3,)),
            test_markers=markers.get("test", (7,)),
        )
        cfg = _apply_expert_operational_overrides(
            master["training"], expert)
        cfg["seed"] += index * 1000
        cfg["extreme_total_mm"] = rates[-1] * max(master["durations_hr"]) * .9
        cfg["expert_name"] = expert["name"]
        cfg["scales"] = {
            "x0": terrain.bounds[0], "y0": terrain.bounds[2],
            "lx": terrain.bounds[1] - terrain.bounds[0], "ly": terrain.bounds[3] - terrain.bounds[2],
            "time_s": cfg["time_end_s"], "rain_m_s": rates[-1] / 3_600_000.,
            "rain_min_m_s": rates[0] / 3_600_000., "duration_s": max(master["durations_hr"]) * 3600.,
            "duration_min_s": min(master["durations_hr"]) * 3600.,
            "rain_log_reference_m_s": master.get("rain_reference_mm_hr", 1.) / 3_600_000.,
            "duration_log_reference_s": master.get("duration_reference_hr", 1.) * 3600.,
            "depth_m": rates[-1] * max(master["durations_hr"]) / 1000.,
            "velocity_m_s": expert["velocity_m_s"], "ramp_s": 600.,
            "time_log_reference_s": 600.,
            "wet_velocity_gate_m": master.get("wet_gate_m", 1e-5)}
        folder = output_root / expert["name"]
        print(f"\n=== training {expert['name']} expert: {rates[0]}–{rates[-1]} mm/hr, {len(events)} scenarios ===", flush=True)
        report = _load_completed_expert_report(terrain, events, cfg, folder)
        training_skipped = report is not None
        if training_skipped:
            print(f"skipping completed and passed {expert['name']} expert", flush=True)
        else:
            _, report = train(terrain, events, cfg, folder)
        summaries.append({"name": expert["name"], "rainfall_mm_hr": [rates[0], rates[-1]],
                          "duration_hr": [duration_min_hr, duration_max_hr],
                          "simulation_horizon_hr": cfg["time_end_s"] / 3600.,
                          "observed_events_used": len(selected_observed),
                          "model": str(folder / "model.pt"),
                          "training_skipped": training_skipped, "report": report})
        if master.get("stop_on_failed_expert", False) and not report["physics_training_passed"]:
            print(f"stopping after failed {expert['name']} expert physics gate", flush=True)
            break
    summary = {"all_passed": (len(summaries) == len(master["experts"]) and
                              all(item["report"]["physics_training_passed"] for item in summaries)),
               "completed_experts": len(summaries),
               "requested_experts": len(master["experts"]),
               "domain": {"rainfall_mm_hr": [min(e["intensities_mm_hr"][0] for e in master["experts"]),
                                                max(e["intensities_mm_hr"][-1] for e in master["experts"])],
                          "duration_hr": [duration_min_hr, duration_max_hr],
                          "simulation_horizon_hr": master["training"]["time_end_s"] / 3600.},
               "routing": master.get("routing"),
               "experts": summaries}
    (output_root / "expert_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(config_path="configs/minxiong_experts_0_800.json"):
    path = (ROOT / config_path).resolve()
    master = json.loads(path.read_text(encoding="utf-8-sig"))
    return run_expert_config(master, path.parent)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/minxiong_experts_0_800.json")
