"""Bounded, observable single/conditional PINN training; never runs on import.

Only explicitly configured dry initial conditions and four slip walls are
implemented. Physical/accuracy approval is separate from finishing optimization.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .model import ParametricPINN
from .physics import residuals
from .contracts import PHYSICS_VERSION


class TrainingBudgetExceeded(RuntimeError):
    pass


def _finite_number(value, label, minimum=0, strictly=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number, not a boolean/string")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if strictly else result < minimum):
        raise ValueError(f"invalid {label}")
    return result


def validate_config(config: dict) -> dict:
    """Validate before any optimization; return a detached canonical copy."""
    cfg = copy.deepcopy(config)
    if cfg.get("conditions", {}).get("initial") != "dry":
        raise ValueError("only explicit initial='dry' is implemented")
    if cfg["conditions"].get("boundaries") != dict.fromkeys(("west", "east", "south", "north"), "wall"):
        raise ValueError("only explicit wall conditions on all four edges are implemented")
    end = _finite_number(cfg["time"]["end_s"], "end_s", strictly=True)
    cfg["time"]["end_s"] = end
    cfg["time"]["output_interval_s"] = _finite_number(cfg["time"]["output_interval_s"], "output_interval_s", strictly=True)
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise ValueError("explicit training/validation/test scenarios are required")
    seen, pairs = set(), {}
    for scenario in scenarios:
        sid = str(scenario["scenario_id"])
        if not sid or sid in seen:
            raise ValueError("scenario_id must be nonempty and unique")
        seen.add(sid)
        scenario["scenario_id"] = sid
        if scenario.get("split") not in {"train", "validation", "test"}:
            raise ValueError("scenario split must be train, validation or test")
        rate = _finite_number(scenario["rainfall_mm_hr"], "rainfall_mm_hr")
        duration = _finite_number(scenario["duration_s"], "duration_s")
        if duration > end:
            raise ValueError("rain duration exceeds simulation end")
        key = (rate, duration)
        if key in pairs:
            raise ValueError("duplicate rain combination (including across splits)")
        pairs[key] = scenario["split"]
        scenario["rainfall_mm_hr"], scenario["duration_s"] = rate, duration
    train_cases = [s for s in scenarios if s["split"] == "train"]
    if not train_cases:
        raise ValueError("at least one training scenario required")
    if not cfg.get("model", {}).get("parameterized", True) and len(train_cases) != 1:
        raise ValueError("three-input model requires exactly one training scenario")
    opts = cfg.setdefault("training", {})
    defaults = {"adam_steps": 20, "lbfgs_steps": 0, "interior_points": 32,
                "ic_points": 16, "bc_points": 8, "learning_rate": .001,
                "seed": 42, "max_seconds": 60., "threads": 1,
                "scenarios_per_batch": min(4, len(train_cases))}
    for key, default in defaults.items():
        opts.setdefault(key, default)
    for key in ("adam_steps", "lbfgs_steps", "interior_points", "ic_points", "bc_points", "threads", "scenarios_per_batch", "seed"):
        value = opts[key]
        minimum = 0 if key in {"adam_steps", "lbfgs_steps", "seed"} else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if opts["adam_steps"] + opts["lbfgs_steps"] == 0:
        raise ValueError("a training run must execute at least one optimizer step")
    opts["learning_rate"] = _finite_number(opts["learning_rate"], "learning_rate", strictly=True)
    opts["max_seconds"] = _finite_number(opts["max_seconds"], "max_seconds", strictly=True)
    if not isinstance(opts.get("allow_long_training", False), bool):
        raise ValueError("allow_long_training must be an actual boolean")
    for key, default in (("max_reference_rows", 100000), ("max_training_reference_points", 4096)):
        opts.setdefault(key, default)
        if isinstance(opts[key], bool) or not isinstance(opts[key], int) or opts[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for name, weight in opts.get("loss_weights", {}).items():
        if name not in {"pde_loss", "ic_loss", "bc_loss", "data_loss", "conservation_loss"}:
            raise ValueError(f"unknown loss weight: {name}")
        opts["loss_weights"][name] = _finite_number(weight, f"loss weight {name}")
    threshold = cfg.setdefault("thresholds", {})
    dry = _finite_number(threshold.get("dry_depth_m", 1e-4), "dry_depth_m")
    flood = _finite_number(threshold.get("flood_depth_m", .01), "flood_depth_m", strictly=True)
    if dry >= flood:
        raise ValueError("dry_depth_m must be less than flood_depth_m")
    threshold.update(dry_depth_m=dry, flood_depth_m=flood)
    json.dumps(cfg, allow_nan=False)
    if not opts.get("allow_long_training", False) and (opts["adam_steps"] > 1000 or opts["lbfgs_steps"] > 100 or opts["max_seconds"] > 600):
        raise ValueError("long training requires explicit training.allow_long_training=true")
    # Constructing the model here validates every fixed normalization scale.
    model_cfg = cfg.setdefault("model", {})
    probe = ParametricPINN(cfg["scales"], **model_cfg)
    del probe
    return cfg


def _read_reference(config, terrain):
    path = config.get("reference_csv")
    if not path:
        return [], None
    if not terrain.metadata.get("synthetic", False):
        provenance = config.get("reference_provenance", {})
        expected = {"crs": terrain.metadata["crs"],
                    "vertical_datum": terrain.metadata["vertical_datum"],
                    "grid_id": terrain.metadata["grid_id"]}
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("real-terrain reference provenance must match target crs, vertical_datum and grid_id")
        units = provenance.get("units", {})
        if any(units.get(key) != value for key, value in {"x": "m", "y": "m", "t": "s", "h": "m", "u": "m/s", "v": "m/s"}.items()):
            raise ValueError("real-terrain reference provenance requires explicit SI units for x,y,t,h,u,v")
        if not provenance.get("source_type") or not provenance.get("source_version") or provenance.get("synthetic") is not False:
            raise ValueError("real-terrain reference requires source_type, source_version and synthetic=false")
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"scenario_id", "x", "y", "t", "h"}.issubset(reader.fieldnames or []):
            raise ValueError("reference requires scenario_id,x,y,t,h")
        rows = []
        for raw in reader:
            rows.append(raw)
            if len(rows) > config["training"]["max_reference_rows"]:
                raise ValueError("reference row cap exceeded; explicitly preprocess/partition reference observations")
    if not rows:
        raise ValueError("reference file contains zero observations")
    cases = {s["scenario_id"]: s for s in config["scenarios"]}
    result = []
    xmin, xmax, ymin, ymax = terrain.bounds
    for raw in rows:
        if raw["scenario_id"] not in cases:
            raise ValueError("reference has unknown scenario_id")
        if raw.get("valid", "1").strip().lower() in {"0", "false"}:
            continue
        values = {k: float(raw[k]) for k in ("x", "y", "t", "h")}
        if not all(math.isfinite(v) for v in values.values()) or values["h"] < 0:
            raise ValueError("reference has nonfinite values or negative h")
        if not (xmin <= values["x"] <= xmax and ymin <= values["y"] <= ymax and 0 <= values["t"] <= config["time"]["end_s"]):
            raise ValueError("reference point outside model space/time domain")
        velocity = []
        for key in ("u", "v"):
            value = raw.get(key, "")
            number = None if value == "" else float(value)
            if number is not None and not math.isfinite(number):
                raise ValueError("nonfinite reference velocity")
            velocity.append(number)
        case = cases[raw["scenario_id"]]
        result.append({**values, "scenario_id": case["scenario_id"], "split": case["split"],
                       "rate": case["rainfall_mm_hr"] / 3_600_000,
                       "duration": case["duration_s"], "u": velocity[0], "v": velocity[1]})
    if not result:
        raise ValueError("reference file has zero valid observations")
    if sum(row["split"] == "train" for row in result) > config["training"]["max_training_reference_points"]:
        raise ValueError("training reference cap exceeded; provide a documented representative training anchor subset")
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def sample_batch(terrain, config, rng, *, all_training=False, scenarios=None):
    """Return fixed numpy point sets. Counts are per selected scenario (BC per edge)."""
    cases = scenarios if scenarios is not None else [s for s in config["scenarios"] if s["split"] == "train"]
    if not cases:
        raise ValueError("cannot sample an empty scenario set")
    if scenarios is None and not all_training:
        count = min(config["training"]["scenarios_per_batch"], len(cases))
        cases = [cases[i] for i in rng.choice(len(cases), size=count, replace=False)]
    opts = config["training"]
    xmin, xmax, ymin, ymax = terrain.bounds
    end = config["time"]["end_s"]
    groups = {"pde": [], "ic": [], "bc": [], "normals": []}
    for case in cases:
        def points(count):
            p = np.empty((count, 5))
            p[:, 0] = rng.uniform(xmin, xmax, count)
            p[:, 1] = rng.uniform(ymin, ymax, count)
            p[:, 2] = rng.uniform(0, end, count)
            p[:, 3] = case["rainfall_mm_hr"] / 3_600_000
            p[:, 4] = case["duration_s"]
            return p
        p = points(opts["interior_points"])
        duration = case["duration_s"]
        # Stratify rain/post-rain and both sides of the source discontinuity.
        if 0 < duration < end:
            half = len(p) // 2
            p[:half, 2] = rng.uniform(0, duration, half)
            p[half:, 2] = rng.uniform(duration, end, len(p) - half)
            if len(p) >= 2:
                epsilon = min(duration, end - duration) * .001
                p[-2:, 2] = [duration - epsilon, duration + epsilon]
        # float32 endpoint rounding must not make a strong-form stop-time point.
        equal = p[:, 2].astype(np.float32) == np.float32(duration)
        p[equal, 2] = np.nextafter(np.float32(duration), np.float32(0.)) if duration > 0 else min(end * .5, .01)
        ic = points(opts["ic_points"])
        ic[:, 2] = 0
        b = points(4 * opts["bc_points"])
        count = opts["bc_points"]
        b[:count, 0], b[count:2 * count, 0] = xmin, xmax
        b[2 * count:3 * count, 1], b[3 * count:, 1] = ymin, ymax
        normals = np.repeat([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]], count, axis=0)
        for key, value in (("pde", p), ("ic", ic), ("bc", b), ("normals", normals)):
            groups[key].append(value)
    return {**{k: np.concatenate(v) for k, v in groups.items()},
            "scenario_ids": [s["scenario_id"] for s in cases]}


def _tensor(model, array, grad=False):
    parameter = next(model.parameters())
    # Retain large GIS coordinates before the model subtracts its local origin.
    return torch.as_tensor(array, dtype=torch.float64, device=parameter.device).clone().detach().requires_grad_(grad)


def loss_components(model, terrain, config, batch, reference_rows=()):
    """Differentiable, dimensionless losses; only caller-selected training references."""
    training_ids = {s["scenario_id"] for s in config["scenarios"] if s["split"] == "train"}
    if not set(batch["scenario_ids"]).issubset(training_ids):
        raise ValueError("held-out scenario cannot enter training loss (including PDE/IC/BC)")
    s = model.scales
    physical = residuals(model, _tensor(model, batch["pde"], True), terrain)
    residual_scales = physical.new_tensor([s["h_scale"] / s["t_scale"],
                                          s["h_scale"] * s["u_scale"] / s["t_scale"],
                                          s["h_scale"] * s["u_scale"] / s["t_scale"]])
    normalized = physical / residual_scales
    pde_parts = normalized.square().mean(dim=0)
    ic_pred = model(_tensor(model, batch["ic"]))
    outputs_scale = ic_pred.new_tensor([s["h_scale"], s["u_scale"], s["u_scale"]])
    ic = (ic_pred / outputs_scale).square().mean()
    bc_pred = model(_tensor(model, batch["bc"]))
    normal_velocity = (bc_pred[:, 1:] * _tensor(model, batch["normals"])).sum(dim=1)
    # Normalize wall discharge; auxiliary velocity preserves original slip-wall constraint.
    bc = ((bc_pred[:, 0] * normal_velocity / (s["h_scale"] * s["u_scale"])).square()
          + (normal_velocity / s["u_scale"]).square()).mean()
    data = physical.new_zeros(())
    if reference_rows:
        if any(row["split"] != "train" for row in reference_rows):
            raise ValueError("held-out reference cannot enter training loss")
        points = [[r["x"], r["y"], r["t"], r["rate"], r["duration"]] for r in reference_rows]
        pred = model(_tensor(model, points))
        errors = [(pred[:, 0] - pred.new_tensor([r["h"] for r in reference_rows])) / s["h_scale"]]
        data = errors[0].square().mean()
        for column, key in ((1, "u"), (2, "v")):
            indexes = [i for i, r in enumerate(reference_rows) if r[key] is not None]
            if indexes:
                target = pred.new_tensor([reference_rows[i][key] for i in indexes])
                data = data + ((pred[indexes, column] - target) / s["u_scale"]).square().mean()
    # Monte-Carlo spatial/temporal quadrature of full-horizon global continuity.
    # Uniform IC locations are reused at t=end. Boundary samples are uniform in
    # time on each edge, with one equal-sized group per side and per scenario.
    # This is a training constraint, not a substitute for exported refined balance.
    end_points = np.array(batch["ic"], copy=True)
    end_points[:, 2] = config["time"]["end_s"]
    end_depth = model(_tensor(model, end_points))[:, 0]
    count_ic = config["training"]["ic_points"]
    count_bc = config["training"]["bc_points"]
    xmin, xmax, ymin, ymax = terrain.bounds
    lengths = physical.new_tensor([ymax-ymin, ymax-ymin, xmax-xmin, xmax-xmin])
    area = (xmax-xmin)*(ymax-ymin)
    mass_terms = []
    for index in range(len(batch["scenario_ids"])):
        si = slice(index*count_ic, (index+1)*count_ic)
        sb = slice(index*4*count_bc, (index+1)*4*count_bc)
        outward = (bc_pred[sb, 0] * normal_velocity[sb]).reshape(4, count_bc)
        boundary_volume = (outward.mean(dim=1)*lengths).sum()*config["time"]["end_s"]
        rate = float(batch["ic"][index*count_ic, 3])
        duration = float(batch["ic"][index*count_ic, 4])
        rain_volume = area*rate*min(config["time"]["end_s"], duration)
        storage = area*(end_depth[si].mean()-ic_pred[si, 0].mean())
        mass_terms.append(((storage+boundary_volume-rain_volume)/(area*s["h_scale"])).square())
    conservation = torch.stack(mass_terms).mean()
    parts = {"pde_loss": pde_parts.mean(), "ic_loss": ic, "bc_loss": bc,
             "data_loss": data, "conservation_loss": conservation,
             "pde_mass_loss": pde_parts[0], "pde_x_loss": pde_parts[1], "pde_y_loss": pde_parts[2]}
    weights = config["training"].get("loss_weights", {})
    total = sum(parts[key] * float(weights.get(key, default)) for key, default in
                (("pde_loss", 1.), ("ic_loss", 5.), ("bc_loss", 5.), ("data_loss", 50.), ("conservation_loss", 1.)))
    return total, parts


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _diagnose(model, terrain, config, references, output):
    """Independent fixed audit samples, all splits named and never optimized."""
    rows, case_metrics = [], []
    rng = np.random.default_rng(config["training"]["seed"] + 100_003)
    opts = copy.deepcopy(config)
    opts["training"].update(interior_points=16, ic_points=16, bc_points=4)
    s = model.scales
    rsc = np.array([s["h_scale"] / s["t_scale"], s["h_scale"] * s["u_scale"] / s["t_scale"], s["h_scale"] * s["u_scale"] / s["t_scale"]])
    for case in config["scenarios"]:
        batch = sample_batch(terrain, opts, rng, scenarios=[case])
        with torch.enable_grad():
            residual = residuals(model, _tensor(model, batch["pde"], True), terrain).detach().cpu().numpy()
        with torch.no_grad():
            pred_ic = model(_tensor(model, batch["ic"])).cpu().numpy()
            pred_bc = model(_tensor(model, batch["bc"])).cpu().numpy()
        if not all(np.isfinite(a).all() for a in [residual, pred_ic, pred_bc]):
            raise FloatingPointError("nonfinite independent audit prediction")
        normalized = residual / rsc
        for point, raw, norm in zip(batch["pde"], residual, normalized):
            rows.append({"scenario_id": case["scenario_id"], "split": case["split"],
                         "x": point[0], "y": point[1], "t": point[2],
                         "mass_residual_m_s": raw[0], "x_residual_m2_s2": raw[1], "y_residual_m2_s2": raw[2],
                         "mass_normalized": norm[0], "x_normalized": norm[1], "y_normalized": norm[2]})
        normal_velocity = (pred_bc[:, 1:] * batch["normals"]).sum(axis=1)
        metric = {"scenario_id": case["scenario_id"], "split": case["split"],
                  "pde_rms_normalized": np.sqrt(np.mean(normalized ** 2, axis=0)).tolist(),
                  "pde_p95_normalized": np.percentile(np.abs(normalized), 95, axis=0).tolist(),
                  "ic_h_rmse_m": float(np.sqrt(np.mean(pred_ic[:, 0] ** 2))),
                  "ic_velocity_rmse_m_s": float(np.sqrt(np.mean(pred_ic[:, 1:] ** 2))),
                  "bc_normal_velocity_rmse_m_s": float(np.sqrt(np.mean(normal_velocity ** 2))),
                  "bc_discharge_rmse_m2_s": float(np.sqrt(np.mean((pred_bc[:, 0] * normal_velocity) ** 2))),
                  "reference_sample_count": 0, "depth_mae_m": None, "depth_rmse_m": None,
                  "relative_l2": None, "velocity_rmse_m_s": None,
                  "flood_iou": None, "flood_precision": None, "flood_recall": None, "flood_f1": None,
                  "reference_status": "not_evaluated"}
        refs = [r for r in references if r["scenario_id"] == case["scenario_id"]]
        if refs:
            with torch.no_grad():
                predictions = model(_tensor(model, [[r["x"], r["y"], r["t"], r["rate"], r["duration"]] for r in refs])).cpu().numpy()
            if not np.isfinite(predictions).all():
                raise FloatingPointError("nonfinite reference-comparison prediction")
            targets = np.array([r["h"] for r in refs])
            err = predictions[:, 0] - targets
            denom = np.linalg.norm(targets)
            metric.update(reference_sample_count=len(refs), depth_mae_m=float(np.abs(err).mean()),
                          depth_rmse_m=float(np.sqrt(np.mean(err ** 2))),
                          relative_l2=float(np.linalg.norm(err) / denom) if denom > 1e-12 else None,
                          reference_status="comparison_completed")
            verr = []
            for column, key in ((1, "u"), (2, "v")):
                verr.extend((predictions[i, column] - r[key]) ** 2 for i, r in enumerate(refs) if r[key] is not None)
            if verr:
                metric["velocity_rmse_m_s"] = float(np.sqrt(np.mean(verr)))
            threshold = float(config.get("thresholds", {}).get("flood_depth_m", .01))
            p, actual = predictions[:, 0] >= threshold, targets >= threshold
            intersection, union = np.count_nonzero(p & actual), np.count_nonzero(p | actual)
            metric["flood_iou"] = float(intersection / union) if union else None
            metric["flood_precision"] = float(intersection / p.sum()) if p.any() else None
            metric["flood_recall"] = float(intersection / actual.sum()) if actual.any() else None
            metric["flood_f1"] = float(2 * intersection / (p.sum() + actual.sum())) if p.any() or actual.any() else None
        case_metrics.append(metric)
    with (output / "physics_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metrics = {"schema_version": "1.0", "scenarios": case_metrics,
               "split_counts": {split: sum(c["split"] == split for c in config["scenarios"]) for split in ("train", "validation", "test")},
               "reference_type": config.get("reference_provenance", {}).get("source_type", "unspecified" if references else "none"),
               "reference_validation_status": "comparison_only_not_accuracy_approval" if references else "not_completed",
               "held_out_scenario_status": "physics_audited_not_accuracy_validated" if any(c["split"] != "train" for c in config["scenarios"]) else "not_evaluated",
               "water_balance_status": "requires_time_series_export", "acceptance_passed": None}
    _write_json(output / "validation_metrics.json", metrics)
    # Keep spatial residual evidence, rather than a single scalar loss.
    plt = _plotting()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name in zip(axes, ("mass_normalized", "x_normalized", "y_normalized")):
        scatter = axis.scatter([r["x"] for r in rows], [r["y"] for r in rows],
                               c=[abs(r[name]) for r in rows], s=12)
        axis.set(title=name + " (all audit scenarios/times)", xlabel="x [m]", ylabel="y [m]")
        figure.colorbar(scatter, ax=axis)
    figure.tight_layout(); figure.savefig(output / "pde_residuals.png", dpi=120); plt.close(figure)
    if references:
        figure, axis = plt.subplots(figsize=(6, 4))
        for case in config["scenarios"]:
            refs = [r for r in references if r["scenario_id"] == case["scenario_id"]]
            if not refs:
                continue
            with torch.no_grad():
                pred = model(_tensor(model, [[r["x"], r["y"], r["t"], r["rate"], r["duration"]] for r in refs])).cpu().numpy()
            scatter = axis.scatter([r["x"] for r in refs], [r["y"] for r in refs],
                                   c=pred[:, 0]-np.array([r["h"] for r in refs]), s=20)
        axis.set(title="Depth error, all reference scenarios/times", xlabel="x [m]", ylabel="y [m]")
        figure.colorbar(scatter, ax=axis, label="h predicted − reference [m]")
        figure.tight_layout(); figure.savefig(output / "reference_error.png", dpi=120); plt.close(figure)
    return metrics


def _plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flood_pinn_mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "flood_pinn_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def train(terrain, config: dict, out_dir):
    """Execute an explicitly bounded run and return ``(model, report)``.

    A normal return with training_completed=False denotes a time-budget stop.
    Other exceptions are re-raised after writing a failed report and last finite
    weights. No result is automatically added to a validated inference domain.
    """
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_history.csv").exists() or (output / "training_report.json").exists():
        raise FileExistsError("use a new output directory; existing training evidence is preserved")
    config = validate_config(config)
    references, reference_hash = _read_reference(config, terrain)
    opts = config["training"]
    torch.manual_seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.set_num_threads(opts["threads"])
    device_name = opts.get("device", "cpu")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParametricPINN(config["scales"], **config["model"]).to(device_name)
    train_refs = [r for r in references if r["split"] == "train"]
    rng = np.random.default_rng(opts["seed"])
    start = time.perf_counter()
    report = {"schema_version": "1.0", "model_version": "conditional-swe-v1",
              "physics_version": PHYSICS_VERSION, "config": config,
              "terrain_fingerprint": terrain.fingerprint, "reference_sha256": reference_hash,
              "device": str(next(model.parameters()).device), "torch_version": str(torch.__version__),
              "training_completed": False, "converged": False,
              "physics_checks_passed": None, "reference_validation_status": "not_completed",
              "held_out_scenario_status": "not_evaluated", "acceptance_passed": None,
              "training_scenario_ids": [s["scenario_id"] for s in config["scenarios"] if s["split"] == "train"],
              "training_reference_count": len(train_refs), "reference_sample_count": len(references),
              "validated_domain": None, "termination_reason": "running", "nonfinite_detected": False,
              "assumptions": ["explicit dry initial condition", "four slip walls", "no infiltration/drainage/building model"],
              "adam_steps_completed": 0, "lbfgs_iterations_completed": 0, "closure_evaluations": 0}
    fields = ["phase", "optimizer_step", "closure_eval", "total_loss", "pde_loss", "ic_loss", "bc_loss", "data_loss",
              "conservation_loss", "pde_mass_loss", "pde_x_loss", "pde_y_loss", "learning_rate", "elapsed_s",
              "gradient_norm_before_clip", "gradient_norm_after_clip", "nonfinite", "scenario_ids"]
    history = []
    last_finite = copy.deepcopy(model.state_dict())
    optimizer_step = 0

    def budget():
        if time.perf_counter() - start > opts["max_seconds"]:
            raise TrainingBudgetExceeded("configured training wall-clock budget exhausted")

    def evaluate(optimizer, batch, phase, writer, stream):
        nonlocal last_finite
        budget()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = loss_components(model, terrain, config, batch, train_refs)
        if not torch.isfinite(loss):
            report["nonfinite_detected"] = True
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            report["nonfinite_detected"] = True
            raise FloatingPointError("nonfinite training gradient")
        # LBFGS line search requires the true objective gradient, not clipped gradients.
        norm = torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))
        if not torch.isfinite(norm):
            report["nonfinite_detected"] = True
            raise FloatingPointError("nonfinite aggregate gradient norm")
        if phase == "adam":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        after = torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))
        if not torch.isfinite(after):
            report["nonfinite_detected"] = True
            raise FloatingPointError("nonfinite gradient norm after clipping")
        row = {"phase": phase, "optimizer_step": optimizer_step,
               "closure_eval": report["closure_evaluations"] if phase == "lbfgs" else 0,
               "total_loss": float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()},
               "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_s": time.perf_counter() - start,
               "gradient_norm_before_clip": float(norm), "gradient_norm_after_clip": float(after),
               "nonfinite": False, "scenario_ids": ";".join(batch["scenario_ids"])}
        if not train_refs:
            row["data_loss"] = None
        writer.writerow(row); stream.flush(); history.append(row)
        last_finite = copy.deepcopy(model.state_dict())
        budget()
        return loss

    failure = None
    try:
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            adam = torch.optim.Adam(model.parameters(), lr=opts["learning_rate"])
            scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=max(1, opts["adam_steps"] // 3), gamma=.5)
            for optimizer_step in range(1, opts["adam_steps"] + 1):
                batch = sample_batch(terrain, config, rng)
                evaluate(adam, batch, "adam", writer, stream)
                adam.step(); scheduler.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite weights after Adam update")
                report["adam_steps_completed"] += 1
            if opts["lbfgs_steps"]:
                fixed = sample_batch(terrain, config, rng, all_training=True)
                lbfgs = torch.optim.LBFGS(model.parameters(), lr=.1, max_iter=opts["lbfgs_steps"],
                                         history_size=20, line_search_fn="strong_wolfe")
                def closure():
                    report["closure_evaluations"] += 1
                    return evaluate(lbfgs, fixed, "lbfgs", writer, stream)
                optimizer_step = opts["adam_steps"] + 1
                lbfgs.step(closure)
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite weights after LBFGS update")
                report["lbfgs_iterations_completed"] = lbfgs.state[next(iter(model.parameters()))].get("n_iter", 0)
            budget()
        report["training_completed"] = True
        report["termination_reason"] = "configured_steps_completed"
    except TrainingBudgetExceeded as exc:
        model.load_state_dict(last_finite)
        report["termination_reason"] = "time_budget_exhausted"
        report["error"] = str(exc)
    except Exception as exc:
        model.load_state_dict(last_finite)
        report["termination_reason"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["nonfinite_detected"] = report["nonfinite_detected"] or isinstance(exc, FloatingPointError)
        torch.save(last_finite, output / "last_finite_state.pt")
        failure = exc
    finally:
        report["elapsed_training_s"] = time.perf_counter() - start
        report["history_rows"] = len(history)
        _write_json(output / "training_report.json", report)
    if failure is not None:
        raise failure
    model.eval()
    try:
        metrics = _diagnose(model, terrain, config, references, output)
    except Exception as exc:
        report["termination_reason"] = "diagnostics_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["physics_checks_passed"] = False
        report["nonfinite_detected"] = isinstance(exc, FloatingPointError)
        _write_json(output / "training_report.json", report)
        raise
    report["reference_validation_status"] = metrics["reference_validation_status"]
    report["held_out_scenario_status"] = metrics["held_out_scenario_status"]
    report["elapsed_total_s"] = time.perf_counter() - start
    report["convergence_note"] = "Optimization completion is not a physics/accuracy acceptance result; no validated domain was inferred."
    _write_json(output / "training_report.json", report)
    # Plotting is a diagnostic artifact, never used as a pass criterion.
    plt = _plotting()
    if history:
        figure, axis = plt.subplots(figsize=(8, 4))
        for name in ("total_loss", "pde_loss", "ic_loss", "bc_loss", "data_loss"):
            y = [row[name] if row[name] is not None else np.nan for row in history]
            if np.isfinite(y).any():
                axis.semilogy(np.arange(len(y)), y, label=name)
        axis.set(xlabel="Objective evaluation (LBFGS closures included)", ylabel="Dimensionless loss")
        axis.legend(); figure.tight_layout(); figure.savefig(output / "loss.png", dpi=120); plt.close(figure)
    return model, report
