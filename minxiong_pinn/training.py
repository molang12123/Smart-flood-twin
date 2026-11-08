from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flood_pinn.physics import residuals
from .model import ConditionalFloodPINN


def _points(terrain, rng, case, count, end_s, slope_fraction):
    xy = terrain.sample_points(rng, count, slope_fraction=slope_fraction)
    p = np.empty((count, 5), dtype=np.float64)
    p[:, :2] = xy
    p[:, 2] = rng.uniform(0, end_s, count)
    p[:, 3] = case["mean_intensity_mm_hr"] / 3_600_000.
    p[:, 4] = case["duration_s"]
    duration = min(case["duration_s"], end_s)
    if 0 < duration < end_s and count >= 4:
        half = count // 2
        p[:half, 2] = rng.uniform(0, duration, half)
        p[half:, 2] = rng.uniform(duration, end_s, count - half)
        epsilon = max(1e-3, min(duration, end_s - duration) * 1e-5)
        p[-2:, 2] = (duration - epsilon, duration + epsilon)
    return p


def _tensor(array, *, grad=False):
    return torch.as_tensor(array, dtype=torch.float64).requires_grad_(grad)


def _fv_residuals(model, terrain, case, rng, cfg):
    """Differentiable midpoint finite-volume balance on active 20 m cells."""
    count = cfg["interior_points"]
    xy = terrain.sample_control_volumes(rng, count)
    dt = min(float(cfg.get("weak_dt_s", 600.)), cfg["time_end_s"])
    t0 = rng.uniform(0, cfg["time_end_s"] - dt, count); t1 = t0 + dt; tm = (t0 + t1) / 2
    rate = case["mean_intensity_mm_hr"] / 3_600_000.; duration = case["duration_s"]

    def make(coords, times):
        p = np.empty((count, 5)); p[:, :2] = coords; p[:, 2] = times
        p[:, 3] = rate; p[:, 4] = duration
        return _tensor(p)

    centre0, centre1 = model(make(xy, t0)), model(make(xy, t1))
    east = xy + np.array([terrain.dx / 2, 0]); west = xy - np.array([terrain.dx / 2, 0])
    north = xy + np.array([0, terrain.dy / 2]); south = xy - np.array([0, terrain.dy / 2])
    ve, vw, vn, vs = (model(make(coords, tm)) for coords in (east, west, north, south))
    h0, u0, v0 = centre0.T; h1, u1, v1 = centre1.T
    he, ue, ve_y = ve.T; hw, uw, vw_y = vw.T; hn, un, vn_y = vn.T; hs, us, vs_y = vs.T
    gravity = 9.81
    mass = ((h1 - h0) / dt + (he * ue - hw * uw) / terrain.dx +
            (hn * vn_y - hs * vs_y) / terrain.dy)
    mx = ((h1 * u1 - h0 * u0) / dt +
          (he * ue.square() + .5 * gravity * he.square() - hw * uw.square() - .5 * gravity * hw.square()) / terrain.dx +
          (hn * un * vn_y - hs * us * vs_y) / terrain.dy)
    my = ((h1 * v1 - h0 * v0) / dt +
          (he * ue * ve_y - hw * uw * vw_y) / terrain.dx +
          (hn * vn_y.square() + .5 * gravity * hn.square() - hs * vs_y.square() - .5 * gravity * hs.square()) / terrain.dy)
    data = terrain.query(xy[:, 0], xy[:, 1])
    zx = mx.new_tensor(data["zx"]); zy = my.new_tensor(data["zy"]); n = mx.new_tensor(data["n"])
    hmid = .5 * (h0 + h1); umid = .5 * (u0 + u1); vmid = .5 * (v0 + v1)
    speed = torch.sqrt(umid.square() + vmid.square() + 1e-12)
    friction = gravity * n.square() * hmid * speed / hmid.clamp_min(cfg["h_regularization_m"]).pow(4 / 3)
    mx = mx + gravity * hmid * zx + friction * umid
    my = my + gravity * hmid * zy + friction * vmid
    rain_seconds = np.maximum(0., np.minimum(t1, duration) - np.minimum(t0, duration))
    mass = mass - mass.new_tensor(rate * rain_seconds / dt)
    return torch.stack((mass, mx, my), dim=1), rate, duration


def _loss(model, terrain, cases, rng, cfg):
    pde_losses, bc_losses, mass_losses = [], [], []
    end_s = cfg["time_end_s"]
    for case in cases:
        if cfg.get("physics_form", "strong") == "finite_volume":
            raw, rate, duration = _fv_residuals(model, terrain, case, rng, cfg)
            points = np.array([[0., 0., 0., rate, duration]])
            multiplier = 1
        else:
            multiplier = (cfg.get("extreme_residual_candidate_multiplier", 1)
                          if case.get("total_mm", 0) >= cfg.get("extreme_total_mm", math.inf) else 1)
            points = _points(terrain, rng, case, cfg["interior_points"] * multiplier, end_s,
                             cfg.get("slope_sampling_fraction", 0.))
            raw = residuals(model, _tensor(points, grad=True), terrain,
                            h_reg=cfg["h_regularization_m"])
        rain_depth = points[0, 3] * min(points[0, 4], end_s)
        continuity_scale = max(cfg["continuity_scale_m_s"], points[0, 3])
        momentum_scale = max(cfg["momentum_scale_m2_s2"],
                             9.81 * rain_depth * cfg.get("momentum_slope_scale", .1))
        scales = raw.new_tensor([continuity_scale, momentum_scale, momentum_scale])
        normalized_square = (raw / scales).square()
        if multiplier > 1:
            scores = normalized_square.detach().mean(dim=1)
            indexes = torch.topk(scores, cfg["interior_points"], sorted=False).indices
            normalized_square = normalized_square[indexes]
        pde_losses.append(normalized_square.mean())

        xy, normals = terrain.sample_boundary(rng, cfg["boundary_points"])
        b = np.empty((len(xy), 5)); b[:, :2] = xy
        b[:, 2] = rng.uniform(0, end_s, len(xy)); b[:, 3] = points[0, 3]; b[:, 4] = points[0, 4]
        pred_b = model(_tensor(b))
        normal = (pred_b[:, 1:] * _tensor(normals)).sum(dim=1)
        bc_losses.append((normal / model.scales["velocity_m_s"]).square().mean())

        event_mass = []
        audit_times = sorted(set((min(points[0, 4] * .5, end_s),
                                  min(points[0, 4], end_s), end_s)))
        for audit_time in audit_times:
            xy_m = terrain.sample_points(rng, cfg["mass_points"])
            m = np.empty((len(xy_m), 5)); m[:, :2] = xy_m
            m[:, 2] = audit_time; m[:, 3] = points[0, 3]; m[:, 4] = points[0, 4]
            mean_depth = model(_tensor(m))[:, 0].mean()
            expected = points[0, 3] * min(audit_time, points[0, 4])
            # Relative volume error at rain-midpoint, stop time and horizon.
            event_mass.append(((mean_depth - expected) / max(expected, 1e-6)).square())
        mass_losses.append(torch.stack(event_mass).mean())
    parts = {"pde": torch.stack(pde_losses).mean(), "boundary": torch.stack(bc_losses).mean(),
             "mass": torch.stack(mass_losses).mean()}
    total = (cfg["loss_weights"]["pde"] * parts["pde"] +
             cfg["loss_weights"]["boundary"] * parts["boundary"] +
             cfg["loss_weights"]["mass"] * parts["mass"])
    return total, parts


def evaluate(model, terrain, events, cfg, *, seed=9001):
    rng = np.random.default_rng(seed)
    rows = []
    for case in events:
        loss, parts = _loss(model, terrain, [case], rng, cfg)
        rows.append({"event_id": case["event_id"], "split": case["split"],
                     "rainfall_mm_hr": case["mean_intensity_mm_hr"],
                     "duration_s": case["duration_s"], "loss": float(loss.detach()),
                     **{f"{k}_loss": float(v.detach()) for k, v in parts.items()}})
    return rows


def train(terrain, events, cfg: dict, output_dir: str | Path):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.set_num_threads(cfg["threads"])
    model = ConditionalFloodPINN(cfg["scales"], cfg["width"], cfg["layers"],
                                 cfg.get("log_conditions", False))
    if cfg.get("initial_model_path"):
        initial = torch.load(cfg["initial_model_path"], map_location="cpu", weights_only=True)
        if initial["terrain_fingerprint"] != terrain.fingerprint:
            raise ValueError("initial checkpoint terrain does not match")
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    training = [e for e in events if e["split"] == "train"]
    validation = [e for e in events if e["split"] == "validation"]
    history, best, best_state = [], math.inf, None
    started = time.perf_counter()
    extreme = max(training, key=lambda event: event["total_mm"])
    probabilities = np.asarray([math.sqrt(max(event["total_mm"], .1)) for event in training])
    probabilities /= probabilities.sum()
    for step in range(1, cfg["adam_steps"] + 1):
        if cfg["scenarios_per_batch"] == 1:
            chosen = [extreme]
        else:
            eligible = [i for i, event in enumerate(training) if event is not extreme]
            p = probabilities[eligible]; p /= p.sum()
            selected = rng.choice(eligible, cfg["scenarios_per_batch"] - 1, replace=False, p=p)
            chosen = [extreme] + [training[i] for i in selected]
        optimizer.zero_grad(set_to_none=True)
        total, parts = _loss(model, terrain, chosen, rng, cfg)
        if not torch.isfinite(total): raise FloatingPointError("training loss became nonfinite")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        optimizer.step()
        record = {"step": step, "elapsed_s": time.perf_counter() - started,
                  "total_loss": float(total.detach()), "gradient_norm": float(grad),
                  **{f"{k}_loss": float(v.detach()) for k, v in parts.items()}}
        history.append(record)
        if step == 1 or step % cfg["validation_interval"] == 0 or step == cfg["adam_steps"]:
            metrics = evaluate(model, terrain, validation, cfg, seed=cfg["seed"] + step)
            score = float(np.mean([r["loss"] for r in metrics]))
            record["validation_loss"] = score
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"step={step:5d} train={record['total_loss']:.5g} val={score:.5g} "
                  f"pde={record['pde_loss']:.5g} mass={record['mass_loss']:.5g}", flush=True)
        if time.perf_counter() - started >= cfg["max_seconds"]:
            break
    if best_state is not None: model.load_state_dict(best_state)
    all_metrics = evaluate(model, terrain, events, cfg, seed=cfg["seed"] + 9999)
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({k for row in history for k in row}); writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(history)
    with (output / "event_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, all_metrics[0]); writer.writeheader(); writer.writerows(all_metrics)
    checkpoint = {"model_state": model.state_dict(), "scales": cfg["scales"],
                  "width": cfg["width"], "layers": cfg["layers"],
                  "log_conditions": model.log_conditions,
                  "terrain_fingerprint": terrain.fingerprint,
                  "training_rain_domain": {"rainfall_mm_hr": [min(e["mean_intensity_mm_hr"] for e in training), max(e["mean_intensity_mm_hr"] for e in training)],
                                           "duration_s": [min(e["duration_s"] for e in training), max(e["duration_s"] for e in training)]}}
    torch.save(checkpoint, output / "model.pt")
    split_summary = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in all_metrics if r["split"] == split]
        split_summary[split] = {"events": len(subset), "mean_loss": float(np.mean([r["loss"] for r in subset])),
                                "worst_loss": float(max(r["loss"] for r in subset)),
                                "mean_pde_loss": float(np.mean([r["pde_loss"] for r in subset])),
                                "mean_mass_loss": float(np.mean([r["mass_loss"] for r in subset]))}
    report = {"completed_steps": len(history), "requested_steps": cfg["adam_steps"],
              "elapsed_s": time.perf_counter() - started, "best_validation_loss": best,
              "splits": split_summary, "terrain": terrain.metadata(),
              "assumptions": cfg["assumptions"],
              "external_flood_observation_validation": "not_available",
              "success_thresholds": cfg["success_thresholds"]}
    thresholds = cfg["success_thresholds"]
    test = split_summary["test"]
    worst_mass = max(float(row["mass_loss"]) for row in all_metrics)
    worst_pde = max(float(row["pde_loss"]) for row in all_metrics)
    report["worst_all_event_mass_loss"] = worst_mass
    report["worst_all_event_pde_loss"] = worst_pde
    report["physics_training_passed"] = bool(test["mean_pde_loss"] <= thresholds["test_mean_pde_loss"] and
                                             test["mean_mass_loss"] <= thresholds["test_mean_mass_loss"] and
                                             worst_mass <= thresholds["worst_all_event_mass_loss"] and
                                             worst_pde <= thresholds["worst_all_event_pde_loss"])
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    from .reporting import generate_report
    generate_report(output)
    return model, report
