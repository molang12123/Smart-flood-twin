from __future__ import annotations

import csv
import hashlib
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


def _fv_start_times(rng, count, end_s, dt, duration, slices=0):
    """Return repeated time slices spanning early rain, stop and recession."""
    max_start = end_s - dt
    if isinstance(slices, bool) or int(slices) != slices or slices < 0:
        raise ValueError("fv_time_slices must be a nonnegative integer")
    if slices == 0:
        return rng.uniform(0, max_start, count)
    groups = np.array_split(np.arange(count), min(int(slices), count))
    phases = ("early", "during", "stop", "post", "late")
    result = np.empty(count, dtype=np.float64)
    for index, group in enumerate(groups):
        phase = phases[index % len(phases)]
        if phase == "early":
            low, high = 0., min(max_start, 1800.)
        elif phase == "during":
            low, high = 0., min(max_start, max(0., duration - dt))
        elif phase == "stop" and dt < duration < end_s:
            low = max(0., duration - dt + 1e-6)
            high = min(max_start, duration - 1e-6)
        elif phase == "post" and duration < end_s:
            low, high = min(duration, max_start), max_start
        elif phase == "late":
            low, high = .75 * max_start, max_start
        else:
            low, high = 0., max_start
        value = low if high <= low else float(rng.uniform(low, high))
        result[group] = value
    rng.shuffle(result)
    return result


def _state_slice_times(count, end_s, duration):
    values = np.asarray((min(.5 * duration, end_s), min(duration, end_s), end_s),
                        dtype=np.float64)
    return np.resize(values, count)


def _conservative_state(state):
    h, u, v = state.unbind(dim=1)
    return torch.stack((h, h * u, h * v), dim=1)


def _hll_flux(left, right, *, axis, gravity=9.81):
    """HLL shallow-water flux for x (axis=0) or y (axis=1) faces."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (x) or 1 (y)")
    h_left, u_left, v_left = left.unbind(dim=1)
    h_right, u_right, v_right = right.unbind(dim=1)
    dry_epsilon = torch.finfo(left.dtype).eps
    u_left = torch.where(h_left > dry_epsilon, u_left, torch.zeros_like(u_left))
    v_left = torch.where(h_left > dry_epsilon, v_left, torch.zeros_like(v_left))
    u_right = torch.where(h_right > dry_epsilon, u_right, torch.zeros_like(u_right))
    v_right = torch.where(h_right > dry_epsilon, v_right, torch.zeros_like(v_right))
    left = torch.stack((h_left, u_left, v_left), dim=1)
    right = torch.stack((h_right, u_right, v_right), dim=1)
    conservative_left = _conservative_state(left)
    conservative_right = _conservative_state(right)
    if axis == 0:
        flux_left = torch.stack((h_left * u_left,
                                 h_left * u_left.square() + .5 * gravity * h_left.square(),
                                 h_left * u_left * v_left), dim=1)
        flux_right = torch.stack((h_right * u_right,
                                  h_right * u_right.square() + .5 * gravity * h_right.square(),
                                  h_right * u_right * v_right), dim=1)
        normal_left, normal_right = u_left, u_right
    else:
        flux_left = torch.stack((h_left * v_left, h_left * u_left * v_left,
                                 h_left * v_left.square() + .5 * gravity * h_left.square()), dim=1)
        flux_right = torch.stack((h_right * v_right, h_right * u_right * v_right,
                                  h_right * v_right.square() + .5 * gravity * h_right.square()), dim=1)
        normal_left, normal_right = v_left, v_right
    c_left = torch.sqrt((gravity * h_left).clamp_min(0.))
    c_right = torch.sqrt((gravity * h_right).clamp_min(0.))
    zeros = torch.zeros_like(h_left)
    speed_left = torch.minimum(zeros, torch.minimum(normal_left - c_left,
                                                    normal_right - c_right))
    speed_right = torch.maximum(zeros, torch.maximum(normal_left + c_left,
                                                     normal_right + c_right))
    denominator = (speed_right - speed_left).clamp_min(dry_epsilon)
    middle = ((speed_right[:, None] * flux_left - speed_left[:, None] * flux_right +
               (speed_left * speed_right)[:, None] *
               (conservative_right - conservative_left)) / denominator[:, None])
    result = torch.where((speed_left >= 0.)[:, None], flux_left,
                         torch.where((speed_right <= 0.)[:, None], flux_right, middle))
    both_dry = (h_left <= dry_epsilon) & (h_right <= dry_epsilon)
    return torch.where(both_dry[:, None], torch.zeros_like(result), result)


def _hydrostatic_face(left, right, z_left, z_right, *, axis, centre_side,
                      gravity=9.81):
    """Audusse hydrostatic reconstruction with centre-cell source correction."""
    if centre_side not in {"left", "right"}:
        raise ValueError("centre_side must be 'left' or 'right'")
    h_left, u_left, v_left = left.unbind(dim=1)
    h_right, u_right, v_right = right.unbind(dim=1)
    face_bed = torch.maximum(z_left, z_right)
    reconstructed_left_h = torch.relu(h_left + z_left - face_bed)
    reconstructed_right_h = torch.relu(h_right + z_right - face_bed)
    reconstructed_left = torch.stack((reconstructed_left_h, u_left, v_left), dim=1)
    reconstructed_right = torch.stack((reconstructed_right_h, u_right, v_right), dim=1)
    flux = _hll_flux(reconstructed_left, reconstructed_right, axis=axis, gravity=gravity)
    if centre_side == "left":
        centre_h, reconstructed_centre_h = h_left, reconstructed_left_h
    else:
        centre_h, reconstructed_centre_h = h_right, reconstructed_right_h
    correction = .5 * gravity * (centre_h.square() - reconstructed_centre_h.square())
    correction_vector = torch.zeros_like(flux)
    correction_vector[:, axis + 1] = correction
    return flux + correction_vector


def _fv_residuals(model, terrain, case, rng, cfg):
    """Well-balanced finite-volume residual on active DEM cells."""
    count = cfg["interior_points"]
    xy = terrain.sample_control_volumes(rng, count)
    dt = min(float(cfg.get("weak_dt_s", 600.)), cfg["time_end_s"])
    if dt <= 0 or cfg["time_end_s"] < dt:
        raise ValueError("weak_dt_s and time_end_s must define a positive interval")
    rate = case["mean_intensity_mm_hr"] / 3_600_000.
    duration = case["duration_s"]
    t0 = _fv_start_times(rng, count, cfg["time_end_s"], dt, duration,
                         cfg.get("fv_time_slices", 0))
    t1 = t0 + dt
    tm = (t0 + t1) / 2
    front_fraction = float(cfg.get("wet_dry_front_fraction", 0.))
    if not 0. <= front_fraction <= 1.:
        raise ValueError("wet_dry_front_fraction must be between zero and one")
    front_count = int(round(count * front_fraction))
    if front_count:
        accumulated = rate * np.minimum(tm[:front_count], duration)
        levels = terrain.water_level_for_mean_depth(accumulated)
        xy[:front_count] = terrain.sample_control_volumes_near_levels(
            rng, levels, window_cells=cfg.get("wet_dry_front_window_cells", 256))

    def make(coords, times):
        p = np.empty((len(coords), 5), dtype=np.float64)
        p[:, :2] = coords; p[:, 2] = times; p[:, 3] = rate; p[:, 4] = duration
        return _tensor(p)

    centre0 = model(make(xy, t0)); centre1 = model(make(xy, t1))
    east_xy = xy + np.array([terrain.dx, 0.]); west_xy = xy - np.array([terrain.dx, 0.])
    north_xy = xy + np.array([0., terrain.dy]); south_xy = xy - np.array([0., terrain.dy])
    mid_coords = np.concatenate((xy, east_xy, west_xy, north_xy, south_xy), axis=0)
    mid_values = model(make(mid_coords, np.tile(tm, 5)))
    centre_mid, east, west, north, south = torch.split(mid_values, count, dim=0)
    z_centre = centre_mid.new_tensor(terrain.query(xy[:, 0], xy[:, 1])["z"])
    z_east = centre_mid.new_tensor(terrain.query(east_xy[:, 0], east_xy[:, 1])["z"])
    z_west = centre_mid.new_tensor(terrain.query(west_xy[:, 0], west_xy[:, 1])["z"])
    z_north = centre_mid.new_tensor(terrain.query(north_xy[:, 0], north_xy[:, 1])["z"])
    z_south = centre_mid.new_tensor(terrain.query(south_xy[:, 0], south_xy[:, 1])["z"])
    gravity = 9.81
    flux_east = _hydrostatic_face(centre_mid, east, z_centre, z_east,
                                  axis=0, centre_side="left", gravity=gravity)
    flux_west = _hydrostatic_face(west, centre_mid, z_west, z_centre,
                                  axis=0, centre_side="right", gravity=gravity)
    flux_north = _hydrostatic_face(centre_mid, north, z_centre, z_north,
                                   axis=1, centre_side="left", gravity=gravity)
    flux_south = _hydrostatic_face(south, centre_mid, z_south, z_centre,
                                   axis=1, centre_side="right", gravity=gravity)
    residual = ((_conservative_state(centre1) - _conservative_state(centre0)) / dt +
                (flux_east - flux_west) / terrain.dx +
                (flux_north - flux_south) / terrain.dy)
    hmid, umid, vmid = centre_mid.unbind(dim=1)
    n = residual.new_tensor(terrain.query(xy[:, 0], xy[:, 1])["n"])
    speed = torch.sqrt(umid.square() + vmid.square() + 1e-12)
    friction = (gravity * n.square() * speed /
                hmid.clamp_min(cfg["h_regularization_m"]).pow(1 / 3))
    rain_seconds = np.maximum(0., np.minimum(t1, duration) - np.minimum(t0, duration))
    mass = residual[:, 0] - residual.new_tensor(rate * rain_seconds / dt)
    momentum_x = residual[:, 1] + friction * umid
    momentum_y = residual[:, 2] + friction * vmid
    return torch.stack((mass, momentum_x, momentum_y), dim=1), rate, duration


def _loss(model, terrain, cases, rng, cfg, *, robust_pde=False):
    pde_losses, pde_objectives, pde_components, bc_losses, mass_losses = [], [], [], [], []
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
        pde_objectives.append(
            torch.log1p(normalized_square).mean() if robust_pde
            else normalized_square.mean())
        pde_components.append(normalized_square.mean(dim=0))

        xy, normals = terrain.sample_boundary(rng, cfg["boundary_points"])
        b = np.empty((len(xy), 5)); b[:, :2] = xy
        if cfg.get("fv_time_slices", 0) and cfg.get("physics_form") == "finite_volume":
            b[:, 2] = _state_slice_times(len(xy), end_s, points[0, 4])
        else:
            b[:, 2] = rng.uniform(0, end_s, len(xy))
        b[:, 3] = points[0, 3]; b[:, 4] = points[0, 4]
        pred_b = model(_tensor(b))
        normal = (pred_b[:, 1:] * _tensor(normals)).sum(dim=1)
        bc_losses.append((normal / model.scales["velocity_m_s"]).square().mean())

        event_mass = []
        audit_times = sorted(set((min(points[0, 4] * .5, end_s),
                                  min(points[0, 4], end_s), end_s)))
        for audit_time in audit_times:
            expected = points[0, 3] * min(audit_time, points[0, 4])
            if (getattr(model, "exact_hydrostatic_mass", False) or
                    getattr(model, "condition_weighted_mass_projection", False)):
                # The static prior is exact; multiplicative corrections use
                # hydrostatic-volume-weighted quadrature and exact full-DEM
                # deployment renormalisation. A sparse area audit is badly
                # biased when only a few lowest cells are wet.
                event_mass.append(raw.sum() * 0.)
                continue
            if cfg.get("stratified_mass_sampling", False) and hasattr(terrain, "sample_mass_points"):
                xy_m, mass_weights = terrain.sample_mass_points(
                    rng, cfg["mass_points"], strata=cfg.get("mass_sampling_strata", 16))
            else:
                xy_m = terrain.sample_points(rng, cfg["mass_points"])
                mass_weights = None
            m = np.empty((len(xy_m), 5)); m[:, :2] = xy_m
            m[:, 2] = audit_time; m[:, 3] = points[0, 3]; m[:, 4] = points[0, 4]
            depths = model(_tensor(m))[:, 0]
            mean_depth = (depths.mean() if mass_weights is None else
                          (depths * depths.new_tensor(mass_weights)).sum())
            # Relative volume error at rain-midpoint, stop time and horizon.
            event_mass.append(((mean_depth - expected) / max(expected, 1e-6)).square())
        mass_losses.append(torch.stack(event_mass).mean())
    component_mean = torch.stack(pde_components).mean(dim=0)
    parts = {"pde": torch.stack(pde_losses).mean(),
             "pde_objective": torch.stack(pde_objectives).mean(),
             "pde_continuity": component_mean[0],
             "pde_momentum_x": component_mean[1],
             "pde_momentum_y": component_mean[2],
             "boundary": torch.stack(bc_losses).mean(),
             "mass": torch.stack(mass_losses).mean()}
    total = (cfg["loss_weights"]["pde"] * parts["pde_objective"] +
             cfg["loss_weights"]["boundary"] * parts["boundary"] +
             cfg["loss_weights"]["mass"] * parts["mass"])
    return total, parts


def _event_evaluation_seed(base_seed, event, repeat):
    signature = json.dumps({
        "base_seed": int(base_seed), "repeat": int(repeat),
        "event_id": event.get("event_id"), "split": event.get("split"),
        "rainfall_mm_hr": float(event["mean_intensity_mm_hr"]),
        "duration_s": float(event["duration_s"]),
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(signature.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def evaluate(model, terrain, events, cfg, *, seed=9001, repeats=1,
             interior_points=None):
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats <= 0:
        raise ValueError("evaluation repeats must be a positive integer")
    eval_cfg = cfg
    if interior_points is not None:
        if (isinstance(interior_points, bool) or int(interior_points) != interior_points or
                interior_points <= 0):
            raise ValueError("evaluation interior_points must be a positive integer")
        eval_cfg = dict(cfg)
        eval_cfg["interior_points"] = int(interior_points)
    rows = []
    for case in events:
        results = []
        for repeat in range(int(repeats)):
            rng = np.random.default_rng(_event_evaluation_seed(seed, case, repeat))
            loss, parts = _loss(model, terrain, [case], rng, eval_cfg,
                                robust_pde=False)
            results.append((float(loss.detach()),
                            {key: float(value.detach()) for key, value in parts.items()}))
        rows.append({"event_id": case["event_id"], "split": case["split"],
                     "rainfall_mm_hr": case["mean_intensity_mm_hr"],
                     "duration_s": case["duration_s"],
                     "loss": float(np.mean([result[0] for result in results])),
                     **{f"{key}_loss": float(np.mean([
                         result[1][key] for result in results]))
                        for key in results[0][1]}})
    return rows


def _training_identity(terrain, events, cfg):
    """Describe every input that can change a resumed optimization result.

    Step/time limits and checkpoint cadence are deliberately operational: a
    stopped run may extend them without invalidating its optimizer trajectory.
    Everything else is compared exactly before any saved state is loaded.
    """
    operational_keys = {"adam_steps", "max_seconds", "checkpoint_interval",
                        "resume_training"}
    config = {key: cfg[key] for key in sorted(cfg) if key not in operational_keys}
    event_fields = ("event_id", "split", "mean_intensity_mm_hr", "duration_s",
                    "total_mm")
    event_signatures = [
        {key: event.get(key) for key in event_fields}
        for event in events
    ]
    return {"terrain_fingerprint": terrain.fingerprint,
            "events": event_signatures, "config": config}


def _scheduled_hydrostatic_blend(cfg, step):
    target = float(cfg.get("hydrostatic_learned_blend", 1.))
    start = cfg.get("hydrostatic_blend_start_step", 0)
    ramp = cfg.get("hydrostatic_blend_ramp_steps", 0)
    if (isinstance(start, bool) or int(start) != start or start < 0 or
            isinstance(ramp, bool) or int(ramp) != ramp or ramp < 0):
        raise ValueError("hydrostatic blend schedule steps must be nonnegative integers")
    if not cfg.get("hydrostatic_depth_baseline", False):
        return target
    if step <= int(start):
        return 0.
    if int(ramp) == 0:
        return target
    progress = min(1., (step - int(start)) / int(ramp))
    return target * progress


def _clip_gradient_norms(model, cfg):
    """Clip independent output networks without one suppressing the other."""
    limit = float(cfg["gradient_clip"])
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("gradient_clip must be positive and finite")
    if not cfg.get("separate_output_gradient_clipping", False):
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
        return float(total), None, None
    if not getattr(model, "separate_output_networks", False):
        raise ValueError(
            "separate output gradient clipping requires separate output networks")
    depth_limit = float(cfg.get("depth_gradient_clip", limit))
    velocity_limit = float(cfg.get("velocity_gradient_clip", limit))
    if (not math.isfinite(depth_limit) or depth_limit <= 0 or
            not math.isfinite(velocity_limit) or velocity_limit <= 0):
        raise ValueError("depth and velocity gradient clips must be positive and finite")
    depth = float(torch.nn.utils.clip_grad_norm_(
        model.depth_net.parameters(), depth_limit))
    velocity = float(torch.nn.utils.clip_grad_norm_(
        model.velocity_net.parameters(), velocity_limit))
    return math.hypot(depth, velocity), depth, velocity


def _atomic_torch_save(value, path):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary); temporary.replace(path)


def _save_training_state(path, *, identity, model, optimizer, step, best,
                         best_state, best_blend, current_blend, history, rng,
                         elapsed_s):
    _atomic_torch_save({"schema_version": 2, "identity": identity,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": int(step), "best": float(best),
                        "best_state": best_state, "best_blend": float(best_blend),
                        "current_blend": float(current_blend), "history": history,
                        "rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(),
                        "elapsed_s": float(elapsed_s)}, path)


def train(terrain, events, cfg: dict, output_dir: str | Path):
    output = Path(output_dir)
    state_path = output / "training_state.pt"
    resume_requested = bool(cfg.get("resume_training", False))
    resume_from_state = output.exists() and resume_requested and state_path.is_file()
    if output.exists():
        if not resume_from_state:
            raise FileExistsError(f"training output already exists: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.set_num_threads(cfg["threads"])
    model = ConditionalFloodPINN(
        cfg["scales"], cfg["width"], cfg["layers"], cfg.get("log_conditions", False),
        log1p_conditions=cfg.get("log1p_conditions", False),
        wet_velocity_gate=cfg.get("wet_velocity_gate", False),
        terrain_features=cfg.get("terrain_features", False), terrain=terrain,
        enhanced_time_features=cfg.get("enhanced_time_features", False),
        mass_conserving_projection=cfg.get("mass_conserving_projection", False),
        mass_projection_anchor_points=cfg.get("mass_projection_anchor_points", 512),
        mass_projection_strata=cfg.get("mass_projection_strata", 32),
        mass_projection_seed=cfg.get("mass_projection_seed", 20261113),
        separate_output_networks=cfg.get("separate_output_networks", False),
        hard_mask_boundary=cfg.get("hard_mask_boundary", False),
        hydrostatic_depth_baseline=cfg.get("hydrostatic_depth_baseline", False),
        hydrostatic_learned_blend=cfg.get("hydrostatic_learned_blend", 1.0),
        hydrostatic_correction_mode=cfg.get("hydrostatic_correction_mode", "convex"),
        hydrostatic_uniform_floor=cfg.get("hydrostatic_uniform_floor", 0.0),
        terrain_local_coordinates=cfg.get("terrain_local_coordinates", False))
    if (getattr(model, "mass_conserving_projection", False) and
            cfg.get("physics_form", "strong") != "finite_volume"):
        raise ValueError("mass-conserving projection requires finite_volume physics")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    training = [e for e in events if e["split"] == "train"]
    validation = [e for e in events if e["split"] == "validation"]
    testing = [e for e in events if e["split"] == "test"]
    if not training or not validation or not testing:
        raise ValueError("training, validation, and test event splits must all be nonempty")
    identity = _training_identity(terrain, events, cfg)
    identity["mass_projection_anchor_fingerprint"] = getattr(
        model, "mass_projection_anchor_fingerprint", None)
    current_blend = float(getattr(model, "hydrostatic_learned_blend", 1.))
    history, best, best_state, best_blend = [], math.inf, None, current_blend
    previous_elapsed, start_step = 0., 1
    if resume_from_state:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("schema_version") != 2 or state.get("identity") != identity:
            raise ValueError("training checkpoint does not match terrain, events, or configuration")
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        history = list(state["history"]); best = float(state["best"])
        best_state = state.get("best_state"); rng.bit_generator.state = state["rng_state"]
        best_blend = float(state.get("best_blend", current_blend))
        current_blend = float(state.get("current_blend", current_blend))
        model.hydrostatic_learned_blend = current_blend
        torch.set_rng_state(state["torch_rng_state"])
        previous_elapsed = float(state.get("elapsed_s", 0.)); start_step = int(state["step"]) + 1
        print(f"resuming training at step {start_step}", flush=True)
    else:
        if cfg.get("initial_model_path"):
            initial = torch.load(cfg["initial_model_path"], map_location="cpu", weights_only=True)
            if initial["terrain_fingerprint"] != terrain.fingerprint:
                raise ValueError("initial checkpoint terrain does not match")
            model.load_state_dict(initial["model_state"])
        elif cfg.get("baseline_initialization", False):
            model.zero_output_layers()
    started = time.perf_counter() - previous_elapsed
    extreme = max(training, key=lambda event: event["total_mm"])
    zero_rain = [event for event in training if event["mean_intensity_mm_hr"] == 0.]
    maximum_duration = max(event["duration_s"] for event in training)
    long_duration = [event for event in training
                     if event["duration_s"] >= .75 * maximum_duration and event is not extreme]
    low_rain_limit = float(cfg.get("low_rain_anchor_max_mm_hr", 1.))
    if not math.isfinite(low_rain_limit) or low_rain_limit <= 0:
        raise ValueError("low_rain_anchor_max_mm_hr must be positive and finite")
    low_rain = [event for event in training
                if 0. < event["mean_intensity_mm_hr"] <= low_rain_limit]
    short_duration_max_s = float(cfg.get("short_duration_anchor_max_hr", 2.)) * 3600.
    if not math.isfinite(short_duration_max_s) or short_duration_max_s <= 0:
        raise ValueError("short_duration_anchor_max_hr must be positive and finite")
    short_duration = [event for event in training
                      if event["duration_s"] <= short_duration_max_s and
                      event["mean_intensity_mm_hr"] > low_rain_limit]
    probabilities = np.asarray([math.sqrt(max(event["total_mm"], .1)) for event in training])
    probabilities /= probabilities.sum()
    last_step = start_step - 1
    for step in range(start_step, cfg["adam_steps"] + 1):
        last_step = step
        current_blend = _scheduled_hydrostatic_blend(cfg, step)
        model.hydrostatic_learned_blend = current_blend
        batch_size = min(cfg["scenarios_per_batch"], len(training))
        sampling = cfg.get("scenario_sampling", "extreme_weighted")
        if sampling == "uniform":
            anchors = []
            if cfg.get("always_include_zero_rain", False) and zero_rain:
                anchors.append(zero_rain[int(rng.integers(0, len(zero_rain)))])
            if cfg.get("always_include_extreme", False) and extreme not in anchors:
                anchors.append(extreme)
            if cfg.get("stratified_scenario_anchors", False):
                for pool in (long_duration, low_rain, short_duration):
                    available = [event for event in pool if event not in anchors]
                    if available and len(anchors) < batch_size:
                        anchors.append(available[int(rng.integers(0, len(available)))])
            anchors = anchors[:batch_size]
            eligible = [event for event in training if event not in anchors]
            remaining = batch_size - len(anchors)
            selected = (rng.choice(len(eligible), remaining, replace=False)
                        if remaining else np.empty(0, dtype=int))
            chosen = anchors + [eligible[int(i)] for i in selected]
        elif sampling == "extreme_weighted" and batch_size == 1:
            chosen = [extreme]
        elif sampling == "extreme_weighted":
            eligible = [i for i, event in enumerate(training) if event is not extreme]
            p = probabilities[eligible]; p /= p.sum()
            selected = rng.choice(eligible, batch_size - 1, replace=False, p=p)
            chosen = [extreme] + [training[i] for i in selected]
        else:
            raise ValueError(f"unsupported scenario_sampling: {sampling}")
        optimizer.zero_grad(set_to_none=True)
        robust_pde = step <= int(cfg.get("robust_pde_warmup_steps", 0))
        total, parts = _loss(model, terrain, chosen, rng, cfg,
                             robust_pde=robust_pde)
        if not torch.isfinite(total): raise FloatingPointError("training loss became nonfinite")
        total.backward()
        depth_warmup = step <= int(cfg.get("depth_warmup_steps", 0))
        depth_gradient_scale = float(cfg.get("depth_gradient_scale", 1.))
        if not 0 < depth_gradient_scale <= 1:
            raise ValueError("depth_gradient_scale must be in (0, 1]")
        if depth_warmup or depth_gradient_scale < 1.:
            scale = 0. if depth_warmup else depth_gradient_scale
            model.scale_depth_gradients(scale)
        grad, depth_grad, velocity_grad = _clip_gradient_norms(model, cfg)
        optimizer.step()
        record = {"step": step, "elapsed_s": time.perf_counter() - started,
                  "total_loss": float(total.detach()), "gradient_norm": grad,
                  "depth_warmup": depth_warmup,
                  "robust_pde_warmup": robust_pde,
                  "hydrostatic_learned_blend": current_blend,
                  **{f"{k}_loss": float(v.detach()) for k, v in parts.items()}}
        if depth_grad is not None:
            record["depth_gradient_norm"] = depth_grad
            record["velocity_gradient_norm"] = velocity_grad
        history.append(record)
        validate = step == 1 or step % cfg["validation_interval"] == 0 or step == cfg["adam_steps"]
        if validate:
            validation_seed = int(cfg.get("validation_seed", cfg["seed"] + 50_000))
            metrics = evaluate(
                model, terrain, validation, cfg, seed=validation_seed,
                repeats=cfg.get("validation_repeats", 1),
                interior_points=cfg.get("validation_interior_points"))
            score = float(np.mean([r["loss"] for r in metrics])); record["validation_loss"] = score
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_blend = current_blend
            phase = (("robust-" if robust_pde else "") +
                     ("velocity-warmup" if depth_warmup else "joint"))
            print(f"step={step:5d} phase={phase} train={record['total_loss']:.5g} val={score:.5g} "
                  f"pde={record['pde_loss']:.5g} mass={record['mass_loss']:.5g}", flush=True)
        checkpoint_interval = int(cfg.get("checkpoint_interval", cfg["validation_interval"]))
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if validate or step % checkpoint_interval == 0:
            _save_training_state(state_path, identity=identity, model=model, optimizer=optimizer,
                                 step=step, best=best, best_state=best_state,
                                 best_blend=best_blend, current_blend=current_blend,
                                 history=history, rng=rng,
                                 elapsed_s=time.perf_counter() - started)
        if time.perf_counter() - started >= cfg["max_seconds"]:
            break
    if last_step >= 1:
        _save_training_state(state_path, identity=identity, model=model, optimizer=optimizer,
                             step=last_step, best=best, best_state=best_state,
                             best_blend=best_blend, current_blend=current_blend,
                             history=history, rng=rng,
                             elapsed_s=time.perf_counter() - started)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.hydrostatic_learned_blend = best_blend
    all_metrics = evaluate(
        model, terrain, events, cfg, seed=cfg["seed"] + 9999,
        repeats=cfg.get("final_evaluation_repeats", 1),
        interior_points=cfg.get("final_evaluation_interior_points"))
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({k for row in history for k in row}); writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(history)
    with (output / "event_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, all_metrics[0]); writer.writeheader(); writer.writerows(all_metrics)
    checkpoint = {"model_state": model.state_dict(), "scales": cfg["scales"],
                  "width": cfg["width"], "layers": cfg["layers"],
                  "log_conditions": model.log_conditions,
                  "log1p_conditions": getattr(model, "log1p_conditions", False),
                  "wet_velocity_gate": getattr(model, "wet_velocity_gate", False),
                  "terrain_features": getattr(model, "terrain_features", False),
                  "enhanced_time_features": getattr(model, "enhanced_time_features", False),
                  "mass_conserving_projection": getattr(model, "mass_conserving_projection", False),
                  "mass_projection_anchor_points": getattr(model, "mass_projection_anchor_points", 512),
                  "mass_projection_strata": getattr(model, "mass_projection_strata", 32),
                  "mass_projection_seed": getattr(model, "mass_projection_seed", 20261113),
                  "mass_projection_anchor_fingerprint": getattr(model, "mass_projection_anchor_fingerprint", None),
                  "separate_output_networks": getattr(model, "separate_output_networks", False),
                  "hard_mask_boundary": getattr(model, "hard_mask_boundary", False),
                  "hydrostatic_depth_baseline": getattr(model, "hydrostatic_depth_baseline", False),
                  "hydrostatic_learned_blend": getattr(model, "hydrostatic_learned_blend", 1.0),
                  "hydrostatic_correction_mode": getattr(model, "hydrostatic_correction_mode", "convex"),
                  "hydrostatic_uniform_floor": getattr(model, "hydrostatic_uniform_floor", 0.0),
                  "terrain_local_coordinates": getattr(
                      model, "terrain_local_coordinates", False),
                  "terrain_feature_version": (getattr(model, "terrain_feature_metadata", None) or {}).get("version"),
                  "terrain_fingerprint": terrain.fingerprint,
                  "training_rain_domain": {"rainfall_mm_hr": [min(e["mean_intensity_mm_hr"] for e in training), max(e["mean_intensity_mm_hr"] for e in training)],
                                           "duration_s": [min(e["duration_s"] for e in training), max(e["duration_s"] for e in training)]}}
    torch.save(checkpoint, output / "model.pt")
    split_summary = {}
    velocity_scale = float(model.scales["velocity_m_s"])
    for split in ("train", "validation", "test"):
        subset = [r for r in all_metrics if r["split"] == split]
        mean_boundary_loss = float(np.mean([r["boundary_loss"] for r in subset]))
        split_summary[split] = {"events": len(subset), "mean_loss": float(np.mean([r["loss"] for r in subset])),
                                "worst_loss": float(max(r["loss"] for r in subset)),
                                "mean_pde_loss": float(np.mean([r["pde_loss"] for r in subset])),
                                "mean_continuity_loss": float(np.mean([r["pde_continuity_loss"] for r in subset])),
                                "mean_momentum_x_loss": float(np.mean([r["pde_momentum_x_loss"] for r in subset])),
                                "mean_momentum_y_loss": float(np.mean([r["pde_momentum_y_loss"] for r in subset])),
                                "mean_boundary_loss": mean_boundary_loss,
                                "rms_boundary_normal_velocity_m_s": velocity_scale * math.sqrt(mean_boundary_loss),
                                "mean_mass_loss": float(np.mean([r["mass_loss"] for r in subset]))}
    report = {"completed_steps": len(history), "requested_steps": cfg["adam_steps"],
              "elapsed_s": time.perf_counter() - started, "best_validation_loss": best,
              "splits": split_summary, "terrain": terrain.metadata(),
              "baseline_initialization": bool(cfg.get("baseline_initialization", False)),
              "depth_warmup_steps": int(cfg.get("depth_warmup_steps", 0)),
              "depth_gradient_scale": float(cfg.get("depth_gradient_scale", 1.)),
              "separate_output_gradient_clipping": bool(
                  cfg.get("separate_output_gradient_clipping", False)),
              "depth_gradient_clip": float(cfg.get(
                  "depth_gradient_clip", cfg["gradient_clip"])),
              "velocity_gradient_clip": float(cfg.get(
                  "velocity_gradient_clip", cfg["gradient_clip"])),
              "hydrostatic_learned_blend": float(model.hydrostatic_learned_blend),
              "terrain_local_coordinates": bool(getattr(
                  model, "terrain_local_coordinates", False)),
              "evaluation_sampling": {
                  "wet_dry_front_fraction": float(cfg.get("wet_dry_front_fraction", 0.)),
                  "wet_dry_front_window_cells": int(cfg.get("wet_dry_front_window_cells", 256)),
                  "validation_repeats": int(cfg.get("validation_repeats", 1)),
                  "final_evaluation_repeats": int(cfg.get("final_evaluation_repeats", 1)),
                  "description": "configured fraction of control volumes targets elevations near the hydrostatic wet/dry front; the remainder is uniform over valid interior cells",
              },
              "mass_constraint_method": (
                  "condition-weighted hydrostatic projection; sparse mass loss is identically zero and deployment uses exact full-DEM renormalization"
                  if getattr(model, "condition_weighted_mass_projection", False)
                  else "stratified sampled domain mean"),
              "resumable_checkpoint": str(state_path),
              "assumptions": cfg["assumptions"],
              "external_flood_observation_validation": "not_available",
              "success_thresholds": cfg["success_thresholds"]}
    thresholds = cfg["success_thresholds"]
    test = split_summary["test"]
    worst_mass = max(float(row["mass_loss"]) for row in all_metrics)
    worst_pde = max(float(row["pde_loss"]) for row in all_metrics)
    worst_continuity = max(float(row["pde_continuity_loss"]) for row in all_metrics)
    worst_boundary = velocity_scale * math.sqrt(
        max(float(row["boundary_loss"]) for row in all_metrics))
    report["worst_all_event_mass_loss"] = worst_mass
    report["worst_all_event_pde_loss"] = worst_pde
    report["worst_all_event_continuity_loss"] = worst_continuity
    report["worst_all_event_boundary_rms_normal_velocity_m_s"] = worst_boundary
    report["physics_training_passed"] = bool(test["mean_pde_loss"] <= thresholds["test_mean_pde_loss"] and
                                             test["mean_continuity_loss"] <= thresholds.get("test_mean_continuity_loss", math.inf) and
                                             test["rms_boundary_normal_velocity_m_s"] <= thresholds.get("test_boundary_rms_normal_velocity_m_s", math.inf) and
                                             test["mean_mass_loss"] <= thresholds["test_mean_mass_loss"] and
                                             worst_mass <= thresholds["worst_all_event_mass_loss"] and
                                             worst_pde <= thresholds["worst_all_event_pde_loss"] and
                                             worst_continuity <= thresholds.get("worst_all_event_continuity_loss", math.inf) and
                                             worst_boundary <= thresholds.get("worst_all_event_boundary_rms_normal_velocity_m_s", math.inf))
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    from .reporting import generate_report
    generate_report(output)
    return model, report
