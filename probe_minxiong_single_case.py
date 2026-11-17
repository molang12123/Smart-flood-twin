"""Overfit one fixed-rain case to separate representation from curriculum failure."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import train
from probe_minxiong_low import BASE_CONFIG, probe_config


ROOT = Path(__file__).resolve().parent
RAINFALL_MM_HR = 5.0
DURATION_HR = 6.0


def single_case_setup(physics="finite_volume"):
    if physics not in {"finite_volume", "strong", "strong_finetune",
                       "strong_local", "strong_local_finetune",
                       "strong_local_wide", "strong_plane", "strong_balanced"}:
        raise ValueError(
            "physics must be finite_volume, strong, strong_finetune, strong_local, "
            "strong_local_finetune, strong_local_wide, strong_plane, or "
            "strong_balanced")
    master = probe_config("uniform_strict")
    expert = master["experts"][0]
    cfg = copy.deepcopy(master["training"])
    cfg.update({
        "adam_steps": 1000,
        "max_seconds": 1800,
        "scenarios_per_batch": 1,
        "always_include_extreme": True,
        "stratified_scenario_anchors": False,
        "validation_interval": 100,
        "final_evaluation_repeats": 3,
        "expert_name": "single_5mm_hr_6hr",
    })
    if physics in {"strong", "strong_finetune", "strong_local",
                   "strong_local_finetune", "strong_local_wide",
                   "strong_plane", "strong_balanced"}:
        cfg.update({
            "adam_steps": 6000,
            "physics_form": "strong",
            "mass_conserving_projection": False,
            "wet_dry_front_fraction": 0.0,
            "expert_name": "single_5mm_hr_6hr_strong",
        })
    if physics in {"strong_local", "strong_local_finetune", "strong_local_wide",
                   "strong_plane"}:
        cfg.update({
            "terrain_local_coordinates": True,
            "expert_name": "single_5mm_hr_6hr_strong_local",
        })
    if physics == "strong_local_wide":
        cfg.update({
            "adam_steps": 12000,
            "max_seconds": 3600,
            "width": 192,
            "layers": 8,
            "expert_name": "single_5mm_hr_6hr_strong_local_wide",
        })
    if physics == "strong_plane":
        # Isolate the differentiable local DEM plane while retaining the
        # previously successful plain wide network and local cell coordinates.
        cfg.update({
            "adam_steps": 1000,
            "width": 192,
            "layers": 8,
            "terrain_local_plane_feature": True,
            "expert_name": "single_5mm_hr_6hr_strong_plane",
        })
    if physics == "strong_balanced":
        cfg.update({
            "adam_steps": 1000,
            "width": 192,
            "layers": 8,
            "network_architecture": "residual",
            "terrain_local_coordinates": False,
            "terrain_local_plane_feature": True,
            "conditioned_velocity_scale": True,
            "interior_points": 512,
            "validation_interior_points": 1024,
            "final_evaluation_interior_points": 2048,
            "interface_points": 256,
            "interface_high_relief_fraction": 0.5,
            "interface_h_wet_m": 0.001,
            "interface_start_step": 100,
            "interface_ramp_steps": 300,
            "expert_name": "single_5mm_hr_6hr_strong_balanced",
        })
        cfg["loss_weights"] = dict(cfg["loss_weights"], interface=0.2)
        cfg["success_thresholds"] = dict(
            cfg["success_thresholds"],
            test_mean_interface_depth_loss=0.02,
            test_mean_interface_normal_flux_loss=0.05,
            test_mean_interface_tangential_flux_loss=0.05,
            worst_all_event_interface_loss=0.1,
        )
    if physics == "strong_finetune":
        initial = (ROOT / "outputs" / "minxiong" / "v10_probes" /
                   "single_5x6_strong" / "model.pt")
        if not initial.is_file():
            raise FileNotFoundError(
                "run the strong single-case probe before strong_finetune")
        cfg.update({
            "learning_rate": 0.00001,
            "initial_model_path": str(initial),
            "expert_name": "single_5mm_hr_6hr_strong_finetune",
        })
    if physics == "strong_local_finetune":
        initial = (ROOT / "outputs" / "minxiong" / "v10_probes" /
                   "single_5x6_strong_local" / "model.pt")
        if not initial.is_file():
            raise FileNotFoundError(
                "run the strong_local single-case probe before strong_local_finetune")
        cfg.update({
            "learning_rate": 0.00001,
            "initial_model_path": str(initial),
            "expert_name": "single_5mm_hr_6hr_strong_local_finetune",
        })
    terrain = MaskedRasterTerrain(
        (BASE_CONFIG.parent / master["terrain_tif"]).resolve(),
        manning_n=master["manning_n"])
    rates = expert["intensities_mm_hr"]
    cfg["scales"] = {
        "x0": terrain.bounds[0], "y0": terrain.bounds[2],
        "lx": terrain.bounds[1] - terrain.bounds[0],
        "ly": terrain.bounds[3] - terrain.bounds[2],
        "time_s": cfg["time_end_s"],
        "rain_m_s": rates[-1] / 3_600_000.,
        "rain_min_m_s": 0.,
        "duration_s": max(master["durations_hr"]) * 3600.,
        "duration_min_s": min(master["durations_hr"]) * 3600.,
        "rain_log_reference_m_s": master["rain_reference_mm_hr"] / 3_600_000.,
        "duration_log_reference_s": master["duration_reference_hr"] * 3600.,
        "depth_m": rates[-1] * max(master["durations_hr"]) / 1000.,
        "velocity_m_s": expert["velocity_m_s"],
        "ramp_s": 600., "time_log_reference_s": 600.,
        "wet_velocity_gate_m": master["wet_gate_m"],
    }
    events = [{
        "event_id": f"single_5x6_{split}", "split": split,
        "mean_intensity_mm_hr": RAINFALL_MM_HR,
        "duration_s": DURATION_HR * 3600.,
        "total_mm": RAINFALL_MM_HR * DURATION_HR,
    } for split in ("train", "validation", "test")]
    suffixes = {
        "finite_volume": "single_5x6",
        "strong": "single_5x6_strong",
        "strong_finetune": "single_5x6_strong_finetune",
        "strong_local": "single_5x6_strong_local",
        "strong_local_finetune": "single_5x6_strong_local_finetune",
        "strong_local_wide": "single_5x6_strong_local_wide",
        "strong_plane": "single_5x6_strong_plane",
        "strong_balanced": "single_5x6_strong_balanced",
    }
    suffix = suffixes[physics]
    output = ROOT / "outputs" / "minxiong" / "v10_probes" / suffix
    return terrain, events, cfg, output


def main(physics="finite_volume"):
    terrain, events, cfg, output = single_case_setup(physics)
    _, report = train(terrain, events, cfg, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "finite_volume")
