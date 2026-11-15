"""Overfit one fixed-rain case to separate representation from curriculum failure."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import train
from probe_minxiong_low import BASE_CONFIG, probe_config


ROOT = Path(__file__).resolve().parent
RAINFALL_MM_HR = 5.0
DURATION_HR = 6.0


def single_case_setup():
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
    output = ROOT / "outputs" / "minxiong" / "v10_probes" / "single_5x6"
    return terrain, events, cfg, output


def main():
    terrain, events, cfg, output = single_case_setup()
    _, report = train(terrain, events, cfg, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
