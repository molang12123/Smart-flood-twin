from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from minxiong_pinn.rainfall import add_design_events, extract_events
from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import train


ROOT = Path(__file__).resolve().parent


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
        cfg = copy.deepcopy(master["training"])
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
        _, report = train(terrain, events, cfg, folder)
        summaries.append({"name": expert["name"], "rainfall_mm_hr": [rates[0], rates[-1]],
                          "duration_hr": [duration_min_hr, duration_max_hr],
                          "simulation_horizon_hr": cfg["time_end_s"] / 3600.,
                          "observed_events_used": len(selected_observed),
                          "model": str(folder / "model.pt"), "report": report})
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
