from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from minxiong_pinn.rainfall import add_design_events, extract_events
from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import train


ROOT = Path(__file__).resolve().parent


def main(config_path="configs/minxiong_experts.json"):
    path = (ROOT / config_path).resolve()
    master = json.loads(path.read_text(encoding="utf-8-sig")); base = path.parent
    resolve = lambda value: (base / value).resolve()
    terrain = MaskedRasterTerrain(resolve(master["terrain_tif"]), manning_n=master["manning_n"])
    observed = extract_events(resolve(master["rainfall_zip"]))
    output_root = resolve(master["output_root"]); output_root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index, expert in enumerate(master["experts"]):
        rates = expert["intensities_mm_hr"]
        selected_observed = [event for event in observed if rates[0] <= event["mean_intensity_mm_hr"] <= rates[-1]]
        events = add_design_events(selected_observed, rates, master["durations_hr"])
        cfg = copy.deepcopy(master["training"])
        cfg["seed"] += index * 1000
        cfg["extreme_total_mm"] = rates[-1] * max(master["durations_hr"]) * .9
        cfg["scales"] = {
            "x0": terrain.bounds[0], "y0": terrain.bounds[2],
            "lx": terrain.bounds[1] - terrain.bounds[0], "ly": terrain.bounds[3] - terrain.bounds[2],
            "time_s": cfg["time_end_s"], "rain_m_s": rates[-1] / 3_600_000.,
            "rain_min_m_s": rates[0] / 3_600_000., "duration_s": max(master["durations_hr"]) * 3600.,
            "duration_min_s": min(master["durations_hr"]) * 3600.,
            "depth_m": rates[-1] * max(master["durations_hr"]) / 1000.,
            "velocity_m_s": expert["velocity_m_s"], "ramp_s": 600.}
        folder = output_root / expert["name"]
        print(f"\n=== training {expert['name']} expert: {rates[0]}–{rates[-1]} mm/hr, {len(events)} scenarios ===", flush=True)
        _, report = train(terrain, events, cfg, folder)
        summaries.append({"name": expert["name"], "rainfall_mm_hr": [rates[0], rates[-1]],
                          "model": str(folder / "model.pt"), "report": report})
    summary = {"all_passed": all(item["report"]["physics_training_passed"] for item in summaries),
               "experts": summaries}
    (output_root / "expert_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/minxiong_experts.json")
