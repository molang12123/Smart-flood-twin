"""Press F5 in VS Code to train or predict with the fixed Minxiong DEM."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from minxiong_pinn.predict import load_model, predict
from minxiong_pinn.rainfall import extract_events
from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import train


ROOT = Path(__file__).resolve().parent


def main(config_path="configs/minxiong.json"):
    path = (ROOT / config_path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    base = path.parent
    resolve = lambda value: (base / value).resolve()
    terrain = MaskedRasterTerrain(resolve(cfg["terrain_tif"]), manning_n=cfg["manning_n"])
    events = extract_events(resolve(cfg["rainfall_zip"]), minimum_total_mm=cfg["minimum_event_total_mm"])
    catalog = resolve(cfg["event_catalog_csv"]); catalog.parent.mkdir(parents=True, exist_ok=True)
    with catalog.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, events[0]); writer.writeheader(); writer.writerows(events)
    action = cfg["action"]
    if action in {"train", "train_and_predict"}:
        model, report = train(terrain, events, cfg["training"], resolve(cfg["training_output_dir"]))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["physics_training_passed"]:
            raise RuntimeError("训练已完成，但独立测试物理门槛尚未通过；请查看 training_report.json")
        checkpoint = {"training_rain_domain": torch_domain(events), "terrain_fingerprint": terrain.fingerprint}
        checkpoint_path = resolve(cfg["training_output_dir"]) / "model.pt"
    elif action == "predict":
        checkpoint_path = resolve(cfg["model_path"])
        model, checkpoint = load_model(checkpoint_path, terrain)
    else:
        raise ValueError("action must be train, predict, or train_and_predict")
    if action in {"predict", "train_and_predict"}:
        if action == "train_and_predict": model, checkpoint = load_model(checkpoint_path, terrain)
        request = cfg["prediction"]
        target = resolve(request["output_dir"])
        if target.exists():
            target = target.with_name(target.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        summary = predict(model, checkpoint, terrain, request["rainfall_mm_hr"],
                          request["duration_min"] * 60., [value * 60. for value in request["times_min"]],
                          target, batch_size=request["batch_size"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def torch_domain(events):
    training = [e for e in events if e["split"] == "train"]
    return {"rainfall_mm_hr": [min(e["mean_intensity_mm_hr"] for e in training), max(e["mean_intensity_mm_hr"] for e in training)],
            "duration_s": [min(e["duration_s"] for e in training), max(e["duration_s"] for e in training)]}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/minxiong.json")
