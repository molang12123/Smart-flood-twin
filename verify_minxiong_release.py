"""Read-only verification of the deployed Minxiong expert bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minxiong_pinn.expert_bundle import load_expert_bundle
from minxiong_pinn.terrain import MaskedRasterTerrain


ROOT = Path(__file__).resolve().parent


def parser():
    value = argparse.ArgumentParser(
        description="Verify model hashes, audit evidence, DEM identity and domain")
    value.add_argument(
        "--terrain", type=Path,
        default=ROOT / "data" / "minxiong" / "裁剪后民雄.tif")
    value.add_argument(
        "--manifest", type=Path,
        default=ROOT / "models" / "minxiong" / "experts" / "manifest.json")
    value.add_argument("--manning-n", type=float, default=.03)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    terrain = MaskedRasterTerrain(args.terrain, manning_n=args.manning_n)
    model, checkpoint = load_expert_bundle(args.manifest, terrain)
    probes = (0., 30., 40., 50., 300.)
    result = {
        "verified": True,
        "bundle_id": checkpoint["bundle_id"],
        "acceptance_profile": checkpoint["acceptance_profile"],
        "terrain_active_cells": terrain.active_count,
        "training_rain_domain": checkpoint["training_rain_domain"],
        "simulation_hours_max": model.scales["time_s"] / 3600.,
        "routing": checkpoint["routing"],
        "routing_probes": {
            f"{rain:g}": {
                "label": model.routing_label_for_rainfall(rain),
                "weights": model.expert_weights_for_rainfall(rain),
            }
            for rain in probes
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
