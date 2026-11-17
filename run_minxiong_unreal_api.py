"""Run the local Unreal Engine HTTP bridge for Minxiong flood prediction."""
from __future__ import annotations

import argparse
from pathlib import Path

from minxiong_pinn.unreal_api import UnrealFloodService, make_server


ROOT = Path(__file__).resolve().parent


def parser():
    value = argparse.ArgumentParser(
        description="Local HTTP API for Unreal Engine Minxiong flood prediction")
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8765)
    value.add_argument("--terrain", type=Path,
                       default=ROOT / "data" / "minxiong" / "裁剪后民雄.tif")
    value.add_argument("--model", type=Path,
                       default=ROOT / "models" / "minxiong" / "model.pt")
    value.add_argument("--expert-manifest", type=Path,
                       default=ROOT / "models" / "minxiong" / "experts" / "manifest.json")
    value.add_argument("--output-root", type=Path,
                       default=ROOT / "outputs" / "minxiong" / "unreal_predictions")
    value.add_argument("--manning-n", type=float, default=.03)
    value.add_argument("--max-frames", type=int, default=500)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    service = UnrealFloodService.from_paths(
        terrain_path=args.terrain, fallback_model_path=args.model,
        expert_manifest_path=args.expert_manifest,
        output_root=args.output_root, manning_n=args.manning_n,
        max_frames=args.max_frames)
    server = make_server(service, args.host, args.port)
    host, port = server.server_address[:2]
    domain = service.health()["active_model_domain"]
    print(f"Unreal API ready: http://{host}:{port}", flush=True)
    print(
        "Active model range: "
        f"{domain['rainfall_mm_hr'][0]:g}-{domain['rainfall_mm_hr'][1]:g} mm/hr, "
        f"rain duration {domain['duration_hours'][0]:g}-{domain['duration_hours'][1]:g} h, "
        f"simulation <= {domain['simulation_hours_max']:g} h",
        flush=True)
    if service.checkpoint.get("bundle_fallback_reason"):
        print("Expert bundle is not active; requests outside the fallback range "
              "will be rejected.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Unreal API.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

