"""Explicit CLI actions; importing the package never starts training."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

from .contracts import PredictionError, write_json
from .terrain import TerrainGrid


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("terrain"), dict):
        raise ValueError("Config and its terrain entry must be JSON objects")
    terrain = config["terrain"]
    def resolve(key):
        value = terrain.get(key)
        return str((path.parent / value).resolve()) if value else None
    grid = TerrainGrid.from_files(resolve("csv"), resolve("metadata"),
                                  roughness_path=resolve("roughness"),
                                  manning_n=terrain.get("manning_n"))
    if config.get("reference_csv"):
        config["reference_csv"] = str((path.parent / config["reference_csv"]).resolve())
    return grid, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate-input", help="Check standardized files without training")
    check.add_argument("config")
    train = sub.add_parser("train", help="Explicit bounded training; no default full-length run")
    train.add_argument("config")
    train.add_argument("--output", required=True)
    predict = sub.add_parser("predict", help="Inference only; no optimizer/training call")
    predict.add_argument("config")
    predict.add_argument("--model", required=True)
    predict.add_argument("--rainfall-mm-hr", type=float, required=True)
    predict.add_argument("--duration-min", type=float, required=True)
    predict.add_argument("--times", type=float, nargs="+", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--requested-only", action="store_true")
    predict.add_argument("--allow-unvalidated", action="store_true", help="Research only, clearly labeled")
    predict.add_argument("--batch-size", type=int, default=4096)
    assessment = sub.add_parser("assess", help="Compare reports to predeclared diagnostic limits")
    assessment.add_argument("--metrics", required=True)
    assessment.add_argument("--simulation", required=True)
    assessment.add_argument("--standards", required=True)
    assessment.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # Writable non-secret cache, no matplotlib attempt to write into read-only HOME.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/flood-pinn-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/flood-pinn-cache")
    try:
        if args.command == "assess":
            from .assessment import assess
            assessment = assess(json.loads(Path(args.metrics).read_text()),
                                json.loads(Path(args.simulation).read_text()),
                                json.loads(Path(args.standards).read_text()))
            path = Path(args.output)
            if path.exists():
                raise FileExistsError("Existing assessment is preserved; select a new output file")
            write_json(path, assessment)
            print(json.dumps({"status": assessment["status"], "output": str(path)}))
            return 0 if assessment["status"] == "passed" else 2
        grid, config = load_config(args.config)
        if args.command == "validate-input":
            from .training import validate_config, _read_reference
            checked = validate_config(config)
            references, _ = _read_reference(checked, grid)
            print(json.dumps({"status": "valid", "grid_id": grid.metadata["grid_id"],
                              "fingerprint": grid.fingerprint, "cells": grid.nx*grid.ny,
                              "synthetic": grid.metadata["synthetic"], "reference_samples": len(references)}))
        elif args.command == "train":
            folder = Path(args.output)
            if folder.exists() and any(folder.iterdir()):
                raise ValueError("Training output directory must be empty; previous runs are preserved")
            from .training import train as run_train
            from .checkpoint import save_checkpoint
            model, report = run_train(grid, config, folder)
            if not report.get("training_completed"):
                print(json.dumps({"status": "training_not_completed", "report": report}, allow_nan=False))
                return 2
            metadata = save_checkpoint(model, grid, config, folder / "model", report)
            print(json.dumps({"status": "training_completed", "model_id": metadata["model_id"],
                              "validated_domain": metadata["validated_domain"], "report": str(folder / "training_report.json")}))
        else:
            from .predict import Predictor
            start = time.perf_counter()
            predictor = Predictor.from_checkpoint(args.model, grid, conditions=config["conditions"])
            load_seconds = time.perf_counter()-start
            result = predictor.predict(
                args.rainfall_mm_hr, args.duration_min, args.times,
                output_mode="requested" if args.requested_only else "complete",
                batch_size=args.batch_size, output_dir=args.output,
                allow_unvalidated=args.allow_unvalidated)
            for _ in result.frames_iterator:
                pass
            summary = result.finalize()
            import resource
            write_json(Path(args.output) / "benchmark_report.json", {
                "device": str(next(predictor.model.parameters()).device), "load_seconds": load_seconds,
                "inference_export_seconds": time.perf_counter()-start-load_seconds,
                "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "rss_scope": "whole process including loading and plotting, Linux KiB",
                "cells": grid.nx*grid.ny, "frames": len(result.actual_output_times_s),
                "batch_size": args.batch_size, "includes_exchange_and_export": True})
            from .export import refresh_manifest
            refresh_manifest(args.output)
            print(json.dumps({"status": summary["status"], "validation_status": result.validation_status,
                              "output": args.output}, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, FloatingPointError) as exc:
        error = exc.as_dict() if isinstance(exc, PredictionError) else {"code": "WORKFLOW_ERROR", "message": str(exc)}
        print(json.dumps(error, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
