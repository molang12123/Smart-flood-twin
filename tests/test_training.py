import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flood_pinn.model import ParametricPINN
from flood_pinn.training import _read_reference, loss_components, sample_batch, train, validate_config


class FlatTerrain:
    bounds = (0., 2., 0., 2.)
    fingerprint = "unit-test-flat-terrain"
    metadata = {"synthetic": True}

    def query(self, x, y):
        return {k: np.zeros_like(x) for k in ("z", "zx", "zy", "n")}


def configuration():
    return {
        "model": {"width": 8, "layers": 2, "parameterized": True},
        "scales": dict(x_origin=0., y_origin=0., x_scale=2., y_scale=2.,
                       t_scale=1., r_scale=1e-4, duration_scale=1., h_scale=.01, u_scale=.1),
        "time": {"end_s": 1., "output_interval_s": .2},
        "conditions": {"initial": "dry", "boundaries": dict.fromkeys(("west", "east", "south", "north"), "wall")},
        "scenarios": [
            {"scenario_id": "a", "rainfall_mm_hr": 50., "duration_s": .4, "split": "train"},
            {"scenario_id": "b", "rainfall_mm_hr": 90., "duration_s": .6, "split": "train"},
            {"scenario_id": "v", "rainfall_mm_hr": 75., "duration_s": .5, "split": "validation"},
            {"scenario_id": "t", "rainfall_mm_hr": 60., "duration_s": .3, "split": "test"},
        ],
        "training": {"adam_steps": 2, "lbfgs_steps": 2, "interior_points": 8,
                     "ic_points": 4, "bc_points": 2, "threads": 1,
                     "learning_rate": .001, "max_seconds": 60.},
    }


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.terrain = FlatTerrain()

    def test_validates_conditions_ranges_and_split_pairs(self):
        for mutate in (
            lambda c: c["conditions"].update(initial="unknown"),
            lambda c: c["conditions"]["boundaries"].update(west="open"),
            lambda c: c["scenarios"][0].update(duration_s=2.),
            lambda c: c["scenarios"][1].update(rainfall_mm_hr=50., duration_s=.4),
            lambda c: c["training"].update(adam_steps=15000),
            lambda c: c["training"].update(loss_weights={"pde_loss": float("nan")}),
            lambda c: c["training"].update(allow_long_training="false"),
            lambda c: c["training"].update(learning_rate=".001"),
            lambda c: c["time"].update(end_s=True),
            lambda c: c["thresholds"].update(flood_depth_m=-1.),
        ):
            cfg = configuration(); cfg["thresholds"] = {}; mutate(cfg)
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_frozen_objective_and_no_heldout_physics(self):
        cfg = validate_config(configuration())
        batch = sample_batch(self.terrain, cfg, np.random.default_rng(3), all_training=True)
        self.assertEqual(set(batch["scenario_ids"]), {"a", "b"})
        self.assertFalse(np.any(batch["pde"][:, 2].astype(np.float32) == batch["pde"][:, 4].astype(np.float32)))
        model = ParametricPINN(cfg["scales"], **cfg["model"])
        values, gradients = [], []
        for _ in range(2):
            model.zero_grad()
            objective, _ = loss_components(model, self.terrain, cfg, batch)
            objective.backward()
            values.append(objective.item())
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        self.assertEqual(values[0], values[1])
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
        heldout = sample_batch(self.terrain, cfg, np.random.default_rng(0), scenarios=[cfg["scenarios"][2]])
        with self.assertRaisesRegex(ValueError, "held-out scenario"):
            loss_components(model, self.terrain, cfg, heldout)

    def test_full_horizon_conservation_includes_rain(self):
        cfg = validate_config(configuration())
        batch = sample_batch(self.terrain, cfg, np.random.default_rng(7), all_training=True)
        class AnalyticRain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.tensor(0.))
                self.scales = cfg["scales"]
            def forward(self, p):
                h = p[:, 3:4] * torch.minimum(p[:, 2:3], p[:, 4:5]) + self.marker*0
                return torch.cat([h, h*0, h*0], dim=1)
        _, parts = loss_components(AnalyticRain(), self.terrain, cfg, batch)
        self.assertLess(parts["conservation_loss"].item(), 1e-15)

    def test_short_training_artifacts_and_reference_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reference = folder / "reference.csv"
            reference.write_text("scenario_id,x,y,t,h\na,1,1,.2,.00001\nv,1,1,.2,.00002\nt,1,1,.2,.00003\n")
            cfg = configuration(); cfg["reference_csv"] = str(reference)
            model, report = train(self.terrain, cfg, folder / "run")
            self.assertTrue(report["training_completed"])
            self.assertFalse(report["converged"])
            self.assertEqual(report["training_reference_count"], 1)
            self.assertEqual(report["training_scenario_ids"], ["a", "b"])
            self.assertIsNone(report["validated_domain"])
            self.assertEqual(report["reference_validation_status"], "comparison_only_not_accuracy_approval")
            metrics = json.loads((folder / "run" / "validation_metrics.json").read_text())
            self.assertEqual(metrics["split_counts"], {"train": 2, "validation": 1, "test": 1})
            for artifact in ("training_history.csv", "training_report.json", "physics_diagnostics.csv", "validation_metrics.json", "loss.png"):
                self.assertTrue((folder / "run" / artifact).is_file(), artifact)
            with (folder / "run" / "training_history.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row["phase"] == "lbfgs" for row in rows))
            self.assertTrue(all(set(row["scenario_ids"].split(";")) <= {"a", "b"} for row in rows))
            with self.assertRaises(FileExistsError):
                train(self.terrain, cfg, folder / "run")

    def test_numerical_failure_preserves_report_and_finite_weights(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as folder:
            def invalid_residual(model, points, terrain, **kwargs):
                return torch.full((len(points), 3), float("nan"))
            with patch("flood_pinn.training.residuals", invalid_residual):
                with self.assertRaises(FloatingPointError):
                    train(self.terrain, cfg, folder)
            report = json.loads((Path(folder) / "training_report.json").read_text())
            self.assertEqual(report["termination_reason"], "failed")
            self.assertTrue(report["nonfinite_detected"])
            self.assertFalse(report["training_completed"])
            self.assertTrue((Path(folder) / "last_finite_state.pt").exists())

    def test_time_budget_not_misreported_as_completion(self):
        cfg = configuration(); cfg["training"]["max_seconds"] = 1e-12
        with tempfile.TemporaryDirectory() as folder:
            _, report = train(self.terrain, cfg, folder)
            self.assertFalse(report["training_completed"])
            self.assertEqual(report["termination_reason"], "time_budget_exhausted")

    def test_diagnostic_failure_is_persisted(self):
        cfg = configuration(); cfg["training"].update(adam_steps=1, lbfgs_steps=0)
        with tempfile.TemporaryDirectory() as folder:
            with patch("flood_pinn.training._diagnose", side_effect=FloatingPointError("bad audit")):
                with self.assertRaises(FloatingPointError):
                    train(self.terrain, cfg, folder)
            report = json.loads((Path(folder) / "training_report.json").read_text())
            self.assertTrue(report["training_completed"])
            self.assertEqual(report["termination_reason"], "diagnostics_failed")
            self.assertFalse(report["physics_checks_passed"])

    def test_oversized_reference_fails_before_training(self):
        cfg = configuration(); cfg["training"]["max_training_reference_points"] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.csv"
            path.write_text("scenario_id,x,y,t,h\na,1,1,.1,.01\na,1,1,.2,.02\n")
            cfg["reference_csv"] = str(path)
            with self.assertRaisesRegex(ValueError, "reference cap"):
                train(self.terrain, cfg, Path(folder) / "run")

    def test_real_terrain_reference_requires_aligned_provenance(self):
        class RealTerrain(FlatTerrain):
            metadata = {"synthetic": False, "crs": "EPSG:3826", "vertical_datum": "test-datum", "grid_id": "test-grid"}
        cfg = validate_config(configuration())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"reference.csv"
            path.write_text("scenario_id,x,y,t,h\na,1,1,.1,.01\n")
            cfg["reference_csv"] = str(path)
            with self.assertRaisesRegex(ValueError, "provenance"):
                _read_reference(cfg, RealTerrain())
            cfg["reference_provenance"] = {
                **RealTerrain.metadata, "source_type": "unit_test", "source_version": "1",
                "units": {"x": "m", "y": "m", "h": "m", "t": "s", "u": "m/s", "v": "m/s"}}
            rows, digest = _read_reference(cfg, RealTerrain())
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
