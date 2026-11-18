import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import train_minxiong_experts as expert_training
from minxiong_pinn.training import _training_identity


class CompletedExpertSkipTests(unittest.TestCase):
    def setUp(self):
        self.terrain = SimpleNamespace(fingerprint="fixed-terrain")
        self.events = [
            {"event_id": f"event-{split}", "split": split,
             "mean_intensity_mm_hr": 10., "duration_s": 3600.,
             "total_mm": 10.}
            for split in ("train", "validation", "test")
        ]
        self.cfg = {
            "resume_training": True, "adam_steps": 10, "seed": 42,
            "scales": {"rain_m_s": 10. / 3_600_000.},
            "success_thresholds": {"test_mean_pde_loss": .02},
            "assumptions": ["fixed terrain", "closed boundary"],
        }

    def _write_artifacts(self, folder, *, state_changes=None,
                         report_changes=None, model_changes=None):
        folder = Path(folder)
        folder.mkdir()
        identity = _training_identity(self.terrain, self.events, self.cfg)
        identity["mass_projection_anchor_fingerprint"] = None
        best_state = {
            "layer.weight": torch.tensor([[1., 2.], [3., 4.]]),
            "nested": {"layer.bias": torch.tensor([.5, -.5])},
        }
        state = {"schema_version": 2, "identity": identity, "step": 10,
                 "best": .125, "best_state": best_state}
        report = {
            "requested_steps": 10, "completed_steps": 10,
            "physics_training_passed": True,
            "terrain": {"fingerprint": self.terrain.fingerprint},
            "best_validation_loss": .125,
            "success_thresholds": copy.deepcopy(
                self.cfg["success_thresholds"]),
            "assumptions": list(self.cfg["assumptions"]),
        }
        model = {
            "terrain_fingerprint": self.terrain.fingerprint,
            "mass_projection_anchor_fingerprint": None,
            "model_state": copy.deepcopy(best_state),
        }
        if state_changes:
            state_changes(state)
        if report_changes:
            report_changes(report)
        if model_changes:
            model_changes(model)
        torch.save(state, folder / "training_state.pt")
        torch.save(model, folder / "model.pt")
        import json
        (folder / "training_report.json").write_text(
            json.dumps(report), encoding="utf-8")

    def test_matching_completed_passed_expert_is_reused(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "low"
            self._write_artifacts(folder)
            report = expert_training._load_completed_expert_report(
                self.terrain, self.events, self.cfg, folder)
            self.assertIsNotNone(report)
            self.assertTrue(report["physics_training_passed"])

    def test_resume_disabled_never_skips(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "low"
            self._write_artifacts(folder)
            cfg = dict(self.cfg)
            cfg["resume_training"] = False
            self.assertIsNone(expert_training._load_completed_expert_report(
                self.terrain, self.events, cfg, folder))

    def test_any_inconsistent_artifact_prevents_skip(self):
        cases = {
            "state schema": {"state_changes": lambda value: value.update(
                schema_version=1)},
            "identity config": {"state_changes": lambda value: value[
                "identity"]["config"].update(seed=99)},
            "unfinished state": {"state_changes": lambda value: value.update(step=9)},
            "report request": {"report_changes": lambda value: value.update(
                requested_steps=9)},
            "report completion": {"report_changes": lambda value: value.update(
                completed_steps=9)},
            "failed physics": {"report_changes": lambda value: value.update(
                physics_training_passed=False)},
            "model terrain": {"model_changes": lambda value: value.update(
                terrain_fingerprint="other-terrain")},
            "report terrain": {"report_changes": lambda value: value[
                "terrain"].update(fingerprint="other-terrain")},
            "anchor": {"model_changes": lambda value: value.update(
                mass_projection_anchor_fingerprint="other-anchor")},
            "model state": {"model_changes": lambda value: value[
                "model_state"]["nested"]["layer.bias"].add_(1.)},
            "best validation": {"report_changes": lambda value: value.update(
                best_validation_loss=.126)},
            "success thresholds": {"report_changes": lambda value: value[
                "success_thresholds"].update(test_mean_pde_loss=.03)},
            "assumptions": {"report_changes": lambda value: value[
                "assumptions"].append("stale report")},
        }
        for label, changes in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                folder = Path(root) / "low"
                self._write_artifacts(folder, **changes)
                self.assertIsNone(
                    expert_training._load_completed_expert_report(
                        self.terrain, self.events, self.cfg, folder))

    def test_skipped_expert_remains_in_summary_without_calling_train(self):
        report = {"physics_training_passed": True}
        terrain = SimpleNamespace(bounds=(0., 100., 0., 200.),
                                  fingerprint="fixed-terrain")
        master = {
            "terrain_tif": "terrain.tif", "rainfall_zip": "rain.zip",
            "output_root": "output", "manning_n": .03,
            "durations_hr": [1, 2],
            "experts": [{"name": "low", "intensities_mm_hr": [0, 10],
                         "velocity_m_s": 2.}],
            "training": {"resume_training": True, "seed": 1,
                         "adam_steps": 10, "time_end_s": 7200.},
        }
        with tempfile.TemporaryDirectory() as root, \
                patch.object(expert_training, "MaskedRasterTerrain",
                             return_value=terrain), \
                patch.object(expert_training, "extract_events", return_value=[]), \
                patch.object(expert_training, "add_design_events",
                             return_value=self.events), \
                patch.object(expert_training, "_load_completed_expert_report",
                             return_value=report), \
                patch.object(expert_training, "train") as trainer:
            summary = expert_training.run_expert_config(master, root)
        trainer.assert_not_called()
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["completed_experts"], 1)
        self.assertEqual(summary["experts"][0]["name"], "low")
        self.assertTrue(summary["experts"][0]["training_skipped"])


if __name__ == "__main__":
    unittest.main()
