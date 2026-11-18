import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import minxiong_pinn.expert_bundle as expert_bundle
from minxiong_pinn.expert_bundle import load_expert_bundle, load_preferred_model
from minxiong_pinn.user_input import build_prediction_request


FINGERPRINT = "fixed-minxiong-test-fingerprint"
RANGES = {"low": (0., 50.), "medium": (30., 300.), "high": (200., 800.)}


class MarkerExpert(nn.Module):
    def __init__(self, marker):
        super().__init__(); self.marker = float(marker)
        self.scales = {"time_s": 36 * 3600.}

    def forward(self, points):
        return points.new_full((len(points), 3), self.marker)


def manifest(**changes):
    value = {
        "schema_version": 1,
        "bundle_id": "minxiong-0-800-test",
        "activation_approved": True,
        "physics_training_passed": True,
        "overlap_consistency_passed": True,
        "continuous_domain_audit_passed": True,
        "terrain_fingerprint": FINGERPRINT,
        "time_s": 36 * 3600.,
        "training_rain_domain": {
            "rainfall_mm_hr": [0., 800.], "duration_s": [3600., 24 * 3600.]},
        "routing": {"low_max_mm_hr": 40., "medium_max_mm_hr": 250.},
        "audit_file": "overlap_audit.json",
        "experts": [
            {"model_id": name, "path": f"{name}/model.pt",
             "sha256": hashlib.sha256(name.encode()).hexdigest(),
             "trained_rainfall_mm_hr": list(RANGES[name])}
            for name in ("low", "medium", "high")],
    }
    value.update(changes)
    return value


def two_expert_manifest(**changes):
    value = manifest()
    value.update({
        "bundle_id": "minxiong-0-300-test",
        "training_rain_domain": {
            "rainfall_mm_hr": [0., 300.],
            "duration_s": [3600., 24 * 3600.],
        },
        "routing": {"low_max_mm_hr": 40.},
        "experts": value["experts"][:2],
    })
    value.update(changes)
    return value


def fake_loader(path, _terrain):
    model_id = Path(path).parent.name
    checkpoint = {
        "terrain_fingerprint": FINGERPRINT,
        "training_rain_domain": {
            "rainfall_mm_hr": list(RANGES[model_id]),
            "duration_s": [3600., 24 * 3600.],
        },
    }
    return MarkerExpert({"low": 1, "medium": 2, "high": 3}[model_id]), checkpoint


class ExpertBundleTests(unittest.TestCase):
    def audit(self, value):
        return {
            "activation_approved": True,
            "physics_training_passed": True,
            "overlap_consistency_passed": True,
            "continuous_domain_passed": True,
            "terrain_fingerprint": FINGERPRINT,
            "experts": [
                {"model_id": item["model_id"],
                 "physics_training_passed": True,
                 "checkpoint_sha256": item.get("sha256")}
                for item in value["experts"]
            ],
            "overlap": {"passed": True},
            "continuous_domain": {"passed": True},
        }

    def write_bundle(self, folder, value=None, audit=None):
        value = value or manifest()
        folder = Path(folder)
        for name in RANGES:
            checkpoint = folder / name / "model.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(name.encode())
        audit_path = folder / "overlap_audit.json"
        audit_path.write_text(
            json.dumps(audit or self.audit(value)), encoding="utf-8")
        value.setdefault(
            "audit_sha256", hashlib.sha256(audit_path.read_bytes()).hexdigest())
        path = Path(folder) / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def load(self, folder, value=None, audit=None):
        path = self.write_bundle(folder, value, audit)
        return load_expert_bundle(
            path, SimpleNamespace(fingerprint=FINGERPRINT), model_loader=fake_loader)

    def test_routes_exact_boundaries_and_mixed_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            model, checkpoint = self.load(folder)
            self.assertEqual(checkpoint["training_rain_domain"]["rainfall_mm_hr"], [0., 800.])
            self.assertEqual(
                [model.model_id_for_rainfall(v) for v in (0, 40, 40.001, 250, 250.001, 800)],
                ["low", "low", "medium", "medium", "high", "high"])
            points = torch.zeros((6, 5), dtype=torch.float64)
            points[:, 3] = torch.tensor([0, 40, 41, 250, 251, 800]) / 3_600_000.
            result = model(points)
            torch.testing.assert_close(
                result[:, 0], torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.float64))

    def test_two_expert_bundle_routes_continuous_zero_to_300_domain(self):
        with tempfile.TemporaryDirectory() as folder:
            model, checkpoint = self.load(folder, two_expert_manifest())
            self.assertEqual(tuple(model.experts), ("low", "medium"))
            self.assertEqual(
                checkpoint["training_rain_domain"]["rainfall_mm_hr"],
                [0., 300.])
            self.assertEqual(
                [model.model_id_for_rainfall(value)
                 for value in (0., 40., 40.001, 123.456, 300.)],
                ["low", "low", "medium", "medium", "medium"])
            points = torch.zeros((5, 5), dtype=torch.float64)
            points[:, 3] = torch.tensor(
                [0., 40., 41., 250., 300.], dtype=torch.float64) / 3_600_000.
            result = model(points)
            torch.testing.assert_close(
                result[:, 0],
                torch.tensor([1., 1., 2., 2., 2.], dtype=torch.float64))
            with self.assertRaisesRegex(ValueError, "outside"):
                model.model_id_for_rainfall(300.001)

    def test_unapproved_or_escaping_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            rejected = manifest(physics_training_passed=False)
            with self.assertRaisesRegex(ValueError, "not approved"):
                self.load(folder, rejected)
            escaping = manifest(); escaping["experts"][0]["path"] = "../model.pt"
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.load(folder, escaping)

    def test_continuous_audit_evidence_is_mandatory_and_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = manifest()
            missing.pop("continuous_domain_audit_passed")
            with self.assertRaisesRegex(ValueError, "continuous_domain"):
                self.load(folder, missing)

        cases = {
            "top-level gate": lambda value: value.update(
                continuous_domain_passed=False),
            "nested overlap": lambda value: value["overlap"].update(
                passed=False),
            "nested continuous": lambda value: value["continuous_domain"].update(
                passed=False),
            "terrain": lambda value: value.update(
                terrain_fingerprint="other-terrain"),
            "expert": lambda value: value["experts"][0].update(
                physics_training_passed=False),
        }
        for label, change in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                value = manifest()
                audit = self.audit(value)
                change(audit)
                with self.assertRaises(ValueError):
                    self.load(folder, value, audit)

        with tempfile.TemporaryDirectory() as folder:
            value = manifest(audit_file="../overlap_audit.json")
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.load(folder, value)

    def test_checkpoint_sha256_is_required_and_verified_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            value = manifest()
            value["experts"][0].pop("sha256")
            with self.assertRaisesRegex(ValueError, "sha256 is invalid"):
                self.load(folder, value)
        with tempfile.TemporaryDirectory() as folder:
            value = manifest()
            value["experts"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                self.load(folder, value)

    def test_audit_file_and_checkpoint_evidence_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            value = manifest(audit_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "audit file sha256 mismatch"):
                self.load(folder, value)
        with tempfile.TemporaryDirectory() as folder:
            value = manifest()
            audit = self.audit(value)
            audit["experts"][0]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "evidence differs"):
                self.load(folder, value, audit)

    def test_invalid_expert_bundle_still_falls_back_to_single_model(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "manifest.json"
            bundle.write_text("{}", encoding="utf-8")
            fallback_model = MarkerExpert(9)
            fallback_checkpoint = {"training_rain_domain": {
                "rainfall_mm_hr": [.5, 23.],
                "duration_s": [3600., 37 * 3600.],
            }}
            with patch.object(
                    expert_bundle, "load_expert_bundle",
                    side_effect=ValueError("incomplete manifest")), patch.object(
                    expert_bundle, "load_model",
                    return_value=(fallback_model, fallback_checkpoint)):
                model, checkpoint = load_preferred_model(
                    bundle, Path(folder) / "fallback.pt",
                    SimpleNamespace(fingerprint=FINGERPRINT))
            self.assertIs(model, fallback_model)
            self.assertEqual(
                checkpoint["training_rain_domain"]["rainfall_mm_hr"],
                [.5, 23.])
            self.assertIn("incomplete manifest",
                          checkpoint["bundle_fallback_reason"])

    def test_zero_rain_is_valid_only_when_checkpoint_includes_it(self):
        checkpoint = {"training_rain_domain": {
            "rainfall_mm_hr": [0., 800.], "duration_s": [3600., 24 * 3600.]}}
        values = dict(
            rainfall_mm_hr="0", rain_duration_value="1", rain_duration_unit="小时",
            simulation_value="36", simulation_unit="小时",
            interval_value="6", interval_unit="小时", checkpoint=checkpoint,
            model_time_s=36 * 3600., active_cells=100)
        request = build_prediction_request(**values)
        self.assertEqual(request.rainfall_mm_hr, 0.)
        values["rainfall_mm_hr"] = "-0.1"
        with self.assertRaisesRegex(ValueError, "非负"):
            build_prediction_request(**values)


if __name__ == "__main__":
    unittest.main()
