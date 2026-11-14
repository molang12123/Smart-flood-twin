import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from minxiong_pinn.expert_bundle import load_expert_bundle
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
        "terrain_fingerprint": FINGERPRINT,
        "time_s": 36 * 3600.,
        "training_rain_domain": {
            "rainfall_mm_hr": [0., 800.], "duration_s": [3600., 24 * 3600.]},
        "routing": {"low_max_mm_hr": 40., "medium_max_mm_hr": 250.},
        "experts": [
            {"model_id": name, "path": f"{name}/model.pt",
             "trained_rainfall_mm_hr": list(RANGES[name])}
            for name in ("low", "medium", "high")],
    }
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
    def load(self, folder, value=None):
        path = Path(folder) / "manifest.json"
        path.write_text(json.dumps(value or manifest()), encoding="utf-8")
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

    def test_unapproved_or_escaping_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            rejected = manifest(physics_training_passed=False)
            with self.assertRaisesRegex(ValueError, "not approved"):
                self.load(folder, rejected)
            escaping = manifest(); escaping["experts"][0]["path"] = "../model.pt"
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.load(folder, escaping)

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
