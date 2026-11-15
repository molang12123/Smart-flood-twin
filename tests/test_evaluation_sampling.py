import unittest
from unittest.mock import patch

import torch

from minxiong_pinn.training import evaluate


class EvaluationSamplingTests(unittest.TestCase):
    def test_each_event_uses_order_independent_stable_repeated_rng(self):
        events = [
            {"event_id": "a", "split": "validation",
             "mean_intensity_mm_hr": 1., "duration_s": 3600.},
            {"event_id": "b", "split": "validation",
             "mean_intensity_mm_hr": 2., "duration_s": 7200.},
        ]

        def fake_loss(_model, _terrain, _cases, rng, _cfg, *, robust_pde):
            self.assertFalse(robust_pde)
            value = torch.tensor(float(rng.random()))
            return value, {"pde": value, "boundary": value * 0.}

        with patch("minxiong_pinn.training._loss", side_effect=fake_loss):
            forward = evaluate(None, None, events, {}, seed=17, repeats=3,
                               interior_points=12)
            reverse = evaluate(None, None, list(reversed(events)), {}, seed=17,
                               repeats=3, interior_points=12)
            changed = evaluate(None, None, events, {}, seed=18, repeats=3)
        by_id = lambda rows: {row["event_id"]: row for row in rows}
        self.assertEqual(by_id(forward), by_id(reverse))
        self.assertNotEqual(by_id(forward)["a"]["loss"],
                            by_id(changed)["a"]["loss"])

    def test_invalid_evaluation_sampling_settings_are_rejected(self):
        event = [{"event_id": "a", "split": "test",
                  "mean_intensity_mm_hr": 1., "duration_s": 3600.}]
        with self.assertRaisesRegex(ValueError, "repeats"):
            evaluate(None, None, event, {}, repeats=0)
        with self.assertRaisesRegex(ValueError, "interior_points"):
            evaluate(None, None, event, {}, interior_points=0)


if __name__ == "__main__":
    unittest.main()
