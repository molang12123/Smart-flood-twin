import unittest

import numpy as np

from minxiong_pinn.training import (
    _continuous_condition_cases, _continuous_condition_sample_count,
)


def _config(count=2):
    return {
        "continuous_condition_samples_per_batch": count,
        "scales": {
            "rain_min_m_s": 0.,
            "rain_m_s": 300. / 3_600_000.,
            "duration_min_s": 3600.,
            "duration_s": 24 * 3600.,
            "rain_log_reference_m_s": 1. / 3_600_000.,
            "duration_log_reference_s": 3600.,
        },
    }


class ContinuousConditionSamplingTests(unittest.TestCase):
    def test_reproducible_off_grid_linear_and_log_draws_cover_domain(self):
        first = _continuous_condition_cases(
            np.random.default_rng(17), _config(), step=1)
        second = _continuous_condition_cases(
            np.random.default_rng(17), _config(), step=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertIn("linear", first[0]["event_id"])
        self.assertIn("log1p", first[1]["event_id"])
        fixed_rain = {0., .5, 1., 2., 5., 10., 20., 30., 40., 50.,
                      75., 100., 150., 200., 250., 300.}
        fixed_duration = {value * 3600. for value in (1, 2, 3, 6, 12, 18, 24)}
        for event in first:
            rain = event["mean_intensity_mm_hr"]
            duration = event["duration_s"]
            self.assertGreater(rain, 0.)
            self.assertLess(rain, 300.)
            self.assertGreater(duration, 3600.)
            self.assertLess(duration, 24 * 3600.)
            self.assertNotIn(rain, fixed_rain)
            self.assertNotIn(duration, fixed_duration)
            self.assertTrue(event["continuous_condition"])
            self.assertEqual(event["split"], "train")
            self.assertAlmostEqual(
                event["total_mm"], rain * duration / 3600.)

    def test_extra_cases_append_without_replacing_zero_and_extreme_anchors(self):
        zero = {"event_id": "zero", "mean_intensity_mm_hr": 0.}
        extreme = {"event_id": "extreme", "mean_intensity_mm_hr": 300.}
        chosen = [zero, extreme]
        chosen += _continuous_condition_cases(
            np.random.default_rng(9), _config(1), step=2)
        self.assertIs(chosen[0], zero)
        self.assertIs(chosen[1], extreme)
        self.assertEqual(len(chosen), 3)
        self.assertTrue(chosen[2]["continuous_condition"])

    def test_zero_default_preserves_legacy_rng_stream(self):
        first = np.random.default_rng(44)
        self.assertEqual(
            _continuous_condition_cases(first, {"scales": {}}, step=1), [])
        first_value = first.random()
        second_value = np.random.default_rng(44).random()
        self.assertEqual(first_value, second_value)
        self.assertEqual(_continuous_condition_sample_count({}), 0)

    def test_count_and_domain_validation(self):
        for value in (-1, 1.5, True, None, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "nonnegative integer"):
                _continuous_condition_sample_count(
                    {"continuous_condition_samples_per_batch": value})
        invalid = _config(1)
        del invalid["scales"]["duration_min_s"]
        with self.assertRaisesRegex(ValueError, "explicit expert"):
            _continuous_condition_cases(
                np.random.default_rng(1), invalid, step=1)
        invalid = _config(1)
        invalid["scales"]["rain_m_s"] = 0.
        with self.assertRaisesRegex(ValueError, "domain is invalid"):
            _continuous_condition_cases(
                np.random.default_rng(1), invalid, step=1)


if __name__ == "__main__":
    unittest.main()
