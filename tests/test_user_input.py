import unittest

from minxiong_pinn.user_input import build_prediction_request
from minxiong_pinn.predict import predict


CHECKPOINT = {"training_rain_domain": {"rainfall_mm_hr": [0.5, 23.0],
                                        "duration_s": [3600.0, 133200.0]}}


def request(**overrides):
    values = dict(rainfall_mm_hr="10", rain_duration_value="2", rain_duration_unit="小时",
                  simulation_value="3", simulation_unit="小时",
                  interval_value="60", interval_unit="分钟", checkpoint=CHECKPOINT,
                  model_time_s=154800., active_cells=212239)
    values.update(overrides)
    return build_prediction_request(**values)


class UserInputTests(unittest.TestCase):
    def test_predict_api_rejects_empty_duplicate_and_reverse_times(self):
        class Model:
            scales = {"time_s": 154800.}
        for times in ([], [0., 0.], [10., 5.]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                predict(Model(), CHECKPOINT, None, 10., 7200., times, "unused")

    def test_regular_times_include_zero_stop_and_end(self):
        result = request()
        self.assertEqual(result.output_times_s, (0., 3600., 7200., 10800.))
        self.assertEqual(result.frame_count, 4)
        self.assertGreater(result.estimated_uncompressed_bytes, 0)

    def test_nondivisible_interval_inserts_rain_stop_and_exact_end(self):
        result = request(rain_duration_value="1.5", simulation_value="2.5",
                         interval_value="1", interval_unit="小时")
        self.assertEqual(result.output_times_s, (0., 3600., 5400., 7200., 9000.))

    def test_simulation_may_end_while_rain_continues(self):
        result = request(rain_duration_value="3", simulation_value="2",
                         interval_value="4", interval_unit="小时")
        self.assertEqual(result.output_times_s, (0., 7200.))

    def test_range_time_and_frame_errors_are_clear(self):
        cases = [
            ({"rainfall_mm_hr": "30"}, "雨势超出"),
            ({"rain_duration_value": ".5"}, "降雨时长超出"),
            ({"simulation_value": "44"}, "超过模型时域"),
            ({"interval_value": "0"}, "输出间隔必须"),
            ({"rainfall_mm_hr": "NaN"}, "有限数值"),
            ({"simulation_value": "10", "interval_value": "1", "interval_unit": "分钟", "max_frames": 500}, "超过上限"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                request(**changes)


if __name__ == "__main__":
    unittest.main()
