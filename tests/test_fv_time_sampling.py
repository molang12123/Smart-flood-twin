import unittest
import numpy as np

from minxiong_pinn.training import _fv_start_times, _state_slice_times


class FiniteVolumeTimeSamplingTests(unittest.TestCase):
    def test_stratified_slices_cover_early_stop_post_and_late(self):
        end_s, duration, dt = 36*3600., 6*3600., 600.
        values = _fv_start_times(np.random.default_rng(7), 100, end_s, dt, duration, slices=5)
        unique = np.unique(values)
        self.assertEqual(len(unique), 5)
        self.assertTrue(any(v <= 1800 for v in unique))
        self.assertTrue(any(v < duration < v + dt for v in unique))
        self.assertTrue(any(v >= duration for v in unique))
        self.assertTrue(any(v >= .75*(end_s-dt) for v in unique))
        self.assertTrue(all(np.count_nonzero(values == v) == 20 for v in unique))

    def test_zero_slices_preserves_independent_uniform_sampling(self):
        values = _fv_start_times(np.random.default_rng(8), 20, 7200., 600., 3600., slices=0)
        self.assertEqual(len(np.unique(values)), 20)

    def test_state_times_repeat_midpoint_stop_and_horizon(self):
        self.assertEqual(set(_state_slice_times(8, 36*3600., 6*3600.)),
                         {3*3600., 6*3600., 36*3600.})


if __name__ == "__main__": unittest.main()
