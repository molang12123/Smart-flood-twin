import unittest
import numpy as np
from flood_pinn.reference import run_fv_A, hll_x, hll_y


class ReferenceTests(unittest.TestCase):
    def test_exact_early_snapshots(self):
        result = run_fv_A([0., .3, .6], nx=4, ny=4)
        for t, h in result.items():
            np.testing.assert_allclose(h, 50/3600000*t, atol=1e-14)

    def test_tiny_and_close_targets_not_omitted(self):
        times = [0., 1e-12, 1e-11, .3, .3+1e-12]
        result, diag = run_fv_A(times, nx=4, ny=4, return_diagnostics=True)
        self.assertEqual(list(result), times)
        self.assertEqual(diag['snapshot_times_s'], times)
        for t, h in result.items():
            np.testing.assert_allclose(h, 50/3600000*t, atol=1e-16)

    def test_invalid_times(self):
        for times in [[], [-1], [float('nan')], [float('inf')], None]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                run_fv_A(times)

    def test_invalid_grid_validated_before_division(self):
        for nx in [0, -1, 1, 2.5, True, float('nan'), float('inf')]:
            with self.subTest(nx=nx), self.assertRaises(ValueError):
                run_fv_A([1], nx=nx)

    def test_invalid_parameters(self):
        for kwargs in [dict(cfl=0), dict(max_dt=0), dict(max_steps=0),
                       dict(rain_duration_s=-1), dict(manning_n=-.1), dict(slope=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_fv_A([1], **kwargs)

    def test_rain_stop_and_mass(self):
        result, diag = run_fv_A([0, .3, .6, 1.1], nx=4, ny=4, slope=0,
                               rain_duration_s=.4, return_diagnostics=True)
        for t, h in result.items():
            np.testing.assert_allclose(h, 50/3600000*min(t, .4), atol=1e-14)
        self.assertAlmostEqual(diag['rain_volume_m3'], 50/3600000*.4)
        self.assertAlmostEqual(diag['mass_balance_error_m3'], 0., places=14)
        self.assertEqual(diag['depth_clip_count'], 0)
        self.assertEqual(diag['momentum_clip_count'], 0)
        self.assertFalse(diag['externally_validated'])

    def test_zero_rain_stays_dry(self):
        for slope in [0., .05]:
            result, diag = run_fv_A([0, 1, 5], nx=4, ny=4, slope=slope,
                                   rainfall_mm_hr=0, return_diagnostics=True)
            for h in result.values():
                np.testing.assert_array_equal(h, 0.)
            self.assertEqual(diag['final_storage_m3'], 0.)

    def test_hll_standard_jump(self):
        flux = hll_x(.001, 0., 0., .002, 0., 0.)[0]
        self.assertAlmostEqual(float(flux), -.5*np.sqrt(9.81*.002)*.001, places=14)

    def test_hll_equal_state_is_physical_flux(self):
        h, u, v = .3, -.4, .25
        np.testing.assert_allclose(hll_x(h, h*u, h*v, h, h*u, h*v),
                                   [h*u, h*u*u+.5*9.81*h*h, h*u*v], atol=1e-15)
        np.testing.assert_allclose(hll_y(h, h*u, h*v, h, h*u, h*v),
                                   [h*v, h*u*v, h*v*v+.5*9.81*h*h], atol=1e-15)

    def test_hll_reflection_and_rotation(self):
        left, right = (.2, .04, -.02), (.3, -.03, .06)
        original = np.asarray(hll_x(*left, *right))
        mirrored = np.asarray(hll_x(right[0], -right[1], right[2], left[0], -left[1], left[2]))
        np.testing.assert_allclose(mirrored, original*[-1, 1, -1])
        rotated = np.asarray(hll_y(left[0], left[2], left[1], right[0], right[2], right[1]))
        np.testing.assert_allclose(rotated, original[[0, 2, 1]])

    def test_step_budget(self):
        with self.assertRaisesRegex(RuntimeError, 'max_steps'):
            run_fv_A([2], nx=4, ny=4, max_dt=.1, max_steps=2)

    def test_flat_solution_invariant_to_grid_and_time_step(self):
        for cells, dt in [(4, .5), (8, .07), (12, .013)]:
            final = run_fv_A([1.1], nx=cells, ny=cells, slope=0,
                             rain_duration_s=.45, max_dt=dt)[1.1]
            np.testing.assert_allclose(final, 50/3600000*.45, rtol=1e-13, atol=1e-16)

    def test_slope_spatial_refinement(self):
        profiles = []
        for cells in [8, 16, 32, 64]:
            snapshots, diag = run_fv_A([20.], nx=cells, ny=4, slope=.01,
                rainfall_mm_hr=100, max_dt=.1, return_diagnostics=True)
            profiles.append(snapshots[20.][0])
            self.assertLess(abs(diag['mass_balance_error_m3']), 1e-12)
            self.assertEqual(diag['depth_clip_count'], 0)
            self.assertEqual(diag['momentum_clip_count'], 0)
        errors = [np.sqrt(np.mean((profiles[i]-profiles[-1].reshape(cells, -1).mean(axis=1))**2))
                  for i, cells in enumerate([8, 16, 32])]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_momentum_clipping_is_reported(self):
        # Deliberately extreme source at the final step exercises reporting,
        # not an admissible physical reference case.
        _, diag = run_fv_A([8.], nx=4, ny=4, slope=1e5, return_diagnostics=True)
        self.assertGreater(diag['momentum_clip_count'], 0)


if __name__ == '__main__':
    unittest.main()
