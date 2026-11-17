import json
from pathlib import Path
import unittest

from minxiong_pinn.rainfall import add_design_events
from probe_minxiong_low import probe_config
from probe_minxiong_single_case import single_case_setup


ROOT = Path(__file__).resolve().parents[1]


class ExpandedExpertDomainTests(unittest.TestCase):
    def test_v10_probes_are_short_isolated_low_expert_runs(self):
        uniform = probe_config("uniform")
        hydro90 = probe_config("hydro90")
        uniform_strict = probe_config("uniform_strict")
        for name, config in (("uniform", uniform), ("hydro90", hydro90),
                             ("uniform_strict", uniform_strict)):
            self.assertEqual(len(config["experts"]), 1)
            self.assertEqual(config["experts"][0]["name"], "low")
            self.assertEqual(config["training"]["adam_steps"], 800)
            self.assertTrue(config["training"]["separate_output_gradient_clipping"])
            self.assertIn(f"v10_probes/{name}", config["output_root"])
        self.assertFalse(uniform["training"]["hydrostatic_depth_baseline"])
        self.assertTrue(hydro90["training"]["hydrostatic_depth_baseline"])
        self.assertEqual(hydro90["training"]["hydrostatic_uniform_floor"], .9)
        self.assertEqual(uniform_strict["training"]["robust_pde_warmup_steps"], 0)
        self.assertEqual(uniform_strict["training"]["learning_rate"], .00003)
        with self.assertRaisesRegex(ValueError, "variant"):
            probe_config("unknown")
        with self.assertRaisesRegex(ValueError, "physics"):
            single_case_setup("unknown")

    def test_expanded_config_covers_requested_domain(self):
        config = json.loads((ROOT / "configs/minxiong_experts_0_800.json").read_text())
        self.assertEqual(config["durations_hr"], [1, 2, 3, 6, 12, 18, 24])
        self.assertEqual(config["training"]["time_end_s"], 36 * 3600)
        self.assertTrue(config["training"]["log1p_conditions"])
        self.assertTrue(config["training"]["terrain_features"])
        self.assertTrue(config["training"]["enhanced_time_features"])
        self.assertTrue(config["training"]["mass_conserving_projection"])
        self.assertTrue(config["training"]["separate_output_networks"])
        self.assertTrue(config["training"]["hard_mask_boundary"])
        self.assertTrue(config["training"]["hydrostatic_depth_baseline"])
        self.assertGreater(config["training"]["hydrostatic_learned_blend"], 0.)
        self.assertLessEqual(config["training"]["hydrostatic_learned_blend"], 1.)
        self.assertEqual(config["training"]["hydrostatic_correction_mode"],
                         "multiplicative")
        self.assertGreater(config["training"]["hydrostatic_uniform_floor"], 0.)
        self.assertGreater(config["training"]["robust_pde_warmup_steps"], 0)
        self.assertGreater(config["training"]["fv_time_slices"], 0)
        ranges = [(item["intensities_mm_hr"][0], item["intensities_mm_hr"][-1])
                  for item in config["experts"]]
        self.assertEqual(ranges, [(0, 50), (30, 300), (200, 800)])
        self.assertEqual(config["routing"]["low_max_mm_hr"], 40)
        self.assertEqual(config["routing"]["medium_max_mm_hr"], 250)

    def test_reduced_config_uses_two_balanced_experts_for_zero_to_300(self):
        config = json.loads(
            (ROOT / "configs/minxiong_experts_0_300.json").read_text())
        self.assertEqual(config["durations_hr"], [1, 2, 3, 6, 12, 18, 24])
        self.assertEqual(config["training"]["time_end_s"], 36 * 3600)
        self.assertEqual(config["routing"]["low_max_mm_hr"], 40.)
        self.assertNotIn("medium_max_mm_hr", config["routing"])
        self.assertEqual(
            [(item["name"], item["intensities_mm_hr"][0],
              item["intensities_mm_hr"][-1])
             for item in config["experts"]],
            [("low", 0, 50), ("medium", 30, 300)])
        training = config["training"]
        self.assertEqual(training["physics_form"], "strong")
        self.assertEqual(training["network_architecture"], "plain")
        self.assertTrue(training["terrain_local_coordinates"])
        self.assertFalse(training["terrain_local_plane_feature"])
        self.assertFalse(training["conditioned_velocity_scale"])
        self.assertFalse(training["mass_conserving_projection"])
        self.assertEqual(training["loss_weights"]["interface"], 0.)
        self.assertEqual(training["continuous_condition_samples_per_batch"], 1)
        self.assertEqual(training["continuity_scale_m_s"], 1e-8)
        self.assertEqual(config["routing"]["overlaps_for_audit_only"],
                         [[30., 50.]])
        continuous_audit = config["continuous_domain_audit"]
        self.assertTrue(continuous_audit["enabled"])
        self.assertGreater(continuous_audit["conditions_per_expert"], 0)
        self.assertGreater(
            continuous_audit["full_domain_conditions_per_expert"], 0)

    def test_zero_rain_is_a_trained_domain_edge(self):
        events = add_design_events([], [0, .5, 5, 50], [1, 3, 24])
        dry = [event for event in events if event["mean_intensity_mm_hr"] == 0]
        self.assertEqual(len(dry), 3)
        self.assertTrue(all(event["split"] == "train" for event in dry))
        self.assertTrue(all(event["total_mm"] == 0 for event in dry))

    def test_expanded_markers_hold_out_complete_interior_scenarios(self):
        events = add_design_events(
            [], [0, .5, 1, 2, 5, 10, 20, 30, 40, 50],
            [1, 2, 3, 6, 12, 18, 24],
            validation_markers=(1, 3), test_markers=(6, 8))
        edges = [event for event in events
                 if event["mean_intensity_mm_hr"] in {0, 50}
                 or event["hours"] in {1, 24}]
        self.assertTrue(all(event["split"] == "train" for event in edges))
        self.assertTrue(any(event["split"] == "validation" for event in events))
        self.assertTrue(any(event["split"] == "test" for event in events))

    def test_negative_rain_and_zero_duration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            add_design_events([], [-.1, 1], [1, 2])
        with self.assertRaisesRegex(ValueError, "positive"):
            add_design_events([], [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            add_design_events([], [0, 2, 1], [1, 2])


if __name__ == "__main__":
    unittest.main()
