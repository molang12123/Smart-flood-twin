from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.terrain import MaskedRasterTerrain
from minxiong_pinn.training import train


class MinxiongTrainingResumeTests(unittest.TestCase):
    def _terrain(self, folder):
        path = Path(folder) / "flat.tif"
        with rasterio.open(path, "w", driver="GTiff", width=3, height=3, count=1,
                           dtype="float32", crs="EPSG:3826",
                           transform=from_origin(0, 60, 20, 20), nodata=-9999.) as dst:
            dst.write(np.zeros((3, 3), dtype=np.float32), 1)
        return MaskedRasterTerrain(path, manning_n=.03)

    def _config(self, steps):
        return {
            "seed": 3, "threads": 1, "width": 4, "layers": 1,
            "adam_steps": steps, "learning_rate": 1e-4,
            "physics_form": "finite_volume", "weak_dt_s": 60.,
            "max_seconds": 120., "scenarios_per_batch": 1,
            "interior_points": 2, "boundary_points": 2, "mass_points": 2,
            "validation_interval": 1, "checkpoint_interval": 1,
            "gradient_clip": 10., "h_regularization_m": .001,
            "time_end_s": 1200., "continuity_scale_m_s": 1e-6,
            "momentum_scale_m2_s2": .005, "momentum_slope_scale": .1,
            "loss_weights": {"pde": 1., "boundary": 1., "mass": 1.},
            "scales": {"x0": 0., "y0": 0., "lx": 60., "ly": 60.,
                       "time_s": 1200., "rain_m_s": 10/3_600_000.,
                       "duration_s": 600., "velocity_m_s": 1., "ramp_s": 60.},
            "success_thresholds": {"test_mean_pde_loss": 1e9,
                                   "test_mean_mass_loss": 1e9,
                                   "worst_all_event_mass_loss": 1e9,
                                   "worst_all_event_pde_loss": 1e9},
            "assumptions": ["test"],
        }

    def test_completed_partial_run_can_resume_without_recreating_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = self._terrain(folder)
            events = [{"event_id": f"event_{split}", "split": split,
                       "mean_intensity_mm_hr": 10., "duration_s": 600.,
                       "total_mm": 10.}
                      for split in ("train", "validation", "test")]
            output = Path(folder) / "training"
            first_config = self._config(1); first_config["resume_training"] = True
            _, first = train(terrain, events, first_config, output)
            self.assertEqual(first["completed_steps"], 1)
            self.assertTrue((output / "training_state.pt").is_file())
            state = torch.load(output / "training_state.pt", map_location="cpu",
                               weights_only=True)
            self.assertEqual(state["schema_version"], 2)

            resumed = self._config(2); resumed["resume_training"] = True
            _, second = train(terrain, events, resumed, output)
            self.assertEqual(second["completed_steps"], 2)
            history = (output / "training_history.csv").read_text(encoding="utf-8")
            self.assertIn("1", history); self.assertIn("2", history)

    def test_resume_rejects_changed_seed_or_event_physics(self):
        with tempfile.TemporaryDirectory() as folder:
            terrain = self._terrain(folder)
            events = [{"event_id": f"event_{split}", "split": split,
                       "mean_intensity_mm_hr": 10., "duration_s": 600.,
                       "total_mm": 10.}
                      for split in ("train", "validation", "test")]
            output = Path(folder) / "training"
            initial = self._config(1); initial["resume_training"] = True
            train(terrain, events, initial, output)

            changed_seed = self._config(2)
            changed_seed["resume_training"] = True
            changed_seed["seed"] = 4
            with self.assertRaisesRegex(ValueError, "does not match"):
                train(terrain, events, changed_seed, output)

            changed_events = [dict(event) for event in events]
            changed_events[0]["mean_intensity_mm_hr"] = 11.
            valid_config = self._config(2); valid_config["resume_training"] = True
            with self.assertRaisesRegex(ValueError, "does not match"):
                train(terrain, changed_events, valid_config, output)


if __name__ == "__main__": unittest.main()
