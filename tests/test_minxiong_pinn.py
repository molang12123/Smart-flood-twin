from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from minxiong_pinn.model import ConditionalFloodPINN
from flood_pinn.physics import residuals
from minxiong_pinn.rainfall import add_design_events, extract_events
from minxiong_pinn.terrain import MaskedRasterTerrain


class MinxiongPINNTests(unittest.TestCase):
    def test_log_conditioning_resolves_full_rainfall_range(self):
        scales = {"x0": 0., "y0": 0., "lx": 1., "ly": 1., "time_s": 1.,
                  "rain_m_s": 700 / 3_600_000., "rain_min_m_s": .5 / 3_600_000.,
                  "duration_s": 37 * 3600., "duration_min_s": 3600.,
                  "depth_m": 26., "velocity_m_s": 20., "ramp_s": 60.}
        model = ConditionalFloodPINN(scales, width=4, layers=1, log_conditions=True)
        captured = []
        handle = model.net[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        points = torch.tensor([[0., 0., 1., .5 / 3_600_000., 3600.],
                               [0., 0., 1., 700 / 3_600_000., 37 * 3600.]], dtype=torch.float64)
        model(points); handle.remove()
        torch.testing.assert_close(captured[0][:, 3:5], torch.tensor([[0., 0.], [1., 1.]]), atol=1e-6, rtol=0)

    def test_design_grid_trains_domain_edges_and_holds_out_interior(self):
        rates = [.5, 1., 2., 5., 10., 20., 50., 100., 200., 350., 500., 700.]
        durations = [1., 2., 3., 6., 12., 18., 24., 30., 37.]
        events = add_design_events([], rates, durations)
        self.assertEqual(len(events), len(rates) * len(durations))
        edges = [event for event in events if event["mean_intensity_mm_hr"] in {.5, 700.}
                 or event["hours"] in {1., 37.}]
        self.assertTrue(all(event["split"] == "train" for event in edges))
        self.assertTrue(any(event["split"] == "validation" for event in events))
        self.assertTrue(any(event["split"] == "test" for event in events))
        self.assertTrue(all(event["source"] == "synthetic_parameter_design" for event in events))

    def test_flat_uniform_rain_is_an_exact_theoretical_solution(self):
        scales = {"x0": 0., "y0": 0., "lx": 100., "ly": 100.,
                  "time_s": 7200., "rain_m_s": 1e-5, "duration_s": 3600.,
                  "depth_m": 1., "velocity_m_s": 1., "ramp_s": 60.}
        model = ConditionalFloodPINN(scales, width=8, layers=2)
        for parameter in model.parameters():
            parameter.data.zero_()

        class Flat:
            def query(self, x, y):
                return {"z": np.zeros_like(x), "zx": np.zeros_like(x),
                        "zy": np.zeros_like(x), "n": np.full_like(x, .03)}

        rate = 10. / 3_600_000.
        points = torch.tensor([[10., 20., 1800., rate, 3600.],
                               [30., 40., 5400., rate, 3600.]], dtype=torch.float64,
                              requires_grad=True)
        result = model(points)
        torch.testing.assert_close(result[:, 0], torch.tensor([.005, .01]), rtol=2e-6, atol=1e-9)
        torch.testing.assert_close(residuals(model, points, Flat()), torch.zeros((2, 3), dtype=torch.float64),
                                   rtol=0, atol=1e-9)

    def test_mask_uses_raster_validity_and_keeps_negative_elevation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dem.tif"
            data = np.array([[-2., 0.], [3., 4.]], dtype=np.float32)
            with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                               dtype="float32", crs="EPSG:3826", transform=from_origin(0, 40, 20, 20),
                               nodata=0.) as dst:
                dst.write(data, 1)
            terrain = MaskedRasterTerrain(path, manning_n=.03)
            self.assertEqual(terrain.active_count, 3)
            self.assertEqual(terrain.z[0, 0], -2)
            self.assertEqual(terrain.query([10], [30])["z"][0], -2)
            with self.assertRaisesRegex(ValueError, "NoData"):
                terrain.query([30], [30])

    def test_rain_event_split_is_chronological(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "rain.zip"
            header = '日/時,' + ','.join(str(i) for i in range(1, 25)) + ',總和\n'
            lines = [header]
            for day in range(1, 32):
                values = ["1.0" if day % 3 == 1 and hour == 1 else "0.0" for hour in range(1, 25)]
                lines.append(str(day) + ',' + ','.join(values) + ',1.0\n')
            lines.append('總和,' + ','.join(['0'] * 25) + '\n')
            with ZipFile(archive, "w") as z:
                z.writestr("rain/C0M760-2024-01-Precipitation-hour.csv", ''.join(lines))
            events = extract_events(archive)
            self.assertEqual(len(events), 11)
            order = {"train": 0, "validation": 1, "test": 2}
            self.assertEqual([e["split"] for e in events], sorted([e["split"] for e in events], key=order.get))
            self.assertTrue(all(e["duration_s"] == 3600 for e in events))


if __name__ == "__main__":
    unittest.main()
