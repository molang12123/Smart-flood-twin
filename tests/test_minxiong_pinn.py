from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import rasterio
from rasterio.transform import from_origin

from minxiong_pinn.rainfall import extract_events
from minxiong_pinn.terrain import MaskedRasterTerrain


class MinxiongPINNTests(unittest.TestCase):
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
