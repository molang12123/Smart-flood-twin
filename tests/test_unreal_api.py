import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from minxiong_pinn.unreal_api import UnrealFloodService, make_server


class _Terrain:
    active_count = 12


class _Model:
    scales = {"time_s": 36 * 3600.}


def _checkpoint(high=300.):
    return {
        "expert_bundle": high == 300.,
        "bundle_id": "approved-0-300" if high == 300. else None,
        "training_rain_domain": {
            "rainfall_mm_hr": [0. if high == 300. else .5, high],
            "duration_s": [3600., 24 * 3600.],
        },
    }


class _Predictor:
    def __init__(self):
        self.calls = []

    def __call__(self, model, checkpoint, terrain, rainfall, duration, times,
                 output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "frame_000.csv").write_text(
            "time_s,row,col,depth_m\n0,0,0,0\n", encoding="utf-8")
        (output / "depth_000.tif").write_bytes(b"test-tif")
        (output / "frame_000.npz").write_bytes(b"test-npz")
        self.calls.append((rainfall, duration, tuple(times), output))
        return {"model_id": "medium", "frames": [
            {"time_s": 0., "max_depth_m": 0., "mean_depth_m": 0.,
             "max_speed_m_s": 0., "csv_file": "frame_000.csv"}]}


def _valid(**changes):
    value = {
        "rainfall_mm_hr": 123.456,
        "duration_hr": 7.25,
        "simulation_hours": 12.5,
        "interval_minutes": 30,
        "output_name": "unreal_case_01",
    }
    value.update(changes)
    return value


class UnrealServiceTests(unittest.TestCase):
    def service(self, folder, *, high=300.):
        predictor = _Predictor()
        service = UnrealFloodService(
            _Terrain(), _Model(), _checkpoint(high), Path(folder) / "outputs",
            predictor=predictor, max_frames=100)
        return service, predictor

    def test_schema_health_and_prediction_return_direct_file_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            service, predictor = self.service(folder)
            self.assertTrue(service.health()["expert_bundle_active"])
            self.assertEqual(
                service.schema()["active_model_domain"]["rainfall_mm_hr"],
                [0., 300.])
            result = service.run_prediction(_valid())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["request"]["duration_hr"], 7.25)
            self.assertEqual(result["request"]["simulation_hours"], 12.5)
            self.assertGreater(result["request"]["frame_count"], 1)
            self.assertEqual(len(result["files"]["csv"]), 1)
            self.assertEqual(len(result["files"]["geotiff"]), 1)
            self.assertEqual(len(result["files"]["npz"]), 1)
            self.assertTrue(Path(result["files"]["csv"][0]).is_file())
            self.assertEqual(len(predictor.calls), 1)

    def test_active_checkpoint_range_is_authoritative_and_path_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            service, predictor = self.service(folder, high=23.)
            health = service.health()
            self.assertTrue(health["fallback_active"])
            self.assertEqual(
                health["active_model_domain"]["rainfall_mm_hr"], [.5, 23.])
            with self.assertRaisesRegex(ValueError, "模型范围"):
                service.run_prediction(_valid(rainfall_mm_hr=100.))
            for name in ("../escape", "nested/name", "..", "a" * 65):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    service.run_prediction(_valid(
                        rainfall_mm_hr=10., output_name=name))
            self.assertFalse((Path(folder) / "escape").exists())
            self.assertFalse(predictor.calls)

    def test_missing_invalid_existing_and_frame_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            service, _ = self.service(folder)
            with self.assertRaisesRegex(ValueError, "missing required"):
                service.run_prediction({"rainfall_mm_hr": 10})
            with self.assertRaisesRegex(ValueError, "finite"):
                service.run_prediction(_valid(rainfall_mm_hr=float("nan")))
            service.run_prediction(_valid())
            with self.assertRaises(FileExistsError):
                service.run_prediction(_valid())
            with self.assertRaisesRegex(ValueError, "超过上限"):
                service.run_prediction(_valid(
                    output_name="too_many", simulation_hours=36,
                    interval_minutes=1))


class UnrealHttpTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        predictor = _Predictor()
        self.service = UnrealFloodService(
            _Terrain(), _Model(), _checkpoint(300.),
            Path(self.folder.name) / "outputs", predictor=predictor,
            max_frames=100)
        self.server = make_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.folder.cleanup()

    def request(self, method, path, body=None, headers=None):
        self.connection.request(method, path, body=body, headers=headers or {})
        response = self.connection.getresponse()
        raw = response.read()
        value = json.loads(raw) if raw else None
        return response, value

    def test_get_options_and_successful_post(self):
        response, health = self.request("GET", "/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), "*")
        self.assertEqual(health["status"], "ready")
        response, schema = self.request("GET", "/schema")
        self.assertEqual(response.status, 200)
        self.assertEqual(schema["endpoint"], "/predict")
        response, _ = self.request("OPTIONS", "/predict")
        self.assertEqual(response.status, 204)
        body = json.dumps(_valid(output_name="http_case"))
        response, result = self.request(
            "POST", "/predict", body,
            {"Content-Type": "application/json"})
        self.assertEqual(response.status, 200)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["files"]["csv"]), 1)

    def test_structured_http_errors(self):
        response, error = self.request("GET", "/unknown")
        self.assertEqual(response.status, 404)
        self.assertEqual(error["error"], "not_found")
        response, error = self.request(
            "POST", "/predict", "not json",
            {"Content-Type": "application/json"})
        self.assertEqual(response.status, 400)
        self.assertEqual(error["error"], "invalid_json")
        body = json.dumps(_valid(output_name="bad/path"))
        response, error = self.request(
            "POST", "/predict", body,
            {"Content-Type": "application/json"})
        self.assertEqual(response.status, 400)
        self.assertEqual(error["error"], "invalid_request")
        body = json.dumps(_valid(output_name="wrong_type"))
        response, error = self.request("POST", "/predict", body)
        self.assertEqual(response.status, 415)
        self.assertEqual(error["error"], "content_type")


if __name__ == "__main__":
    unittest.main()
