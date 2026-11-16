import tempfile
import unittest
from pathlib import Path

import torch

from minxiong_pinn.model import ConditionalFloodPINN
from minxiong_pinn.predict import load_model


def _scales():
    return {
        "x0": 0., "y0": 0., "lx": 100., "ly": 100.,
        "time_s": 7200., "rain_m_s": 800. / 3_600_000.,
        "duration_s": 86400., "depth_m": 1., "velocity_m_s": 2.,
        "ramp_s": 600.,
    }


class _Terrain:
    fingerprint = "fixed-test-terrain"


def _checkpoint(model, *, include_architecture=True):
    checkpoint = {
        "model_state": model.state_dict(), "scales": _scales(),
        "width": 8, "layers": 4,
        "terrain_fingerprint": _Terrain.fingerprint,
        "training_rain_domain": {
            "rainfall_mm_hr": [0., 800.], "duration_s": [3600., 86400.],
        },
    }
    if include_architecture:
        checkpoint["network_architecture"] = model.network_architecture
    return checkpoint


class ResidualNetworkTests(unittest.TestCase):
    def test_default_plain_architecture_loads_legacy_checkpoint(self):
        model = ConditionalFloodPINN(_scales(), width=8, layers=4)
        self.assertEqual(model.network_architecture, "plain")
        self.assertIsInstance(model.net, torch.nn.Sequential)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(_checkpoint(model, include_architecture=False), path)
            loaded, checkpoint = load_model(path, _Terrain())
        self.assertNotIn("network_architecture", checkpoint)
        self.assertEqual(loaded.network_architecture, "plain")
        loaded.load_state_dict(model.state_dict())

    def test_residual_checkpoint_round_trip(self):
        model = ConditionalFloodPINN(
            _scales(), width=8, layers=4, network_architecture="residual")
        points = torch.tensor([
            [25., 30., 1800., 5. / 3_600_000., 3600.],
            [75., 60., 5400., 20. / 3_600_000., 3600.],
        ], dtype=torch.float64)
        expected = model(points)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(_checkpoint(model), path)
            loaded, checkpoint = load_model(path, _Terrain())
        self.assertEqual(checkpoint["network_architecture"], "residual")
        self.assertEqual(loaded.network_architecture, "residual")
        torch.testing.assert_close(loaded(points), expected)

    def test_residual_forward_input_derivatives_and_backward_are_finite(self):
        model = ConditionalFloodPINN(
            _scales(), width=12, layers=6, network_architecture="residual")
        points = torch.tensor([
            [20., 30., 1200., 5. / 3_600_000., 3600.],
            [60., 70., 4800., 30. / 3_600_000., 7200.],
        ], dtype=torch.float64, requires_grad=True)
        values = model(points)
        derivatives = torch.autograd.grad(
            values.sum(), points, create_graph=True, retain_graph=True)[0]
        loss = values.square().mean() + derivatives[:, :3].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(derivatives).all())
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_zero_output_layers_support_residual_separate_networks(self):
        model = ConditionalFloodPINN(
            _scales(), width=8, layers=4, separate_output_networks=True,
            network_architecture="residual")
        model.zero_output_layers()
        points = torch.tensor(
            [[50., 50., 1800., 20. / 3_600_000., 3600.]],
            dtype=torch.float64)
        expected_depth = points[:, 3:4] * points[:, 2:3]
        values = model(points)
        torch.testing.assert_close(values[:, :1], expected_depth.to(values.dtype))
        torch.testing.assert_close(values[:, 1:], torch.zeros_like(values[:, 1:]))
        for network in (model.depth_net, model.velocity_net):
            output = model._output_layer(network)
            torch.testing.assert_close(output.weight, torch.zeros_like(output.weight))
            torch.testing.assert_close(output.bias, torch.zeros_like(output.bias))

    def test_invalid_network_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "plain or residual"):
            ConditionalFloodPINN(
                _scales(), width=8, layers=4, network_architecture="fourier")


if __name__ == "__main__":
    unittest.main()
