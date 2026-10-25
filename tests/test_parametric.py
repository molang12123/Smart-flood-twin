import unittest

import numpy as np
import torch
from torch import nn

from flood_pinn.physics import derivative, manning_source, residuals


class AnalyticTerrain:
    def __init__(self, zx=0.0, zy=0.0, n=0.0):
        self.zx, self.zy, self.n = zx, zy, n

    def query(self, x, y):
        return dict(z=self.zx * x + self.zy * y,
                    zx=np.full_like(x, self.zx), zy=np.full_like(y, self.zy),
                    n=np.full_like(x, self.n))


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.points = torch.tensor([
            [.2, .3, .2, 2e-5, .5], [.7, .6, .8, 4e-5, .5],
        ], dtype=torch.float64, requires_grad=True)

    def test_lake_at_rest_over_sloping_terrain(self):
        class Lake(nn.Module):
            def forward(self, p):
                h = 2 - .1 * p[:, 0:1] + .2 * p[:, 1:2]
                zero = h * 0
                return torch.cat([h, zero, zero], dim=1)
        p = self.points.detach().clone()
        p[:, 3] = 0
        result = residuals(Lake(), p, AnalyticTerrain(.1, -.2))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-13, rtol=0)

    def test_each_sample_has_its_own_rainfall_and_stop(self):
        class RainAccumulation(nn.Module):
            def forward(self, p):
                h = .1 + p[:, 3:4] * torch.minimum(p[:, 2:3], p[:, 4:5])
                return torch.cat([h, h * 0, h * 0], 1)
        result = residuals(RainAccumulation(), self.points, AnalyticTerrain())
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-15, rtol=0)

    def test_conservative_friction_has_depth_factor(self):
        h = torch.tensor([[.001], [.01]], dtype=torch.float64)
        u = torch.full_like(h, .2)
        v = torch.full_like(h, -.1)
        n = torch.full_like(h, .02)
        sx, sy = manning_source(h, u, v, n)
        speed = torch.sqrt(u * u + v * v + 1e-12)
        expected = 9.81 * n.square() * speed / h.pow(1 / 3)
        torch.testing.assert_close(sx, expected * u)
        torch.testing.assert_close(sy, expected * v)

    def test_constant_fields_and_non_grad_input(self):
        class Constant(nn.Module):
            def forward(self, p):
                return torch.tensor([1., 0., 0.], dtype=p.dtype).expand(len(p), 3)
        p = self.points.detach().clone()
        p[:, 3] = 0
        result = residuals(Constant(), p, AnalyticTerrain())
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_physical_derivatives_use_only_x_y_t(self):
        f = (self.points[:, 0:1] ** 2 + 3 * self.points[:, 1:2] +
             7 * self.points[:, 2:3] + 11 * self.points[:, 3:4])
        torch.testing.assert_close(derivative(f, self.points, 0), 2 * self.points[:, 0:1])
        torch.testing.assert_close(derivative(f, self.points, 1), torch.full_like(f, 3))
        torch.testing.assert_close(derivative(f, self.points, 2), torch.full_like(f, 7))
        with self.assertRaises(ValueError):
            derivative(f, self.points, 3)

    def test_bad_physics_parameters_rejected(self):
        for invalid in [0., -1., float('nan')]:
            with self.assertRaises(ValueError):
                manning_source(torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1), h_reg=invalid)


class ParametricNetworkTests(unittest.TestCase):
    @staticmethod
    def scales():
        return dict(x_origin=250000., y_origin=2600000., x_scale=100., y_scale=80.,
                    t_scale=120., r_scale=1e-5, duration_scale=120., h_scale=.1, u_scale=1.)

    def test_parameter_count_and_finite_backpropagation(self):
        from flood_pinn.model import ParametricPINN
        model = ParametricPINN(self.scales())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 83715)
        points = torch.tensor([[250020., 2600030., 10., 1e-5, 60.],
                               [250040., 2600050., 100., 2e-5, 80.]])
        fields = model(points)
        self.assertTrue((fields[:, 0] >= 0).all())
        fields.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_forward_normalization_chain_rule(self):
        from flood_pinn.model import ParametricPINN
        # Network-input hook verifies all physical input scales independently.
        model = ParametricPINN(self.scales(), width=8, layers=2).double()
        captured = []
        first = next(layer for layer in model.modules() if isinstance(layer, nn.Linear))
        handle = first.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        points = torch.tensor([[250050., 2600040., 60., 2e-5, 30.]],
                              dtype=torch.float64, requires_grad=True)
        model(points)
        handle.remove()
        normalized = captured[0]
        torch.testing.assert_close(normalized, torch.tensor([[.5, .5, .5, 2., .25]], dtype=torch.float64))
        gradient = torch.autograd.grad(normalized[:, 0].sum(), points)[0]
        self.assertAlmostEqual(gradient[0, 0].item(), .01)

    def test_single_to_conditional_initialization_preserves_forward(self):
        from flood_pinn.model import ParametricPINN
        single = ParametricPINN(self.scales(), width=8, layers=2, parameterized=False)
        conditional = ParametricPINN(self.scales(), width=8, layers=2)
        conditional.initialize_from_single(single)
        points = torch.tensor([[250020., 2600030., 10., 1e-5, 60.],
                               [250040., 2600050., 100., 2e-5, 80.]])
        torch.testing.assert_close(conditional(points), single(points), atol=1e-7, rtol=1e-6)

    def test_centimetre_offsets_survive_large_gis_origin_with_float32_weights(self):
        from flood_pinn.model import ParametricPINN
        model = ParametricPINN(self.scales(), width=8, layers=2)
        points = torch.tensor([[250000.01, 2600000.02, 1., 1e-5, 60.]],
                              dtype=torch.float64, requires_grad=True)
        captured = []
        hook = model.net[0].register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        model(points); hook.remove()
        self.assertEqual(captured[0].dtype, torch.float32)
        self.assertAlmostEqual(captured[0][0, 0].item(), .01/100., places=10)
        self.assertAlmostEqual(captured[0][0, 1].item(), .02/80., places=10)
        gradient = torch.autograd.grad(captured[0][:, 1].sum(), points)[0]
        self.assertAlmostEqual(gradient[0, 1].item(), 1/80., places=14)


if __name__ == '__main__':
    unittest.main()
