import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class ModelTests(unittest.TestCase):
    def test_structure_and_finite_gradients(self):
        from flood_pinn.case_a import PINN, residuals, s_int
        model = PINN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 83459)
        residual = residuals(model, *s_int(8))
        loss = sum(x.square().mean() for x in residual)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_frozen_closure_same_loss_and_gradient(self):
        from flood_pinn.case_a import PINN, make_lbfgs_closure
        model = PINN(n_hidden=8, n_layers=2)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=2)
        closure = make_lbfgs_closure(model, optimizer, Np=8, Ni=4, Nb=2)
        first = closure().item()
        gradients = [p.grad.clone() for p in model.parameters()]
        second = closure().item()
        self.assertEqual(first, second)
        for expected, parameter in zip(gradients, model.parameters()):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

    def test_legacy_forward_equivalence_and_source_integrity(self):
        from flood_pinn import case_a
        path = Path(__file__).resolve().parents[1]/'legacy/PINN_Case_A_inclined_plane.py'
        source = path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '4c648425b0df8ddeb9b508a2a02918a3534d38b381a808bd72938b6521840bc7')
        original_class = next(node for node in ast.parse(source).body
                              if isinstance(node, ast.ClassDef) and node.name == 'PINN')
        namespace = {name: getattr(case_a, name) for name in
                     ['torch', 'nn', 'x_sc', 'y_sc', 't_sc', 'h_sc', 'u_sc']}
        exec(compile(ast.Module(body=[original_class], type_ignores=[]), '<legacy PINN class>', 'exec'), namespace)
        old = namespace['PINN'](n_hidden=8, n_layers=2)
        new = case_a.PINN(n_hidden=8, n_layers=2)
        new.load_state_dict(old.state_dict())
        x, y, t = torch.rand(9, 1), torch.rand(9, 1), torch.rand(9, 1)*300
        for expected, actual in zip(old(x, y, t), new(x, y, t)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_conservative_manning_wet_and_regularized_dry(self):
        from flood_pinn.case_a import manning_source
        h = torch.tensor([[.02], [1e-6]], dtype=torch.float64)
        u = torch.full_like(h, .3)
        v = torch.full_like(h, -.4)
        fx, fy = manning_source(h, u, v)
        regularized = h.clamp(min=1e-4)
        coefficient = 9.81*.02**2*h*torch.sqrt(u*u+v*v+1e-10)/regularized**(4/3)
        torch.testing.assert_close(fx, coefficient*u)
        torch.testing.assert_close(fy, coefficient*v)
        # At wet depths this is h^(-1/3), not acceleration h^(-4/3).
        wet_expected = 9.81*.02**2*torch.sqrt(u[0]**2+v[0]**2+1e-10)*u[0]/h[0]**(1/3)
        torch.testing.assert_close(fx[0], wet_expected)

    def test_manufactured_constant_field_residual(self):
        from flood_pinn.case_a import residuals, manning_source
        class ConstantField(torch.nn.Module):
            def forward(self, x, y, t):
                zero = 0*(x+y+t)
                return zero+.02, zero+.3, zero-.4
        x, y, t = [torch.ones(5, 1, dtype=torch.float64, requires_grad=True)*value
                   for value in [.4, .6, 200.]]
        continuity, rx, ry = residuals(ConstantField(), x, y, t)
        expected_x, expected_y = manning_source(torch.full_like(x, .02),
                                                torch.full_like(x, .3), torch.full_like(x, -.4))
        torch.testing.assert_close(continuity, torch.zeros_like(x))
        torch.testing.assert_close(rx, expected_x - 9.81*.02*.05)
        torch.testing.assert_close(ry, expected_y)

    def test_no_training_or_outputs_at_import(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            command = f'import sys; sys.path.insert(0, {str(root)!r}); import flood_pinn.case_a'
            result = subprocess.run([sys.executable, '-c', command], cwd=directory,
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_rainfall_endpoint_and_missing_anchors(self):
        from flood_pinn.case_a import rainfall, PINN, compute_loss
        values = rainfall(torch.tensor([[0.], [179.], [180.], [181.]]))
        self.assertGreater(values[1].item(), 0.)
        self.assertEqual(values[2].item(), 0.)
        with self.assertRaisesRegex(ValueError, 'reference anchors'):
            compute_loss(PINN(n_hidden=4, n_layers=1), Np=2, Ni=2, Nb=2)


if __name__ == '__main__':
    unittest.main()
