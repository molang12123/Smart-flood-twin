import unittest
from types import SimpleNamespace
import numpy as np
from flood_pinn.exchange import build_faces,face_fluxes,integrate_exchange,cell_sides


def grid():
    x,y=np.meshgrid([1.,3.],[1.5,4.5],indexing='xy')
    return SimpleNamespace(nx=2,ny=2,dx=2.,dy=3.,x_origin=0.,y_origin=0.,
                           bounds=(0.,4.,0.,6.),x=x.ravel(),y=y.ravel(),z=np.zeros((2,2)))


def rain():
    return SimpleNamespace(intensity_mm_hr=0.,duration_s=.5)


class ExchangeTests(unittest.TestCase):
    def test_topology_and_constant_flux(self):
        terrain=grid(); faces=build_faces(terrain)
        self.assertEqual(len(faces.ids),12)
        self.assertEqual(sum(faces.boundary==''),4)
        self.assertTrue(np.array_equal(faces.ids,np.arange(12)))
        evaluate=lambda p: np.tile([2.,3.,4.],(len(p),1))
        q=face_fluxes(terrain,evaluate,0.,rain())
        np.testing.assert_allclose(q[faces.axis=='x'],18.)
        np.testing.assert_allclose(q[faces.axis=='y'],16.)
        net=np.zeros(4)
        for side,(indices,sign) in cell_sides(terrain).items(): net+=sign*q[indices]
        np.testing.assert_allclose(net,0,atol=1e-14)
        out=sum(q[faces.boundary=='east'])+sum(q[faces.boundary=='north'])
        incoming=sum(q[faces.boundary=='west'])+sum(q[faces.boundary=='south'])
        self.assertAlmostEqual(out,incoming)
    def test_reversal_volume_not_net_sign(self):
        terrain=grid()
        def evaluate(p):
            return np.column_stack([np.ones(len(p)),p[:,2]-.5,np.zeros(len(p))])
        result=integrate_exchange(terrain,evaluate,rain(),0.,1.,time_substeps=1)
        faces=build_faces(terrain); mask=faces.axis=='x'
        np.testing.assert_allclose(result['V_forward'][mask],.125*3,atol=1e-14)
        np.testing.assert_allclose(result['V_reverse'][mask],.125*3,atol=1e-14)
        np.testing.assert_allclose(result['V_signed'],0,atol=1e-14)
        np.testing.assert_allclose(result['error_est'],0,atol=1e-14)
    def test_quadratic_spatial_flux(self):
        terrain=grid(); faces=build_faces(terrain)
        def evaluate(p): return np.column_stack([np.ones(len(p)),p[:,1]**2,np.zeros(len(p))])
        q=face_fluxes(terrain,evaluate,0.,rain())
        np.testing.assert_allclose(q[:3],9.,atol=1e-14)
        np.testing.assert_allclose(q[3:6],63.,atol=1e-14)
    def test_invalid_quadrature(self):
        evaluate=lambda p: np.tile([1.,0.,0.],(len(p),1))
        for n in [0,-1,1.5,True]:
            with self.assertRaises(ValueError): integrate_exchange(grid(),evaluate,rain(),0,1,time_substeps=n)
