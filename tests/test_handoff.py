import unittest
import numpy as np
from flood_pinn.coordinates import to_ue, from_ue, velocity_to_ue
from flood_pinn.assessment import assess


class HandoffTests(unittest.TestCase):
    def test_anchor_axis_units_and_roundtrip(self):
        origin=np.array([200000.,2600000.,30.])
        points=origin+np.array([[0,0,0],[20,0,0],[0,20,0],[0,0,1]])
        matrix=np.diag([1,-1,1])
        expected=np.array([[0,0,0],[2000,0,0],[0,-2000,0],[0,0,100]])
        np.testing.assert_allclose(to_ue(points,origin,matrix),expected)
        np.testing.assert_allclose(from_ue(expected,origin,matrix),points)
        np.testing.assert_allclose(velocity_to_ue([[1,2,0]],matrix),[[100,-200,0]])
        with self.assertRaises(ValueError): to_ue(points,origin,np.ones((3,3)))

    def test_no_missing_evidence_is_counted_as_pass(self):
        self.assertEqual(assess({}, {})['status'],'not_evaluated')
        m={'scenarios':[{'scenario_id':'test','depth_rmse_m':.1}]}
        self.assertEqual(assess(m,{}, {'depth_rmse_m_max':.01})['status'],'failed')
        self.assertEqual(assess(m,{}, {'depth_rmse_m_max':1.})['status'],'not_evaluated')
