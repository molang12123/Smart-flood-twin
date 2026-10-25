import unittest
import numpy as np
from flood_pinn.rainfall import Rainfall,output_times


class RainfallTests(unittest.TestCase):
    def test_units_and_stopping_integral(self):
        a=Rainfall.from_duration(36,2,'min');b=Rainfall.from_duration(36,1/30,'h')
        self.assertEqual(a,b);self.assertAlmostEqual(a.rate_ms,1e-5)
        np.testing.assert_array_equal(a.rate([0,119,120,121]),[1e-5,1e-5,0,0])
        self.assertAlmostEqual(float(a.integral(119,121)),1e-5)
        self.assertAlmostEqual(float(a.integral(0,300)),.0012)
        self.assertEqual(float(Rainfall(0,120).integral(0,300)),0)
        self.assertEqual(float(Rainfall(36,0).integral(0,300)),0)

    def test_invalid_inputs(self):
        for intensity,duration in [(-1,2),(1,-1),(float('inf'),2),(1,float('nan'))]:
            with self.assertRaises(ValueError): Rainfall(intensity,duration)
        with self.assertRaises(ValueError): Rainfall.from_duration(1,2,'guess')
        with self.assertRaises(ValueError): Rainfall(1,2).integral(2,1)

    def test_complete_and_requested_times(self):
        np.testing.assert_array_equal(output_times(300,500,180),[0,90,180,240,300])
        np.testing.assert_array_equal(output_times(300,30,180,[180,150,150],mode='requested'),[150,180])
        for args in [(0,1,0),(10,0,1),(10,1,11)]:
            with self.assertRaises(ValueError): output_times(*args)
        with self.assertRaises(ValueError): output_times(10,1,2,[-1])
        with self.assertRaises(ValueError): output_times(10,1,2,[],mode='requested')
        with self.assertRaises(ValueError): output_times(10,1,2,mode='unknown')
        np.testing.assert_array_equal(output_times(10,50,0),[0,5,10])
        np.testing.assert_array_equal(output_times(10,50,10),[0,5,10])
        times=output_times(1,.1,.3)
        self.assertEqual(np.count_nonzero(np.abs(times-.3)<1e-12),1)
        self.assertEqual(times[-1],1.)

if __name__=='__main__': unittest.main()
