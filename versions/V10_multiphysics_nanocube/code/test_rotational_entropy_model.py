import unittest
import numpy as np
from scipy.stats import qmc
from rotational_entropy_model import (orientation_fraction, rotate, body_rotate,
                                      basin_grid, evaluate_samples)


class RotationalEntropyTests(unittest.TestCase):
    def test_symmetry_and_entropy_ratio(self):
        for delta in (1,2,5,10):
            f = orientation_fraction('face',2,delta)
            t = orientation_fraction('tip',2,delta)
            self.assertGreater(f,0)
            self.assertLess(t,1)
            self.assertAlmostEqual(t/f,np.pi/(3*np.deg2rad(delta)),places=10)
            self.assertLess(-2*np.log(t),-2*np.log(f))

    def test_rotation_preserves_norm_and_twist_projection(self):
        u=qmc.Sobol(10,scramble=True,seed=42).random_base2(10)
        m=np.tile(np.array([1,1,1])/np.sqrt(3),(len(u),1))
        axis=np.array([1.,0,0])
        spun=rotate(m,np.broadcast_to(axis,m.shape),u[:,0]*2*np.pi)
        np.testing.assert_allclose(np.linalg.norm(spun,axis=1),1,atol=1e-12)
        np.testing.assert_allclose(spun@axis,m@axis,atol=1e-12)
        for kind in ('face','tip'):
            lab=body_rotate(m,u[:,4:7],axis,kind,2,5)
            np.testing.assert_allclose(np.linalg.norm(lab,axis=1),1,atol=1e-12)

    def test_zero_coupling_partition(self):
        u=qmc.Sobol(10,scramble=True,seed=13).random_base2(12)
        grid=basin_grid(48)
        for kind in ('face','tip'):
            values=evaluate_samples(kind,250,2,5,u,grid,coupling=0)
            self.assertAlmostEqual(values[0],0,places=12)
            self.assertAlmostEqual(values[1],0,places=12)
            self.assertAlmostEqual(values[4],1,places=12)

    def test_quadrature_normalization_and_resolution(self):
        from scipy.constants import Boltzmann
        from scipy.special import logsumexp
        results=[]
        for order in (48,64):
            d,w,a=basin_grid(order)
            self.assertAlmostEqual(w.sum(),1,places=12)
            np.testing.assert_allclose(np.linalg.norm(d,axis=1),1,atol=1e-12)
            p=np.exp(np.log(w)-a/(Boltzmann*200))
            results.append((logsumexp(np.log(w)-a/(Boltzmann*200)),(p@a)/p.sum()/(Boltzmann*200)))
        np.testing.assert_allclose(results[0],results[1],atol=2e-5,rtol=0)


if __name__=='__main__':
    unittest.main()
