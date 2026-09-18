import unittest
import numpy as np
from scipy.constants import Boltzmann
from brownian_configuration_hopping import brownian_generator,path_maximum,compute
import geometry_model as g


class ConfigurationHoppingTests(unittest.TestCase):
    def test_isolated_first_rank_brownian_time(self):
        s=g.cubic_easy_axis_states();tau=2e-6
        Q,_=brownian_generator(250,np.zeros(64),0.,s,np.array([1.,0,0]),tau)
        moments=np.repeat(s,8,axis=0)
        np.testing.assert_allclose(tau*Q@moments,-moments,atol=1e-14)
        np.testing.assert_allclose(Q.sum(axis=1),0,atol=1e-8)

    def test_path_maximum_matches_dense_rotation(self):
        rng=np.random.default_rng(31)
        for _ in range(12):
            m,other,n=rng.normal(size=(3,3));m/=np.linalg.norm(m);other/=np.linalg.norm(other);n/=np.linalg.norm(n)
            axis=np.eye(3)[rng.integers(3)]
            theta=np.linspace(0,np.pi/2,20001)
            rotated=(m[None,:]*np.cos(theta)[:,None]+np.cross(axis,m)[None,:]*np.sin(theta)[:,None]
                     +axis[None,:]*np.dot(axis,m)*(1-np.cos(theta))[:,None])
            es=rotated@other-3*(rotated@n)*np.dot(other,n)
            self.assertAlmostEqual(path_maximum(1.,m,other,axis,n),max(es),places=8)

    def test_detailed_balance_for_both_geometries(self):
        for n in (np.array([1.,0,0]),np.ones(3)/np.sqrt(3)):
            r=g.center_distance_at_gap_m(n,3e-9)
            energy,C,s,n=g.easy_axis_pair_energies_J(r*n)
            Q,_=brownian_generator(200,energy,C,s,n,4e-6)
            weights=np.exp(-(energy-energy.min())/(Boltzmann*200));weights/=weights.sum()
            flux=weights[:,None]*Q
            np.testing.assert_allclose(flux,flux.T,rtol=1e-11,atol=1e-9)

    def test_disabled_brownian_restores_old_model(self):
        for kind in ('face','tip'):
            r=compute(200,kind,mobility=0)
            self.assertAlmostEqual(r['Udd_time_mean_kBT'],r['Neel_only_Udd_time_mean_kBT'],places=12)

    def test_finite_window_and_unchanged_equilibrium(self):
        r=compute(200,'tip')
        self.assertLess(r['Udd_time_mean_kBT'],-2.8)
        self.assertLess(abs(r['Udd_time_mean_kBT']-r['Udd_eq_kBT']),2e-6)
        self.assertAlmostEqual(r['Udd_eq_kBT'],-2.812575258415973,places=10)
        short=compute(200,'tip',window_s=1e-12)
        self.assertLess(abs(short['Udd_time_mean_kBT']),1e-5)
        self.assertLess(r['Udd_time_mean_kBT'],short['Udd_time_mean_kBT'])


if __name__=='__main__':unittest.main()
