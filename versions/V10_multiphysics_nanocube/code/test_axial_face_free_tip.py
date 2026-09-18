import unittest
import numpy as np
from scipy.constants import Boltzmann
from scipy.special import i0e,i1e
from axial_face_free_tip import face_model,tip_model,cage_cost,compute


class AxialFreeTests(unittest.TestCase):
    def test_tip_zero_and_weak_coupling(self):
        self.assertAlmostEqual(tip_model(0,250)['F_orient_J']/Boltzmann/250,0,places=12)
        kbt=Boltzmann*250
        r=tip_model(.001*kbt,250)
        self.assertAlmostEqual(r['Udd_J']/kbt,-2e-6/3,places=11)
        self.assertAlmostEqual(r['F_orient_J']/kbt,-1e-6/3,places=11)

    def test_tip_energy_partition_derivative_and_quadrature(self):
        C=5e-21
        r=tip_model(C,250,80)
        hi=tip_model(C,250.01)
        lo=tip_model(C,249.99)
        thermodynamic_u=r['F_orient_J']-250*(hi['F_orient_J']-lo['F_orient_J'])/.02
        self.assertAlmostEqual((thermodynamic_u-r['Udd_J'])/(Boltzmann*250),0,places=7)
        self.assertAlmostEqual((r['Udd_J']-tip_model(C,250,160)['Udd_J'])/(Boltzmann*250),0,places=10)

    def test_face_limits(self):
        C=1e-20
        short=face_model(C,250,1e-7)
        self.assertAlmostEqual(short['p_parallel_mean'],.5,places=7)
        self.assertLess(short['Udd_J'],0)  # Fast azimuthal alignment survives slow Neel flips.
        long=face_model(C,300,1000)
        self.assertAlmostEqual(long['p_parallel_end'],long['p_parallel_eq'],places=10)
        x=2*C/3/(Boltzmann*300)
        self.assertGreaterEqual(long['F_orient_J'],long['F_orient_eq_J']-1e-30)
        self.assertLessEqual(long['Udd_J'],0)

    def test_three_stage_bookkeeping(self):
        for r in compute(200):
            kbt=Boltzmann*200
            self.assertLess(r['Udd_J'],0)
            self.assertEqual(r['stage1_Udd_J'],r['Udd_J'])
            self.assertEqual(r['stage2_F_orient_J'],r['F_orient_J'])
            self.assertNotEqual(r['UvdW_J'],0)
            self.assertAlmostEqual((r['stage2_F_orient_J']-
                                    r['stage1_Udd_J']-r['orientation_entropy_cost_J'])/kbt,0,places=12)
            self.assertAlmostEqual((r['stage3_F_with_rotation_J']-
                                    r['stage2_F_orient_J']-r['rotation_constraint_F_J'])/kbt,0,places=12)
        self.assertEqual(cage_cost('tip',250)[0],0)


if __name__=='__main__':
    unittest.main()
