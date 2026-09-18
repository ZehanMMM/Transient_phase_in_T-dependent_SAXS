"""Run with: python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_pair_free_reference.py"""
import unittest
import numpy as np
from scipy.constants import Boltzmann
from scipy.linalg import expm

from pair_free_reference import (compute_point, equilibrium, ReversibleEvolution,
                                 state_thermodynamics)


class FreeReferenceTests(unittest.TestCase):
    def test_uncoupled_reference(self):
        row, initial, peq, pend, pmean = compute_point(250, [1, 0, 0], coupling_scale=0)
        for key in ("Udd_time_mean_kBT", "delta_Fmag_eq_kBT",
                    "delta_Fmag_time_mean_kBT"):
            self.assertAlmostEqual(row[key], 0, places=10)
        np.testing.assert_allclose(pend, initial, atol=1e-12)
        np.testing.assert_allclose(pmean, peq, atol=1e-12)

    def test_spectral_and_exact_window_mean(self):
        q = np.array([[-2., 2.], [1., -1.]])
        pi = np.array([1/3, 2/3])
        p0 = np.array([1., 0.])
        ev = ReversibleEvolution(q, pi, p0)
        for t in (1e-8, 0.1, 20):
            np.testing.assert_allclose(ev.at(t), p0 @ expm(q*t), atol=1e-12)
            block = np.zeros((4, 4))
            block[:2, :2] = q
            block[:2, 2:] = np.eye(2)
            exact = p0 @ expm(block*t)[:2, 2:] / t
            np.testing.assert_allclose(ev.mean(t), exact, atol=1e-12)

    def test_kl_identity(self):
        energies = Boltzmann*250*np.array([-3., -1., 2.])
        peq, feq = equilibrium(energies, 250)
        for p in (np.array([1., 0., 0.]), np.ones(3)/3, peq):
            _, f, kl = state_thermodynamics(p, energies/(Boltzmann*250), peq)
            self.assertAlmostEqual(f-feq, kl, places=12)
            self.assertGreaterEqual(kl, -1e-12)

    def test_window_limits_and_probability(self):
        for direction in ([1, 0, 0], [1, 1, 1]):
            short, p0, _, _, pmean = compute_point(250, direction, window_s=1e-8)
            np.testing.assert_allclose(pmean, p0, atol=1e-8)
            self.assertAlmostEqual(short["Udd_time_mean_kBT"], 0, places=6)
            long, _, peq, pend, _ = compute_point(300, direction, window_s=100)
            np.testing.assert_allclose(peq, pend, atol=1e-9)
            self.assertLess(long["free_energy_identity_error_kBT"], 1e-9)
            self.assertGreaterEqual(long["delta_Fmag_time_mean_kBT"],
                                    long["delta_Fmag_eq_kBT"] - 1e-9)
            self.assertTrue(np.isnan(long["assembly_delta_G_J"]))

    def test_uniform_mean_and_thermodynamic_temperature_dependence(self):
        low, p0, _, _, _ = compute_point(200, [1, 0, 0])
        high, _, _, _, _ = compute_point(300, [1, 0, 0])
        self.assertAlmostEqual(low["U_initial_kBT"], 0, places=12)
        self.assertAlmostEqual(high["U_initial_kBT"], 0, places=12)
        self.assertLess(low["Udd_eq_J"], high["Udd_eq_J"])
        self.assertLess(low["Udd_time_mean_J"], 0)
        self.assertLess(high["Udd_time_mean_J"], 0)


if __name__ == "__main__":
    unittest.main()
