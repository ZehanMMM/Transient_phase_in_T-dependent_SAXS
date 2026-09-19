"""Run with: python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_binodal_phase_diagram.py"""
import unittest

import numpy as np
from scipy.constants import Boltzmann

import binodal_phase_diagram as bpd
import geometry_model as geometry


class PairEnergyTests(unittest.TestCase):
    def test_geometry_is_the_measured_one(self):
        """19 nm centre spacing on a 16 nm cube, i.e. the inherited 3 nm gap."""
        for choice in ('inherited', 'converged'):
            energies = bpd.pair_energies(choice)
            self.assertAlmostEqual(energies['centre_distance_nm'], 19.0, places=6)
            self.assertAlmostEqual(
                energies['centre_distance_nm'] - geometry.PARAMS.particle_size_nm,
                bpd.SURFACE_GAP_NM, places=6)
        with self.assertRaises(ValueError):
            bpd.pair_energies('npvdw')

    def test_converged_van_der_waals_is_the_one_in_use(self):
        """The fitted diagram runs on the converged sum, not the inherited 4^3."""
        converged = bpd.pair_energies('converged')
        inherited = bpd.pair_energies('inherited')
        self.assertEqual(converged['vdw_J'], bpd.CONVERGED_VDW_J)
        self.assertAlmostEqual(converged['vdw_J'] / inherited['vdw_J'], 1.875,
                               places=2)
        # Same geometry and same dipolar spectrum; only the vdW term differs.
        np.testing.assert_allclose(converged['dipole_J'], inherited['dipole_J'],
                                   rtol=0, atol=0)

    def test_dipolar_energy_sums_to_zero_but_the_mayer_term_does_not(self):
        """The whole mechanism: <U_dd> = 0 exactly, <exp(-U_dd/kT)> >> 1."""
        energies = bpd.pair_energies('converged')
        dipole = np.asarray(energies['dipole_J'])
        self.assertAlmostEqual(float(dipole.sum()) / abs(energies['prefactor_J']),
                               0.0, places=12)
        for temperature, expected in ((300.0, 4.40), (200.0, 7.53)):
            term = bpd.dipolar_mayer_term(temperature, energies)
            self.assertAlmostEqual(term, expected, places=2)
            # Jensen: ln<exp(-U/kT)> >= -<U>/kT = 0, strictly here.
            self.assertGreater(term, 0.0)


class CohesionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energies = bpd.pair_energies('converged')

    def test_cohesion_interpolates_between_the_two_pure_limits(self):
        for temperature in (200.0, 250.0, 300.0):
            terms = bpd.dense_phase_epsilon(temperature, self.energies, 100.0, 0.05)
            blocked = terms['blocked_weight']
            self.assertAlmostEqual(
                terms['epsilon_kBT'],
                (1.0 - blocked) * terms['epsilon_unblocked_kBT']
                + blocked * terms['epsilon_frozen_kBT'], places=12)
            self.assertLessEqual(terms['epsilon_frozen_kBT'],
                                 terms['epsilon_kBT'] + 1e-12)
            self.assertLessEqual(terms['epsilon_kBT'],
                                 terms['epsilon_unblocked_kBT'] + 1e-12)
            # The frozen limit is the van der Waals term alone.
            self.assertAlmostEqual(
                terms['epsilon_frozen_kBT'],
                -self.energies['vdw_J'] / (Boltzmann * temperature), places=12)

    def test_cohesion_is_non_monotonic_with_an_interior_maximum(self):
        """Re-entrance lives here: cooling deepens the well, then freezes it out."""
        grid = np.arange(200.0, 300.01, 1.0)
        cohesion = np.array([
            bpd.dense_phase_epsilon(float(t), self.energies, 100.0, 0.0485)[
                'epsilon_kBT'] for t in grid])
        peak = int(np.argmax(cohesion))
        self.assertGreater(peak, 2)
        self.assertLess(peak, len(grid) - 3)
        self.assertGreater(cohesion[peak], cohesion[0] + 0.2)
        self.assertGreater(cohesion[peak], cohesion[-1] + 0.2)


class FreeEnergyTests(unittest.TestCase):
    def test_chemical_potential_and_pressure_are_the_derivatives_of_f(self):
        """mu = d(eta f)/d eta and P = eta^2 df/d eta, checked numerically."""
        step = 1e-7
        for attraction in (0.0, 8.0, 20.0):
            for eta in (0.01, 0.1, 0.3, 0.5):
                gradient = ((bpd.free_energy(eta + step, attraction)
                             - bpd.free_energy(eta - step, attraction))
                            / (2.0 * step))
                # free_energy drops the ideal-gas -1, so mu sits one unit
                # below f + eta df/deta.  A constant cancels in coexistence.
                self.assertAlmostEqual(
                    float(bpd.chemical_potential(eta, attraction))
                    - (float(bpd.free_energy(eta, attraction)) + eta * gradient),
                    -1.0, places=4)
                self.assertAlmostEqual(
                    float(bpd.pressure(eta, attraction)) / (eta ** 2 * gradient),
                    1.0, places=4)

    def test_coexistence_satisfies_its_own_equations(self):
        for attraction in (12.0, 16.0, 22.0, 28.0):
            found = bpd.coexistence_by_continuation(np.array([attraction]))[0]
            self.assertIsNotNone(found)
            dilute, dense = found
            self.assertLess(dilute, dense)
            self.assertAlmostEqual(
                float(bpd.chemical_potential(dilute, attraction)),
                float(bpd.chemical_potential(dense, attraction)), places=6)
            self.assertAlmostEqual(float(bpd.pressure(dilute, attraction)),
                                   float(bpd.pressure(dense, attraction)),
                                   places=8)

    def test_stronger_attraction_widens_the_dome(self):
        branches = bpd.coexistence_by_continuation(
            np.array([12.0, 16.0, 20.0, 25.0]))
        dilute = [b[0] for b in branches]
        dense = [b[1] for b in branches]
        self.assertTrue(all(np.diff(dilute) < 0))
        self.assertTrue(all(np.diff(dense) > 0))

    def test_critical_attraction_matches_noro_frenkel(self):
        """A short-range attractive colloid condenses near 2-3 kBT per bond."""
        critical = bpd.critical_attraction(1.0, 30.0, 300)
        epsilon = bpd.critical_epsilon_kBT(critical)
        self.assertTrue(2.0 < epsilon < 3.0, f'eps_c = {epsilon}')
        self.assertIsNone(bpd.coexistence_by_continuation(
            np.array([critical * 0.9]))[0])
        self.assertIsNotNone(bpd.coexistence_by_continuation(
            np.array([critical * 1.3]))[0])


class RegistrationTests(unittest.TestCase):
    def test_registration_cost_follows_the_tilt_cone(self):
        for tilt in (10.0, 15.5, 20.0, 30.0):
            fraction = 6.0 * (1.0 - np.cos(np.deg2rad(tilt))) / 2.0
            self.assertAlmostEqual(bpd.registration_cost_kBT(tilt),
                                   -2.0 * np.log(fraction), places=12)
        self.assertAlmostEqual(bpd.registration_cost_kBT(15.5), 4.431, places=3)
        # A wider cone is cheaper; a cone wide enough to cover SO(3) is free.
        self.assertGreater(bpd.registration_cost_kBT(10.0),
                           bpd.registration_cost_kBT(30.0))
        self.assertEqual(bpd.registration_cost_kBT(90.0), 0.0)
        for bad in (0.0, -5.0, 120.0):
            with self.assertRaises(ValueError):
                bpd.registration_cost_kBT(bad)

    def test_tilt_cone_follows_from_the_measured_spacing(self):
        """+/-15.5 deg is where a rounded-cube corner meets the neighbour."""
        params = geometry.PARAMS
        half = params.particle_size_nm / 2.0 - params.roundness_nm
        support = lambda u: half * np.sum(np.abs(u)) + params.roundness_nm
        available = bpd.EFFECTIVE_DIAMETER_NM / 2.0
        self.assertAlmostEqual(available, 9.5, places=6)
        # Face fits; edge and vertex do not, so the dense phase is face-registered.
        self.assertLess(support(np.array([1.0, 0.0, 0.0])), available)
        self.assertGreater(support(np.array([1.0, 1.0, 0.0]) / np.sqrt(2)), available)
        self.assertGreater(support(np.array([1.0, 1.0, 1.0]) / np.sqrt(3)), available)
        angle = np.deg2rad(15.5)
        self.assertAlmostEqual(support(np.array([np.cos(angle), np.sin(angle), 0.0])),
                               available, delta=0.02)


class ConcentrationTests(unittest.TestCase):
    def test_effective_and_core_fractions_differ_by_the_spacing_cubed(self):
        table = bpd.concentration_table(0.0209)
        ratio = (bpd.EFFECTIVE_DIAMETER_NM / bpd.CORE_EDGE_NM) ** 3
        self.assertAlmostEqual(0.0209 / table['core_fraction'][0], ratio, places=9)
        self.assertAlmostEqual(table['mg_per_mL'][0], 65.0, delta=1.0)
        # The sample constant is the 1.25 % core fraction pushed through it.
        self.assertAlmostEqual(
            bpd.concentration_table(bpd.SAMPLE_VOLUME_FRACTION)['core_fraction'][0],
            0.0125, places=9)


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energies = bpd.pair_energies('converged')
        cls.fit = bpd.fit_size_cv(bpd.SAMPLE_VOLUME_FRACTION, 250.0, 270.0,
                                  100.0, cls.energies)

    def test_threshold_reproduces_the_target_concentration(self):
        threshold, attraction = bpd.epsilon_threshold(bpd.SAMPLE_VOLUME_FRACTION)
        branches = bpd.coexistence_by_continuation(np.array([attraction]))[0]
        self.assertIsNotNone(branches)
        self.assertAlmostEqual(branches[0] / bpd.SAMPLE_VOLUME_FRACTION, 1.0,
                               delta=0.05)
        self.assertAlmostEqual(
            threshold,
            attraction * bpd.CLOSE_PACKED_FRACTION / (bpd.MAX_COORDINATION / 2.0),
            places=12)

    def test_fit_reproduces_both_observed_edges(self):
        self.assertAlmostEqual(self.fit['lower_edge_K'], 250.0, delta=0.5)
        self.assertEqual(self.fit['upper_edge_K'], 270.0)
        self.assertTrue(0.0 < self.fit['size_cv'] < 0.10)
        self.assertAlmostEqual(self.fit['size_cv'], 0.0485, delta=0.005)

    def test_width_is_independent_of_the_registry_cost(self):
        """Pinning the upper edge fixes the cut level, so only CV sets the width.

        This is why the registry cost can be read off as a free check rather
        than being another fitted knob.
        """
        grid = np.arange(200.0, 300.01, 0.25)
        cohesion = np.array([
            bpd.dense_phase_epsilon(float(t), self.energies, 100.0,
                                    self.fit['size_cv'])['epsilon_kBT']
            for t in grid])
        widths = []
        for shift in (-0.5, 0.0, 0.5):
            level = float(np.interp(270.0, grid, cohesion))
            inside = grid[cohesion >= level]
            widths.append(inside.max() - inside.min())
            # shifting the cost shifts eps and the level together
            cohesion = cohesion + shift
        self.assertAlmostEqual(widths[0], widths[1], places=9)
        self.assertAlmostEqual(widths[1], widths[2], places=9)

    def test_registry_cost_was_not_fitted_and_matches_the_geometry(self):
        """The one genuinely independent check in the whole construction."""
        self.assertAlmostEqual(self.fit['geometric_cost_kBT'],
                               bpd.registration_cost_kBT(15.5), places=12)
        self.assertAlmostEqual(self.fit['fitted_tilt_deg'], 15.74, delta=0.15)
        self.assertLess(abs(self.fit['fitted_tilt_deg']
                            - self.fit['geometric_tilt_deg']), 1.0)
        # Soft ligands can only widen the cone relative to rigid cores.
        self.assertGreater(self.fit['fitted_tilt_deg'], self.fit['geometric_tilt_deg'])

    def test_a_broad_size_distribution_would_falsify_the_model(self):
        """CV ~ 10 % gives a window far wider than the observed 20 K."""
        grid = np.arange(200.0, 300.01, 0.5)
        for size_cv, floor in ((0.0485, 15.0), (0.10, 45.0)):
            cohesion = np.array([
                bpd.dense_phase_epsilon(float(t), self.energies, 100.0, size_cv)[
                    'epsilon_kBT'] for t in grid])
            level = float(np.interp(270.0, grid, cohesion))
            inside = grid[cohesion >= level]
            width = inside.max() - inside.min()
            if size_cv < 0.06:
                self.assertLess(width, 25.0)
            else:
                self.assertGreater(width, floor)


if __name__ == '__main__':
    unittest.main()
