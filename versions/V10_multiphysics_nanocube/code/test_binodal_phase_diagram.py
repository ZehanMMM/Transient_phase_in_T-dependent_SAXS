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


class DecompositionTests(unittest.TestCase):
    """The plotted budget must be the same eps the binodal is built from."""

    @classmethod
    def setUpClass(cls):
        cls.energies = bpd.pair_energies('converged')

    def test_parts_sum_to_the_net_cohesion(self):
        grid = np.arange(210.0, 290.1, 10.0)
        parts = bpd.energy_decomposition(self.energies, 100.0, 0.0485, 15.5,
                                         grid)
        rebuilt = (parts['vdw_kBT'] + parts['collected_kBT']
                   - parts['registration_kBT'])
        self.assertTrue(np.allclose(parts['net_kBT'], rebuilt, atol=1e-12))
        for row, collected in zip(parts['rows'], parts['collected_kBT']):
            self.assertAlmostEqual(row['epsilon_kBT'],
                                   row['vdw_kBT'] + collected, places=12)

    def test_monotone_inputs_give_a_non_monotone_net(self):
        """The whole mechanism in one test: two monotone terms, one peak."""
        grid = np.arange(200.0, 300.1, 2.0)
        parts = bpd.energy_decomposition(self.energies, 100.0, 0.0485, 15.5,
                                         grid)
        # cooling deepens both wells and freezes the moments, without exception
        for key in ('vdw_kBT', 'dipole_kBT', 'blocked_weight'):
            self.assertTrue(np.all(np.diff(parts[key]) < 0), key)
        # yet what the condensate collects turns over inside the window
        peak = parts['temperature_K'][int(np.argmax(parts['net_kBT']))]
        self.assertGreater(peak, 250.0)
        self.assertLess(peak, 270.0)
        self.assertLess(parts['net_kBT'][0], parts['net_kBT'].max())
        self.assertLess(parts['net_kBT'][-1], parts['net_kBT'].max())


class LossFractionTests(unittest.TestCase):
    """What a bond keeps when its moments freeze, and what the data allows."""

    @classmethod
    def setUpClass(cls):
        cls.energies = bpd.pair_energies('converged')

    def test_parallel_moments_are_exactly_neutral_on_a_face_bond(self):
        """Magic angle: (s.n)^2 = 1/3 for a <100> bond, so U_dd vanishes.

        Worth pinning because it is easy to assume the aligned state is the
        attractive one.  It is not -- the attractive states are head-to-tail.
        """
        direction = np.asarray(bpd.LINKS['face'], float)
        distance = geometry.center_distance_at_gap_m(
            direction, bpd.SURFACE_GAP_NM * 1e-9, geometry.PARAMS)
        table, _, axes, _ = geometry.easy_axis_pair_energies_J(
            distance * direction, geometry.PARAMS)
        matrix = np.asarray(table).reshape(8, 8)
        self.assertTrue(np.allclose(np.diag(matrix), 0.0, atol=1e-30))
        deepest = np.unravel_index(int(np.argmin(matrix)), matrix.shape)
        along = float((axes[deepest[0]] @ direction)
                      * (axes[deepest[1]] @ direction))
        self.assertGreater(along, 0.0)          # head-to-tail, not opposed
        self.assertAlmostEqual(float(axes[deepest[0]] @ axes[deepest[1]]),
                               -1.0 / 3.0, places=9)

    def test_frustration_collapses_what_a_frozen_bond_keeps(self):
        """One moment cannot be deepest for six bonds at once."""
        kept = [bpd.frustrated_quench_kBT(250.0, z) for z in (1, 2, 4, 6)]
        self.assertTrue(all(a > b for a, b in zip(kept, kept[1:])))
        self.assertAlmostEqual(kept[0], 7.51, delta=0.02)
        self.assertAlmostEqual(kept[-1], 2.64, delta=0.02)
        self.assertTrue(all(value > 0.0 for value in kept))

    def test_the_quench_is_exhaustive_and_therefore_reproducible(self):
        self.assertEqual(bpd.frustrated_quench_kBT(250.0, 5),
                         bpd.frustrated_quench_kBT(250.0, 5))
        with self.assertRaises(ValueError):
            bpd.frustrated_quench_kBT(250.0, 7)

    def test_retained_share_is_nearly_temperature_independent(self):
        """Why lambda can be a constant rather than a function of T."""
        shares = [bpd.frustrated_quench_kBT(t, 6) / bpd.dipolar_mayer_term(
            t, self.energies) for t in (200.0, 250.0, 300.0)]
        for share in shares:
            self.assertGreater(share, 0.43)
            self.assertLess(share, 0.50)
        self.assertLess(max(shares) - min(shares), 0.07)

    def test_lambda_spans_one_down_to_a_half_and_never_inverts(self):
        previous = 1.0
        for f in (0.0, 0.07, 0.21, 0.42, 1.0):
            lam = bpd.loss_fraction(250.0, self.energies, f, 6)
            self.assertLessEqual(lam, previous + 1e-12)
            previous = lam
        self.assertEqual(bpd.loss_fraction(250.0, self.energies, 0.0), 1.0)
        self.assertAlmostEqual(bpd.loss_fraction(250.0, self.energies, 1.0, 6),
                               0.532, delta=0.005)
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                bpd.loss_fraction(250.0, self.energies, bad)

    def test_a_frozen_bond_keeps_a_floor_rather_than_nothing(self):
        """lambda < 1 leaves (1-lambda)*D behind; it is never repulsive."""
        for lam in (1.0, 0.9, 0.53):
            terms = bpd.dense_phase_epsilon(250.0, self.energies, 100.0, 0.05,
                                            lam)
            floor = terms['epsilon_frozen_kBT'] - terms['vdw_kBT']
            self.assertAlmostEqual(floor, (1.0 - lam) * terms['dipole_kBT'],
                                   places=12)
            self.assertGreaterEqual(floor, -1e-12)

    def test_the_observed_lower_edge_rules_out_mostly_correlated_freezing(self):
        """f >~ 0.3 cannot put the lower edge at 250 K, whatever CV is.

        This is the quantitative constraint on where the moments freeze: with
        lambda = 0.53 (everything freezing inside the aggregate) the coldest
        reachable edge is about 244 K, well above the observed value.
        """
        common = dict(dwell_s=100.0, energies=self.energies,
                      threshold_kBT=2.844, grid_step_K=2.0)
        keep = bpd.fit_size_cv(bpd.SAMPLE_VOLUME_FRACTION, 250.0, 270.0,
                               loss_fraction=1.0, **common)
        self.assertIsNotNone(keep)
        self.assertAlmostEqual(keep['size_cv'], 0.0485, delta=0.006)
        self.assertIsNone(bpd.fit_size_cv(bpd.SAMPLE_VOLUME_FRACTION, 250.0,
                                          270.0, loss_fraction=0.53, **common))


class OrientationalBookkeepingTests(unittest.TestCase):
    """Per-bond vs per-particle counting of the same orientational entropy.

    These pin the inconsistency rather than hide it: the cohesion charges the
    isolated-pair cost on every one of six bonds, and that is 3.6x more
    orientational entropy than a single particle orientation can actually owe.
    """

    def test_the_pair_limit_reproduces_the_analytic_cost(self):
        pair = bpd.orientational_cost_kBT(15.5, 1)
        self.assertAlmostEqual(pair['per_bond_kBT'],
                               bpd.registration_cost_kBT(15.5), delta=0.05)
        self.assertAlmostEqual(abs(pair['surplus_per_bond_kBT']), 0.0,
                               delta=0.05)
        self.assertAlmostEqual(pair['allowed_fraction'],
                               6.0 * (1 - np.cos(np.deg2rad(15.5))) / 2.0,
                               delta=0.002)

    def test_the_per_particle_cost_saturates_by_the_second_neighbour(self):
        """Two non-collinear bonds already lock all three rotational axes."""
        costs = [bpd.orientational_cost_kBT(15.5, z)['per_particle_kBT']
                 for z in (1, 2, 3, 6)]
        self.assertLess(costs[0], costs[1])                  # 1 -> 2 costs
        self.assertLess(costs[1], costs[2] + 1e-9)
        self.assertAlmostEqual(costs[2], costs[3], places=9)  # 3 == 6, saturated
        self.assertAlmostEqual(costs[3], 3.57, delta=0.05)
        self.assertLess(costs[3] - costs[1], 0.2)             # 2 is nearly there

    def test_the_per_bond_share_falls_as_one_over_coordination(self):
        saturated = bpd.orientational_cost_kBT(15.5, 6)['per_particle_kBT']
        for z in (3, 6):
            share = bpd.orientational_cost_kBT(15.5, z)['per_bond_kBT']
            self.assertAlmostEqual(share, 2.0 * saturated / z, places=9)
        self.assertAlmostEqual(bpd.orientational_cost_kBT(15.5, 6)['per_bond_kBT'],
                               1.19, delta=0.03)

    def test_orientation_explains_only_a_quarter_of_the_fitted_cost(self):
        """The rest must scale per contact -- ligands, not geometry.

        This is why the fitted-vs-geometric tilt agreement is NOT an
        independent check: both sides used the pair bookkeeping.
        """
        cage = bpd.orientational_cost_kBT(15.5, 6)
        self.assertAlmostEqual(cage['surplus_per_bond_kBT'], 3.24, delta=0.05)
        explained = cage['per_bond_kBT'] / bpd.registration_cost_kBT(15.5)
        self.assertLess(explained, 0.30)
        self.assertGreater(explained, 0.24)

    def test_it_is_seeded_and_guards_its_input(self):
        self.assertEqual(bpd.orientational_cost_kBT(15.5, 4),
                         bpd.orientational_cost_kBT(15.5, 4))
        for bad in (0, 7):
            with self.assertRaises(ValueError):
                bpd.orientational_cost_kBT(15.5, bad)


class LeverRuleTests(unittest.TestCase):
    """Volume share of the dense phase vs particle share -- not the same number.

    Conflating them produced a bogus "at most 6.7 % of particles can be in the
    aggregate" bound earlier in this project.  The volume share really is
    capped at phi / phi_dense; the particle share is not capped at all.
    """

    def test_particle_share_runs_to_one_while_volume_share_stays_small(self):
        phi, dense = bpd.SAMPLE_VOLUME_FRACTION, 0.31
        for dilute in (0.02, 0.01, 1e-4, 0.0):
            volume_share = (phi - dilute) / (dense - dilute)
            particle_share = dense * volume_share / phi
            self.assertLessEqual(volume_share, phi / dense + 1e-9)
            self.assertLessEqual(particle_share, 1.0 + 1e-9)
        # the deep-quench limit: every particle in a dense phase occupying
        # only phi / phi_dense of the volume
        self.assertAlmostEqual(phi / dense, 0.0675, delta=0.002)
        self.assertAlmostEqual(dense * (phi / (dense - 0.0)) / phi, 1.0,
                               places=9)

    def test_outside_coexistence_nothing_is_aggregated(self):
        self.assertEqual(bpd.aggregated_particle_fraction(0.5), 0.0)
        self.assertEqual(bpd.aggregated_particle_fraction(1.0), 0.0)

    def test_inside_the_dome_the_share_is_finite_and_bounded(self):
        share = bpd.aggregated_particle_fraction(2.9)
        self.assertGreater(share, 0.0)
        self.assertLess(share, 1.0)


class InheritedFrozenEnergyTests(unittest.TestCase):
    """The history integral that replaced the scalar lambda."""

    @classmethod
    def setUpClass(cls):
        cls.energies = bpd.pair_energies('converged')
        cls.grid = np.arange(200.0, 300.1, 10.0)

    def test_uniform_freezing_contributes_exactly_nothing(self):
        """Every row of U sums to zero, so an unbiased frozen moment is neutral."""
        table = bpd.inherited_frozen_energy(self.grid, self.energies,
                                            lambda t: 0.0, step_K=5.0)
        self.assertTrue(np.allclose(table['frozen_kBT'], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(table['frozen_energy_J'], 0.0, atol=1e-30))

    def test_blocked_weight_matches_the_accurate_quadrature(self):
        """b^2 must come from `neel_blocked_fraction`, not the coarse grid.

        The inherited lognormal grid is ~30 % out here; this pins the fix.
        """
        table = bpd.inherited_frozen_energy(self.grid, self.energies,
                                            lambda t: 0.0, step_K=0.5)
        for temperature, value in zip(self.grid, table['blocked_weight']):
            reference = bpd.neel_blocked_fraction(float(temperature), 100.0,
                                                  0.048480) ** 2
            self.assertAlmostEqual(value, reference, places=4)

    def test_the_ceiling_agrees_with_the_scalar_lambda_it_replaced(self):
        """At x = 1 the history integral must reproduce b^2 (1-lambda) D."""
        table = bpd.inherited_frozen_energy(self.grid, self.energies,
                                            lambda t: 1.0, step_K=0.5)
        for temperature, value in zip(self.grid, table['frozen_kBT']):
            terms = bpd.dense_phase_epsilon(float(temperature), self.energies,
                                            100.0, 0.048480)
            lam = bpd.loss_fraction(float(temperature), self.energies, 1.0, 6)
            scalar = terms['blocked_weight'] * (1.0 - lam) * terms['dipole_kBT']
            self.assertAlmostEqual(value, scalar, delta=0.02 + 0.05 * scalar)

    def test_the_frozen_energy_only_accumulates_on_cooling(self):
        table = bpd.inherited_frozen_energy(self.grid, self.energies,
                                            lambda t: 1.0, step_K=1.0)
        joules = table['frozen_energy_J']
        self.assertTrue(np.all(np.diff(joules) >= -1e-30))   # rises towards 0
        self.assertLessEqual(joules[-1], 0.0)                # attractive
        # At the warm end only the 0.8 % of bonds already blocked carry
        # anything, and what they carry is b^2 Q_6 -- small, but not zero.
        warm = table['frozen_kBT'][-1]
        expected = (bpd.neel_blocked_fraction(300.0, 100.0, 0.048480) ** 2
                    * bpd.frustrated_quench_kBT(300.0, 6))
        self.assertLess(warm, 0.05)
        self.assertAlmostEqual(warm, expected, delta=0.25 * expected)

    def test_what_a_frozen_bond_keeps_is_a_fixed_energy(self):
        """Q_z(T) T is constant, so the inherited term scales like 1/T."""
        product = [bpd.frustrated_quench_kBT(t, 6) * t
                   for t in (200.0, 250.0, 300.0)]
        self.assertLess(max(product) / min(product) - 1.0, 0.01)
        # at z = 1 the ground state is less isolated and it drifts more
        single = [bpd.frustrated_quench_kBT(t, 1) * t
                  for t in (200.0, 250.0, 300.0)]
        self.assertGreater(max(single) / min(single) - 1.0, 0.02)


class SelfConsistentCoolingTests(unittest.TestCase):
    """The closure: where the moments froze is decided by the model itself."""

    @classmethod
    def setUpClass(cls):
        cls.energies = bpd.pair_energies('converged')

    def sweep(self, **kwargs):
        return bpd.self_consistent_cooling(self.energies, step_K=5.0, **kwargs)

    def test_the_sweep_is_re_entrant_at_the_fitted_geometry(self):
        window = [r['temperature_K'] for r in self.sweep(tilt_deg=15.65)
                  if r['two_phase']]
        self.assertTrue(window)
        self.assertGreater(min(window), 200.0)      # it really does redissolve
        self.assertLess(max(window), 300.0)         # and is single-phase warm
        self.assertAlmostEqual(min(window), 250.0, delta=6.0)
        self.assertAlmostEqual(max(window), 268.0, delta=6.0)

    def test_the_closure_picks_a_negligible_frozen_term(self):
        """f is an output here, and it comes out near zero, not assumed so."""
        sweep = self.sweep(tilt_deg=15.65)
        coldest = sweep[-1]
        ceiling = (coldest['blocked_weight']
                   * bpd.frustrated_quench_kBT(coldest['temperature_K'], 6))
        self.assertGreater(ceiling, 2.0)
        self.assertLess(coldest['frozen_kBT'], 0.10)
        self.assertLess(coldest['frozen_kBT'] / ceiling, 0.05)

    def test_a_deeper_quench_runs_the_feedback_away(self):
        """More aggregation -> more freezing in place -> more aggregation.

        Past about 17 degrees of tilt the loop no longer closes on a window:
        the dense phase never redissolves within the swept range.
        """
        window = [r['temperature_K'] for r in self.sweep(tilt_deg=18.5)
                  if r['two_phase']]
        self.assertTrue(window)
        self.assertEqual(min(window), 200.0)        # open at the cold end
        coldest = self.sweep(tilt_deg=18.5)[-1]
        ceiling = (coldest['blocked_weight']
                   * bpd.frustrated_quench_kBT(coldest['temperature_K'], 6))
        self.assertGreater(coldest['frozen_kBT'] / ceiling, 0.5)

    def test_the_explicit_march_is_step_size_converged(self):
        coarse = bpd.self_consistent_cooling(self.energies, tilt_deg=15.65,
                                             step_K=4.0)
        fine = bpd.self_consistent_cooling(self.energies, tilt_deg=15.65,
                                           step_K=2.0)
        self.assertAlmostEqual(coarse[-1]['frozen_kBT'],
                               fine[-1]['frozen_kBT'], delta=0.005)
        self.assertAlmostEqual(max(r['aggregated_fraction'] for r in coarse),
                               max(r['aggregated_fraction'] for r in fine),
                               delta=0.02)


class CoolingTableTests(unittest.TestCase):
    """The plotted columns must be the trajectory, not a re-derivation."""

    @classmethod
    def setUpClass(cls):
        cls.table = bpd.cooling_energy_table(bpd.pair_energies('converged'),
                                             step_K=5.0)

    def test_the_plotted_total_is_the_two_attractions(self):
        rebuilt = -(-self.table['vdw_kBT'] + self.table['annealed_kBT']
                    + self.table['frozen_kBT'])
        self.assertTrue(np.allclose(self.table['total_self'], rebuilt,
                                    atol=1e-12))

    def test_the_threshold_carries_the_registration_constant(self):
        """Moving a T-independent constant between eps and the level is free."""
        net = self.table['threshold_sum_kBT'] - self.table['registration_kBT']
        self.assertAlmostEqual(net, 2.85, delta=0.15)

    def test_the_window_is_where_the_lever_rule_says(self):
        cold, warm = self.table['window']
        inside = self.table['aggregated_fraction'] > 0.0
        self.assertEqual(float(self.table['temperature_K'][inside].min()), cold)
        self.assertEqual(float(self.table['temperature_K'][inside].max()), warm)
        self.assertTrue(np.all(self.table['aggregated_fraction'][~inside] == 0.0))


if __name__ == '__main__':
    unittest.main()
