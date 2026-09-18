"""Run with: python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_configuration_averaged_assembly.py"""
import unittest

import numpy as np
from scipy.constants import Boltzmann
from scipy.linalg import expm

import configuration_averaged_assembly as assembly
import geometry_model as geometry
from brownian_configuration_hopping import path_maximum as inherited_path_maximum
from pair_energy_model import HYDRODYNAMIC_DIAMETER_M


class GeometryAndRotationTests(unittest.TestCase):
    def test_path_maximum_reduces_to_inherited_quarter_turn(self):
        rng = np.random.default_rng(5150)
        for _ in range(12):
            moment, other, link = rng.normal(size=(3, 3))
            moment /= np.linalg.norm(moment)
            other /= np.linalg.norm(other)
            link /= np.linalg.norm(link)
            axis = np.eye(3)[rng.integers(3)] * rng.choice([-1.0, 1.0])
            self.assertAlmostEqual(
                assembly.path_maximum(1.0, moment, other, axis, link, np.pi / 2),
                inherited_path_maximum(1.0, moment, other, axis, link), places=12)

    def test_path_maximum_matches_dense_scan_at_arbitrary_angle(self):
        rng = np.random.default_rng(77)
        for _ in range(10):
            moment, other, link, axis = rng.normal(size=(4, 3))
            for vector in (moment, other, link, axis):
                vector /= np.linalg.norm(vector)
            angle = rng.uniform(-2.5, 2.5)
            theta = np.linspace(0.0, angle, 40001)
            rotated = np.array([assembly.rotation_matrix(axis, t) @ moment for t in theta])
            energies = rotated @ other - 3.0 * (rotated @ link) * np.dot(other, link)
            self.assertAlmostEqual(
                assembly.path_maximum(1.0, moment, other, axis, link, angle),
                float(energies.max()), places=7)

    def test_rotation_decay_rank_and_first_rank_calibration(self):
        expected = {'free': 4.0, 'face': 2.0, 'tip': 3.0}
        for kind, rank in expected.items():
            self.assertAlmostEqual(
                assembly.rotation_decay_rank(assembly.MOVES[kind]), rank, places=10)
        # At zero coupling the generator must decay every relaxable moment
        # component at exactly 1/tau_B, whichever move set is used.
        states = geometry.cubic_easy_axis_states()
        tau_B = 3.0e-6
        for kind in assembly.CONTACTS:
            geom = dict(states=states, dipole_J=np.zeros(64), prefactor_J=0.0,
                        link=assembly.LINKS['face'])
            structure = assembly.registry_structure(geom, assembly.MOVES[kind])
            generator = assembly.generator_from_structure(
                structure, 250.0, 1.0 / (structure['rank'] * tau_B))
            moments = np.repeat(states, 8, axis=0)
            axis = np.asarray(assembly.MOVES[kind][0][0], dtype=float)
            axis = axis / np.linalg.norm(axis)
            transverse = moments - np.outer(moments @ axis, axis)
            if kind == 'free':
                np.testing.assert_allclose(tau_B * generator @ moments, -moments,
                                           atol=1e-14)
            else:
                np.testing.assert_allclose(tau_B * generator @ transverse, -transverse,
                                           atol=1e-14)
            np.testing.assert_allclose(generator.sum(axis=1), 0.0, atol=1e-8)

    def test_contact_preserving_moves_stay_on_the_easy_axes(self):
        states = geometry.cubic_easy_axis_states()
        for kind in assembly.CONTACTS:
            for axis, angle in assembly.MOVES[kind]:
                rotated = states @ assembly.rotation_matrix(axis, angle).T
                # A cube symmetry operation permutes the eight <111> states.
                overlap = np.abs(rotated @ states.T)
                np.testing.assert_allclose(np.sort(overlap, axis=1)[:, -1], 1.0,
                                           atol=1e-12)

    def test_neel_structure_reproduces_inherited_generator(self):
        structures = assembly.build_structures('dimer')
        for kind in assembly.CONTACTS:
            geom = structures['geometry'][kind]
            built = assembly.generator_from_structure(
                structures['neel'][kind], 237.0,
                1.0 / (3.0 * geometry.PARAMS.attempt_time_s))
            inherited = geometry.neel_pair_generator(
                237.0, geometry.PARAMS.zfc_fc_activation_barrier_J, geom['dipole_J'],
                geom['prefactor_J'], geom['states'], geom['link'], geometry.PARAMS)
            scale = max(float(np.max(np.abs(inherited))), 1e-300)
            np.testing.assert_allclose(built / scale, inherited / scale, atol=1e-12)

    def test_inherited_energies_are_untouched(self):
        """The vdW and gap models must be the inherited ones, unscaled in dimer mode."""
        for kind in assembly.BOUND:
            geom = assembly.contact_geometry(kind, 'dimer')
            direction = assembly.LINKS[kind]
            distance = geometry.center_distance_at_gap_m(
                direction, assembly.SURFACE_GAP_NM * 1e-9, geometry.PARAMS)
            self.assertAlmostEqual(geom['center_distance_m'], distance, places=18)
            self.assertAlmostEqual(
                geom['vdw_J'],
                geometry.pair_vdw_energy_J(distance * direction, geometry.PARAMS),
                places=30)
            self.assertEqual(geom['bond_multiplicity'], 1.0)
            self.assertEqual(geom['report_factor'], 0.5)

    def test_superlattice_scaling_is_the_inherited_coordination_convention(self):
        for kind in assembly.BOUND:
            dimer = assembly.contact_geometry(kind, 'dimer')
            lattice = assembly.contact_geometry(kind, 'superlattice')
            factor = assembly.COORDINATION[kind] / 2.0
            np.testing.assert_allclose(lattice['energy_J'], factor * dimer['energy_J'],
                                       rtol=1e-14)
            self.assertAlmostEqual(lattice['report_factor'], 1.0, places=14)
            self.assertAlmostEqual(dimer['report_factor'] * dimer['energy_J'].min(),
                                   0.5 * dimer['energy_J'].min(), places=30)


class BindingKineticsTests(unittest.TestCase):
    def test_reaction_volume_is_the_detailed_balance_identity(self):
        for length in (0.2, 0.5, 1.5):
            volume = assembly.site_volume_m3(length)
            self.assertAlmostEqual(volume,
                                   4.0 * np.pi * HYDRODYNAMIC_DIAMETER_M * (length * 1e-9) ** 2,
                                   places=34)
            for temperature in (200.0, 273.0, 300.0):
                escape = assembly.smoluchowski_rate_m3ps(temperature) / volume
                diffusive = (assembly.relative_diffusion_m2ps(temperature)
                             / (length * 1e-9) ** 2)
                self.assertAlmostEqual(escape / diffusive, 1.0, places=12)

    def test_generator_satisfies_detailed_balance_in_both_modes(self):
        for mode in ('dimer', 'superlattice'):
            structures = assembly.build_structures(mode)
            binding = assembly.Binding(assembly=mode)
            for temperature in (200.0, 250.0, 300.0):
                model = assembly.joint_model(temperature, binding, structures)
                scale = float(np.max(np.abs(model['generator'])))
                self.assertLess(model['detailed_balance_relative_error'], 1e-12)
                np.testing.assert_allclose(model['generator'].sum(axis=1), 0.0,
                                           atol=1e-11 * scale)
                np.testing.assert_allclose(
                    model['equilibrium'] @ model['generator'], 0.0,
                    atol=1e-11 * scale * float(model['equilibrium'].max()))

    def test_on_off_ratio_reproduces_the_boltzmann_weight(self):
        structures = assembly.build_structures('dimer')
        binding = assembly.Binding(volume_fraction=3.0e-3, escape_length_nm=0.8)
        temperature = 271.0
        model = assembly.joint_model(temperature, binding, structures)
        kbt = Boltzmann * temperature
        free_rows = assembly.block('free')
        for kind in assembly.BOUND:
            rows = assembly.block(kind)
            on = model['generator'][free_rows, rows]
            off = model['generator'][rows, free_rows]
            energy = structures['geometry'][kind]['energy_J']
            np.testing.assert_allclose(on / off,
                                       model['alpha'][kind] * np.exp(-energy / kbt),
                                       rtol=1e-12)

    def test_rejects_unphysical_binding_parameters(self):
        for bad in (dict(volume_fraction=0.0), dict(volume_fraction=0.9),
                    dict(escape_length_nm=0.0), dict(escape_length_nm=9.0),
                    dict(mobility=-1.0), dict(contact_barrier_kBT=-1.0),
                    dict(assembly='crystal')):
            with self.assertRaises(ValueError):
                assembly.Binding(**bad)


class PopulationAndEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = {mode: assembly.build_structures(mode)
                          for mode in ('dimer', 'superlattice')}

    def test_propagator_matches_expm_and_the_window_mean_integral(self):
        structures = self.structures['dimer']
        model = assembly.joint_model(260.0, assembly.Binding(), structures)
        initial = np.zeros(192)
        initial[assembly.block('free')] = 1.0 / 64.0
        for window in (1.0e-7, 1.0e-4, 20.0):
            end, mean, drift = assembly.propagate(model['generator'], initial, window)
            self.assertLess(drift, 1e-3)
            reference = initial @ expm(model['generator'] * window)
            np.testing.assert_allclose(end, reference / reference.sum(), atol=1e-11)
            # Composite Gauss-Legendre on log-spaced subintervals.  A uniform
            # rule cannot resolve the microsecond binding transient inside a
            # 20 s window and is wrong by 1e-4, not the propagator.
            quadrature = np.zeros(192)
            edges = np.r_[0.0, np.geomspace(window * 1e-12, window, 48)]
            nodes, weights = np.polynomial.legendre.leggauss(10)
            for left, right in zip(edges[:-1], edges[1:]):
                middle, half = (left + right) / 2.0, (right - left) / 2.0
                for node, weight in zip(nodes, weights):
                    quadrature += (half * weight * initial
                                   @ expm(model['generator'] * (middle + half * node)))
            quadrature /= window
            np.testing.assert_allclose(mean, quadrature / quadrature.sum(), atol=1e-9)

    def test_short_window_returns_the_initial_colloidal_state(self):
        for mode in ('dimer', 'superlattice'):
            row = assembly.compute(250.0, assembly.Binding(assembly=mode),
                                   self.structures[mode], window_s=1.0e-14)[0]
            self.assertAlmostEqual(row['p_free_window'], 1.0, places=6)
            self.assertAlmostEqual(row['U_total_per_NC_window_kBT'], 0.0, places=6)

    def test_long_window_reproduces_the_closed_form_equilibrium(self):
        for mode in ('dimer', 'superlattice'):
            binding = assembly.Binding(assembly=mode)
            for temperature in (200.0, 250.0, 300.0):
                row, equilibrium, _ = assembly.compute(
                    temperature, binding, self.structures[mode], window_s=20.0)
                closed = assembly.equilibrium_populations(
                    temperature, binding, self.structures[mode])
                self.assertAlmostEqual(row['p_bound_eq'], closed['p_bound'], places=12)
                self.assertAlmostEqual(row['U_total_per_NC_eq_J'],
                                       closed['U_total_per_NC_J'],
                                       delta=1e-6 * abs(closed['U_total_per_NC_J']) + 1e-30)
                self.assertAlmostEqual(row['p_bound_end'], row['p_bound_eq'], places=4)
                self.assertAlmostEqual(float(equilibrium.sum()), 1.0, places=12)

    def test_total_energy_is_the_configuration_average_of_the_same_distribution(self):
        structures = self.structures['dimer']
        report = assembly.per_state_report_energy(structures)
        for temperature in (210.0, 290.0):
            row, equilibrium, window_mean = assembly.compute(
                temperature, assembly.Binding(), structures)
            self.assertAlmostEqual(row['U_total_per_NC_eq_J'],
                                   float(equilibrium @ report), places=30)
            self.assertAlmostEqual(row['U_total_per_NC_window_J'],
                                   float(window_mean @ report), places=30)
            # Free states carry no energy, so the average cannot exceed the
            # conditional bound energy in magnitude.
            self.assertGreaterEqual(row['U_total_per_NC_eq_J'],
                                    row['U_bound_conditional_eq_J'] - 1e-30)
            self.assertLessEqual(row['U_total_per_NC_eq_J'], 0.0)

    def test_cooling_increases_the_bound_population_and_deepens_the_energy(self):
        binding = assembly.Binding()
        previous = None
        for temperature in (300.0, 275.0, 250.0, 225.0, 200.0):
            row = assembly.compute(temperature, binding, self.structures['dimer'])[0]
            if previous is not None:
                self.assertGreater(row['p_bound_eq'], previous['p_bound_eq'])
                self.assertLess(row['U_total_per_NC_eq_kBT'],
                                previous['U_total_per_NC_eq_kBT'])
            previous = row

    def test_concentration_enters_only_through_alpha(self):
        structures = self.structures['dimer']
        low = assembly.equilibrium_populations(
            260.0, assembly.Binding(volume_fraction=1.0e-4), structures)
        high = assembly.equilibrium_populations(
            260.0, assembly.Binding(volume_fraction=1.0e-2), structures)
        self.assertLess(low['p_bound'], high['p_bound'])
        ratio_low = low['p_bound'] / low['p_free']
        ratio_high = high['p_bound'] / high['p_free']
        self.assertAlmostEqual(ratio_high / ratio_low, 100.0, places=6)


class LifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = assembly.build_structures('superlattice')
        cls.binding = assembly.Binding(assembly='superlattice')

    def test_direct_escape_only_lifetime_is_one_over_k_off(self):
        temperature = 250.0
        model = assembly.joint_model(temperature, self.binding, self.structures,
                                     channels=())
        deepest = assembly.deepest_bound_start(model['energy_J'])
        _, lifetime = assembly.survival_and_lifetime(
            model['generator'], model['equilibrium'], deepest, 20.0)
        energy = model['energy_J'][assembly.bound_indices()].min()
        expected = 1.0 / (model['escape_per_s']
                          * np.exp(energy / (Boltzmann * temperature)))
        self.assertAlmostEqual(lifetime / expected, 1.0, places=8)

    def test_survival_matches_expm_on_the_absorbing_block(self):
        model = assembly.joint_model(285.0, self.binding, self.structures)
        index = assembly.bound_indices()
        deepest = assembly.deepest_bound_start(model['energy_J'])
        for window in (1.0, 20.0, 200.0):
            survival, _ = assembly.survival_and_lifetime(
                model['generator'], model['equilibrium'], deepest, window)
            reference = float((deepest @ expm(
                model['generator'][np.ix_(index, index)] * window)).sum())
            # Two independent stiff solvers on a block spanning ~15 decades in
            # k_off; agreement to 1e-6 is the floor either can support.
            self.assertAlmostEqual(survival, reference, places=6)

    def test_opening_a_channel_can_only_shorten_the_lifetime(self):
        for temperature in (200.0, 250.0, 300.0):
            row = assembly.compute(temperature, self.binding, self.structures)[0]
            direct = row['pair_lifetime_direct_escape_s']
            self.assertLessEqual(row['pair_lifetime_neel_blocked_s'], direct + 1e-12)
            self.assertLessEqual(row['pair_lifetime_rotation_blocked_s'], direct + 1e-12)
            self.assertLessEqual(row['pair_lifetime_deepest_s'],
                                 min(row['pair_lifetime_neel_blocked_s'],
                                     row['pair_lifetime_rotation_blocked_s']) + 1e-12)
            self.assertGreater(row['pair_lifetime_deepest_s'], 0.0)

    def test_blocking_temperature_separates_transient_from_persistent_bonds(self):
        window = 20.0
        hot = assembly.compute(300.0, self.binding, self.structures, window)[0]
        cold = assembly.compute(200.0, self.binding, self.structures, window)[0]
        self.assertLess(hot['pair_lifetime_deepest_s'], window)
        self.assertGreater(cold['pair_lifetime_deepest_s'], 1.0e4 * window)
        self.assertLess(hot['pair_survival_deepest'], 0.05)
        self.assertGreater(cold['pair_survival_deepest'], 0.99)
        self.assertEqual(hot['regime'], 'fast Neel and Brownian')
        self.assertEqual(cold['regime'], 'Neel blocked, Brownian only')

    def test_extra_rotation_barrier_lengthens_the_lifetime_towards_direct_escape(self):
        free_rotation = assembly.compute(
            300.0, assembly.Binding(assembly='superlattice'), self.structures)[0]
        blocked = assembly.compute(
            300.0, assembly.Binding(assembly='superlattice', contact_barrier_kBT=25.0),
            self.structures)[0]
        self.assertGreater(blocked['pair_lifetime_deepest_s'],
                           free_rotation['pair_lifetime_deepest_s'])
        self.assertAlmostEqual(
            blocked['pair_lifetime_deepest_s']
            / blocked['pair_lifetime_rotation_blocked_s'], 1.0, places=4)

    def test_half_crossing_interpolates(self):
        data = [dict(temperature_K=t, value=v) for t, v in
                ((200.0, 1.0), (250.0, 0.75), (260.0, 0.25), (300.0, 0.0))]
        self.assertAlmostEqual(assembly.half_crossing(data, 'value'), 255.0, places=10)


class FaceTipCompetitionTests(unittest.TestCase):
    """The <100>/<111> magic-angle cancellation and the twist entropy."""

    @classmethod
    def setUpClass(cls):
        cls.structures = assembly.build_structures('dimer')

    def test_parallel_face_dipole_energy_is_identically_zero(self):
        """(s.n)^2 = 1/3 for every <111> against a <100> bond, exactly."""
        geom = self.structures['geometry']['face']
        states, link = geom['states'], geom['link']
        np.testing.assert_allclose((states @ link) ** 2, 1.0 / 3.0, atol=1e-15)
        parallel = assembly.dipole_manifold_energies(geom, 'parallel')
        dipole = parallel - geom['vdw_J']
        np.testing.assert_allclose(dipole / abs(geom['prefactor_J']), 0.0, atol=1e-14)

    def test_parallel_tip_dipole_energy_reaches_minus_two_C(self):
        geom = self.structures['geometry']['tip']
        states, link = geom['states'], geom['link']
        np.testing.assert_allclose(np.sort(np.unique(np.round(states @ link, 12))),
                                   [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], atol=1e-12)
        parallel = assembly.dipole_manifold_energies(geom, 'parallel')
        dipole = (parallel - geom['vdw_J']) / geom['prefactor_J']
        self.assertAlmostEqual(float(dipole.min()), -2.0, places=12)
        self.assertEqual(int(np.sum(np.abs(dipole + 2.0) < 1e-12)), 2)
        # The tip minimum is also the global minimum over all 64 states.
        free = (assembly.dipole_manifold_energies(geom, 'free')
                - geom['vdw_J']) / geom['prefactor_J']
        self.assertAlmostEqual(float(free.min()), float(dipole.min()), places=12)

    def test_manifold_selection(self):
        for kind in assembly.BOUND:
            geom = self.structures['geometry'][kind]
            free = assembly.dipole_manifold_energies(geom, 'free')
            parallel = assembly.dipole_manifold_energies(geom, 'parallel')
            self.assertEqual(free.shape, (64,))
            self.assertEqual(parallel.shape, (8,))
            np.testing.assert_allclose(
                parallel, free.reshape(8, 8)[np.arange(8), np.arange(8)], atol=0)
        with self.assertRaises(ValueError):
            assembly.dipole_manifold_energies(
                self.structures['geometry']['face'], 'antiparallel')

    def test_twist_entropy_is_two_log_pi_over_three_delta_and_tilt_free(self):
        for twist in (1.0, 2.0, 5.0, 10.0):
            expected = 2.0 * np.log(np.pi / (3.0 * np.deg2rad(twist)))
            self.assertAlmostEqual(assembly.twist_entropy_advantage_kBT(twist),
                                   expected, places=12)
            for tilt in (0.5, 2.0, 5.0):
                self.assertAlmostEqual(
                    assembly.twist_entropy_advantage_kBT(twist, tilt), expected,
                    places=12)
        self.assertGreater(assembly.twist_entropy_advantage_kBT(1.0),
                           assembly.twist_entropy_advantage_kBT(10.0))

    def test_competition_bookkeeping_and_sign_convention(self):
        for manifold in assembly.MANIFOLDS:
            for temperature in (200.0, 250.0, 300.0):
                row = assembly.face_tip_competition(
                    temperature, self.structures, manifold, 5.0)
                self.assertAlmostEqual(
                    row['d_F_total_kBT'],
                    row['d_F_dip_kBT'] + row['d_F_twist_kBT'], places=12)
                self.assertLess(row['d_F_twist_kBT'], 0.0)  # entropy favours tip
                self.assertGreater(row['d_vdW_kBT'], 0.0)   # vdW favours face
                self.assertEqual(row['preferred_with_entropy'],
                                 'tip' if row['d_F_total_kBT'] < 0 else 'face')
                self.assertAlmostEqual(row['d_F_total_J'],
                                       row['d_F_total_kBT'] * row['kBT_J'],
                                       places=30)

    def test_manifold_inverts_the_preferred_contact(self):
        """Free dipoles favour face on energy; parallel dipoles favour tip."""
        for temperature in (200.0, 250.0, 300.0):
            free = assembly.face_tip_competition(
                temperature, self.structures, 'free', 5.0)
            parallel = assembly.face_tip_competition(
                temperature, self.structures, 'parallel', 5.0)
            self.assertGreater(free['d_F_dip_kBT'], 0.0)
            self.assertLess(parallel['d_F_dip_kBT'], 0.0)
            self.assertEqual(parallel['preferred_with_entropy'], 'tip')
            # The face annealed free energy cannot beat its own deepest state.
            self.assertGreaterEqual(free['face_f_dip_kBT'],
                                    free['face_u_min_kBT'] - 1e-12)

    def test_face_parallel_binding_is_van_der_waals_only(self):
        row = assembly.face_tip_competition(250.0, self.structures, 'parallel', 5.0)
        self.assertAlmostEqual(row['face_u_min_kBT'], row['face_vdw_kBT'], places=12)
        self.assertAlmostEqual(row['face_f_dip_kBT'], row['face_vdw_kBT'], places=12)

    def test_twist_registry_weight_matches_the_entropy_advantage(self):
        for twist in (1.0, 2.0, 5.0, 10.0):
            self.assertEqual(assembly.twist_registry_weight('tip', twist), 1.0)
            face = assembly.twist_registry_weight('face', twist)
            self.assertAlmostEqual(face, (3.0 * np.deg2rad(twist) / np.pi) ** 2,
                                   places=15)
            self.assertAlmostEqual(-np.log(face),
                                   assembly.twist_entropy_advantage_kBT(twist),
                                   places=12)
        self.assertLess(assembly.twist_registry_weight('face', 1.0),
                        assembly.twist_registry_weight('face', 10.0))
        with self.assertRaises(ValueError):
            assembly.twist_registry_weight('edge', 5.0)

    def test_restricted_channel_closed_form_against_a_direct_orbit_sum(self):
        """Each orbit binds on its own, so the reference must do the same."""
        binding = assembly.Binding(volume_fraction=4.0e-3, escape_length_nm=0.7)
        kbt = Boltzmann * 263.0
        for channel in assembly.BOUND:
            for manifold in assembly.MANIFOLDS:
                row = assembly.restricted_channel_populations(
                    263.0, self.structures, channel, manifold, 5.0, binding)
                geom = self.structures['geometry'][channel]
                energies = np.asarray(geom['energy_J'])
                orbits = assembly.manifold_orbits(geom, manifold)
                alpha = (assembly.COORDINATION['tip']
                         * assembly.site_volume_m3(binding.escape_length_nm)
                         * assembly.number_density_per_m3(binding.volume_fraction)
                         * assembly.twist_registry_weight(channel, 5.0))
                total = sum(len(orbit) for orbit in orbits)
                fraction, energy = 0.0, 0.0
                for orbit in orbits:
                    values = energies[orbit]
                    boltzmann = np.exp(-values / kbt)
                    ratio = alpha * boltzmann.mean()
                    bound = ratio / (1.0 + ratio)
                    fraction += len(orbit) / total * bound
                    energy += (len(orbit) / total * bound * geom['report_factor']
                               * float(boltzmann @ values / boltzmann.sum()))
                self.assertAlmostEqual(row['p_pair'], fraction, places=14)
                self.assertAlmostEqual(row['p_pair'] + row['p_free'], 1.0, places=14)
                self.assertAlmostEqual(row['U_total_per_NC_J'], energy, places=30)
                self.assertAlmostEqual(
                    row['U_bound_conditional_per_NC_J'] * row['p_pair'],
                    row['U_total_per_NC_J'], places=30)

    def test_particle_freeze_temperature_inverts_tau_N(self):
        params = geometry.PARAMS
        for factor in (0.6, 1.0, 1.4):
            freeze = assembly.particle_freeze_temperature_K(factor, 20.0)
            tau = params.attempt_time_s * np.exp(
                factor * params.zfc_fc_activation_barrier_J
                / (Boltzmann * freeze))
            self.assertAlmostEqual(tau / 20.0, 1.0, places=9)
        self.assertAlmostEqual(
            assembly.particle_freeze_temperature_K(1.0, 20.0),
            assembly.neel_freeze_temperature_K(20.0), places=9)

    def test_history_frozen_weight_is_b_squared(self):
        """A pair freezes when the SECOND moment does, so the weight is b^2."""
        grid = [300.0, 270.0, 250.0, 220.0, 200.0]
        rows = assembly.history_cooling_trajectory(grid, self.structures,
                                                   'face', 0.10, 20.0, 0.0)
        for row in rows:
            blocked = assembly.neel_blocked_fraction(row['temperature_K'], 20.0,
                                                     0.10)
            # Sharp per-size cut against the soft survival criterion: same
            # physics, so the two must agree to a few percent.
            self.assertAlmostEqual(row['frozen_pair_weight'], blocked ** 2,
                                   delta=0.12)
            self.assertTrue(0.0 <= row['frozen_pair_weight'] <= 1.0)

    def test_inheriting_a_bound_distribution_prevents_disassembly(self):
        """The whole point: uniform was an assumption, not a consequence."""
        grid = [300.0, 250.0, 200.0]
        bound = assembly.history_cooling_trajectory(grid, self.structures,
                                                    'face', 0.10, 20.0, 1.0)
        dispersed = assembly.history_cooling_trajectory(grid, self.structures,
                                                        'face', 0.10, 20.0, 0.0)
        for hot, cold in zip(bound, dispersed):
            self.assertLess(hot['U_pair_excess_kBT'], cold['U_pair_excess_kBT'])
            self.assertGreaterEqual(hot['U_pair_excess_J'], -1e-30)
        # Froze-while-bound keeps essentially all of its binding.
        self.assertLess(bound[-1]['U_pair_excess_kBT'], 0.2)
        self.assertGreater(dispersed[-1]['U_pair_excess_kBT'], 5.0)
        self.assertLess(bound[-1]['repulsive_weight'], 1e-3)
        self.assertGreater(dispersed[-1]['repulsive_weight'], 0.2)

    def test_computed_branch_tracks_the_dispersed_one_at_low_phi(self):
        """p_bound at the freeze temperatures is tiny, so uniform is earned."""
        grid = [280.0, 240.0, 200.0]
        computed = assembly.history_cooling_trajectory(grid, self.structures,
                                                       'face', 0.10, 20.0, None)
        dispersed = assembly.history_cooling_trajectory(grid, self.structures,
                                                        'face', 0.10, 20.0, 0.0)
        for left, right in zip(computed, dispersed):
            self.assertAlmostEqual(left['U_pair_kBT'], right['U_pair_kBT'],
                                   delta=0.2)
        with self.assertRaises(ValueError):
            assembly.history_cooling_trajectory(grid, self.structures, 'face',
                                                0.10, 20.0, 1.5)

    def test_blocked_fraction_matches_adaptive_quadrature(self):
        """The fixed 400-node rule must reproduce scipy.quad on the sigmoid."""
        from scipy.integrate import quad

        params = geometry.PARAMS
        for cv in (0.05, 0.10, 0.15):
            sigma = np.sqrt(np.log1p(cv ** 2))
            for temperature in (200.0, 250.0, 270.0, 300.0):
                exponent = (params.zfc_fc_activation_barrier_J
                            / (Boltzmann * temperature))

                def integrand(variate):
                    factor = np.exp(-0.5 * sigma ** 2 + sigma * variate) ** 3
                    tau = params.attempt_time_s * np.exp(
                        min(factor * exponent, 700.0))
                    return (np.exp(-20.0 / tau) * np.exp(-variate ** 2 / 2.0)
                            / np.sqrt(2.0 * np.pi))

                exact = quad(integrand, -8.0, 8.0, limit=400)[0]
                self.assertAlmostEqual(
                    assembly.neel_blocked_fraction(temperature, 20.0, cv),
                    exact, places=6)

    def test_blocked_fraction_limits_and_monotonicity(self):
        """b rises on cooling; a size spread broadens it at both ends."""
        for cv in (0.0, 0.10):
            values = [assembly.neel_blocked_fraction(t, 20.0, cv)
                      for t in (300.0, 275.0, 250.0, 225.0, 200.0)]
            for low, high in zip(values[:-1], values[1:]):
                self.assertGreater(high, low)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        # Monodisperse is sharper: nearly unblocked hot, nearly blocked cold.
        self.assertLess(assembly.neel_blocked_fraction(300.0, 20.0, 0.0),
                        assembly.neel_blocked_fraction(300.0, 20.0, 0.10))
        self.assertGreater(assembly.neel_blocked_fraction(200.0, 20.0, 0.0),
                           assembly.neel_blocked_fraction(200.0, 20.0, 0.10))
        # A longer window blocks less.
        self.assertLess(assembly.neel_blocked_fraction(250.0, 150.0, 0.10),
                        assembly.neel_blocked_fraction(250.0, 1.0, 0.10))
        with self.assertRaises(ValueError):
            assembly.neel_blocked_fraction(250.0, 20.0, -0.1)

    def test_gradual_cooling_has_no_step_and_brackets_correctly(self):
        grid = np.arange(200.0, 301.0, 1.0)
        for cv in (0.0, 0.10):
            rows = assembly.gradual_cooling_trajectory(grid, self.structures,
                                                       'face', cv)
            weights = np.array([[r['weight_both_free'], r['weight_half_frozen'],
                                 r['weight_both_frozen']] for r in rows])
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
            self.assertTrue((weights >= -1e-15).all())
            energies = np.array([r['U_pair_kBT'] for r in rows])
            # Continuous: no 1 K step may exceed 0.5 kBT anywhere.
            self.assertLess(np.abs(np.diff(energies)).max(), 0.5)
            for row in rows:
                # The mixture must lie between the two pure components.
                self.assertLessEqual(row['U_pair_J'],
                                     row['U_pair_both_frozen_J'] + 1e-30)
                self.assertGreaterEqual(row['U_pair_J'],
                                        row['U_pair_equilibrium_J'] - 1e-30)
                self.assertGreaterEqual(row['U_pair_excess_J'], -1e-30)

    def test_gradual_cooling_recovers_both_pure_limits(self):
        """b -> 0 gives equilibrium, b -> 1 gives the van der Waals term."""
        geom = self.structures['geometry']['face']
        hot = assembly.gradual_cooling_trajectory([400.0], self.structures,
                                                  'face', 0.0)[0]
        self.assertLess(hot['blocked_fraction'], 1e-6)
        self.assertAlmostEqual(hot['U_pair_J'], hot['U_pair_equilibrium_J'],
                               places=28)
        cold = assembly.gradual_cooling_trajectory([120.0], self.structures,
                                                   'face', 0.0)[0]
        self.assertGreater(cold['blocked_fraction'], 1.0 - 1e-6)
        self.assertAlmostEqual(cold['U_pair_J'], geom['vdw_J'], places=28)
        self.assertAlmostEqual(cold['U_pair_both_frozen_J'], geom['vdw_J'],
                               places=30)

    def test_one_mobile_moment_fully_recovers_the_face_binding(self):
        """Z_j is j-independent for a <100> bond, so half-frozen == annealed.

        Consequence: only pairs with BOTH moments blocked lose their dipolar
        binding, so the controlling variable is b^2, not b.  The identity is
        specific to face; tip violates it.
        """
        face = self.structures['geometry']['face']
        tip = self.structures['geometry']['tip']
        for temperature in (200.0, 250.0, 300.0):
            kbt = Boltzmann * temperature
            columns = np.exp(-np.asarray(face['energy_J']).reshape(8, 8) / kbt)
            partition = columns.sum(axis=0)
            np.testing.assert_allclose(partition / partition.mean(), 1.0,
                                       rtol=1e-12)
            row = assembly.gradual_cooling_trajectory(
                [temperature], self.structures, 'face', 0.10)[0]
            self.assertAlmostEqual(row['U_pair_half_frozen_J'],
                                   row['U_pair_equilibrium_J'], places=28)
            # So the three-term mixture collapses to the b^2 law.
            blocked = row['blocked_fraction']
            collapsed = ((1.0 - blocked ** 2) * row['U_pair_equilibrium_J']
                         + blocked ** 2 * row['U_pair_both_frozen_J'])
            self.assertAlmostEqual(row['U_pair_J'], collapsed, places=28)
            # Tip breaks the identity: Z_j differs between the classes.
            tip_columns = np.exp(-np.asarray(tip['energy_J']).reshape(8, 8) / kbt)
            tip_partition = tip_columns.sum(axis=0)
            self.assertGreater(tip_partition.max() / tip_partition.min(), 2.0)

    def test_freeze_temperature_is_where_tau_N_equals_the_window(self):
        params = geometry.PARAMS
        for window in (1.0, 20.0, 150.0):
            freeze = assembly.neel_freeze_temperature_K(window)
            tau = params.attempt_time_s * np.exp(
                params.zfc_fc_activation_barrier_J / (Boltzmann * freeze))
            self.assertAlmostEqual(tau / window, 1.0, places=9)
        self.assertGreater(assembly.neel_freeze_temperature_K(20.0),
                           assembly.BLOCKING_TEMPERATURE_K)
        with self.assertRaises(ValueError):
            assembly.neel_freeze_temperature_K(1e-12)

    def test_cooling_trajectory_freezes_and_then_stops_updating(self):
        grid = [300.0, 280.0, 260.0, 240.0, 200.0]
        for inherit in ('dispersed', 'annealed'):
            rows = assembly.cooling_trajectory(grid, self.structures, 'face',
                                               inherit)
            freeze = rows[0]['freeze_temperature_K']
            above = [r for r in rows if not r['blocked']]
            below = [r for r in rows if r['blocked']]
            self.assertTrue(above and below)
            for row in above:
                self.assertGreater(row['temperature_K'], freeze)
                self.assertAlmostEqual(row['U_pair_J'],
                                       row['U_pair_equilibrium_J'], places=30)
                self.assertAlmostEqual(row['U_pair_excess_kBT'], 0.0, places=12)
            # Frozen means constant in joules, however far the cooling goes.
            for row in below[1:]:
                self.assertAlmostEqual(row['U_pair_J'], below[0]['U_pair_J'],
                                       places=30)
            # A frozen distribution can never beat the equilibrium mean.
            for row in rows:
                self.assertGreaterEqual(row['U_pair_excess_J'], -1e-30)
        with self.assertRaises(ValueError):
            assembly.cooling_trajectory(grid, self.structures, 'face', 'random')

    def test_random_freeze_leaves_exactly_the_van_der_waals_term(self):
        """Sum of U_dd over the 64 states is zero, so the dipole averages out."""
        rows = assembly.cooling_trajectory([200.0, 230.0], self.structures,
                                           'face', 'dispersed')
        geom = self.structures['geometry']['face']
        for row in rows:
            self.assertTrue(row['blocked'])
            self.assertAlmostEqual(row['U_pair_J'], geom['vdw_J'], places=30)
            # 24 of the 64 frozen states are net repulsive.
            self.assertAlmostEqual(row['repulsive_weight'], 24.0 / 64.0, places=12)
        ordered = assembly.cooling_trajectory([200.0], self.structures, 'face',
                                              'annealed')[0]
        # The ordered inheritance keeps a tiny repulsive tail from the
        # Boltzmann distribution at the freeze temperature, not exactly zero.
        self.assertLess(ordered['repulsive_weight'], 1e-4)
        self.assertLess(ordered['U_pair_J'], rows[0]['U_pair_J'])

    def test_bound_face_population_concentrates_on_the_deepest_level(self):
        """At 300 K the bound face pair is 92 % in its lowest dipole level."""
        geom = self.structures['geometry']['face']
        energies = np.asarray(geom['energy_J'])
        deepest = np.abs(energies - energies.min()) < 1e-30
        self.assertEqual(int(deepest.sum()), 8)
        for temperature, expected in ((300.0, 0.9215), (200.0, 0.9836)):
            weights = np.exp(-energies / (Boltzmann * temperature))
            self.assertAlmostEqual(float(weights[deepest].sum() / weights.sum()),
                                   expected, places=4)
        # The lowest state is head-to-tail along the bond, antiparallel across.
        states, link = geom['states'], geom['link']
        index = int(np.argmin(energies))
        first, second = states[index // 8], states[index % 8]
        self.assertAlmostEqual((first @ link) * (second @ link), 1.0 / 3.0, places=12)
        transverse_1 = first - (first @ link) * link
        transverse_2 = second - (second @ link) * link
        # Unit moments, so the two transverse parts are (0,-1,-1)/sqrt(3) and
        # (0,+1,+1)/sqrt(3): antiparallel, dot product -2/3.
        self.assertAlmostEqual(float(transverse_1 @ transverse_2), -2.0 / 3.0,
                               places=12)

    def test_blocked_groups_follow_the_orientational_entropy_model(self):
        """A tip contact is never really blocked; a face contact fully is."""
        self.assertEqual(assembly.BLOCKED_ROTATION, {'tip': 'full', 'face': 'none'})
        tip = assembly.manifold_orbits(self.structures['geometry']['tip'], 'blocked')
        self.assertEqual(len(tip), 1)
        np.testing.assert_array_equal(tip[0], np.arange(64))
        face = assembly.manifold_orbits(self.structures['geometry']['face'], 'blocked')
        self.assertEqual([len(o) for o in face], [1] * 64)
        np.testing.assert_array_equal(np.sort(np.concatenate(face)), np.arange(64))

    def test_blocked_tip_still_reaches_head_to_tail(self):
        """The whole point: a frozen tip cube can turn its moment onto the bond."""
        geom = self.structures['geometry']['tip']
        for temperature in (200.0, 250.0):
            kbt = Boltzmann * temperature
            blocked = assembly.orbit_statistics(geom, 'blocked', kbt)
            free = assembly.orbit_statistics(geom, 'free', kbt)
            self.assertAlmostEqual(blocked['quenched_free_energy_J'],
                                   free['quenched_free_energy_J'], places=30)
            self.assertAlmostEqual(blocked['annealed_free_energy_J'],
                                   free['annealed_free_energy_J'], places=30)
            # The -2C head-to-tail state is inside the single orbit.
            dipole = (blocked['min_energy_J'] - geom['vdw_J']) / geom['prefactor_J']
            self.assertAlmostEqual(dipole, -2.0, places=12)

    def test_blocked_face_loses_its_dipolar_binding_exactly(self):
        """Sum of U_dd over the 64 states is zero, so a frozen face keeps vdW."""
        geom = self.structures['geometry']['face']
        dipole = np.asarray(geom['energy_J']) - geom['vdw_J']
        self.assertAlmostEqual(float(dipole.sum()) / abs(geom['prefactor_J']),
                               0.0, places=12)
        for temperature in (200.0, 300.0):
            kbt = Boltzmann * temperature
            blocked = assembly.orbit_statistics(geom, 'blocked', kbt)
            self.assertAlmostEqual(blocked['quenched_free_energy_J'],
                                   geom['vdw_J'], places=30)
            self.assertLess(blocked['annealed_free_energy_J'],
                            blocked['quenched_free_energy_J'])

    def test_bondaxis_orbits_partition_the_states_and_are_rotation_closed(self):
        expected = {'face': [16, 16, 16, 16],
                    'tip': [1, 3, 3, 1, 3, 9, 9, 3, 3, 9, 9, 3, 1, 3, 3, 1]}
        for channel in assembly.BOUND:
            geom = self.structures['geometry'][channel]
            orbits = assembly.manifold_orbits(geom, 'bondaxis')
            self.assertEqual([len(o) for o in orbits], expected[channel])
            np.testing.assert_array_equal(
                np.sort(np.concatenate(orbits)), np.arange(64))
            member = {index: number for number, orbit in enumerate(orbits)
                      for index in orbit}
            states = geom['states']
            lookup = {tuple(np.sign(state).astype(int)): index
                      for index, state in enumerate(states)}
            for axis, angle in assembly.MOVES[channel]:
                matrix = assembly.rotation_matrix(axis, angle)
                for first in range(8):
                    moved = lookup[tuple(np.sign(matrix @ states[first]).astype(int))]
                    for second in range(8):
                        self.assertEqual(member[8 * first + second],
                                         member[8 * moved + second])
                        self.assertEqual(member[8 * second + first],
                                         member[8 * second + moved])

    def test_blocking_leaves_the_ensemble_free_energy_unchanged(self):
        """w_orbit = |orbit|/64 makes the annealed average manifold free."""
        for channel in assembly.BOUND:
            geom = self.structures['geometry'][channel]
            for temperature in (200.0, 250.0, 300.0):
                kbt = Boltzmann * temperature
                free = assembly.orbit_statistics(geom, 'free', kbt)
                blocked = assembly.orbit_statistics(geom, 'blocked', kbt)
                self.assertEqual((free['states'], blocked['states']), (64, 64))
                self.assertAlmostEqual(blocked['annealed_free_energy_J'] / kbt,
                                       free['annealed_free_energy_J'] / kbt,
                                       places=12)
                # Jensen: a quenched average can only sit above the annealed
                # one, with equality exactly when there is a single orbit.
                self.assertGreaterEqual(blocked['quenched_free_energy_J'] / kbt,
                                        blocked['annealed_free_energy_J'] / kbt
                                        - 1e-12)
                if len(blocked['orbits']) > 1:
                    self.assertGreater(blocked['quenched_free_energy_J'] / kbt,
                                       blocked['annealed_free_energy_J'] / kbt)
                else:
                    self.assertAlmostEqual(blocked['quenched_free_energy_J'] / kbt,
                                           blocked['annealed_free_energy_J'] / kbt,
                                           places=12)
                self.assertAlmostEqual(free['quenched_free_energy_J'] / kbt,
                                       free['annealed_free_energy_J'] / kbt,
                                       places=12)

    def test_manifold_switches_at_the_blocking_temperature(self):
        self.assertEqual(assembly.BLOCKING_TEMPERATURE_K,
                         geometry.PARAMS.blocking_temperature_K)
        for temperature in (200.0, 249.9, 250.0):
            self.assertEqual(assembly.manifold_for_temperature(temperature), 'blocked')
        for temperature in (250.1, 275.0, 300.0):
            self.assertEqual(assembly.manifold_for_temperature(temperature), 'free')

    def test_bondaxis_blocking_is_not_parallel_locking(self):
        """Half the bond-axis face orbits still reach the -4C/3 well."""
        geom = self.structures['geometry']['face']
        kbt = Boltzmann * 200.0
        statistics = assembly.orbit_statistics(geom, 'bondaxis', kbt)
        deep = [r for r in statistics['orbits'] if r['min_energy_J'] / kbt < -5.0]
        self.assertEqual(len(deep), 2)
        self.assertAlmostEqual(sum(r['weight'] for r in deep), 0.5, places=12)
        # The hard parallel restriction would have removed that well entirely.
        parallel = assembly.orbit_statistics(geom, 'parallel', kbt)
        self.assertAlmostEqual(parallel['min_energy_J'], geom['vdw_J'], places=30)
        self.assertLess(statistics['min_energy_J'], parallel['min_energy_J'])

    def test_blocking_switches_the_single_pair_preference_to_tip(self):
        """Below T_B the frozen face loses U_dd and the tip keeps it."""
        for twist in (5.0, 10.0):
            cold = assembly.face_tip_competition(
                200.0, self.structures, assembly.manifold_for_temperature(200.0), twist)
            hot = assembly.face_tip_competition(
                300.0, self.structures, assembly.manifold_for_temperature(300.0), twist)
            self.assertEqual(cold['dipole_manifold'], 'blocked')
            self.assertEqual(hot['dipole_manifold'], 'free')
            self.assertEqual(cold['preferred_single_pair'], 'tip')
            self.assertLess(cold['d_F_total_quenched_kBT'], -2.0)
            # The ensemble branch is unmoved by the switch, by the orbit identity.
            self.assertAlmostEqual(
                cold['d_F_dip_kBT'],
                assembly.face_tip_competition(200.0, self.structures, 'free',
                                              twist)['d_F_dip_kBT'], places=12)

    def test_restricted_tip_is_twist_independent_and_face_is_not(self):
        binding = assembly.Binding()
        tips = [assembly.restricted_channel_populations(
                    250.0, self.structures, 'tip', 'free', twist, binding)['p_pair']
                for twist in (1.0, 5.0, 10.0)]
        np.testing.assert_allclose(tips, tips[0], rtol=1e-14)
        faces = [assembly.restricted_channel_populations(
                     250.0, self.structures, 'face', 'free', twist, binding)['p_pair']
                 for twist in (1.0, 5.0, 10.0)]
        self.assertLess(faces[0], faces[1])
        self.assertLess(faces[1], faces[2])

    def test_restricted_odds_ratio_reproduces_the_competition_free_energy(self):
        """p_tip/p_face in the dilute limit must equal exp(-dF_total)."""
        # Strictly dilute, so no orbit saturates and the quenched ensemble
        # average coincides with the annealed one to machine precision.
        binding = assembly.Binding(volume_fraction=1.0e-9)
        for manifold in assembly.MANIFOLDS:
            for temperature in (200.0, 250.0, 300.0):
                rows = {c: assembly.restricted_channel_populations(
                            temperature, self.structures, c, manifold, 5.0, binding)
                        for c in assembly.BOUND}
                odds = ((rows['tip']['p_pair'] / rows['tip']['p_free'])
                        / (rows['face']['p_pair'] / rows['face']['p_free']))
                competition = assembly.face_tip_competition(
                    temperature, self.structures, manifold, 5.0)
                # Residual is the O(alpha) saturation correction, which
                # vanishes as phi -> 0; at phi = 1e-9 it is below 1e-7.
                self.assertAlmostEqual(
                    float(np.log(odds)), -competition['d_F_total_kBT'], delta=1e-7)

    def test_parallel_manifold_makes_tip_the_majority_channel(self):
        binding = assembly.Binding()
        for temperature in (200.0, 250.0, 300.0):
            face = assembly.restricted_channel_populations(
                temperature, self.structures, 'face', 'parallel', 5.0, binding)
            tip = assembly.restricted_channel_populations(
                temperature, self.structures, 'tip', 'parallel', 5.0, binding)
            self.assertGreater(tip['p_pair'], 100.0 * face['p_pair'])
            self.assertLess(tip['U_total_per_NC_kBT'], face['U_total_per_NC_kBT'])
            # Face keeps only its van der Waals binding in this manifold.
            self.assertAlmostEqual(face['U_bound_conditional_per_NC_kBT'],
                                   0.5 * face['vdW_pair_kBT'], places=10)


if __name__ == '__main__':
    unittest.main()
