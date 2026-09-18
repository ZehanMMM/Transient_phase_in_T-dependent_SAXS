import unittest
import numpy as np
from pair_free_reference import compute_point
from finite_window_rotational_entropy import add_entropy, roots, SEPARATED_STAGES, updated_rotation_fraction


class FiniteRotationTests(unittest.TestCase):
    def test_updated_rotational_domain_changes_entropy_only(self):
        for kind in ('face','tip'):
            base=compute_point(250,[1,0,0] if kind=='face' else [1,1,1])[0]
            old=add_entropy(base,kind)
            new=add_entropy(base,kind,rotation_domain='axial-face-free-tip')
            for field in base:
                if not np.isnan(base[field]):self.assertEqual(new[field],old[field])
            if kind=='tip':self.assertEqual(new['rotation_free_energy_cost_kBT'],0.)
            else:self.assertAlmostEqual(new['rotation_free_energy_cost_kBT'],12.609592002049173,places=10)
            self.assertAlmostEqual(new['Fmag_plus_rotation_time_mean_kBT']-
                new['Fmag_plus_rotation_equilibrium_kBT'],
                base['delta_Fmag_time_mean_kBT']-base['delta_Fmag_eq_kBT'],places=12)

    def test_updated_face_twist_is_unrestricted(self):
        from rotational_entropy_model import orientation_fraction
        self.assertAlmostEqual(updated_rotation_fraction('face',2)/orientation_fraction('face',2,5),9.)
        self.assertEqual(updated_rotation_fraction('tip',2),1.)

    def test_separated_stages_use_legacy_fields_without_vdw(self):
        base=compute_point(250,[1,0,0])[0]
        row=add_entropy(base,'face')
        fields=[s[0]+'_kBT' for s in SEPARATED_STAGES]
        self.assertEqual(row[fields[0]],base['Udd_time_mean_kBT'])
        self.assertEqual(row[fields[1]],base['delta_Fmag_time_mean_kBT'])
        self.assertAlmostEqual(row[fields[2]],
            row['partial_contact_F_with_rotation_kBT']-row['vdW_kBT'],places=12)

    def test_no_energy_or_probability_penalty(self):
        base=compute_point(250,[1,0,0])[0]
        new=add_entropy(base,'face')
        self.assertEqual(new['Udd_time_mean_J'],base['Udd_time_mean_J'])
        self.assertAlmostEqual(new['Fmag_plus_rotation_time_mean_kBT']-
                               base['delta_Fmag_time_mean_kBT'],
                               new['rotation_free_energy_cost_kBT'],places=12)
        self.assertAlmostEqual(new['partial_contact_F_with_rotation_J']/base['kBT_J'],
                               new['partial_contact_F_with_rotation_kBT'],places=12)
        self.assertTrue(np.isnan(new['assembly_delta_G_J']))
        self.assertAlmostEqual(new['partial_contact_F_with_rotation_kBT']-
                               new['partial_contact_F_with_rotation_equilibrium_kBT'],
                               base['relaxation_free_energy_excess_time_mean_kBT'],places=12)

    def test_entropy_once_per_particle_not_per_neighbour(self):
        base=compute_point(250,[1,0,0])[0]
        f,t=add_entropy(base,'face'),add_entropy(base,'tip')
        self.assertAlmostEqual(t['rotation_free_energy_cost_kBT']-
                               f['rotation_free_energy_cost_kBT'],
                               -2*np.log(12),places=12)

    def test_crossing_interpolation(self):
        self.assertEqual(roots([200,250,300],[-1,1,-1]),[225.,275.])


if __name__=='__main__':
    unittest.main()
