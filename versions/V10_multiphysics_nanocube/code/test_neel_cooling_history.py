import unittest
import numpy as np
from neel_cooling_history import calculate,Protocol


class CoolingHistoryTests(unittest.TestCase):
    def test_probability_inherited_through_every_ramp(self):
        frames,trace,rows=calculate('tip',Protocol(end_K=280.))
        for frame_index in range(1,3):
            old=rows[(frame_index-1)*64:frame_index*64]
            new=rows[frame_index*64:(frame_index+1)*64]
            np.testing.assert_array_equal([r['p_hold_end'] for r in old],
                                          [r['p_previous_hold_end'] for r in new])
        self.assertEqual(frames[1]['hold_start_s']-frames[0]['hold_end_s'],120.)
        self.assertTrue(any(r['stage']=='ramp' for r in trace))
        for name in ('p_hold_start','p_hold_mean','p_hold_end'):
            p=np.array([r[name] for r in rows]).reshape(-1,64)
            np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
            self.assertGreaterEqual(p.min(),0)

    def test_zero_rates_preserve_random_state(self):
        frames,_,rows=calculate('face',Protocol(end_K=280.),rates_scale=0.)
        for r in frames:self.assertLess(abs(r['Udd_hold_mean_kBT']),1e-10)
        np.testing.assert_allclose([r['p_hold_end'] for r in rows],1/64,atol=1e-12)

    def test_start_matches_reset_but_low_temperature_retains_energy(self):
        frames,_,_=calculate('tip')
        self.assertAlmostEqual(frames[0]['Udd_hold_mean_kBT'],frames[0]['Udd_reset_same_window_kBT'],places=12)
        self.assertLess(frames[-1]['Udd_hold_mean_kBT'],-2.)
        self.assertLess(abs(frames[-1]['Udd_reset_same_window_kBT']),.002)

    def test_ramp_step_convergence(self):
        for kind in ('face','tip'):
            coarse,_,_=calculate(kind,Protocol(ramp_substep_K=.25))
            fine,_,_=calculate(kind,Protocol(ramp_substep_K=.125))
            error=max(abs(a['Udd_hold_mean_kBT']-b['Udd_hold_mean_kBT']) for a,b in zip(coarse,fine))
            print(f'{kind} maximum ramp-grid difference: {error:.3g} kBT')
            self.assertLess(error,5e-5)


if __name__=='__main__':unittest.main()
