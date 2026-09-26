# Blocked-dipole condensation: the self-consistent cooling model

Produced by `code/binodal_phase_diagram.py`; checked by
`code/test_binodal_phase_diagram.py` (46 tests). The repository README gives
the overview; this file gives the derivation, the numbers and the record of
what was revised.

---

## 1. Scope

Zero-field, temperature-dependent SAXS of 16 nm Fe3O4 nanocubes at 1.25 vol.%
core in hexane. Observed: single colloidal phase at 300 K, aggregation between
about 270 and 250 K with one broad peak at 19 nm spacing, redissolution below,
reversible with about 10 K of rate-dependent hysteresis at both edges. The
broad peak means the dense phase is amorphous, so the framework is colloidal
fluid-fluid phase separation, not superlattice formation.

## 2. The terms, per face-registered bond

Signed so that negative binds. Geometry is measured: 16 nm cube, 19 nm centre
spacing, 3 nm gap.

| term | 300 K | 250 K | 200 K |
|---|---|---|---|
| van der Waals, U_vdW/kBT (converged) | -2.21 | -2.65 | -3.32 |
| annealed dipolar, -D | -4.40 | -5.64 | -7.53 |
| both moments blocked, b^2 | 0.008 | 0.19 | 0.81 |
| dipolar actually collected, -[(1-b^2)D + Phi] | -4.36 | -4.59 | -1.44 |
| total, vdW + dipolar | -6.57 | -7.24 | -4.76 |

The van der Waals value is the converged sharp-cube Hamaker sum, -9.156e-21 J
(A = 2.0e-20 J). The inherited 4^3 voxel sum gives -4.883e-21 J and is
under-converged by a factor 1.875: its 4 nm voxel cannot resolve a 3 nm gap on
a 1/r^6 kernel.

## 3. The dipolar term

**State space.** Each moment sits on one of the 8 <111> easy axes of its cube,
so a pair has 64 states with energies U_ij. Two identities hold exactly:
the sum over all 64 of U_ij is zero, and every row sums to zero. At 250 K the
spectrum is -7.675 (x8), -3.837 (x16), 0 (x16), +3.837 (x16), +7.675 (x8) kBT.
The zero level is the parallel and antiparallel states, at the magic angle
(s.n)^2 = 1/3; the attractive states are head-to-tail.

**Annealed term.** D = ln[(1/64) sum_ij exp(-U_ij/kBT)]. It is positive
although <U_dd> = 0: at 250 K the pair spends 96 % of its time in the eight
deepest states. Equivalently D = -beta<U>_Boltzmann - D_KL(p || uniform),
7.51 - 1.87 = 5.64 kBT.

**Why b^2.** A bond has three cases: both moments free, (1-b)^2; one free,
2b(1-b); both blocked, b^2. With the partner frozen at j, the conditional
partition function Z_j = sum_i exp(-U_ij/kBT) is the same for all eight j, so
ln(Z_j/8) = D exactly and one mobile moment recovers the whole annealed value.
The first two cases therefore combine to (1-b^2)D.

**Blocked bonds inherit their history.** Bonds freezing between T' and T'+dT'
are a share -db^2/dT' and keep the distribution they had then. A particle that
froze in the dilute phase had no partner: its moment froze into one of eight
body-frame axes with equal weight and is uncorrelated with any later bond, so
the zero row sums make its contribution exactly zero. A particle that froze
inside an aggregate froze out of the Boltzmann distribution over its existing
bonds and keeps Q_z per bond:

    Phi(T) = (1/kBT) int_T^inf (-db^2/dT') x(T') Q_z(T') kB T' dT'

**Frustration.** One moment serves z bonds and cannot be deepest for all of
them. Exhaustively over all 8^z frozen neighbour sets, at 250 K:

| z | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| Q_z (kBT) | 7.51 | 3.85 | 3.12 | 2.64 |

Q_6(T) T is constant to 0.1 % (659.4 K), so the kept energy is fixed in joules
and strengthens on cooling like van der Waals. Q_1 exceeds D because a bond
frozen in place does not pay the 1.87 kBT orientational entropy.

**Closure.** x is the particle share of the dense phase from the lever rule,
x = phi_d (phi - phi_l) / [phi (phi_d - phi_l)]. Phi at T needs x only above
T, so one downward sweep closes the loop.

## 4. Results

At z = 6, CV 4.85 %, dwell 100 s, contact cost 4.39 kBT:

| | |
|---|---|
| two-phase window | 250-268 K (observed 250-270) |
| peak particle share in the dense phase | 12.8 % at 258 K |
| peak volume share of the dense phase | 0.84 % |
| inherited frozen term Phi | <= 0.036 kBT; ceiling 2.68 kBT |
| total, 300 K minus minimum | 0.72 kBT |

The frozen term is small for a structural reason. b^2 rises from 0.19 at
250 K to 0.81 at 200 K, so 62 % of all contacts freeze below the window, where
x = 0 and they contribute nothing. The aggregate exists only in an 18 K window,
during which b^2 rises by 0.12 and x never exceeds 0.13.

Sensitivity:

| | window | peak x | Phi at 200 K |
|---|---|---|---|
| z = 6 | 250-268 K | 12.8 % | 0.036 |
| z = 1 | 239-268 K | 23.6 % | 0.379 |

Positive feedback (more aggregation, more freezing in place, more attraction)
runs away when the aggregate holds more than about 88 % of the particles,
which happens above an effective tilt of about 18.5 degrees; there the dense
phase never redissolves within 200-300 K. At 17 degrees (70 % aggregated) it
still redissolves, at 224 K.

The energy minimum sits 12-16 K above the half-blocking temperature for any CV
between 1 and 5 %, so the cold edge falling at T_B is structural, not fitted.

## 5. What is fitted

CV (4.85 %) and the contact cost C_contact (4.39 kBT) are set against the two
observed edges; the window position is therefore not a prediction.

C_contact matches the orientational registration cost of an isolated pair at
the measured +/-15.5 degree tilt tolerance (4.43 kBT), but that coincidence
proves little. In the dense phase one orientation serves all six contacts, and
the per-particle orientational cost saturates at 3.57 kBT by the second
non-collinear neighbour; spread over six bonds that is 1.19 kBT per bond. The
remaining 3.2 kBT is attributed to ligand-shell compression and conformational
entropy at the 3 nm gap, which does scale per contact. It is constrained by
the data, not derived.

Blocking temperature: K_cubic is calibrated so tau_N = 100 s at 250 K for the
nominal cube (K_cubic = 2.56e5 J/m^3, dE = 8.75e-20 J, tau0 = 0.98 ns). The
ensemble is half blocked at 244 K because b is the survival probability
exp(-t/tau_N), which is 1/e at tau_N = t, and the lognormal is set by the mean
diameter. There is no disagreement to reconcile.

## 6. Limitations

- Nucleation is not modelled, so the rate-dependent hysteresis is not
  reproduced.
- The dense-phase free energy is van der Waals level (Carnahan-Starling plus
  mean field, z_max = 6, eta_cp = 0.64).
- Wall-oriented heterogeneous nucleation, suggested by the anisotropic 2D
  pattern, is not described.
- x(T) is the equilibrium lever-rule share at each step; particles are not
  followed individually.

## 7. Revision history

Earlier conclusions in this project that turned out wrong, recorded because the
reasoning is not reconstructible without them.

1. **The superlattice framing was wrong.** The broad SAXS peak means an
   amorphous dense phase; `configuration_averaged_assembly`'s face/tip analysis
   does not apply to this experiment.
2. **The van der Waals conclusion reversed twice** before the threshold was
   derived rather than assumed; the converged value is the one that works.
3. **A predicted absence of lower-edge hysteresis was falsified** by the data;
   the hysteresis is kinetic at both edges.
4. **The scalar loss fraction lambda was replaced.** lambda = 1 - f Q_z/D was
   an ansatz between two computed endpoints. The history integral replaces it
   and reproduces both endpoints.
5. **Volume share was mistaken for particle share.** phi/phi_dense = 6.7 %
   bounds the volume of the dense phase, not the share of particles in it,
   which runs to one for a deep quench. An argument that most moments must
   freeze in the dilute phase rested on this and was withdrawn.
6. **A sparse scan was misread.** Four points suggested the model had no
   parameter range with both a visible aggregate and redissolution; a finer
   scan found it.
7. **The registration check was not independent.** The fitted and geometric
   costs agreed to 0.24 degrees, but both used isolated-pair bookkeeping.
   Only about a quarter of the cost has a geometric origin in the dense phase.
8. **"The isolated pair is clean" was overstated.** The registration cost is
   unambiguous for a pair, but the b^2 mechanism is not a pair effect: a free
   particle re-randomises its moment by Brownian tumbling (about 6 us) whether
   or not Neel is blocked. The mechanism requires the particle to be caged.
9. **The 6 K "gap" between 244 K and 250 K was not a disagreement**, as
   explained in section 5.
