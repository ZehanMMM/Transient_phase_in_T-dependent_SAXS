# Configuration-averaged colloid/superlattice populations

Produced by `code/configuration_averaged_assembly.py`.
The van der Waals model is the **inherited** V10 4³ sharp-cube voxel Hamaker
sum and the gap model is the **inherited** exact rounded-cube (1.5 nm
Minkowski) centre distance. Nothing in `geometry_model.py` was modified; the
V13 `npvdw` replacement was cancelled and is not used here.

## What was added

The inherited entry points evaluate one fixed pair geometry and report its
magnetic energy. None of them decides whether the pair exists. The contact
geometry is now a label of the configuration space

| axis | states | relaxes by |
|---|---|---|
| contact `c` | `free`, `face`, `tip` | Brownian translation (docking) and rigid rotation |
| dipole `(i,j)` | inherited 8 × 8 lab ⟨111⟩ pairs | Néel inside the cube, **or** rigid cube rotation |

**The face/tip ordering from this 192-state average is refined in the
"Face vs tip" section below**, which resolves the dipole manifold and the
orientational entropy the average leaves out.

3 × 64 = 192 states, one detailed-balanced generator, exact to 8e-19 in the
worst row of the whole run. The closed-form Boltzmann split is reproduced to
2e-15.
Summing over `c` is the cube-configuration average; summing over `(i,j)` is
the dipole-configuration average. The colloid/SL ratio is the marginal over
`c`; the equivalent total system energy is the average of the per-NC energy
over the *same* distribution.

Rigid rotations while bound are restricted to the ones that preserve the
contact: ±90° about the ⟨100⟩ bond for `face`, ±120° about the ⟨111⟩ bond for
`tip`. A free cube keeps the inherited six ⟨100⟩ quarter turns. Each move set
is calibrated so the zero-coupling first-rank decay is exactly 1/τ_B
(λ = 2, 3, 4 respectively).

Binding has one length and one concentration. Detailed balance fixes the
reaction volume from the Smoluchowski encounter rate and the diffusive escape
attempt frequency, `v_site = k_diff/ν_esc = 4π d_h ℓ² = 59.7 nm³` at
ℓ = 0.5 nm; the bound/free weight is `α_c = z_c v_site n` with `n = φ/V`.
**φ and ℓ are assumptions, not measurements**, and both are swept.

Two assembly modes: `dimer` (one bond, mass action) and `superlattice` (the
inherited `coordination_factor_per_nc` convention, z_c/2 bonds per site, a
mean-field scaling of a pair model).

## Answers to the three regimes

### Timescales, from the inherited parameters alone

| T | τ_N (Néel) | τ_B (Brownian) | encounter | 20 s window |
|---|---|---|---|---|
| 300 K | 1.5 s | 7.6e-7 s | 1.8e-6 s | both relaxations fast |
| 250 K | 100 s (= T_B by construction) | 1.6e-6 s | 3.8e-6 s | Néel crossover |
| 200 K | 5.7e4 s | 4.1e-6 s | 9.7e-6 s | Néel blocked, Brownian only |

The three regimes requested are therefore real and well separated, and τ_B
stays microscopic in all of them.

### Regime 1 and 2 have the *same* populations — and that is a result, not an omission

Because τ_B ≪ 20 s at every temperature, the lab-frame dipole label
equilibrates in both regimes: above T_B via Néel *and* rotation, below T_B via
rotation alone, with the moment frozen in the body frame. Detailed balance
then forces the equilibrium colloid/SL split to be the same function of T in
both regimes. The encounter time is microseconds at φ = 1e-2, so the 20 s
window mean and the equilibrium split agree to ~1e-6 (see `window_scan.csv`:
the partition needs ~100 encounter times, i.e. ~1 ms at φ = 1e-2 and ~1 s at
φ = 1e-4, so any window ≥ 1 s is equilibrated).

What separates the regimes is the **bond lifetime**, not the populations.

### Populations and equivalent total energy, dimer mode, φ = 1e-2

| T | p(colloid) | p(SL, face) | p(SL, tip) | U per NC | 20 s bond survival |
|---|---|---|---|---|---|
| 200 K | 0.095 | 0.905 | 0.0005 | −5.11 k_BT | 0.000 |
| 225 K | 0.267 | 0.732 | 0.0011 | −3.65 k_BT | 0.000 |
| 250 K | 0.496 | 0.502 | 0.0016 | −2.24 k_BT | 0.000 |
| 265 K | 0.620 | 0.379 | 0.0018 | −1.59 k_BT | 0.000 |
| 275 K | 0.688 | 0.310 | 0.0019 | −1.25 k_BT | 0.000 |
| 300 K | 0.811 | 0.187 | 0.0019 | −0.69 k_BT | 0.000 |

p_bound passes ½ at **250.4 K**, inside the experimentally flagged 253–273 K
transient-aggregation band, with nothing fitted to it. The crossover moves
logarithmically with concentration: T₅₀ = 166 / 199 / 250 / 340 K for
φ = 1e-4 / 1e-3 / 1e-2 / 1e-1, about 50 K per decade.

**One bond is never enough.** The deepest dimer bond is only −5.7 k_BT per NC
and the diffusive escape prefactor is ~1e8 /s, so no dimer survives 20 s at
any temperature in 200–300 K. At 200 K, 90 % of pairs are bound *at any
instant* but the identity of the partner turns over every ~0.3 ms. An
isolated pair is a rapidly exchanging contact, not a structure.

### Populations and lifetimes, superlattice mode, φ = 1e-2

A lattice site carries −23 to −34 k_BT per NC, so it is bound at every
temperature (p_free < 2e-6 throughout) and the thermodynamic crossover leaves
the window entirely. The regimes separate kinetically:

| T | U per NC | τ_bond | 20 s survival |
|---|---|---|---|
| 300 K | −22.7 k_BT | 3.7 s | 0.005 |
| 275 K | −24.8 k_BT | 41 s | 0.616 |
| 265 K | −25.7 k_BT | 122 s | 0.849 |
| 250 K | −27.3 k_BT | 718 s | 0.973 |
| 225 K | −30.3 k_BT | 2.3e4 s | 0.999 |
| 200 K | −34.1 k_BT | 1.6e6 s | 1.000 |

The 20 s survival passes ½ at **278.5 K**. So: at 300 K the SL bond dissolves
within the observation window (regime 1); below T_B it is effectively frozen
(regime 2); and 250–280 K is a continuous crossover (regime 3), resolved at
1 K spacing in `populations_and_energies.csv`.

### Which relaxation dissolves the bond

`contact_barrier_scan.csv` isolates the mechanism by suppressing the rotation
channel only. At 300 K, superlattice mode:

| extra rotation barrier | τ_bond | Néel blocked | rotation blocked | direct escape only |
|---|---|---|---|---|
| 0 | 3.73 s | 3.74 s | 5.67 s | 11.7 s |
| +5 k_BT | 4.19 s | 4.24 s | 5.67 s | 11.7 s |
| +10 k_BT | 5.61 s | 10.7 s | 5.67 s | 11.7 s |

At the inherited zero contact barrier the **contact-preserving Brownian
rotation** dominates (τ_bond tracks "Néel blocked") and Néel is a ~2×
modifier. At +10 k_BT the rotation channel is shut and τ_bond tracks
"rotation blocked", i.e. it becomes **Néel limited** at 5.7 s — still shorter
than the 20 s window, so the qualitative conclusion for regime 1 survives
either way. The statement "fast Néel breaks the pair up" is the
barrier-limited branch of this model, not the zero-barrier default; the
inherited modules document that zero barrier as an explicitly optimistic
accessibility baseline, and it is not measured.

### Face vs tip, with the blocked rotations taken from the entropy model

**Third and current version of this section.** Earlier versions (a) claimed
tip never competes, (b) replaced the below-T_B dipole state by a hard
parallel restriction, and (c) froze the tip's contact vertex. All three were
wrong in the same way: they guessed what a blocked cube can do instead of
reading it off the orientational model that supplies the entropy term.

Throughout: the **contact is fixed**, the **dipole is never fixed** (always
Boltzmann). T_B switches *which states the dipole can reach*.

**What a blocked cube can still do, from `orientation_fraction` itself:**

| contact | `orientation_fraction` | what that grants | blocked group | orbits |
|---|---|---|---|---|
| tip | 8·cap | all 8 body diagonals, twist **free** over 2π | full cube group | **1 orbit of 64** |
| face | 24·cap·δ/π | twist confined to ±δ ≈ a few degrees | identity | **64 orbits of 1** |

A vertex contact is a point contact: choosing *which* vertex touches is itself
a rigid rotation mapping tip-contact to tip-contact, so a blocked tip cube can
turn its frozen moment onto the bond and reach head-to-tail. A face contact is
registry-locked to ±δ, so it cannot even perform the 90° bond-axis turn, let
alone roll to a different face. **So a tip contact is never really blocked and
a face contact fully is** — and this now agrees with the entropy term instead
of contradicting it.

**Two exact consequences:**

- Tip: blocked ≡ free, to machine precision, and the −2C head-to-tail state
  sits inside the single orbit.
- Face: Σ over all 64 states of U_dd is **exactly zero** by symmetry, so a
  fully frozen face pair has quenched free energy equal to its van der Waals
  term alone, −1.77 k_BT at 200 K. It keeps no dipolar binding at all.

Both are asserted in the tests.

**Result** (F_tip − F_face per pair, negative favours tip):

| branch | manifold | T | ΔF_dip | ΔF_twist (δ=5°) | **ΔF_total** |
|---|---|---|---|---|---|
| ensemble | free | 300 K | +4.86 | −4.97 | **−0.11** |
| ensemble | blocked | 200 K | +7.76 | −4.97 | **+2.79** |
| single pair | free | 300 K | +4.86 | −4.97 | **−0.11** |
| single pair | blocked | 250 K | +0.41 | −4.97 | **−4.56** |
| single pair | blocked | 200 K | +0.23 | −4.97 | **−4.74** |

The single-pair branch now steps by ~5 k_BT at T_B, toward tip. With the
manifold switched at T_B, which contact is preferred over 200–300 K:

| δ | ensemble branch | single-pair branch |
|---|---|---|
| 1° | tip 200–300 K | tip 200–300 K |
| 2° | tip 226–300 K | tip 200–300 K |
| 5° | tip 295–300 K | tip 200–300 K |
| 10° | tip nowhere | **tip 200–250 K, face 250–300 K** |

**δ = 10° on the single-pair branch reproduces the reported experiment**: tip
below T_B, face above, with the changeover at T_B by construction. The
mechanism is now explicit and does not require assuming parallel moments — a
frozen face pair is stranded in whatever state it docked in and its dipolar
binding averages to exactly zero, while a frozen tip pair can still rotate
its moment onto the bond.

**The ensemble branch is still continuous across T_B, and that is a theorem.**
Orbit weights are w = |orbit|/64, and

    Σ_orbits w ⟨exp(−U/k_BT)⟩_orbit ≡ (1/64) Σ_all 64 exp(−U/k_BT)

identically, so no re-partitioning of the 64 states into orbits can change
the dilute-limit ensemble free energy (tests assert 1e-12). Blocking
reshuffles which states a given pair can reach, not how many pairs sit at
each energy. The ensemble and single-pair branches therefore answer different
questions and must not be conflated:

- **ensemble** — what fraction of all pairs is bound in each geometry at
  equilibrium. Continuous at T_B.
- **single pair** — the binding free energy of one pair locked into its own
  orbit; the right criterion if the observed structure is selected by which
  contact nucleates and survives rather than by the equilibrium population.
  This is where the switch lives.

The `bondaxis` convention (face C4, tip C3 about the bond) is retained in the
CSVs as the intermediate case, and `parallel` as the hard-restriction limit.

Caveats: δ is the one parameter that matters and it is not measured (the tilt
cone cancels in every tip-minus-face difference, δ does not); the tip gap is
still pinned at the face value of 3 nm, putting centres at 28.5 nm against
19.0 nm, and closing it would strengthen tip further on both branches; and
the orbit weights assume the moments freeze while the particles are still
dispersed, where the dipolar coupling is ~0.1 k_BT.

### Restricted single-channel assembly, with the manifold switching at T_B

`restricted_channel_populations.csv` and
`restricted_channels_{dimer,superlattice}_twist{1,2,5,10}deg.png`: **if the
system may form only face-to-face, or only tip-to-tip, what fraction is
paired and what is the total system energy?** One figure per twist tolerance
δ, three stacked panels — pair fraction (log), total system energy per NC,
and the bound-pair energy on its own axis (the direct analogue of the
published panel-D curves). A single set of curves, with the dipole manifold
switching at T_B = 250 K, marked by the dashed vertical line; the thin dashed
continuations show what the other manifold would have given, so the size of
the switch is visible rather than hidden in the kink.

The orientational weight enters as `α_c = z_tip · v_site · n · w_c(δ)` with
`w_tip = 1` and `w_face = (3δ/π)²`. Only the face/tip *ratio* is applied,
because that is the part the tilt cone cancels out of; the absolute
orientational volume of a contact is not measured and stays absorbed in
`v_site`, calibrated so the tip channel reproduces the baseline
`α_tip = z_tip v_site n` of the 192-state model.

Dimer mode, φ = 1e-2, manifold switched at T_B:

| δ | T | manifold | p_face | p_tip | p_tip/p_face | U_face | U_tip |
|---|---|---|---|---|---|---|---|
| 1° | 200 K | blocked | 3.5e-3 | 5.2e-3 | 1.5 | −0.020 | −0.007 |
| 1° | 300 K | free | 8.6e-5 | 2.4e-3 | **27.6** | −0.000 | −0.002 |
| 2° | 200 K | blocked | 1.4e-2 | 5.2e-3 | 0.4 | −0.078 | −0.007 |
| 2° | 300 K | free | 3.4e-4 | 2.4e-3 | **6.9** | −0.001 | −0.002 |
| 5° | 200 K | blocked | 7.5e-2 | 5.2e-3 | 0.1 | −0.424 | −0.007 |
| 5° | 300 K | free | 2.1e-3 | 2.4e-3 | 1.1 | −0.008 | −0.002 |
| 10° | 200 K | blocked | 2.1e-1 | 5.2e-3 | 0.03 | −1.169 | −0.007 |
| 10° | 300 K | free | 8.5e-3 | 2.4e-3 | 0.3 | −0.031 | −0.002 |

Reading:

- **Tip is completely twist-independent** — its bond-axis rotation is free —
  so the whole δ dependence sits in the face channel: p_face moves ~3 decades
  from δ = 1° to 10° while p_tip does not move at all. δ is the single knob
  that controls the competition.
- **The switch at T_B is small in the populations** (8 % at δ = 5°, 21 % at
  δ = 10°, both in the direction of less pairing when blocked) for the
  identity reason given above. Anyone expecting a step at T_B in the pair
  fraction should read that section first: the step belongs in the
  single-pair free energy and in the bond lifetime, not in the population.
- **The temperature trend runs the wrong way for the experiment**: tip is
  relatively favoured at *high* T (where its weaker but twist-free binding
  competes with a face channel that the twist cost suppresses) and face at
  *low* T (where the deep −4C/3 face well, still reachable in half the
  blocked orbits, wins). Reproducing a below-T_B tip preference needs one of
  the four routes listed in the previous section.
- Absolute pair fractions are small in dimer mode (one bond, ≤ 5.7 k_BT) and
  scale linearly with φ; the superlattice-mode figures saturate near p = 1
  instead, so there the energy panel is the informative one.

### Cooling trajectory: the superlattice is stranded by its own frozen moments

`cooling_trajectory.csv`, `cooling_trajectory.png`. A **fixed** face-to-face
contact cooled from 300 K, with the dipole distribution freezing when tau_N
exceeds the observation window. The freeze temperature is **266.9 K** for a
20 s window (not T_B = 250 K, which is the 100 s ZFC/FC definition; both are
marked).

**Why selective binding stops working inside a lattice.** Above the freeze
temperature the bound population is enriched in attractive dipole
configurations, and that enrichment does *not* need Néel: a particle whose
frozen moment points the wrong way simply fails to stick, unbinds, rotates
freely in solution and re-docks. Selective binding is translational. A
particle already *inside* a formed superlattice cannot do this — it is
surrounded by z neighbours, its twist is registry-locked to ±δ, and below the
freeze temperature it cannot flip by Néel either. It is stranded with the
moment it froze with.

**What it inherits is the whole question**, and the two limits bracket it:

| inherit | U_pair at 267 K → 200 K | repulsive-bond weight | z=6 lattice per NC at 200 K |
|---|---|---|---|
| `annealed` (ordered, from the bound state) | −8.31 → −11.10 k_BT | <1e-4 | −33.3 k_BT |
| `dispersed` (random, from the colloid) | −8.31 → **−1.77** k_BT | **0.375** | **−5.3 k_BT** |

The random branch is the disassembly mechanism, and it is exact rather than
fitted: **Σ over the 64 states of U_dd is identically zero**, so a pair with a
random frozen dipole has *no* mean dipolar binding and is left with the van
der Waals term alone, −1.77 k_BT at 200 K. The pair energy therefore **jumps
upward by +7.2 k_BT at the freeze temperature** and then stays flat in joules
while the equilibrium reference keeps deepening to −11.3 k_BT — an excess that
grows to +9.5 k_BT by 200 K. Per NC in a z = 6 lattice the binding collapses
from −33.3 to −5.3 k_BT, and **37.5 % of bonds (24 of the 64 states) become
net repulsive**, up to +7.8 k_BT.

So on this picture the low-temperature fate is **disassembly of the
face-to-face superlattice, not conversion to tip-to-tip**, which is the
reading the experiment suggests.

**How strong is the enrichment that is lost?** At 300 K a bound face pair is
**92.2 %** in its single deepest dipole level (8 of 64 states), rising to
98.4 % at 200 K, against 12.5 % for a random dipole — a 7.4× enrichment at
300 K. That lowest level is *head-to-tail along the bond*: both moments have
the same bond-axis component (so the −3(s1·n)(s2·n) term attracts) while their
transverse components are antiparallel (so s1·s2 attracts too). Both
contributions are attractive at once, which is why it sits 6.4 k_BT below the
next level at 300 K and takes essentially all the weight.

Caveats: which branch applies depends on whether the lattice orders its
moments *before* tau_N exceeds the window, which this pair model cannot settle
— it needs the intra-lattice Néel ordering time against the cooling rate. The
`dispersed` branch also assumes the lattice keeps growing or rearranging below
the freeze temperature; bonds already formed in an attractive configuration
survive, so a real sample should show a mixture weighted by how much of the
lattice was built above 267 K.

### Gradual freeze-in: no step, and a binding maximum inside the experimental band

`gradual_cooling.csv`, `gradual_cooling.png`. The step in the previous
section is an idealisation. tau_N crosses the fixed observation window at a
different temperature for every particle size, so the freeze-in is gradual:

    b(T) = < exp(-window / tau0 exp(f dE/k_BT)) >,   f = (D/Dbar)^3

with lognormal diameters. A 10 % spread in D is a 33 % spread in the barrier
and, on an exponential, enormous: **31 % of particles are already Néel-blocked
at 300 K on a 20 s window, and 20 % are still thermally active at 200 K.**

*Quadrature note.* This uses its own 400-node rule over the normal variate,
not the inherited 11-node Gauss-Hermite of
`geometry_model.barrier_distribution`. That rule is built for smooth
integrands; `exp(-window/tau)` is nearly a step in f and the 11-node result is
wrong by 15 % at 270 K (0.373 against 0.436). Checked against adaptive
quadrature in the tests.

A pair is then a three-component mixture, exact rather than interpolated:
both moments free `(1-b)²`, one frozen `2b(1-b)`, both frozen `b²`.

**An exact identity, face only.** For a ⟨100⟩ bond the conditional partition
function Z_j = Σ_i exp(−U_ij/k_BT) is the *same* for all eight frozen partner
states j, so the half-frozen term equals the fully annealed one exactly and
the mixture collapses to

    U(T) = (1 − b²) U_annealed(T) + b² U_frozen

**One mobile moment recovers the whole equilibrium dipolar binding**; only a
pair with *both* moments blocked loses it. The controlling variable is
therefore **b², not b**. It follows from the C4 stabiliser of the bond plus
the (i,j) → (−i,−j) symmetry of a bilinear energy, and it fails for tip, where
Z_j differs by 2.9× between the head-to-tail states and the rest. Both are
asserted in the tests.

**Result: the binding strength peaks and then falls**, with no discontinuity
anywhere:

| size CV | T of strongest binding | U_pair there | U_pair at 200 K | rise |
|---|---|---|---|---|
| monodisperse | 276 K | −7.93 k_BT | −1.78 k_BT | +6.16 |
| 5 % | 285 K | −7.22 k_BT | −2.69 k_BT | +4.53 |
| **10 %** | **272 K** | **−6.90 k_BT** | **−5.25 k_BT** | **+1.65** |
| 15 % | 241 K | −6.99 k_BT | −6.66 k_BT | +0.34 |

The maximum sits at 272–285 K for CV ≤ 10 %, i.e. at the top of or just above
the experimentally flagged 253–273 K transient-aggregation band, and the
face-to-face contact weakens monotonically below it. Over the same interval
the weight of net-repulsive bonds climbs from ~4 % at 300 K to 24 % at 200 K
(CV = 10 %), or to 37 % monodisperse.

So the cooling picture is: assemble on the way down, reach the strongest
face-to-face binding near the top of the aggregation band, then weaken
continuously as more and more pairs have *both* moments frozen — a gradual
disassembly rather than a transition to tip-to-tip.

The size distribution controls how much of the effect survives, and it is the
one number here worth measuring: at CV = 15 % the rise is only +0.34 k_BT and
the mechanism is essentially washed out, while monodisperse gives +6.2 k_BT.
The inherited `diameter_coefficient_of_variation` is 0 and is flagged in
`geometry_model` as a placeholder for the sample's measured TEM distribution.

### The frozen distribution is INHERITED, not uniform — and that decides everything

`history_cooling.csv`, `history_cooling.png`. This corrects a physical error
in the previous section, which handed every frozen pair a *uniform* dipole
distribution. A pair that froze at temperature T_pair inherits the
distribution it actually had then:

    q(T_pair) = p_bound(T_pair) · Boltzmann_bound(T_pair)
              + (1 − p_bound(T_pair)) · uniform

Uniform is only correct for a pair that froze while still **dispersed**,
where the free pair has essentially no dipolar coupling (~0.1 k_BT) so its
Boltzmann distribution *is* uniform. A pair that froze while already **bound**
inherits the attractive bias and keeps it.

T_pair is set by the **second** moment to freeze — the exact consequence of
the face identity above, since one mobile moment already recovers the whole
equilibrium binding. With independent sizes the variable is m = min(f₁, f₂),
density 2(1−F)p, so the frozen weight integrates to exactly b².

| branch | excess at 300 K | at 250 K | at 200 K | repulsive weight at 200 K |
|---|---|---|---|---|
| froze while **bound** | +0.01 | +0.04 | **+0.12 k_BT** | <1e-4 |
| computed p_bound(T_freeze) | +0.67 | +2.47 | **+6.37 k_BT** | 0.25 |
| froze while **dispersed** | +0.67 | +2.48 | **+6.52 k_BT** | 0.26 |

So the mechanism lives or dies on *where the moments froze*:

- Froze while bound → the superlattice keeps its ordered moments and loses
  only 0.12 k_BT by 200 K. **No disassembly.**
- Froze while dispersed → inherits uniform, loses 6.5 k_BT and 26 % of bonds
  turn net repulsive. **Disassembly.**

**The computed branch sits on top of the dispersed one**, and that is the
substantive result rather than an assumption: p_bound at the relevant freeze
temperatures is tiny. The freeze temperatures are high — f = 1 gives 267 K but
f = 1.25 gives 334 K and f = 1.5 gives 400 K — so the larger half of a
CV = 10 % distribution froze *above room temperature*, long before any
assembly, and p_face there is ~2e-3 at φ = 1e-2 with δ = 5°. Essentially every
pair therefore froze while dispersed, the inherited distribution really is
uniform, and the disassembly branch is the one that applies.

That conclusion is concentration- and assembly-dependent by construction: in
`superlattice` mode p_bound ≈ 1, so if a lattice already exists above the
freeze temperature its interior froze ordered and nothing happens to it. The
discriminator is whether assembly precedes or follows the freeze-in, and with
freeze temperatures of 270–400 K it follows.

### The re-entrant aggregation window, and what the pair model can and cannot do

Experiment: pure colloidal at 300 K, aggregation observed between ~270 and
~250 K, aggregation gone again on further cooling. This is the strongest
constraint in the file and it is worth being precise about which parts of it
this model earns.

**Two structural limits, both provable, both negative for a pair model.**

1. *A pair-level bound fraction cannot vanish on cooling.* The best frozen
   dipole configuration at 200 K has 144× the affinity of the *annealed*
   average at 270 K. So if anything binds at 270 K, the 8/64 pairs that froze
   into a good configuration are certainly bound at 200 K, and p_bound is
   floored near 8/64 = 12.5 %. Scanning α over six decades, the best the pair
   model produces is a shallow maximum — at α = 1e-2 the peak sits at 266 K
   with p = 0.67 → 0.72 → 0.56 across 300 → 266 → 200 K. A 20 % modulation,
   not an appearance and disappearance.
2. *Selective binding defeats the frozen-dipole penalty at the pair level.* A
   pair with a badly frozen moment simply fails to stick, unbinds, rotates and
   re-docks, so the bound ensemble is re-enriched. This is the same orbit
   identity as above.

**What breaks both limits is coordination.** A lattice site cannot be
re-selected — its moment is shared by z neighbours and one frozen moment
cannot satisfy all of them. Per bond, P(attractive) = 24/64 = 0.375, so the
bulk is strongly frustrated, and a 12³ simple-cubic sample of frozen random
⟨111⟩ moments gives a per-NC energy that collapses to the van der Waals floor:

| Hamaker A | ordered lattice, 200 K | frozen random lattice, 200 K | 
|---|---|---|
| 9 zJ | −31.2 k_BT | **−2.8 ± 7.5 k_BT** |
| 14 zJ | −32.3 k_BT | −3.7 ± 7.3 k_BT |
| 20 zJ (inherited) | −34.1 k_BT | −5.6 ± 7.4 k_BT |
| 29 zJ | −36.2 k_BT | −8.1 ± 7.0 k_BT |

21 % of sites are net repulsive in every case. **At the low end of the
reported Lifshitz range the frozen lattice is held by only ~2.8 k_BT per NC
with a 7.5 k_BT spread, i.e. marginally — it dissolves. At the inherited
20 zJ it does not.**

**A three-part reading that the model does support**, with both edges kinetic:

| T | what the model says | source |
|---|---|---|
| 300 K, colloidal | bond lifetime 3.7 s < 20 s window, so no *persistent* lattice however favourable the thermodynamics | `populations_and_energies.csv` |
| 270–250 K, aggregated | lifetime crosses the window (122 s at 265 K, 20 s survival 0.85) while the dipoles are still mostly mobile (b² = 0.23–0.34), so the lattice orders and binds at −23 to −25 k_BT per NC | same + `history_cooling.csv` |
| below 250 K, colloidal again | b² passes ⅓, the lattice is stranded frustrated-random, and the van der Waals floor alone is marginal | `history_cooling.csv` + the table above |

So the upper edge is the **bond lifetime against the observation window** and
the lower edge is **dipole freeze-in plus lattice frustration**. Neither edge
is thermodynamic, which is why a purely equilibrium pair model missed both.

**What this costs in assumptions.** The lower edge needs A near 9 zJ rather
than the inherited 20 zJ. That is a documented uncertainty — `geometry_model`
calls 2.0e-20 J "a representative value ... should be replaced when the
experimental solvent and its optical data are specified", citing the 9–29 zJ
Faure range — so it is a legitimate parameter to pin, not a free fit. But it
is a prediction: **the mechanism requires the low end of that range**, and an
independent Lifshitz calculation for the actual solvent would confirm or kill
it.

**What is still missing.** The frustration number above is a 12³ random sample,
not a model: it has no lattice relaxation, no defect structure, no growth
kinetics, and no distinction between surface and bulk sites. Turning this
reading into a prediction of the window edges needs a lattice model with
frozen moments, which is outside the pair framework used everywhere else in
this file.

### Audit: the inherited 4³ voxel van der Waals sum is under-converged by 1.9×

This is independent of every modelling choice above and holds whichever vdW
branch is eventually adopted, so it is recorded on its own.

Refining the *same* sharp-cube Hamaker integral with the displacement-
multiplicity sum (`rotational_entropy_model.central_vdw`, which is already in
the tree as a diagnostic) converges to a value 1.9× deeper than the inherited
4³ voxel result for the face pair at the inherited 3 nm gap:

| grid per axis | E_vdW (J) | k_BT at 300 K |
|---|---|---|
| 8 | −7.337e-21 | −1.771 |
| 16 | −8.606e-21 | −2.078 |
| 32 | −9.037e-21 | −2.182 |
| 48 | −9.124e-21 | −2.203 |
| 64 | −9.156e-21 | −2.210 |
| **inherited 4³ (with the 1e-19 m² floor)** | **−4.883e-21** | **−1.179** |

The cause is geometric: the voxel edge is 4 nm while the surface gap is 3 nm,
and a 1/r⁶ kernel is dominated by the closest surface elements, so a 4³ grid
cannot resolve the near-contact region. The `vdw_d2_floor_m2 = 1e-19` guard
(3.2 nm) sits right at the gap scale and truncates exactly the dominant
contribution.

For scale, `versions/V13_nanoparticle_vdw/INTEGRATION.md` reports −2.0560e-20 J
(−4.96 k_BT at 300 K) for the same geometry from the calibrated `npvdw`
potential — 4.2× the inherited value. The extra factor beyond convergence is
a different body (sextic superellipsoid), the eps1 energy scale and the
repulsive term, i.e. the calibration to the published Fig. S28E curve.

**Why this now matters for the mechanism.** The converged distributed-dipole
calculation gives −5.65 k_BT per face bond at 300 K (the point-dipole value,
−6.40 k_BT, is only 13 % high, so the dipolar term is not an artefact). The
dipolar-to-vdW ratio is therefore 4.8 with the inherited vdW, 2.6 with the
converged sharp-cube value, and 1.14 with the `npvdw` value. Which term drives
aggregation on cooling is decided entirely by that choice:

| vdW per NC (z/2 = 3) | frozen dipolar disorder, per NC | can the disorder break the lattice? |
|---|---|---|
| −3.5 k_BT (inherited 4³) | sd ≈ 7 k_BT | yes |
| −6.6 k_BT (converged sharp cube) | sd ≈ 7 k_BT | marginal |
| −14.9 k_BT (`npvdw`) | sd ≈ 7 k_BT | no, needs 2.3σ |

So there is a quantitative tension: **a vdW strong enough to explain
vdW-driven aggregation on cooling — which is what the literature reports for
comparable systems — is too strong for frozen dipolar disorder to dissolve.**
The mechanism needs the vdW binding per NC in roughly −3 to −7 k_BT, i.e. the
inherited or converged sharp-cube magnitude, not the calibrated `npvdw` one.
Pinning the vdW magnitude is therefore now the pivotal open number, and the
three candidates span a factor of four.

Nothing here has been changed in `geometry_model.py`; the inherited 4³ sum is
still what every figure in this file uses.

## Files

| file | contents |
|---|---|
| `populations_and_energies.csv` | 101 temperatures × 2 modes: populations, energies, all timescales, lifetimes, survivals, diagnostics |
| `state_probabilities.csv` | all 192 state probabilities (equilibrium and 20 s mean) at every temperature |
| `volume_fraction_sweep.csv` | closed-form populations, φ = 1e-4…1e-1 on a 120–700 K grid |
| `contact_barrier_scan.csv` | lifetime decomposition at 0, 5, 10 k_BT extra rotation barrier |
| `window_scan.csv` | 1 µs…150 s windows at φ = 1e-4 and 1e-2 |
| `face_tip_competition.csv` | (F_tip − F_face) decomposed, free vs parallel dipole manifold × δ = 1, 2, 5, 10° |
| `restricted_channel_populations.csv` | only face, or only tip, allowed: pair fraction and total energy × mode × manifold (free / blocked / bondaxis / parallel) × δ |
| `cooling_trajectory.csv` | fixed face contact cooled from 300 K, dipole frozen at tau_N = window, both inheritance limits |
| `gradual_cooling.csv` | the same with a gradual freeze-in: lognormal sizes CV = 0, 5, 10, 15 %, three-component mixture |
| `history_cooling.csv` | frozen distribution inherited from the pair freeze temperature; bound / computed / dispersed branches |
| `populations_*`, `total_energy_*`, `timescales_*`, `persistence_*`, `configuration_density_*` | figures, png + pdf |
| `summary.txt`, `model_config.json` | tabulated summary and the full parameter/assumption record |

## Assumptions and limits

- φ and ℓ are not measured here. φ shifts the thermodynamic crossover
  logarithmically; it cannot change its slope.
- `superlattice` mode is a mean-field z_c/2 scaling of a pair model: all z_c
  neighbours are assumed to share the same relative configuration, and α_c
  keeps the dimer reaction volume. It is not a many-body calculation.
- U(free) = 0. The dipolar coupling at the mean free separation is reported as
  `free_state_dipole_diagnostic` (0.08 k_BT at 300 K rising to 0.12 k_BT at
  200 K, mean separation 74.3 nm at φ = 1e-2) so the size of that
  approximation is visible.
- No ligand/solvent PMF, no face↔tip structural conversion path, no positional
  or librational entropy beyond `v_site`, constant M_s, CV = 0, inherited
  effective cubic K.
- The propagator renormalises the probability leak from scaling-and-squaring
  (‖Q‖ × window reaches ~1e10). The leak is ≤1.6e-6 across the whole 20 s run
  and reaches ~1e-5 only in the 150 s window scan; it is reported per row as
  `propagator_normalisation_drift`, and the distribution itself is verified
  against composite Gauss–Legendre quadrature to 1e-9 in the tests.

## Verification

`code/test_configuration_averaged_assembly.py`, 23 tests, all passing:
the generalised `path_maximum` reduces to the inherited quarter-turn version
and matches a dense scan at arbitrary angle; the Néel structure reproduces
`geometry_model.neel_pair_generator` to machine precision; the rotation
attempt calibration gives a 1/τ_B first-rank decay for all three move sets;
the reaction-volume identity `k_diff/v_site = D_rel/ℓ²` holds to 1e-12; the
generator satisfies detailed balance and `k_on/k_off` reproduces the Boltzmann
weight; the propagator matches `expm` and a composite quadrature; the long
window reproduces the closed-form equilibrium and the short window returns the
colloidal initial state; direct-escape-only lifetimes equal 1/k_off; opening
a channel can only shorten a lifetime; and the inherited vdW and gap values
are asserted unchanged. All six inherited V10 test modules still pass.
