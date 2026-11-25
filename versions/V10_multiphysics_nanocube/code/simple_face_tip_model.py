"""Simplest two-cube model: face-to-face against tip-to-tip energy on cooling.

Two 16 nm Fe3O4 nanocubes, 3.0 nm surface gap, in two configurations:

  face-to-face   centres along <100>, 19.0 nm apart
  tip-to-tip     centres along <111>, 28.52 nm apart (rounded corners)

A moment that flips among the eight <111> easy axes within the observation
time has zero time-averaged dipolar energy with any partner, because the
eight easy-axis vectors sum to zero.  A blocked moment is fixed to the
crystal axes, so Brownian rotation of the cubes (microseconds) can bring two
blocked moments head to tail.  For <111> moments that is the tip-to-tip
configuration, E_dd = -2 mu0 mu^2 / (4 pi r^3).  In the face-to-face
configuration two parallel <111> moments make the magic angle (54.7 deg) with
the <100> line of centres, so E_dd = 0 exactly.

Dipole freezing is continuous.  The blocked fraction after an observation
time t is the Neel survival probability

    b(T) = exp(-t / tau_N),   tau_N = tau0 exp(dE / kBT),
    dE = kB T_B ln(t_ZFC / tau0),

and t = t_ZFC = 100 s, so b(T_B) = 1/e with no new parameter.  A pair keeps
its head-to-tail energy only if both moments are blocked, probability b^2;
if either flips, the average is zero.  Hence

    E_face(T) = E_vdW,face
    E_tip(T)  = E_vdW,tip + b(T)^2 E_dd,tip

Energies, not free energies: the Boltzmann bias of fluctuating moments
towards attractive states, the particle-size distribution, concentration and
many-particle effects are not included.  `crossover_K` reports where the
two curves cross; `summary` also lists it for other windows and for b
instead of b^2, so the sensitivity is explicit.

The van der Waals energy is the same Hamaker integral as the inherited
voxel sum, over two aligned sharp cubes, but converged: the six-dimensional
integral is reduced to three dimensions with the cube autocorrelation
prod_i (L - |s_i|) and evaluated on graded Gauss-Legendre panels.
geometry_model.py is only read, not modified.

Outputs (outputs/simple_face_tip_model/):
  face_tip_pair_energy           energy curves (JACS column)
  face_tip_pair_energy_insets    the same with the two configurations drawn in
  face_tip_configurations        the two configurations alone
  face_tip_pair_energy.csv, summary.json

Usage:  python simple_face_tip_model.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from scipy.constants import Boltzmann, mu_0
from scipy.optimize import brentq

import geometry_model as geometry

OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'simple_face_tip_model'
PARAMS = geometry.PARAMS
GAP_M = 3.0e-9
DIRECTIONS = {'face': np.array([1.0, 0.0, 0.0]),
              'tip': np.ones(3) / np.sqrt(3.0)}
WINDOW_S = PARAMS.zfc_fc_observation_time_s     # 100 s, defines T_B
BLOCKED_POWER = 2                               # both moments must be blocked
OBSERVED_WINDOW_K = (250.0, 270.0)              # zero-field SAXS, cooling

BLUE, ORANGE, MOMENT = '#3C3C8C', '#C0552F', '#1A1A1A'
JACS_COLUMN_IN = 240 / 72
MARGINS_IN = dict(left=0.42, right=0.10, bottom=0.36, top=0.05)
STYLE = {'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 8,
         'axes.linewidth': 0.8, 'axes.labelpad': 2, 'axes.grid': False,
         'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
         'xtick.major.size': 3, 'ytick.major.size': 3,
         'xtick.major.pad': 2, 'ytick.major.pad': 2,
         'legend.fontsize': 7, 'pdf.fonttype': 42}


# ---------------------------------------------------------------- vdW
def _graded_nodes(edge_m: float, nodes_per_panel: int):
    """Gauss-Legendre nodes on [-L, L], graded towards -L and towards 0.

    The integrand is largest where s_i -> -L (the facing surfaces or
    corners, since every centre vector here has nonnegative components) and
    the autocorrelation weight has a kink at s_i = 0.
    """
    near = -edge_m + edge_m * np.geomspace(1e-5, 1.0, 28)
    upper = edge_m * np.geomspace(1e-3, 1.0, 14)
    edges = np.unique(np.concatenate([[-edge_m], near, [0.0], upper]))
    x, w = np.polynomial.legendre.leggauss(nodes_per_panel)
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def cube_pair_vdw_J(centre_vector_m, edge_m: float = PARAMS.particle_size_nm * 1e-9,
                    hamaker_J: float = PARAMS.hamaker_J,
                    nodes_per_panel: int = 8) -> float:
    """-A/pi^2 * int int dV_a dV_b / r^6 for two aligned sharp cubes.

    With s = r_a - r_b the double volume integral is int g(s) / |s + d|^6 d^3s
    where g(s) = prod_i (L - |s_i|) is the overlap volume of a cube with its
    copy shifted by s.
    """
    d = np.asarray(centre_vector_m, float)
    s, w = _graded_nodes(edge_m, nodes_per_panel)
    g = (edge_m - np.abs(s)) * w
    sy, sz = s[:, None] + d[1], s[None, :] + d[2]
    gyz = g[:, None] * g[None, :]
    total = 0.0
    for sx, gx in zip(s + d[0], g):
        total += gx * np.sum(gyz / (sx * sx + sy * sy + sz * sz) ** 3)
    return float(-hamaker_J / np.pi ** 2 * total)


# ---------------------------------------------------------------- dipoles
def moment_Am2() -> float:
    return PARAMS.saturation_magnetization_Apm * (PARAMS.particle_size_nm * 1e-9) ** 3


def frozen_dipolar_J(direction, distance_m: float) -> float:
    """Two parallel moments along [111] (body frame), centres along `direction`."""
    m = np.ones(3) / np.sqrt(3.0)
    r = np.asarray(direction, float)
    prefactor = mu_0 * moment_Am2() ** 2 / (4 * np.pi * distance_m ** 3)
    return float(prefactor * (m @ m - 3 * (m @ r) ** 2))


def neel_time_s(temperature_K):
    """tau_N = tau0 exp(dE / kBT), dE = kB T_B ln(t_ZFC / tau0)."""
    t = np.asarray(temperature_K, float)
    return PARAMS.attempt_time_s * np.exp(
        PARAMS.zfc_fc_activation_barrier_J / (Boltzmann * t))


def blocked_fraction(temperature_K, window_s: float = WINDOW_S):
    """Probability that a moment has not flipped within the window."""
    return np.exp(-window_s / neel_time_s(temperature_K))


# ---------------------------------------------------------------- model
def build(temperatures_K=np.linspace(200.0, 300.0, 1001),
          window_s: float = WINDOW_S, power: int = BLOCKED_POWER) -> dict:
    geom = {}
    for name, direction in DIRECTIONS.items():
        distance = geometry.center_distance_at_gap_m(direction, GAP_M, PARAMS)
        geom[name] = dict(
            distance_nm=distance * 1e9,
            vdw_J=cube_pair_vdw_J(distance * direction),
            vdw_J_check=cube_pair_vdw_J(distance * direction, nodes_per_panel=12),
            vdw_J_inherited_voxel=geometry.pair_vdw_energy_J(distance * direction, PARAMS),
            dipolar_frozen_J=frozen_dipolar_J(direction, distance))
    face_d = geom['face']['distance_nm'] * 1e-9
    prefactor = mu_0 * moment_Am2() ** 2 / (4 * np.pi * face_d ** 3)
    # best face state if the two frozen moments need not be parallel
    geom['face']['dipolar_best_nonparallel_J'] = -4.0 / 3.0 * prefactor

    t = np.asarray(temperatures_K, float)
    kbt = Boltzmann * t
    b = blocked_fraction(t, window_s)
    face = np.full_like(t, geom['face']['vdw_J'])
    tip_dd = b ** power * geom['tip']['dipolar_frozen_J']
    tip = geom['tip']['vdw_J'] + tip_dd
    table = dict(temperature_K=t, blocked_fraction=b, face_J=face, tip_J=tip,
                 face_kBT=face / kbt, tip_kBT=tip / kbt, tip_dipolar_kBT=tip_dd / kbt,
                 tip_vdw_kBT=geom['tip']['vdw_J'] / kbt)
    return dict(geometry=geom, table=table, window_s=window_s, power=power,
                blocking_K=PARAMS.blocking_temperature_K)


def crossover_K(geom: dict, window_s: float = WINDOW_S,
                power: int = BLOCKED_POWER) -> float:
    """T where E_face = E_tip."""
    def gap(temperature):
        b = blocked_fraction(temperature, window_s) ** power
        return (geom['tip']['vdw_J'] + b * geom['tip']['dipolar_frozen_J']
                - geom['face']['vdw_J'])
    return float(brentq(gap, 150.0, 320.0))


def summary(model: dict) -> dict:
    g = model['geometry']
    rows = {}
    for temperature in (300.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 200.0):
        kbt = Boltzmann * temperature
        b = float(blocked_fraction(temperature, model['window_s']))
        tip = g['tip']['vdw_J'] + b ** model['power'] * g['tip']['dipolar_frozen_J']
        rows[f'{temperature:g}'] = dict(blocked_fraction=b,
                                        face_kBT=g['face']['vdw_J'] / kbt,
                                        tip_kBT=tip / kbt,
                                        neel_time_s=float(neel_time_s(temperature)))
    return dict(
        blocking_temperature_K=model['blocking_K'],
        window_s=model['window_s'], blocked_power=model['power'],
        moment_Am2=moment_Am2(),
        barrier_J=PARAMS.zfc_fc_activation_barrier_J,
        geometry=g,
        energies=rows,
        crossover_K=crossover_K(g, model['window_s'], model['power']),
        crossover_sensitivity_K={f't={w:g}s,b^{p}': crossover_K(g, w, p)
                                 for w in (20.0, 100.0, 200.0) for p in (1, 2)},
    )


# ---------------------------------------------------------------- drawings
CUBE_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
CUBE_NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0],
                         [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)


def _view(screen_x, screen_y_hint, tilt_deg):
    """Rows: screen x, screen y, depth (towards the viewer)."""
    ex = np.asarray(screen_x, float); ex /= np.linalg.norm(ex)
    ey = np.asarray(screen_y_hint, float); ey -= ey @ ex * ex; ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    a = np.radians(tilt_deg)           # tilt about screen x for a 3D look
    ey, ez = np.cos(a) * ey - np.sin(a) * ez, np.sin(a) * ey + np.cos(a) * ez
    return np.vstack([ex, ey, ez])


def _draw_cube(ax, centre, edge, view, colour):
    corners = np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5)
                        for k in (-.5, .5)]) * edge + centre
    screen = corners @ view.T
    light = np.array([0.35, 0.8, 0.5]); light /= np.linalg.norm(light)
    for face, normal in zip(CUBE_FACES, CUBE_NORMALS):
        n_screen = view @ normal
        if n_screen[2] <= 1e-9:
            continue                    # back face
        shade = 0.55 + 0.45 * max(n_screen @ light, 0.0)
        rgb = 1 - shade * (1 - np.array(matplotlib.colors.to_rgb(colour)))
        rgb = 1 - (1 - rgb) * 0.45      # pale fill
        ax.add_patch(Polygon(screen[list(face), :2], closed=True, facecolor=rgb,
                             edgecolor=colour, lw=0.7, joinstyle='round', zorder=2))
    return screen[:, 2].mean()


def _draw_moment(ax, centre, direction, length, view):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    tail = (centre - 0.5 * length * d) @ view.T
    head = (centre + 0.5 * length * d) @ view.T
    ax.add_patch(FancyArrowPatch(tail[:2], head[:2], arrowstyle='-|>',
                                 mutation_scale=7, lw=1.1, color=MOMENT,
                                 shrinkA=0, shrinkB=0, zorder=5))


# The four body diagonals of the cube.  A flipping moment visits both senses
# of each, i.e. all eight <111> easy-axis directions.
BODY_DIAGONALS = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]


def _draw_flipping_moments(ax, centre, half_length, view):
    """Dashed double-headed arrows through the centre along the four body
    diagonals: a moment that flips among the eight <111> directions within
    the observation time."""
    for axis in BODY_DIAGONALS:
        d = np.asarray(axis, float) / np.sqrt(3.0)
        tail = (centre - half_length * d) @ view.T
        head = (centre + half_length * d) @ view.T
        ax.add_patch(FancyArrowPatch(tail[:2], head[:2], arrowstyle='<|-|>',
                                     mutation_scale=5.0, lw=0.8, color=MOMENT,
                                     alpha=0.75, linestyle=(0, (2.2, 1.4)),
                                     shrinkA=0, shrinkB=0, zorder=5))


def draw_configuration(ax, kind: str, edge_nm: float = 16.0):
    """Two cubes in data units of nm.

    Face-to-face: the regime where it is preferred is T > T_B, so each moment
    is drawn flipping, as dashed double arrows along the four body diagonals.
    Tip-to-tip: the blocked regime, one solid head-to-tail moment per cube.
    """
    m = np.ones(3) / np.sqrt(3.0)
    if kind == 'face':
        distance = 19.0
        view = _view([1, -0.25, 0], [0, 0, 1], tilt_deg=33)  # body diagonals >= 27 deg apart on screen
        centres = [np.zeros(3), distance * DIRECTIONS['face']]
        colour = BLUE
    elif kind == 'tip':
        distance = 28.517
        view = _view(m, [0, 0, 1], tilt_deg=25)
        centres = [np.zeros(3), distance * m]
        colour = ORANGE
    else:
        raise ValueError(kind)
    order = sorted(centres, key=lambda c: (c @ view.T)[2])   # far cube first
    for c in order:
        _draw_cube(ax, c, edge_nm, view, colour)
    for c in centres:
        if kind == 'face':
            _draw_flipping_moments(ax, c, 0.62 * edge_nm, view)
        else:
            _draw_moment(ax, c, m, 0.95 * edge_nm, view)
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')


def _fit_limits(ax, pad=1.5):
    xs, ys = [], []
    for patch in ax.patches:
        if isinstance(patch, Polygon):
            xy = patch.get_xy(); xs += list(xy[:, 0]); ys += list(xy[:, 1])
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def plot_configurations(stem: Path, height_in: float = 1.15):
    width = JACS_COLUMN_IN
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height_in))
        boxes = {'face': [0.02, 0.18, 0.38, 0.80], 'tip': [0.43, 0.18, 0.55, 0.80]}
        for kind, label, colour in (('face', 'face-to-face', BLUE),
                                    ('tip', 'tip-to-tip', ORANGE)):
            ax = fig.add_axes(boxes[kind])
            draw_configuration(ax, kind)
            _fit_limits(ax)
            box = boxes[kind]
            fig.text(box[0] + box[2] / 2, 0.06, label, ha='center', va='center',
                     color=colour, fontsize=8)
        for ext in ('png', 'pdf'):
            fig.savefig(f'{stem}.{ext}', dpi=600)
        plt.close(fig)


# ---------------------------------------------------------------- energy plot
def plot(model: dict, stem: Path, height_in: float = 2.6, insets: bool = False):
    tab, tb = model['table'], model['blocking_K']
    t = tab['temperature_K']
    margin, width = MARGINS_IN, JACS_COLUMN_IN
    axes_h = height_in - margin['bottom'] - margin['top']
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height_in))
        rect = [margin['left'] / width, margin['bottom'] / height_in,
                (width - margin['left'] - margin['right']) / width, axes_h / height_in]
        ax = fig.add_axes(rect)
        ax.axvspan(*OBSERVED_WINDOW_K, color='#F4EAA2', alpha=.75, zorder=0)
        ax.axvline(tb, color='#C9A227', lw=1.0, ls=(0, (5, 2)), zorder=1)
        ax.axhline(0.0, color='0.6', lw=0.6, zorder=1)
        ax.plot(t, tab['face_kBT'], color=BLUE, lw=1.6, zorder=4, label='face-to-face')
        ax.plot(t, tab['tip_kBT'], color=ORANGE, lw=1.6, zorder=5, label='tip-to-tip')
        ax.set(xlabel='Temperature (K)', xlim=(200, 300),
               ylabel=r'Pair energy  ($k_BT$)', ylim=(-5.0, 1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_label_coords(-0.085, 0.5)
        for spine in ax.spines.values():
            spine.set_zorder(6)
        ax.legend(loc='upper left', ncol=1, frameon=True, fancybox=False,
                  facecolor='white', edgecolor='none', framealpha=1,
                  handlelength=1.8, handletextpad=0.5, borderpad=0.4,
                  borderaxespad=0)
        if insets:
            # each drawing sits just under its own curve, right of the band
            def data_to_fig(x, y):
                return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
            for kind, (x0, x1, y0, y1) in (('tip', (273, 299, -1.95, -0.35)),
                                           ('face', (273, 299, -4.85, -2.85))):
                (fx0, fy0), (fx1, fy1) = data_to_fig(x0, y0), data_to_fig(x1, y1)
                sub = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
                draw_configuration(sub, kind)
                _fit_limits(sub)
                sub.set_anchor('C')
        for ext in ('png', 'pdf'):
            fig.savefig(f'{stem}.{ext}', dpi=600)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = build()
    info = summary(model)
    plot(model, OUT / 'face_tip_pair_energy')
    plot(model, OUT / 'face_tip_pair_energy_insets', insets=True)
    plot_configurations(OUT / 'face_tip_configurations')
    tab = model['table']
    keys = ('temperature_K', 'blocked_fraction', 'face_J', 'tip_J', 'face_kBT',
            'tip_kBT', 'tip_dipolar_kBT', 'tip_vdw_kBT')
    with open(OUT / 'face_tip_pair_energy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(0, len(tab['temperature_K']), 10):
            writer.writerow([f"{np.atleast_1d(tab[k])[i if np.ndim(tab[k]) else 0]:.6g}"
                             for k in keys])
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: info[k] for k in ('energies', 'crossover_K',
                                           'crossover_sensitivity_K')}, indent=2))


if __name__ == '__main__':
    main()
