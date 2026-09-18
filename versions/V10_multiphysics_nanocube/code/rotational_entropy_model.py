"""Continuous-spin/cage-orientation equilibrium sensitivity, not an SL simulation.

Face: each body axis tilts in a small cone and twists in +/- delta.
Tip: each body diagonal tilts in the same cone and twists freely.
SO(3) measure is normalized. Cubic symmetry gives 24 face registries and
8 body-diagonal tip cones. Magnetic and orientational averages are JOINT.
Two particles are independently caged relative to a fixed structural frame.
There is no 20 s trajectory, ligand PMF, positional integration or SC lattice.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, mu_0
from scipy.special import logsumexp
from scipy.stats import qmc

import geometry_model as geometry

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "rotational_entropy"


def orientation_fraction(kind, tilt_deg, twist_halfwidth_deg):
    """Fraction of normalized SO(3), union of non-overlapping cube registries."""
    if not 0 < tilt_deg <= 5 or not 0 < twist_halfwidth_deg < 45:
        raise ValueError("Use 0 < tilt <= 5 degrees and 0 < twist halfwidth < 45")
    cap = (1 - np.cos(np.deg2rad(tilt_deg))) / 2
    if kind == "face":
        return 24 * cap * np.deg2rad(twist_halfwidth_deg) / np.pi
    if kind == "tip":
        return 8 * cap
    raise ValueError(kind)


def rotate(vectors, axes, angles):
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    return (vectors*c + np.cross(axes, vectors)*s
            + axes*np.sum(axes*vectors, axis=1)[:, None]*(1-c))


def body_rotate(moments, uniforms, direction, kind, tilt_deg, twist_deg):
    """Sample exact conditional Haar measure in axis cone x twist interval."""
    ref = np.array([0., 1., 0.])
    e1 = np.cross(direction, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    azimuth = 2*np.pi*uniforms[:, 0]
    axes = np.cos(azimuth)[:, None]*e1 + np.sin(azimuth)[:, None]*e2
    tilt = np.arccos(1-uniforms[:, 1]*(1-np.cos(np.deg2rad(tilt_deg))))
    twist = (2*np.pi*uniforms[:, 2] if kind == "tip" else
             (2*uniforms[:, 2]-1)*np.deg2rad(twist_deg))
    spun = rotate(moments, np.broadcast_to(direction, moments.shape), twist)
    return rotate(spun, axes, tilt)


def basin_grid(order):
    """Continuous-angle quadrature of one octant, mirrored to all 8 basins."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    z = (nodes+1)/2
    phi = (nodes+1)*np.pi/4
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    rr = np.sqrt(1-zz**2)
    directions = np.stack([rr*np.cos(pp), rr*np.sin(pp), zz], axis=-1).reshape(-1, 3)
    measure = np.outer(weights/2, weights/2).ravel()
    params = geometry.PARAMS
    ani = (geometry.single_particle_anisotropy_J(directions, params)
           + params.magnetocrystalline_anisotropy_Jpm3*params.particle_volume_m3/3)
    return directions, measure, ani


def draw_isolated(uniform_index, uniform_sign, grid, temperature):
    directions, measure, ani = grid
    logs = np.log(measure)-ani/(Boltzmann*temperature)
    probability = np.exp(logs-logsumexp(logs))
    cdf = np.cumsum(probability)
    cdf[-1] = 1
    index = np.searchsorted(cdf, uniform_index, side="right")
    signs = geometry.cubic_easy_axis_states()[np.minimum((uniform_sign*8).astype(int), 7)]*np.sqrt(3)
    return directions[index]*signs, ani[index], float(probability@ani)


def central_vdw(direction, grid_n=48):
    """Converged-displacement sharp-cube sum at NOMINAL posture only.

    This term is a diagnostic, not used in the main magnetic+rotation curves.
    It does not include twist-dependent vdW, ligand PMF or rounded core volume.
    """
    p = geometry.PARAMS
    r = geometry.center_distance_at_gap_m(direction, 3e-9, p)
    indices = np.arange(1-grid_n, grid_n)
    disp = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"), axis=-1).reshape(-1, 3)
    multiplicity = np.prod(grid_n-np.abs(disp), axis=1).astype(float)/grid_n**6
    dr = r*direction-disp*(p.particle_size_nm*1e-9/grid_n)
    return -(p.hamaker_J/np.pi**2)*p.particle_volume_m3**2*np.sum(
        multiplicity/np.sum(dr*dr, axis=1)**3)


def evaluate_samples(kind, temperature, tilt, delta, uniform, grid, coupling=1.0):
    direction = np.array([1., 0, 0]) if kind == "face" else np.ones(3)/np.sqrt(3)
    r = geometry.center_distance_at_gap_m(direction, 3e-9, geometry.PARAMS)
    m1, a1, isolated = draw_isolated(uniform[:, 0], uniform[:, 1], grid, temperature)
    m2, a2, _ = draw_isolated(uniform[:, 2], uniform[:, 3], grid, temperature)
    s1 = body_rotate(m1, uniform[:, 4:7], direction, kind, tilt, delta)
    s2 = body_rotate(m2, uniform[:, 7:10], direction, kind, tilt, delta)
    mu = geometry.PARAMS.saturation_magnetization_Apm*geometry.PARAMS.particle_volume_m3
    pref = coupling*mu_0*mu**2/(4*np.pi*r**3)
    edd = pref*(np.sum(s1*s2, axis=1)-3*(s1@direction)*(s2@direction))
    kbt = Boltzmann*temperature
    lw = -edd/kbt
    lz = logsumexp(lw)-np.log(len(lw))
    p = np.exp(lw-logsumexp(lw))
    return np.array([lz, p@edd/kbt, (p@(a1+a2)-2*isolated)/kbt,
                     isolated/kbt, 1/(np.sum(p*p)*len(p))])


def compute(temperatures, deltas, tilt=2., power=16, repeats=4, order=48):
    grid = basin_grid(order)
    streams = [qmc.Sobol(10, scramble=True, seed=7201+i).random_base2(power)
               for i in range(repeats)]
    rows = []
    vdw = {g: central_vdw(np.array([1., 0, 0]) if g == "face" else np.ones(3)/np.sqrt(3))
           for g in ("face", "tip")}
    for t in temperatures:
        tip_cache = None
        for delta in deltas:
            for kind in ("face", "tip"):
                if kind == "tip" and tip_cache is not None:
                    samples = tip_cache
                else:
                    samples = np.array([evaluate_samples(kind, t, tilt, delta, u, grid)
                                        for u in streams])
                    if kind == "tip":
                        tip_cache = samples
                # Pool equal-size importance samples across independent scrambles.
                logz = logsumexp(samples[:, 0])-np.log(repeats)
                pool_weights = np.exp(samples[:, 0]-logsumexp(samples[:, 0]))
                udd, uani = pool_weights@samples[:, 1], pool_weights@samples[:, 2]
                fraction = orientation_fraction(kind, tilt, delta)
                geom_cost = -2*np.log(fraction)
                f_cond = -logz
                f = f_cond+geom_cost
                row = dict(geometry=kind, temperature_K=float(t), tilt_halfangle_deg=tilt,
                           face_twist_halfwidth_deg=float(delta), orientation_fraction_per_particle=fraction,
                           conditional_magnetic_F_kBT=f_cond, geometric_rotation_cost_kBT=geom_cost,
                           magnetic_plus_rotation_F_kBT=f, Udd_kBT=udd, delta_Uani_kBT=uani,
                           delta_Umag_kBT=udd+uani,
                           total_magnetic_rotational_entropy_change_over_kB=udd+uani-f,
                           F_scramble_standard_error_kBT=float(np.std(-samples[:, 0], ddof=1)/np.sqrt(repeats)),
                           nominal_posture_vdW_kBT=vdw[kind]/(Boltzmann*t),
                           partial_F_with_nominal_vdW_kBT=f+vdw[kind]/(Boltzmann*t),
                           minimum_importance_ESS_fraction=float(samples[:, 4].min()),
                           assembly_delta_G_J=np.nan)
                for key, value in list(row.items()):
                    if key.endswith("_kBT"):
                        row[key[:-4]+"_J"] = value*Boltzmann*t
                rows.append(row)
        print(f"T={t:g} K complete", flush=True)
    return rows


def export(rows, out):
    with (out/"continuous_spin_rotation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = []
    deltas = sorted(set(r["face_twist_halfwidth_deg"] for r in rows))
    temperatures = sorted(set(r["temperature_K"] for r in rows))
    for delta in deltas:
        for t in temperatures:
            face = next(r for r in rows if r["geometry"] == "face" and r["temperature_K"] == t and r["face_twist_halfwidth_deg"] == delta)
            tip = next(r for r in rows if r["geometry"] == "tip" and r["temperature_K"] == t and r["face_twist_halfwidth_deg"] == delta)
            differences.append(dict(temperature_K=t, face_twist_halfwidth_deg=delta,
                tip_minus_face_magnetic_rotation_kBT=tip["magnetic_plus_rotation_F_kBT"]-face["magnetic_plus_rotation_F_kBT"],
                tip_minus_face_nominal_vdW_kBT=tip["partial_F_with_nominal_vdW_kBT"]-face["partial_F_with_nominal_vdW_kBT"],
                pure_geometric_tip_advantage_kBT=2*np.log(np.pi/(3*np.deg2rad(delta))),
                combined_sampling_SE_kBT=np.hypot(tip["F_scramble_standard_error_kBT"], face["F_scramble_standard_error_kBT"])))
    with (out/"tip_minus_face.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    plt.rcParams.update({"font.family":"Arial", "font.size":13, "axes.linewidth":1.6,
                         "axes.grid":False, "pdf.fonttype":42})
    for add_vdw in (False, True):
        fig, ax = plt.subplots(figsize=(7,8))
        ax.set_box_aspect(1)
        key = "tip_minus_face_nominal_vdW_kBT" if add_vdw else "tip_minus_face_magnetic_rotation_kBT"
        for delta, color in zip(deltas, ["#C54A36", "#D49238", "#307C94", "#555555"]):
            data = [r for r in differences if r["face_twist_halfwidth_deg"] == delta]
            t = np.array([r["temperature_K"] for r in data])
            y = np.array([r[key] for r in data])
            err = np.array([r["combined_sampling_SE_kBT"] for r in data])
            ax.plot(t,y,lw=2.2,color=color,label=f"Face twist: +/- {delta:g} deg")
            ax.fill_between(t,y-2*err,y+2*err,color=color,alpha=.13)
        ax.axhline(0,color="black",lw=1)
        ax.set(xlabel="Temperature (K)",ylabel=r"$(F_{tip}-F_{face}) / k_BT$",xlim=(200,300),
               title="Continuous spins + body rotation" + ("\n+ nominal-posture vdW diagnostic" if add_vdw else "\nMagnetic and rotational terms only"))
        ax.text(.03,.97,"Positive: face favored",ha="left",va="top",transform=ax.transAxes)
        ax.legend(loc="upper left",bbox_to_anchor=(0,-.16),frameon=False)
        fig.subplots_adjust(bottom=.28,left=.17,right=.96,top=.89)
        name = "rotation_with_nominal_vdw" if add_vdw else "rotation_magnetic_competition"
        for ext in ("png","pdf"):
            fig.savefig(out/f"{name}.{ext}",dpi=300,bbox_inches="tight")
        plt.close(fig)
    return differences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-power",type=int,default=16)
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--step-K",type=float,default=5)
    args=parser.parse_args()
    if not 8 <= args.sample_power <= 20 or args.repeats < 2 or not 0 < args.step_K <= 100:
        parser.error("Invalid sampling/grid settings")
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    ts=np.unique(np.r_[np.arange(200,300,args.step_K),250,300])
    rows=compute(ts,[1.,2.,5.,10.],power=args.sample_power,repeats=args.repeats)
    differences=export(rows,out)
    config=dict(tilt_halfangle_deg=2,face_twist_halfwidths_deg=[1,2,5,10],
                tip_twist="free 0..2pi",samples_per_scramble=2**args.sample_power,
                scrambles=args.repeats,angular_quadrature_order=48,
                reference="two uncoupled particles with free SO(3) body orientations",
                cage_count=2,symmetry_face=24,symmetry_tip=8,
                approximation="independent local cages, fixed centres, point dipoles, inherited constant effective cubic K",
                not_included=["SC lattice", "many-body magnetic correlations", "20 s dynamics",
                              "positional entropy", "ligand/solvent PMF", "orientation-dependent vdW"],
                vdW_diagnostic="48-per-axis sharp cube at nominal posture, not in main curves",
                parameter_status="tilt/twist widths are sensitivity assumptions, not fitted or measured")
    (out/"model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    for r in differences:
        if r["temperature_K"] in (200,250,300):
            print(r)


if __name__ == "__main__":
    main()
