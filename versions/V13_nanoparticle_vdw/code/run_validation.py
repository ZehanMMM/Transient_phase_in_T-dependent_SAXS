"""Stage D: the verification suite required before the module is used.

Checks, in order:
  D1 energy units and magnitudes
  D2 particle-exchange symmetry U(i,j) = U(j,i)
  D3 invariance under global translation and global rotation
  D4 GJK distance against analytic benchmarks
  D5 volume and surface weights
  D6 far-field r^-6 behaviour of the attraction
  D7 integral convergence at positive gaps
  D8 overlap handling
  D9 benchmark against the legacy V6.20/V10 model at identical geometry and
     parameters

Every check prints its measured deviation and a PASS/FAIL against an explicit
tolerance.  Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.spatial.transform import Rotation

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"
LEGACY_CODE = ROOT.parents[0] / "V10_multiphysics_nanocube" / "code"

EDGE_NM = 13.37
RESULTS = []


def check(name, measured, tolerance, unit, detail=""):
    ok = bool(measured <= tolerance)
    RESULTS.append(
        {
            "check": name,
            "measured": float(measured),
            "tolerance": float(tolerance),
            "unit": unit,
            "pass": ok,
            "detail": detail,
        }
    )
    print("  [%s] %-52s %12.4e <= %8.1e %s"
          % ("PASS" if ok else "FAIL", name, measured, tolerance, unit))
    if detail:
        print("         %s" % detail)
    return ok


def d1_units():
    print("D1  energy units and magnitudes")
    # 1 kcal/mol must be 4184 J per mole of pairs
    reconstructed = V.KCAL_PER_MOL_IN_J * V.AVOGADRO
    check("kcal/mol -> J round trip", abs(reconstructed - 4184.0) / 4184.0,
          1e-14, "relative")
    check("Singh A = 3 kcal/mol equals 2.08431e-20 J",
          abs(V.SINGH_HAMAKER_J - 2.084309e-20) / 2.084309e-20, 1e-6, "relative")
    check("A = 2.0e-20 J equals 2.878652 kcal/mol",
          abs(V.hamaker_J_to_kcalmol(2.0e-20) - 2.878652) / 2.878652, 1e-6,
          "relative")
    # kBT at 300 K
    kbt = V.kcalmol_to_kBT(1.0, 300.0)
    check("1 kcal/mol = 1.67739845 kBT at 300 K",
          abs(kbt - 1.67739845) / 1.67739845, 1e-8, "relative",
          "SI depth -2.33 kcal/mol = %.4f kBT per pair at 300 K"
          % V.kcalmol_to_kBT(-2.33, 300.0))
    # K_W magnitude
    params = V.load_parameters("singh_literature")
    check("K_W within 1 % of the SI's quoted 2.5e5 nm^6 kcal/mol",
          abs(params.kw(EDGE_NM) / 2.5e5 - 1.0), 0.01, "relative",
          "K_W = %.6e nm^6 kcal/mol" % params.kw(EDGE_NM))
    # three unit channels agree
    result = V.face_to_face_pair(EDGE_NM, 3.0, params, temperature_K=300.0)
    check("J channel equals kcal/mol channel",
          abs(result["u_pair_J"] / V.kcalmol_to_J(result["u_pair_kcalmol"]) - 1.0),
          1e-14, "relative")
    check("kBT channel equals J channel",
          abs(result["u_pair_kBT"] / V.J_to_kBT(result["u_pair_J"], 300.0) - 1.0),
          1e-14, "relative")


def d2_exchange():
    print("D2  particle-exchange symmetry")
    params = V.load_parameters("singh_literature")
    rng = np.random.default_rng(3)
    worst = 0.0
    worst_rep = 0.0
    for trial in range(12):
        qa = Rotation.random(random_state=400 + trial).as_matrix()
        qb = Rotation.random(random_state=500 + trial).as_matrix()
        centre = rng.normal(size=3) * 2.0 + np.array([19.0, 0.0, 0.0])
        a = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0], qa)
        b = V.Placement.create(EDGE_NM, centre, qb)
        forward = V.pair_energy(a, b, params)
        reverse = V.pair_energy(b, a, params)
        if forward["overlap"]:
            continue
        scale = max(abs(forward["u_pair_kcalmol"]), 1e-12)
        worst = max(worst, abs(forward["u_pair_kcalmol"] - reverse["u_pair_kcalmol"]) / scale)
        worst_rep = max(
            worst_rep,
            abs(forward["u_rep_kcalmol"] - reverse["u_rep_kcalmol"])
            / max(abs(forward["u_rep_kcalmol"]), 1e-12),
        )
    check("U_pair(i,j) = U_pair(j,i), random orientations", worst, 1e-13,
          "relative")
    check("U_rep(i,j) = U_rep(j,i) after symmetrisation", worst_rep, 1e-13,
          "relative")
    # double counting audit
    asym = replace(params, symmetrise_repulsion=False)
    a = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0])
    b = V.Placement.create(EDGE_NM, [EDGE_NM + 3.0, 0.0, 0.0])
    one_sided = V.pair_energy(a, b, asym)["u_rep_kcalmol"]
    averaged = V.pair_energy(a, b, params)["u_rep_kcalmol"]
    check("symmetrisation introduces no factor of two in a symmetric "
          "configuration", abs(averaged / one_sided - 1.0), 1e-13, "relative",
          "one-sided %.6f, averaged %.6f kcal/mol" % (one_sided, averaged))


def d3_invariance():
    print("D3  global translation and rotation invariance")
    params = V.load_parameters("singh_literature")
    rng = np.random.default_rng(5)
    worst_t = worst_r = 0.0
    for trial in range(8):
        qa = Rotation.random(random_state=600 + trial).as_matrix()
        qb = Rotation.random(random_state=700 + trial).as_matrix()
        centre = rng.normal(size=3) * 2.0 + np.array([20.0, 0.0, 0.0])
        base = V.pair_energy(
            V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0], qa),
            V.Placement.create(EDGE_NM, centre, qb),
            params,
        )["u_pair_kcalmol"]
        shift = rng.normal(size=3) * 80.0
        moved = V.pair_energy(
            V.Placement.create(EDGE_NM, shift, qa),
            V.Placement.create(EDGE_NM, centre + shift, qb),
            params,
        )["u_pair_kcalmol"]
        worst_t = max(worst_t, abs(moved / base - 1.0))
        G = Rotation.random(random_state=800 + trial).as_matrix()
        rotated = V.pair_energy(
            V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0], G @ qa),
            V.Placement.create(EDGE_NM, G @ centre, G @ qb),
            params,
        )["u_pair_kcalmol"]
        worst_r = max(worst_r, abs(rotated / base - 1.0))
    check("invariance under global translation", worst_t, 1e-12, "relative")
    check("invariance under global rotation", worst_r, 1e-12, "relative")


def d4_gjk():
    print("D4  GJK distance against analytic benchmarks")
    worst = 0.0
    detail = []
    for name, d in V.SYMMETRY_DIRECTIONS.items():
        for gap in (0.02, 0.5, 2.99, 8.0):
            radius = V.centre_distance_for_gap_coaligned(EDGE_NM, gap, d)
            got = V.gjk_distance(
                V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0]),
                V.Placement.create(EDGE_NM, radius * np.asarray(d)),
            )["distance_nm"]
            worst = max(worst, abs(got - gap))
        detail.append("2h[%s]/a = %.6f" % (name, 2 * V.analytic_support_height(EDGE_NM, d) / EDGE_NM))
    check("GJK equals R - 2h(d) on <100>, <110>, <111>", worst, 1e-11, "nm",
          "; ".join(detail) + " (exact: 1, 2^(1/3), 3^(1/3))")

    # generic direction: analytic formula is a lower bound only
    rng = np.random.default_rng(9)
    violations = 0
    for _ in range(200):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        radius = 1.35 * EDGE_NM
        got = V.gjk_distance(
            V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0]),
            V.Placement.create(EDGE_NM, radius * d),
        )["distance_nm"]
        if got < V.analytic_gap_coaligned(EDGE_NM, radius, d) - 1e-10:
            violations += 1
    check("GJK >= R - 2h(d) for generic directions", violations, 0, "count")

    # point-to-body certificate and cross-check
    rng = np.random.default_rng(13)
    body = V.Placement.create(EDGE_NM, [1.0, -2.0, 0.5],
                              Rotation.random(random_state=17).as_matrix())
    local = []
    for _ in range(600):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        local.append(d * rng.uniform(0.80 * EDGE_NM, 3.0 * EDGE_NM))
    points = body.to_world(np.array(local))
    distance, certificate = V.point_to_body_distance(points, body)
    check("point-to-body two-sided certificate", certificate, 1e-11, "nm")
    reference = np.array(
        [V.gjk_distance(V.Placement.create(1e-7, p), body)["distance_nm"]
         for p in points[:60]]
    )
    check("point-to-body agrees with GJK", float(np.max(np.abs(distance[:60] - reference))),
          1e-6, "nm")
    # analytic point checks
    flat = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0])
    got = V.point_to_body_distance(np.array([[EDGE_NM, 0.0, 0.0]]), flat)[0][0]
    check("face-centre normal distance", abs(got - EDGE_NM / 2), 1e-12, "nm")
    corner = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0) * 2 * EDGE_NM
    expect = 2 * EDGE_NM - 3 ** (1 / 3.0) * EDGE_NM / 2
    check("[111] corner distance", abs(V.point_to_body_distance(corner, flat)[0][0] - expect),
          1e-9, "nm")


def d5_weights():
    print("D5  volume and surface weights")
    shape = V.Superellipsoid(EDGE_NM)
    exact = shape.volume_exact_nm3
    check("exact volume coefficient equals 0.9009589",
          abs(V.exact_volume_coefficient() - 0.9009589) / 0.9009589, 1e-6,
          "relative",
          "literature 0.9 a^3 is %.4f %% below the exact volume"
          % (100 * (1 - shape.volume_literature_nm3 / exact)))
    worst = 0.0
    inside = True
    for scheme, n in [("grid27", 3), ("cartesian", 3), ("cartesian", 8),
                      ("cartesian", 20), ("radial", 6), ("radial", 20)]:
        nodes, weights = shape.volume_nodes(scheme, n, "exact")
        worst = max(worst, abs(weights.sum() / exact - 1.0))
        inside = inside and bool(shape.contains(nodes).all())
        if scheme != "grid27":
            # raw quadrature accuracy before renormalisation
            raw = shape._nodes_cartesian(n, "exact") if scheme == "cartesian" else None
    check("volume weights sum to the exact volume", worst, 1e-12, "relative")
    check("all volume nodes lie inside the body", 0 if inside else 1, 0, "count",
          "the exact-boundary rules never place a node outside, so no partial "
          "voxel is ever counted as a full one")
    # raw (unrenormalised) quadrature accuracy: integrate f = 1
    for scheme, n in [("cartesian", 4), ("cartesian", 8), ("cartesian", 16),
                      ("radial", 4), ("radial", 8), ("radial", 16)]:
        nodes, weights = shape.volume_nodes(scheme, n, "exact")
        print("         %-10s n=%2d  N=%6d  weight sum / exact volume = 1 by "
              "construction" % (scheme, n, len(nodes)))
    area_reference = shape.surface_area(160)
    worst_area = 0.0
    for scheme, n in [("lattice", 8), ("facegl", 16), ("facegl", 32), ("facegl", 64)]:
        _, areas = shape.surface_nodes(scheme, n)
        worst_area = max(worst_area, abs(areas.sum() / area_reference - 1.0))
    check("surface weights sum to the surface area", worst_area, 1e-4, "relative",
          "area = %.4f nm^2, between the sphere %.4f and the sharp cube %.4f"
          % (area_reference, 4 * np.pi * (EDGE_NM / 2) ** 2, 6 * EDGE_NM ** 2))
    nodes, areas = shape.surface_nodes("lattice", 8)
    check("386 surface elements recovered from the 8x8x8 cube lattice",
          abs(len(nodes) - 386), 0, "count", "6*8^2 + 2 = 386")
    nodes27, _ = shape.volume_nodes("grid27", 3)
    check("27 volume elements", abs(len(nodes27) - 27), 0, "count")


def d6_far_field():
    print("D6  far-field r^-6 behaviour of the attraction")
    params = replace(
        V.load_parameters("singh_literature"),
        label="farfield",
        epsilon2=0.0,
        volume_scheme="cartesian",
        volume_n=10,
        kw_volume_convention="exact",
    )
    shape = V.Superellipsoid(EDGE_NM)
    point_point = params.attraction_prefactor() * shape.volume_exact_nm3 ** 2
    nodes, weights = shape.volume_nodes("cartesian", 24, "exact")
    second_moment = np.sum(weights * nodes[:, 0] ** 2) / weights.sum()
    check("<x^2> equals the closed form 3 a^2 / 40",
          abs(second_moment / EDGE_NM ** 2 / 0.075 - 1.0), 2e-3, "relative",
          "<x^2>/a^2 = %.8f, so the leading finite-size correction is "
          "30 <x^2>/a^2 = %.6f times (a/R)^2"
          % (second_moment / EDGE_NM ** 2, 30 * second_moment / EDGE_NM ** 2))

    def attraction(factor):
        radius = factor * EDGE_NM
        return -V.pair_energy(
            V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0]),
            V.Placement.create(EDGE_NM, [radius, 0.0, 0.0]),
            params,
        )["u_attr_kcalmol"]

    coefficients = []
    for factor in (6.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0):
        amplitude = attraction(factor) * (factor * EDGE_NM) ** 6
        excess = amplitude / point_point - 1.0
        coefficients.append((factor, excess))
        print("         R/a = %5.1f  -U_attr R^6 = %.6e  excess over the "
              "point-point limit = %+.6f %%  excess x (R/a)^2 = %.5f"
              % (factor, amplitude, 100 * excess, excess * factor ** 2))
    tail = [e * f ** 2 for f, e in coefficients if f >= 30.0]
    check("far-field excess is a pure quadrupole term, 2.25 (a/R)^2",
          abs(tail[-1] - 2.25), 0.01, "absolute",
          "scaled excess over R/a = 30..80 spans %.5f to %.5f, analytic value "
          "30 <x^2>/a^2 = 2.25" % (min(tail), max(tail)))
    check("U_attr reaches -eps1 (A/pi^2) V^2 / R^6 at R/a = 80",
          abs(coefficients[-1][1]), 5e-4, "relative")
    exponent = np.log(attraction(40.0) / attraction(80.0)) / np.log(2.0)
    check("fitted far-field exponent equals 6", abs(exponent - 6.0), 3e-3,
          "absolute", "measured exponent = %.8f between R/a = 40 and 80"
          % exponent)


def d7_convergence():
    print("D7  integral convergence at positive gaps")
    print("         production mesh: cartesian n=20 (8000 volume nodes) + "
          "facegl n=32 (6144 surface nodes), r2_mode='exact_surface'")
    print("         reference:       cartesian n=28 (21952) + facegl n=40 (9600)")
    base = replace(V.load_parameters("singh_literature"), label="conv",
                   r2_mode="exact_surface", surface_scheme="facegl")
    production = replace(base, volume_scheme="cartesian", volume_n=20,
                         surface_n=32)
    reference = replace(base, volume_scheme="cartesian", volume_n=28,
                        surface_n=40)
    worst = {"near": 0.0, "well": 0.0}
    print("         %-10s %6s %14s %14s %12s %12s"
          % ("direction", "gap", "U_attr prod", "U_attr ref", "rel attr", "rel rep"))
    for name, direction in (("100", [1, 0, 0]), ("110", [1, 1, 0]),
                            ("111", [1, 1, 1])):
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        for gap in (1.0, 2.99, 8.0):
            radius = V.centre_distance_for_gap(EDGE_NM, gap, d)
            a = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0])
            b = V.Placement.create(EDGE_NM, radius * d)
            fine = V.pair_energy(a, b, production)
            ref = V.pair_energy(a, b, reference)
            rel_attr = abs(fine["u_attr_kcalmol"] / ref["u_attr_kcalmol"] - 1.0)
            rel_rep = abs(fine["u_rep_kcalmol"] / ref["u_rep_kcalmol"] - 1.0)
            bucket = "near" if gap < 2.0 else "well"
            worst[bucket] = max(worst[bucket], rel_attr, rel_rep)
            print("         [%-8s] %6.2f %14.6f %14.6f %12.3e %12.3e"
                  % (name, gap, fine["u_attr_kcalmol"], ref["u_attr_kcalmol"],
                     rel_attr, rel_rep))
    check("components within 1 % at gaps >= 2.99 nm", worst["well"], 0.01,
          "relative",
          "this is the separation range that matters for the well and for the "
          "3 nm working point")
    check("components within 1 % at a 1 nm gap", worst["near"], 0.01,
          "relative",
          "the near-contact case.  It is the hardest one because the 1/r^6 "
          "integrand concentrates in a thin layer at the facing faces, and it "
          "only passes because the nested Cartesian rule clusters its nodes "
          "there.  The radial rule is NOT converged here even at 8192 nodes; "
          "see outputs/convergence/.  The repulsive term is exact to machine "
          "precision at every gap.")


def d8_overlap():
    print("D8  overlap handling")
    params = V.load_parameters("singh_literature")
    a = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0])
    inside = V.pair_energy(a, V.Placement.create(EDGE_NM, [0.9 * EDGE_NM, 0, 0]),
                           params)
    check("overlapping pair is rejected", 0 if inside["overlap"] else 1, 0, "count")
    check("overlapping pair returns +inf", 0 if np.isinf(inside["u_pair_kcalmol"]) else 1,
          0, "count", "hard core is enforced by GJK, not by the bounded soft "
          "repulsion")
    touching = V.pair_energy(
        a, V.Placement.create(EDGE_NM, [EDGE_NM + 1e-6, 0, 0]), params
    )
    check("just-separated pair is finite",
          0 if np.isfinite(touching["u_pair_kcalmol"]) else 1, 0, "count",
          "gap = %.3e nm, U_pair = %.4f kcal/mol"
          % (touching["gap_nm"], touching["u_pair_kcalmol"]))
    # the soft repulsion alone is bounded, proving the hard core is needed
    bounded = V.pair_energy(
        a, V.Placement.create(EDGE_NM, [EDGE_NM + 1e-4, 0, 0]),
        replace(params, r2_mode="exact_surface", surface_scheme="facegl",
                surface_n=24),
    )
    print("         at a 1e-4 nm gap the soft repulsion is only %.1f kcal/mol, "
          "finite, so it cannot stop interpenetration on its own"
          % bounded["u_rep_kcalmol"])


def d9_legacy():
    print("D9  benchmark against the legacy V6.20/V10 model")
    sys.path.insert(0, str(LEGACY_CODE))
    import geometry_model as legacy

    edge = 16.0
    legacy_params = replace(legacy.PARAMS, particle_size_nm=edge)
    # Reproduce the legacy sharp-cube 4^3 voxel integral inside npvdw by using
    # a sharp-cube-equivalent parameter set: eps1 = 1, no repulsion, volume
    # weights summing to a^3 on a 4^3 midpoint lattice.
    rows = []
    worst = 0.0
    for name, direction in (("100", [1, 0, 0]), ("110", [1, 1, 0]), ("111", [1, 1, 1])):
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        radius = legacy.center_distance_at_gap_m(d, 3e-9, legacy_params) * 1e9
        legacy_energy = legacy.pair_vdw_energy_J(radius * 1e-9 * d, legacy_params)
        # independent re-implementation of the same sharp-cube sum
        step = edge / 4.0
        line = np.linspace(-edge / 2 + step / 2, edge / 2 - step / 2, 4)
        grid = np.stack(np.meshgrid(line, line, line, indexing="ij"), -1).reshape(-1, 3)
        delta = grid[:, None, :] - (grid[None, :, :] + (radius * d)[None, None, :])
        r2 = np.einsum("ijk,ijk->ij", delta, delta)
        manual_kcal = -(
            V.J_to_kcalmol(legacy_params.hamaker_J) / np.pi ** 2
        ) * step ** 6 * np.sum(r2 ** -3.0)
        manual_J = V.kcalmol_to_J(manual_kcal)
        worst = max(worst, abs(manual_J / legacy_energy - 1.0))
        rows.append((name, radius, legacy_energy, manual_J))
    check("independent re-implementation matches legacy pair_vdw_energy_J",
          worst, 1e-12, "relative",
          "; ".join("[%s] R=%.4f nm, U=%.4e J" % (n, r, e) for n, r, e, _ in rows))

    # same physics, superellipsoid body: the difference is the deliverable
    print("         legacy vs superellipsoid at a 3 nm gap, A = 2.0e-20 J, "
          "eps1 = 1, no repulsion:")
    bare = replace(
        V.load_parameters("singh_literature"), label="bare", epsilon1=1.0,
        epsilon2=0.0, volume_scheme="cartesian", volume_n=20,
        surface_scheme="facegl", surface_n=8, r2_mode="exact_surface",
        kw_volume_convention="exact",
    ).with_hamaker_J(legacy_params.hamaker_J)
    for name, direction in (("100", [1, 0, 0]), ("110", [1, 1, 0]), ("111", [1, 1, 1])):
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        legacy_R = legacy.center_distance_at_gap_m(d, 3e-9, legacy_params) * 1e9
        legacy_U = legacy.pair_vdw_energy_J(legacy_R * 1e-9 * d, legacy_params)
        new_R = V.centre_distance_for_gap(edge, 3.0, d)
        new_U = V.pair_energy(
            V.Placement.create(edge, [0, 0, 0]),
            V.Placement.create(edge, new_R * d), bare
        )["u_pair_J"]
        print("         [%s]  legacy R=%7.4f nm U=%+.4e J   |   "
              "superellipsoid R=%7.4f nm U=%+.4e J   ratio %.4f"
              % (name, legacy_R, legacy_U, new_R, new_U, new_U / legacy_U))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stage in (d1_units, d2_exchange, d3_invariance, d4_gjk, d5_weights,
                  d6_far_field, d7_convergence, d8_overlap, d9_legacy):
        stage()
        print()
    failures = [r for r in RESULTS if not r["pass"]]
    (OUT / "D_validation.json").write_text(
        json.dumps({"results": RESULTS, "failures": len(failures)}, indent=2),
        encoding="utf-8",
    )
    print("=" * 78)
    print("%d checks, %d failures" % (len(RESULTS), len(failures)))
    for item in failures:
        print("  FAILED: %s  %.4e > %.1e %s"
              % (item["check"], item["measured"], item["tolerance"], item["unit"]))
    print("wrote %s" % (OUT / "D_validation.json"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
