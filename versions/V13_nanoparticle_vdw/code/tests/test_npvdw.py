"""Unit tests for the nanoparticle vdW module.  Run: pytest -q tests"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import npvdw as V

EDGE = 13.37


# ------------------------------------------------------------------- units --
def test_kcal_per_mole_definition():
    assert V.KCAL_PER_MOL_IN_J * V.AVOGADRO == pytest.approx(4184.0, rel=1e-15)


def test_singh_hamaker_in_joule():
    assert V.SINGH_HAMAKER_J == pytest.approx(2.084309e-20, rel=1e-6)


def test_hamaker_round_trip():
    assert V.hamaker_J_to_kcalmol(2.0e-20) == pytest.approx(2.878652, rel=1e-6)
    assert V.kcalmol_to_J(V.hamaker_J_to_kcalmol(2.0e-20)) == pytest.approx(2.0e-20)


# ---------------------------------------------------------------- geometry --
def test_exact_volume_coefficient():
    assert V.exact_volume_coefficient() == pytest.approx(0.9009589, rel=1e-6)


def test_literature_volume_is_within_0p2_percent_of_exact():
    shape = V.Superellipsoid(EDGE)
    ratio = shape.volume_literature_nm3 / shape.volume_exact_nm3
    assert abs(ratio - 1.0) < 2e-3


@pytest.mark.parametrize("scheme,n", [("grid27", 3), ("cartesian", 4),
                                      ("cartesian", 12), ("radial", 8)])
def test_volume_weights_sum_and_nodes_inside(scheme, n):
    shape = V.Superellipsoid(EDGE)
    nodes, weights = shape.volume_nodes(scheme, n, "exact")
    assert weights.sum() == pytest.approx(shape.volume_exact_nm3, rel=1e-12)
    assert shape.contains(nodes).all()


def test_surface_element_count_is_386_for_n8():
    nodes, areas = V.Superellipsoid(EDGE).surface_nodes("lattice", 8)
    assert len(nodes) == 386 == 6 * 8 ** 2 + 2
    assert areas.min() > 0.0


@pytest.mark.parametrize("scheme,n", [("lattice", 8), ("facegl", 24), ("facegl", 48)])
def test_surface_areas_agree_between_schemes(scheme, n):
    shape = V.Superellipsoid(EDGE)
    _, areas = shape.surface_nodes(scheme, n)
    assert areas.sum() == pytest.approx(shape.surface_area(160), rel=2e-4)


def test_surface_nodes_lie_on_the_surface():
    shape = V.Superellipsoid(EDGE)
    nodes, _ = shape.surface_nodes("lattice", 8)
    radius = np.sum(np.abs(nodes) ** 6, axis=1) ** (1 / 6.0)
    assert np.allclose(radius, shape.half_nm, atol=1e-12)


def test_area_modes_bracket_the_figure_ordering():
    """Fig. S28E colours run blue (small) at face centres to red at corners."""
    shape = V.Superellipsoid(EDGE)
    nodes, cube = shape.surface_nodes("lattice", 8, area_mode="cube_param")
    _, solid = shape.surface_nodes("lattice", 8, area_mode="equal_solid_angle")
    radius = np.linalg.norm(nodes, axis=1)
    face, corner = int(np.argmin(radius)), int(np.argmax(radius))
    assert cube[corner] < cube[face]        # does not match the figure
    assert solid[corner] > solid[face]      # matches the figure
    assert solid.sum() == pytest.approx(cube.sum(), rel=1e-3)


# --------------------------------------------------------------------- GJK --
@pytest.mark.parametrize("name", ["100", "110", "111"])
@pytest.mark.parametrize("gap", [0.05, 2.99, 8.0])
def test_gjk_matches_support_formula_on_symmetry_axes(name, gap):
    d = V.SYMMETRY_DIRECTIONS[name]
    radius = V.centre_distance_for_gap_coaligned(EDGE, gap, d)
    got = V.gjk_distance(
        V.Placement.create(EDGE, [0, 0, 0]), V.Placement.create(EDGE, radius * d)
    )["distance_nm"]
    assert got == pytest.approx(gap, abs=1e-11)


def test_support_heights_are_the_closed_form():
    shape = V.Superellipsoid(EDGE)
    assert 2 * shape.support_height([1, 0, 0]) / EDGE == pytest.approx(1.0)
    assert 2 * shape.support_height(
        np.array([1, 1, 0]) / np.sqrt(2)
    ) / EDGE == pytest.approx(2 ** (1 / 3.0))
    assert 2 * shape.support_height(
        np.array([1, 1, 1]) / np.sqrt(3)
    ) / EDGE == pytest.approx(3 ** (1 / 3.0))


def test_analytic_formula_is_only_a_lower_bound_off_axis():
    d = np.array([0.3, 0.8, 0.52])
    d /= np.linalg.norm(d)
    radius = 1.3 * EDGE
    exact = V.gjk_distance(
        V.Placement.create(EDGE, [0, 0, 0]), V.Placement.create(EDGE, radius * d)
    )["distance_nm"]
    assert exact > V.analytic_gap_coaligned(EDGE, radius, d)
    assert not V.is_symmetry_direction(d)


def test_gjk_is_invariant_and_symmetric():
    qa = Rotation.random(random_state=1).as_matrix()
    qb = Rotation.random(random_state=2).as_matrix()
    centre = np.array([18.0, 3.0, -2.0])
    a = V.Placement.create(EDGE, [0, 0, 0], qa)
    b = V.Placement.create(EDGE, centre, qb)
    base = V.gjk_distance(a, b)["distance_nm"]
    assert V.gjk_distance(b, a)["distance_nm"] == pytest.approx(base, abs=1e-12)
    G = Rotation.random(random_state=3).as_matrix()
    t = np.array([100.0, -40.0, 7.0])
    a2 = V.Placement.create(EDGE, t, G @ qa)
    b2 = V.Placement.create(EDGE, G @ centre + t, G @ qb)
    assert V.gjk_distance(a2, b2)["distance_nm"] == pytest.approx(base, abs=1e-11)


def test_overlap_detection():
    a = V.Placement.create(EDGE, [0, 0, 0])
    assert V.overlaps(a, V.Placement.create(EDGE, [0.95 * EDGE, 0, 0]))
    assert not V.overlaps(a, V.Placement.create(EDGE, [1.05 * EDGE, 0, 0]))


def test_centre_distance_inverse_with_independent_rotations():
    qb = Rotation.random(random_state=4).as_matrix()
    d = np.array([0.5, 0.7, -0.51])
    d /= np.linalg.norm(d)
    radius = V.centre_distance_for_gap(EDGE, 3.0, d, rotation_b=qb)
    got = V.gjk_distance(
        V.Placement.create(EDGE, [0, 0, 0]), V.Placement.create(EDGE, radius * d, qb)
    )["distance_nm"]
    assert got == pytest.approx(3.0, abs=1e-9)


def test_point_to_body_analytic_points():
    body = V.Placement.create(EDGE, [0, 0, 0])
    face = V.point_to_body_distance(np.array([[EDGE, 0.0, 0.0]]), body)[0][0]
    assert face == pytest.approx(EDGE / 2, abs=1e-12)
    corner_point = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3) * 2 * EDGE
    expect = 2 * EDGE - 3 ** (1 / 3.0) * EDGE / 2
    assert V.point_to_body_distance(corner_point, body)[0][0] == pytest.approx(
        expect, abs=1e-9
    )


def test_point_to_body_certificate_is_tight():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    body = V.Placement.create(EDGE, [2.0, -1.0, 0.5],
                              Rotation.random(random_state=5).as_matrix())
    # sample in the body frame so every point is strictly exterior, which is
    # the function's contract
    local = directions * rng.uniform(0.80 * EDGE, 3 * EDGE, (300, 1))
    points = body.to_world(local)
    _, certificate = V.point_to_body_distance(points, body)
    assert certificate < 1e-9


# --------------------------------------------------------------- potential --
def test_kw_matches_the_si_quoted_value():
    params = V.load_parameters("singh_literature")
    assert params.kw(EDGE) == pytest.approx(2.5e5, rel=0.01)


def test_attraction_equals_minus_kw_times_the_bare_sum():
    """With the SI's 27 equal elements, E_attr must be exactly -K_W sum r^-6."""
    params = V.load_parameters("singh_literature")
    result = V.face_to_face_pair(EDGE, 2.99, params)
    element_volume = V.Superellipsoid(EDGE).volume_literature_nm3 / 27
    bare = result["attraction_bare_double_sum_nm6"] / element_volume ** 2
    assert result["u_attr_kcalmol"] == pytest.approx(-params.kw(EDGE) * bare, rel=1e-12)


def test_kw_is_not_rebound_to_the_mesh():
    """Refining the quadrature must not change K_W."""
    params = V.load_parameters("singh_literature")
    fine = replace(params, volume_scheme="cartesian", volume_n=12)
    assert fine.kw(EDGE) == params.kw(EDGE)
    assert fine.kw_element_count == 27


def test_exchange_symmetry():
    params = V.load_parameters("singh_literature")
    qa = Rotation.random(random_state=6).as_matrix()
    qb = Rotation.random(random_state=7).as_matrix()
    a = V.Placement.create(EDGE, [0, 0, 0], qa)
    b = V.Placement.create(EDGE, [19.0, 2.0, -1.0], qb)
    forward = V.pair_energy(a, b, params)
    reverse = V.pair_energy(b, a, params)
    for key in ("u_attr_kcalmol", "u_rep_kcalmol", "u_pair_kcalmol"):
        assert forward[key] == pytest.approx(reverse[key], rel=1e-13)


def test_symmetrisation_does_not_double_count():
    params = V.load_parameters("singh_literature")
    a = V.Placement.create(EDGE, [0, 0, 0])
    b = V.Placement.create(EDGE, [EDGE + 3.0, 0, 0])
    averaged = V.pair_energy(a, b, params)["u_rep_kcalmol"]
    one_sided = V.pair_energy(
        a, b, replace(params, symmetrise_repulsion=False)
    )["u_rep_kcalmol"]
    assert averaged == pytest.approx(one_sided, rel=1e-13)


def test_rigid_motion_invariance_of_the_energy():
    params = V.load_parameters("singh_literature")
    qa = Rotation.random(random_state=8).as_matrix()
    qb = Rotation.random(random_state=9).as_matrix()
    centre = np.array([20.0, 1.0, -3.0])
    base = V.pair_energy(
        V.Placement.create(EDGE, [0, 0, 0], qa),
        V.Placement.create(EDGE, centre, qb), params
    )["u_pair_kcalmol"]
    G = Rotation.random(random_state=10).as_matrix()
    t = np.array([-70.0, 12.0, 5.0])
    moved = V.pair_energy(
        V.Placement.create(EDGE, t, G @ qa),
        V.Placement.create(EDGE, G @ centre + t, G @ qb), params
    )["u_pair_kcalmol"]
    assert moved == pytest.approx(base, rel=1e-12)


def _far_field_params():
    return replace(
        V.load_parameters("singh_literature"), epsilon2=0.0,
        volume_scheme="cartesian", volume_n=12, kw_volume_convention="exact",
    )


def _attraction_at(factor, params):
    radius = factor * EDGE
    return V.pair_energy(
        V.Placement.create(EDGE, [0, 0, 0]),
        V.Placement.create(EDGE, [radius, 0, 0]), params
    )["u_attr_kcalmol"]


def test_second_moment_closed_form():
    """<x^2> = 3 a^2 / 40 exactly for the sextic superellipsoid."""
    shape = V.Superellipsoid(EDGE)
    nodes, weights = shape.volume_nodes("cartesian", 24, "exact")
    second = np.sum(weights * nodes[:, 0] ** 2) / weights.sum()
    assert second / EDGE ** 2 == pytest.approx(0.075, rel=1e-3)


def test_far_field_amplitude_and_quadrupole_correction():
    """U_attr -> -eps1 (A/pi^2) V^2 / R^6 [1 + 2.25 (a/R)^2 + O((a/R)^4)].

    The 2.25 is 30 <x^2> / a^2 with <x^2> = 3 a^2 / 40, the leading
    finite-size correction to the point-point Hamaker limit.
    """
    params = _far_field_params()
    shape = V.Superellipsoid(EDGE)
    point_point = params.attraction_prefactor() * shape.volume_exact_nm3 ** 2
    coefficients = []
    for factor in (30.0, 40.0, 60.0, 80.0):
        amplitude = -_attraction_at(factor, params) * (factor * EDGE) ** 6
        excess = amplitude / point_point - 1.0
        coefficients.append(excess * factor ** 2)
    # the scaled excess must be flat and equal to the analytic 2.25
    assert max(coefficients) - min(coefficients) < 0.01
    assert coefficients[-1] == pytest.approx(2.25, abs=0.01)
    # and the bare amplitude must approach the point-point limit
    amplitude = -_attraction_at(80.0, params) * (80.0 * EDGE) ** 6
    assert amplitude == pytest.approx(point_point, rel=5e-4)


def test_far_field_exponent_is_six():
    params = _far_field_params()
    exponent = np.log(_attraction_at(40.0, params) / _attraction_at(80.0, params)) / np.log(2.0)
    assert exponent == pytest.approx(6.0, abs=3e-3)


def test_overlap_returns_infinite_energy():
    params = V.load_parameters("singh_literature")
    result = V.pair_energy(
        V.Placement.create(EDGE, [0, 0, 0]),
        V.Placement.create(EDGE, [0.9 * EDGE, 0, 0]), params
    )
    assert result["overlap"] and result["hard_core_rejected"]
    assert np.isinf(result["u_pair_kcalmol"])


def test_overall_scale_multiplies_both_terms():
    params = V.load_parameters("singh_literature")
    one = V.face_to_face_pair(EDGE, 3.0, params)
    five = V.face_to_face_pair(EDGE, 3.0, replace(params, overall_scale=5.0))
    for key in ("u_attr_kcalmol", "u_rep_kcalmol", "u_pair_kcalmol"):
        assert five[key] == pytest.approx(5.0 * one[key], rel=1e-14)


def test_kernels_reproduce_direct_evaluation():
    params = V.load_parameters("singh_literature")
    gaps = [1.0, 3.0, 7.0]
    kernels = V.energy_kernels(EDGE, gaps, params)
    attraction, repulsion, total = V.kernel_energies(
        kernels, params.epsilon1, params.epsilon2, params.beta_nm
    )
    for index, gap in enumerate(gaps):
        direct = V.face_to_face_pair(EDGE, gap, params)
        assert attraction[index] == pytest.approx(direct["u_attr_kcalmol"], rel=1e-14)
        assert repulsion[index] == pytest.approx(direct["u_rep_kcalmol"], rel=1e-14)
        assert total[index] == pytest.approx(direct["u_pair_kcalmol"], rel=1e-14)


def test_per_particle_is_exactly_half_the_pair():
    params = V.load_parameters("singh_literature")
    result = V.face_to_face_pair(EDGE, 3.0, params, temperature_K=300.0)
    assert result["u_pair_per_particle_kBT"] == pytest.approx(
        0.5 * result["u_pair_kBT"], rel=1e-15
    )


def test_literature_parameters_have_no_interior_well():
    """Documented finding: (130, 290, 9.56 nm) cannot produce a minimum."""
    params = V.load_parameters("singh_literature")
    _, _, interior = V.find_well(EDGE, params, bracket=(0.3, 14.0), samples=50)
    assert interior is False


def test_configs_round_trip(tmp_path):
    params = V.load_parameters("singh_literature")
    path = V.save_parameters(replace(params, label="tmp"), tmp_path / "tmp.json")
    assert V.load_parameters(path).epsilon1 == params.epsilon1
