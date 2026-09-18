"""Gilbert-Johnson-Keerthi distance and overlap for sextic superellipsoids.

Used for (a) the particle-level surface gap and overlap test, exactly as in
Singh et al. SI p. 24 ("we use the GJK algorithm to efficiently determine if
two nanocubes overlap"), and (b) the point-to-body distances that define r2 in
the continuum form of the repulsive integral.

GJK gives the MINIMUM surface separation of the two bodies.  It does NOT
replace the per-element distances inside the volume or surface integrals:
those are computed pair by pair / node by node in potential.py.

Analytic cross-checks (co-oriented identical particles) are exact because the
Minkowski difference of a centrally symmetric body with itself is that body
scaled by two, so
    gap(R, d) = |R| - 2 h(d)   with   h(d) = (a/2) (sum |d_i|^(6/5))^(5/6)
for any unit centre-offset direction d.  See analytic_gap_coaligned().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .superellipsoid import EXPONENT, Superellipsoid, support, support_height


@dataclass(frozen=True)
class Placement:
    """A particle instance: body shape, world centre, body->world rotation."""

    shape: Superellipsoid
    centre_nm: np.ndarray
    rotation: np.ndarray

    @staticmethod
    def create(edge_nm, centre_nm=(0.0, 0.0, 0.0), rotation=None) -> "Placement":
        rot = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if abs(np.linalg.det(rot) - 1.0) > 1e-9 or not np.allclose(
            rot.T @ rot, np.eye(3), atol=1e-9
        ):
            raise ValueError("rotation must be a proper rotation matrix")
        return Placement(
            Superellipsoid(float(edge_nm)),
            np.asarray(centre_nm, dtype=float),
            rot,
        )

    def support_world(self, direction_world) -> np.ndarray:
        """Support point in world coordinates for a world-frame direction."""
        d_body = self.rotation.T @ np.asarray(direction_world, dtype=float)
        return self.centre_nm + self.rotation @ support(d_body, self.shape.half_nm)

    def to_world(self, points_body) -> np.ndarray:
        return self.centre_nm + np.asarray(points_body, dtype=float) @ self.rotation.T

    def to_body(self, points_world) -> np.ndarray:
        return (np.asarray(points_world, dtype=float) - self.centre_nm) @ self.rotation


# --------------------------------------------------------------------------
# closest point of a simplex to the origin, returned as barycentric weights
# --------------------------------------------------------------------------
def _closest_simplex(points: np.ndarray):
    """Closest point of conv(points) to the origin.

    Returns (closest_point, weights, keep_indices).  Brute-force over all
    non-empty subsets: at most 15 subsets for a 4-simplex, which is
    negligible next to the integral evaluations and is numerically safer than
    a hand-rolled Voronoi-region cascade.
    """
    n = len(points)
    best = None
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        P = points[idx]
        k = len(idx)
        if k == 1:
            lam = np.array([1.0])
        else:
            # minimise |sum lam_i P_i|^2 subject to sum lam = 1
            G = P @ P.T
            A = np.empty((k + 1, k + 1))
            A[:k, :k] = 2.0 * G
            A[:k, k] = 1.0
            A[k, :k] = 1.0
            A[k, k] = 0.0
            b = np.zeros(k + 1)
            b[k] = 1.0
            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            lam = sol[:k]
            if np.any(lam < -1e-12):
                continue  # closest point is not interior to this subset
        point = lam @ P
        dist2 = float(point @ point)
        if best is None or dist2 < best[0] - 1e-18:
            best = (dist2, point, lam, idx)
    dist2, point, lam, idx = best
    return point, lam, idx


def gjk_distance(
    a: Placement,
    b: Placement,
    max_iterations: int = 256,
    tolerance: float = 1e-12,
):
    """Minimum surface separation of two placed superellipsoids.

    Returns a dict with:
      distance_nm    >= 0 separation, 0.0 when the bodies overlap
      overlap        bool
      point_a/point_b witness points on each surface (world frame, nm)
      iterations, residual_nm  convergence diagnostics
    """

    def support_difference(direction):
        pa = a.support_world(direction)
        pb = b.support_world(-np.asarray(direction, dtype=float))
        return pa - pb, pa, pb

    d = b.centre_nm - a.centre_nm
    if np.linalg.norm(d) < 1e-14:
        d = np.array([1.0, 0.0, 0.0])
    w, pa, pb = support_difference(-d)
    simplex = [w]
    wit_a = [pa]
    wit_b = [pb]
    v = w
    residual = np.inf
    for iteration in range(max_iterations):
        if np.linalg.norm(v) < 1e-14:
            return {
                "distance_nm": 0.0,
                "overlap": True,
                "point_a": wit_a[0],
                "point_b": wit_b[0],
                "iterations": iteration,
                "residual_nm": 0.0,
            }
        w, pa, pb = support_difference(-v)
        vv = float(v @ v)
        residual = vv - float(v @ w)
        if residual <= tolerance * max(vv, 1.0):
            break
        if float(v @ w) > 0.0 and len(simplex) >= 1:
            pass
        simplex.append(w)
        wit_a.append(pa)
        wit_b.append(pb)
        P = np.array(simplex)
        v, lam, keep = _closest_simplex(P)
        simplex = [simplex[i] for i in keep]
        wit_a = [wit_a[i] for i in keep]
        wit_b = [wit_b[i] for i in keep]
        if np.linalg.norm(v) < 1e-13:
            return {
                "distance_nm": 0.0,
                "overlap": True,
                "point_a": wit_a[0],
                "point_b": wit_b[0],
                "iterations": iteration,
                "residual_nm": 0.0,
            }
    P = np.array(simplex)
    v, lam, keep = _closest_simplex(P)
    lam_full = np.zeros(len(simplex))
    lam_full[: len(lam)] = 0.0
    A = np.array([wit_a[i] for i in keep])
    B = np.array([wit_b[i] for i in keep])
    point_a = lam @ A
    point_b = lam @ B
    distance = float(np.linalg.norm(v))
    return {
        "distance_nm": distance,
        "overlap": distance < 1e-12,
        "point_a": point_a,
        "point_b": point_b,
        "iterations": iteration + 1,
        "residual_nm": float(residual),
    }


def overlaps(a: Placement, b: Placement) -> bool:
    return bool(gjk_distance(a, b)["overlap"])


# --------------------------------------------------------------------------
# analytic benchmarks for co-oriented identical particles
# --------------------------------------------------------------------------
def analytic_support_height(edge_nm, unit_direction):
    return float(support_height(np.asarray(unit_direction, float), 0.5 * edge_nm))


SYMMETRY_DIRECTIONS = {
    "100": np.array([1.0, 0.0, 0.0]),
    "110": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    "111": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
}


def is_symmetry_direction(direction, tolerance: float = 1e-9) -> bool:
    """True when the closest-point normal is forced parallel to ``direction``.

    For two identical co-oriented particles the Minkowski difference is the
    body scaled by two, so gap = R - 2 h(d) holds only when the outward normal
    at the closest point is parallel to d.  For the sextic superellipsoid that
    is guaranteed exactly for the <100>, <110> and <111> families, where the
    non-zero components of d have equal magnitude.  It is NOT true for a
    generic direction, where gap > R - 2 h(d).
    """
    d = np.abs(np.asarray(direction, dtype=float))
    d = d / np.linalg.norm(d)
    nonzero = d[d > tolerance]
    return bool(np.all(np.abs(nonzero - nonzero[0]) < tolerance))


def analytic_gap_coaligned(edge_nm, centre_distance_nm, direction):
    """Surface gap of two identical co-oriented superellipsoids: R - 2 h(d).

    EXACT only for symmetry directions (see is_symmetry_direction); for any
    other direction this is a strict LOWER bound on the true gap.  Use
    gjk_distance() for generic directions.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return float(centre_distance_nm - 2.0 * analytic_support_height(edge_nm, d))


def centre_distance_for_gap_coaligned(edge_nm, gap_nm, direction):
    """R = gap + 2 h(d); exact for symmetry directions only."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return float(gap_nm + 2.0 * analytic_support_height(edge_nm, d))


def centre_distance_for_gap(
    edge_nm,
    gap_nm,
    direction,
    rotation_a=None,
    rotation_b=None,
    tolerance: float = 1e-12,
):
    """Solve GJK(gap) = gap_nm for the centre distance along ``direction``.

    Works for arbitrary independent orientations.  For co-oriented particles
    along a symmetry direction it reproduces gap + 2 h(d) to machine accuracy.
    """
    from scipy.optimize import brentq

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    a = Placement.create(edge_nm, [0.0, 0.0, 0.0], rotation_a)
    if (
        rotation_b is None
        and rotation_a is None
        and is_symmetry_direction(d)
    ):
        return centre_distance_for_gap_coaligned(edge_nm, gap_nm, d)

    def residual(radius):
        b = Placement.create(edge_nm, radius * d, rotation_b)
        return gjk_distance(a, b)["distance_nm"] - gap_nm

    lo = 0.5 * edge_nm
    hi = 2.0 * edge_nm + 4.0 * max(gap_nm, 1.0)
    while residual(lo) > 0.0:
        lo *= 0.5
        if lo < 1e-6:
            raise RuntimeError("cannot bracket centre distance from below")
    while residual(hi) < 0.0:
        hi *= 1.5
    return float(brentq(residual, lo, hi, xtol=tolerance, rtol=1e-15))


# --------------------------------------------------------------------------
# vectorised point-to-body distance (continuum r2)
# --------------------------------------------------------------------------
def point_to_body_distance(
    points_world: np.ndarray,
    body: Placement,
    iterations: int = 200,
    tolerance: float = 1e-12,
):
    """Exact distance from each world point to the surface of ``body``.

    Batched damped Newton solve of the projection KKT system in the body frame:
        q - x = lam * grad F(x),      F(x) = sum |x_i|^6 - (a/2)^6 = 0
    with grad F = 6 sgn(x_i) |x_i|^5 and Hessian diag(30 |x_i|^4).

    A plain fixed-point iteration n <- normalise(q - S(n)) is NOT used: it
    oscillates for points above the nearly flat faces of a sextic
    superellipsoid, where the radial direction and the true surface normal
    differ strongly.

    CONTRACT: points must lie OUTSIDE the body.  In this module that always
    holds, because the surface nodes of one particle are exterior to the other
    whenever the pair does not overlap, and overlapping pairs are rejected by
    GJK before any integral is evaluated.  Interior points are returned as
    distance 0 and excluded from the certificate, since the
    supporting-hyperplane lower bound is not valid for them.

    Returns (distance_nm, certified_bound_nm).  For the converged normal n and
    an exterior point,
        n.q - h(n)  <=  dist(q, body)  <=  |q - x|
    (left: supporting-hyperplane inequality; right: x is a feasible surface
    point), so the returned bound is a rigorous two-sided certificate on the
    worst exterior point of the batch.
    """
    q = np.atleast_2d(np.asarray(body.to_body(points_world), dtype=float))
    half = body.shape.half_nm
    p = EXPONENT
    limit = half ** p

    # start from the radial projection, which is feasible and cheap
    radial = q / np.linalg.norm(q, axis=1, keepdims=True)
    x = support(radial, half)
    x = np.where(np.abs(x) < 1e-12, 1e-12, x)
    grad = p * np.sign(x) * np.abs(x) ** (p - 1.0)
    lam = np.sum((q - x) * grad, axis=1) / np.sum(grad * grad, axis=1)

    n_pts = len(q)
    eye = np.broadcast_to(np.eye(3), (n_pts, 3, 3))
    for _ in range(iterations):
        grad = p * np.sign(x) * np.abs(x) ** (p - 1.0)
        hess = p * (p - 1.0) * np.abs(x) ** (p - 2.0)
        r1 = q - x - lam[:, None] * grad
        r2 = np.sum(np.abs(x) ** p, axis=1) - limit
        # both residuals are compared on their own scale: r1 is a length, r2
        # is the sextic form whose natural size is (a/2)^6.  Using an absolute
        # tolerance on r2 would demand far below machine precision and the
        # loop would never exit early.
        if max(np.abs(r1).max() / half, np.abs(r2).max() / limit) < tolerance:
            break
        J = np.zeros((n_pts, 4, 4))
        J[:, :3, :3] = -eye - lam[:, None, None] * (
            hess[:, :, None] * np.eye(3)[None, :, :]
        )
        J[:, :3, 3] = -grad
        J[:, 3, :3] = grad
        rhs = np.concatenate([-r1, -r2[:, None]], axis=1)[:, :, None]
        try:
            step = np.linalg.solve(J, rhs)[:, :, 0]
        except np.linalg.LinAlgError:
            break
        # damped update keeps |x| from collapsing through the origin
        scale = np.ones(n_pts)
        big = np.linalg.norm(step[:, :3], axis=1) > 0.5 * half
        scale[big] = 0.5 * half / np.linalg.norm(step[big, :3], axis=1)
        x = x + scale[:, None] * step[:, :3]
        lam = lam + scale * step[:, 3]
        x = np.where(np.abs(x) < 1e-14, 1e-14, x)

    # re-project onto the surface so the upper bound is strictly feasible
    x = support(p * np.sign(x) * np.abs(x) ** (p - 1.0), half)
    diff = q - x
    upper = np.linalg.norm(diff, axis=1)
    n = diff / np.maximum(upper, 1e-300)[:, None]
    lower = np.sum(n * q, axis=1) - np.sum(n * support(n, half), axis=1)
    exterior = np.sum(np.abs(q) ** p, axis=1) > limit
    upper = np.where(exterior, upper, 0.0)
    gap = (upper - lower)[exterior]
    return upper, float(np.max(gap)) if gap.size else 0.0
