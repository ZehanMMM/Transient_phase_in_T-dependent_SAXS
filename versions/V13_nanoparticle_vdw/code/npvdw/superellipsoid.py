"""Sextic superellipsoid particle geometry and integration meshes.

Shape (Singh et al. SI, text above Eqn. 7):
    |x|^6 + |y|^6 + |z|^6 <= (a/2)^6        [particle body frame, nm]

This is a cube-like body with rounded edges AND corners.  It is NOT a
quadric ellipsoid and NOT a Minkowski-rounded cube.

Exact volume (SI p. 22):  V = (2 a^3 / sqrt(pi)) * Gamma(7/6)^3 = 0.900959 a^3
The SI then uses the rounded value 0.9 a^3 inside K_W; both are provided.

PROVENANCE
  support()                 : SI Eqn. 7 (Lagrange-multiplier support function).
  volume_exact()            : SI p. 22 closed form.
  volume_nodes('grid27')    : SI p. 23 "27 identical volume elements" plus the
                              Fig. S28E bottom inset, which shows a 3x3x3 array
                              of element centres viewed down [111].
                              IMPLEMENTATION ASSUMPTION: centres on the regular
                              3x3x3 lattice at multiples of a/3, equal weights
                              V/27.  The SI gives no node coordinates.
  surface_nodes('lattice',8): SI p. 23 "386 surface elements ... (the elements
                              have different surface areas)".
                              IMPLEMENTATION ASSUMPTION: the surface lattice
                              points of an 8x8x8 cube grid number exactly
                              6*8^2 + 2 = 386, and radial projection onto the
                              superellipsoid reproduces the area pattern of the
                              Fig. S28E top inset (smallest elements at face
                              centres, largest at the [111] corners, 3-fold
                              corner cells).  The SI gives neither node
                              coordinates nor areas.
  volume_nodes('cartesian'|'radial'), surface_nodes('facegl')
                            : this project's convergence machinery; no
                              literature counterpart.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gamma

import numpy as np

EXPONENT = 6.0
_LITERATURE_VOLUME_COEFFICIENT = 0.9
_EXACT_VOLUME_COEFFICIENT = (
    gamma(1.0 + 1.0 / EXPONENT) ** 3 / gamma(1.0 + 3.0 / EXPONENT)
)


def exact_volume_coefficient() -> float:
    """V / a^3 for the sextic superellipsoid = 0.9009589..."""
    return float(_EXACT_VOLUME_COEFFICIENT)


def shape_function(points_nm, half_nm: float) -> np.ndarray:
    """F(x) = sum |x_i|^6 - (a/2)^6 ; negative strictly inside."""
    p = np.asarray(points_nm, dtype=float)
    return np.sum(np.abs(p) ** EXPONENT, axis=-1) - half_nm ** EXPONENT


def support(direction, half_nm: float) -> np.ndarray:
    """SI Eqn. 7 support map: argmax_{x in body} d.x, body frame, origin centred.

    Lagrange multipliers on sum |x_i|^6 = (a/2)^6 give
        x_i = (a/2) sgn(d_i) |d_i|^(1/5) / ( sum_j |d_j|^(6/5) )^(1/6).
    Accepts a single direction (3,) or a stack (..., 3); d need not be a unit
    vector.  A zero direction returns the centre.
    """
    d = np.asarray(direction, dtype=float)
    abs_d = np.abs(d)
    numerator = np.sign(d) * abs_d ** (1.0 / (EXPONENT - 1.0))
    scale = np.sum(abs_d ** (EXPONENT / (EXPONENT - 1.0)), axis=-1, keepdims=True)
    scale = np.where(scale <= 0.0, 1.0, scale)
    return half_nm * numerator / scale ** (1.0 / EXPONENT)


def support_height(unit_direction, half_nm: float) -> np.ndarray:
    """h(d) = d.support(d) for |d| = 1; closed form (a/2)*(sum|d_i|^(6/5))^(5/6)."""
    d = np.abs(np.asarray(unit_direction, dtype=float))
    q = EXPONENT / (EXPONENT - 1.0)
    return half_nm * np.sum(d ** q, axis=-1) ** (1.0 - 1.0 / EXPONENT)


@dataclass(frozen=True)
class Superellipsoid:
    """A sextic superellipsoid of edge length ``edge_nm`` centred on the origin."""

    edge_nm: float

    @property
    def half_nm(self) -> float:
        return 0.5 * self.edge_nm

    @property
    def volume_exact_nm3(self) -> float:
        return _EXACT_VOLUME_COEFFICIENT * self.edge_nm ** 3

    @property
    def volume_literature_nm3(self) -> float:
        """The SI's rounded 0.9 a^3, the value that appears inside K_W."""
        return _LITERATURE_VOLUME_COEFFICIENT * self.edge_nm ** 3

    def volume(self, convention: str = "literature") -> float:
        if convention == "literature":
            return self.volume_literature_nm3
        if convention == "exact":
            return self.volume_exact_nm3
        raise ValueError("unknown volume convention %r" % (convention,))

    def support(self, direction) -> np.ndarray:
        return support(direction, self.half_nm)

    def support_height(self, unit_direction) -> np.ndarray:
        return support_height(unit_direction, self.half_nm)

    def contains(self, points_nm) -> np.ndarray:
        return shape_function(points_nm, self.half_nm) <= 0.0

    # ------------------------------------------------------------- volume ---
    def volume_nodes(self, scheme="grid27", n=3, volume_convention="literature"):
        """Return (centres_nm (N,3), weights_nm3 (N,)); sum(weights) == volume.

        'grid27'    literature 3x3x3 midpoint lattice, equal weights V/27.
        'cartesian' nested Gauss-Legendre using the EXACT superellipsoid
                    limits y<=(h^6-|x|^6)^(1/6), z<=(h^6-|x|^6-|y|^6)^(1/6).
                    n nodes per axis, N = n^3, every node strictly inside, so
                    there is no partial-voxel error by construction.
        'radial'    radial x angular product rule, N = n*n*2n.
        """
        if scheme == "grid27":
            return self._nodes_grid27(volume_convention)
        if scheme == "midpoint":
            return self._nodes_midpoint(n, volume_convention)
        if scheme == "cartesian":
            return self._nodes_cartesian(n, volume_convention)
        if scheme == "radial":
            return self._nodes_radial(n, volume_convention)
        raise ValueError("unknown volume scheme %r" % (scheme,))

    def _nodes_grid27(self, volume_convention):
        step = self.edge_nm / 3.0
        line = np.array([-step, 0.0, step])
        grid = np.stack(np.meshgrid(line, line, line, indexing="ij"), axis=-1)
        centres = grid.reshape(-1, 3)
        weights = np.full(27, self.volume(volume_convention) / 27.0)
        return centres, weights

    def _nodes_midpoint(self, n, volume_convention):
        """Equal-weight midpoint lattice, the faithful refinement of grid27.

        n = 3 reproduces the SI's node positions exactly: the lattice centres
        land on multiples of a/3 and all 27 fall inside the body, so the rule
        is identical to 'grid27'.  For n > 3 the corner cells of the bounding
        box fall OUTSIDE the superellipsoid; those are dropped and the
        surviving weights are renormalised to the total volume, so a partial
        boundary cell is never counted as a full one.

        This scheme exists to answer one question: does refining the SI's own
        rule converge?  It does, to the same limit as the Gauss-Legendre
        rules, but only at the first-order rate that a staircase boundary
        allows.  Use 'cartesian' for production work.
        """
        n = int(n)
        step = self.edge_nm / n
        line = (np.arange(n) + 0.5) * step - self.half_nm
        grid = np.stack(
            np.meshgrid(line, line, line, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        centres = grid[self.contains(grid)]
        weights = np.ones(len(centres))
        return centres, self._rescale(weights, volume_convention)

    def _nodes_cartesian(self, n, volume_convention):
        h = self.half_nm
        t, w = np.polynomial.legendre.leggauss(int(n))
        x = h * t
        wx = h * w
        y_half = (h ** EXPONENT - np.abs(x) ** EXPONENT) ** (1.0 / EXPONENT)
        y = y_half[:, None] * t[None, :]
        wy = y_half[:, None] * w[None, :]
        z_half = (
            h ** EXPONENT
            - np.abs(x)[:, None] ** EXPONENT
            - np.abs(y) ** EXPONENT
        ).clip(min=0.0) ** (1.0 / EXPONENT)
        z = z_half[:, :, None] * t[None, None, :]
        wz = z_half[:, :, None] * w[None, None, :]
        X = np.broadcast_to(x[:, None, None], z.shape)
        Y = np.broadcast_to(y[:, :, None], z.shape)
        centres = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
        weights = (wx[:, None, None] * wy[:, :, None] * wz).ravel()
        return centres, self._rescale(weights, volume_convention)

    def _nodes_radial(self, n, volume_convention):
        h = self.half_nm
        n = int(n)
        tr, wr = np.polynomial.legendre.leggauss(n)
        rho = 0.5 * (tr + 1.0)
        w_rho = 0.5 * wr * rho ** 2
        tc, wc = np.polynomial.legendre.leggauss(n)
        sin_t = np.sqrt(np.clip(1.0 - tc ** 2, 0.0, None))
        m = 2 * n
        phi = 2.0 * np.pi * (np.arange(m) + 0.5) / m
        w_phi = 2.0 * np.pi / m
        u = np.stack(
            [
                np.outer(sin_t, np.cos(phi)),
                np.outer(sin_t, np.sin(phi)),
                np.broadcast_to(tc[:, None], (n, m)),
            ],
            axis=-1,
        ).reshape(-1, 3)
        w_ang = (wc[:, None] * np.full(m, w_phi)[None, :]).ravel()
        extent = h / np.sum(np.abs(u) ** EXPONENT, axis=1) ** (1.0 / EXPONENT)
        centres = (
            (rho[:, None, None] * extent[None, :, None]) * u[None, :, :]
        ).reshape(-1, 3)
        weights = (w_rho[:, None] * (w_ang * extent ** 3)[None, :]).ravel()
        return centres, self._rescale(weights, volume_convention)

    def _rescale(self, weights, volume_convention):
        """Renormalise weights to the requested total volume.

        The raw weights already integrate to the exact volume at quadrature
        accuracy.  Renormalising keeps the total particle volume identical
        across refinement levels so that only spatial resolution changes.
        """
        target = self.volume(volume_convention)
        return weights * (target / weights.sum())

    # ------------------------------------------------------------ surface ---
    def surface_nodes(self, scheme="lattice", n=8, refine=48, area_mode="cube_param"):
        """Return (centres_nm (N,3), areas_nm2 (N,)); sum(areas) == surface area.

        'lattice'  literature-style reconstruction: the surface lattice points
                   of an n x n x n cube grid (N = 6 n^2 + 2; n = 8 -> 386),
                   radially projected.  Two area reconstructions are offered
                   because the SI specifies neither node coordinates nor areas:

                   area_mode='cube_param'
                       Voronoi partition of the cube-surface parameter domain,
                       refined ``refine`` times per coarse cell edge.  Gives
                       the LARGEST elements at face centres.  This ordering is
                       OPPOSITE to the Fig. S28E top-inset colour scale.
                   area_mode='equal_solid_angle'
                       equal solid angle per node, mapped to the surface with
                       the exact factor R(u)^2 / cos(gamma) and renormalised to
                       the true total area.  Gives the SMALLEST elements at
                       face centres and the LARGEST at the [111] corners, which
                       MATCHES the Fig. S28E top-inset colour scale.

        'facegl'   n x n Gauss-Legendre per cube face with the analytic surface
                   Jacobian; N = 6 n^2.  This is the converging scheme.
        """
        if scheme == "lattice":
            return self._surface_lattice(n, refine, area_mode)
        if scheme == "facegl":
            return self._surface_facegl(n)
        raise ValueError("unknown surface scheme %r" % (scheme,))

    def outward_normal(self, points_nm):
        """Unit outward normal of the superellipsoid at surface points."""
        p = np.asarray(points_nm, dtype=float)
        g = np.sign(p) * np.abs(p) ** (EXPONENT - 1.0)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def _project(self, u):
        norm = np.sum(np.abs(u) ** EXPONENT, axis=-1) ** (1.0 / EXPONENT)
        return u * (self.half_nm / norm)[..., None]

    @staticmethod
    def _cube_face_frames():
        for axis in range(3):
            for sign in (-1.0, 1.0):
                others = [i for i in range(3) if i != axis]
                yield axis, sign, others

    def _surface_lattice(self, n, refine, area_mode="cube_param"):
        from scipy.spatial import cKDTree

        n = int(n)
        line = 2.0 * np.arange(n + 1) / n - 1.0
        grid = np.stack(
            np.meshgrid(line, line, line, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        params = grid[np.max(np.abs(grid), axis=1) >= 1.0 - 1e-12]
        order = np.lexsort((params[:, 2], params[:, 1], params[:, 0]))
        params = params[order]
        centres = self._project(params)

        if area_mode == "equal_solid_angle":
            radius = np.linalg.norm(centres, axis=1)
            normal = self.outward_normal(centres)
            cos_gamma = np.abs(np.sum(centres * normal, axis=1)) / radius
            weight = radius ** 2 / cos_gamma
            areas = weight * (self.surface_area() / weight.sum())
            return centres, areas
        if area_mode != "cube_param":
            raise ValueError("unknown area_mode %r" % (area_mode,))

        fine = int(n * refine)
        s = 2.0 * np.arange(fine + 1) / fine - 1.0
        S, T = np.meshgrid(s, s, indexing="ij")
        cell_area = []
        cell_param = []
        for axis, sign, others in self._cube_face_frames():
            corner = np.zeros((fine + 1, fine + 1, 3))
            corner[..., axis] = sign
            corner[..., others[0]] = S
            corner[..., others[1]] = T
            X = self._project(corner)
            d1 = X[1:, 1:] - X[:-1, :-1]
            d2 = X[1:, :-1] - X[:-1, 1:]
            cell_area.append(
                0.5 * np.linalg.norm(np.cross(d1, d2), axis=-1).ravel()
            )
            mid = 0.25 * (
                corner[1:, 1:] + corner[:-1, :-1] + corner[1:, :-1] + corner[:-1, 1:]
            )
            cell_param.append(mid.reshape(-1, 3))
        cell_area = np.concatenate(cell_area)
        cell_param = np.vstack(cell_param)
        owner = cKDTree(params).query(cell_param)[1]
        areas = np.bincount(owner, weights=cell_area, minlength=len(params))
        if np.any(areas <= 0.0):
            raise RuntimeError("empty surface element; increase refine")
        return centres, areas

    def _surface_facegl(self, n):
        n = int(n)
        t, w = np.polynomial.legendre.leggauss(n)
        S, T = np.meshgrid(t, t, indexing="ij")
        WS, WT = np.meshgrid(w, w, indexing="ij")
        h = self.half_nm
        centres = []
        areas = []
        for axis, sign, others in self._cube_face_frames():
            u = np.zeros((n, n, 3))
            u[..., axis] = sign
            u[..., others[0]] = S
            u[..., others[1]] = T
            norm = np.sum(np.abs(u) ** EXPONENT, axis=-1) ** (1.0 / EXPONENT)
            x = h * u / norm[..., None]
            tangents = []
            for k in (0, 1):
                e = np.zeros(3)
                e[others[k]] = 1.0
                uc = u[..., others[k]]
                dnorm = (
                    np.abs(uc) ** (EXPONENT - 1.0)
                    * np.sign(uc)
                    / norm ** (EXPONENT - 1.0)
                )
                tangents.append(
                    h * (e / norm[..., None] - u * (dnorm / norm ** 2)[..., None])
                )
            jac = np.linalg.norm(np.cross(tangents[0], tangents[1]), axis=-1)
            centres.append(x.reshape(-1, 3))
            areas.append((jac * WS * WT).ravel())
        return np.vstack(centres), np.concatenate(areas)

    def surface_area(self, n: int = 128) -> float:
        return float(self._surface_facegl(n)[1].sum())
