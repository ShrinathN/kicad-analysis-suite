"""Magnetic cores in the filament solve: a volume integral method with face (facet) unknowns.

The core is a set of occupied cells of a tensor-product grid (planes at every box edge, each
interval split to at most the cell size).  The magnetisation M is represented by its normal
component m_a on every grid face a that touches the core (lowest-order Raviart-Thomas / "facet"
functions W_a: in each cell M_x varies linearly between the cell's two x-faces, and so on).  The
normal flux is continuous from cell to cell by construction, so magnetic charge appears only
where M really diverges: inside cells where it changes (ρ = -div M, uniform per cell) and on the
core's surface (σ = M·n).  This is what makes high permeability work: with one constant M per
cell (the first version) every cell boundary carried spurious charge, like a tiny air gap, and an
ungapped core at μr = 1000 read 2-6x low, converging slowly with the cells (Le-Van, Meunier et
al., IEEE Trans. Magn. 51 (2015): facet elements for this reason).

In a linear material M = χ H; tested with the same functions (Galerkin):

    (K/χ + P) m = G I

K_ab = ∫ W_a · W_b dV (a sparse mass matrix, divided by each cell's χ), P_ab = (1/4π) ∫∫ q_a q_b / r
(the Coulomb energy of the functions' charges: uniform cell charges and surface face charges; it
vanishes for flux that circulates without leaving the core), G_ak = ∫ W_a · H_k dV with H_k the
field of filament k for 1 A (Biot-Savart, Gauss points in the cells).  By reciprocity the core's
flux through filament k is μ0 ∫ M · H_k dV = μ0 G_k · m, so the core adds

    ΔL = μ0 Gᵀ (K/χ + P)⁻¹ G

to the filaments' partial inductance matrix: symmetric, positive, and for a real constant χ just
added to L (the reduced model, SPICE and maps are unchanged).

The charge integrals: ∫∫ 1/r between pieces (cells, boundary faces) by Gauss quadrature over one
piece of the other's exact potential (closed forms F3: ∂x∂y∂z F3 = 1/r, F2: ∂x∂y F2 = 1/r), far
pairs as point charges; equal relative geometries computed once.

Units: metres, SI.
"""
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import linalg

from .. import THREADS
from ..loop.peec import MEM_SHARE, _lg, available_ram, rect_integral
from .kernels import _hash

MU0 = 4e-7 * math.pi
FAR_Q = 4.0                          # piece pairs farther than this x the larger piece: point charges
_G3 = np.polynomial.legendre.leggauss(3)
_G2 = np.polynomial.legendre.leggauss(2)
_G4 = np.polynomial.legendre.leggauss(4)


# ---- closed-form potentials ------------------------------------------------------------
def _ln(u, r):
    return _lg(u, r, r * r - u * u)


def _atan(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, np.arctan(a / np.where(b != 0, b, 1.0)), 0.0)


def _F3(x, y, z):
    """∂x∂y∂z F3 = 1/r."""
    r = np.sqrt(x * x + y * y + z * z)
    return (x * y * _ln(z, r) + y * z * _ln(x, r) + z * x * _ln(y, r)
            - x * x / 2 * _atan(y * z, x * r) - y * y / 2 * _atan(x * z, y * r) - z * z / 2 * _atan(x * y, z * r))


def _F2(x, y, z):
    """∂x∂y F2 = 1/r (z: the distance from the rectangle's plane)."""
    r = np.sqrt(x * x + y * y + z * z)
    return x * _ln(y, r) + y * _ln(x, r) - z * _atan(x * y, z * r)


def box_potential(P, lo, hi):
    """∫ dV' / |P - r'| over boxes [lo, hi] (unit charge density) at points P (broadcast, (..., 3))."""
    tot = 0.0
    for sx, X in ((1, hi[..., 0] - P[..., 0]), (-1, lo[..., 0] - P[..., 0])):
        for sy, Y in ((1, hi[..., 1] - P[..., 1]), (-1, lo[..., 1] - P[..., 1])):
            for sz, Z in ((1, hi[..., 2] - P[..., 2]), (-1, lo[..., 2] - P[..., 2])):
                tot = tot + sx * sy * sz * _F3(X, Y, Z)
    return tot


def rect_potential(P, lo, hi, axis):
    """∫ dA' / |P - r'| over axis-normal rectangles [lo, hi] (lo[axis] = hi[axis]) at points P."""
    u, v = [a for a in range(3) if a != axis]
    dz = P[..., axis] - lo[..., axis]
    tot = 0.0
    for su, U in ((1, hi[..., u] - P[..., u]), (-1, lo[..., u] - P[..., u])):
        for sv, V in ((1, hi[..., v] - P[..., v]), (-1, lo[..., v] - P[..., v])):
            tot = tot + su * sv * _F2(U, V, dz)
    return tot


def core_bytes(n_dof, n_pieces):
    """Peak memory of Core and delta_L: C (pieces²), T C (dof x pieces), P and the factor (dof² each)."""
    return int(8 * (n_pieces ** 2 + n_dof * n_pieces + 3 * n_dof ** 2))


# ---- the grid --------------------------------------------------------------------------
def _edges(bounds, h, fine=()):
    """Sorted grid planes on one axis: every box bound, each interval split into pieces of at most
    h, or of at most hf where it overlaps a refinement span (a0, a1, hf)."""
    b = np.unique(np.round(np.asarray(list(bounds) + [q for a0, a1, _ in fine for q in (a0, a1)], float), 12))
    out = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        hh = min([h] + [hf for a0, a1, hf in fine if a0 < c - 1e-12 and a1 > a + 1e-12])
        n = max(1, int(math.ceil((c - a) / hh - 1e-9)))
        out += list(a + (c - a) * np.arange(1, n + 1) / n)
    return np.array(out)


@dataclass
class Grid:
    """Occupied cells of a tensor-product grid; chi per cell (μr - 1, real)."""
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    chi: np.ndarray                   # (nx, ny, nz), 0 where empty
    occ: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.occ is None:
            self.occ = self.chi > 0

    @property
    def shape(self):
        return self.occ.shape

    @property
    def n_cells(self):
        return int(self.occ.sum())


def grid_from_boxes(boxes, h, refine=()):
    """boxes: [(lo (3,), hi (3,), chi)], metres.  Later boxes override earlier ones (an air gap
    is a box with chi 0 inside a core box).  h: largest cell size; refine: [(lo, hi, h_fine)]
    boxes where the cells are at most h_fine (the grid is a tensor product: the refinement spans
    the core along the other axes)."""
    E = []
    for a in range(3):
        fine = [(lo[a], hi[a], hf) for lo, hi, hf in refine]
        E.append(_edges([q for lo, hi, _ in boxes for q in (lo[a], hi[a])], h, fine))
    lo_all = np.min([lo for lo, _, _ in boxes], axis=0)
    hi_all = np.max([hi for _, hi, _ in boxes], axis=0)
    E = [e[(e >= lo_all[a] - 1e-12) & (e <= hi_all[a] + 1e-12)] for a, e in enumerate(E)]
    cx, cy, cz = [(e[:-1] + e[1:]) / 2 for e in E]
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    chi = np.zeros(X.shape)
    for lo, hi, c in boxes:
        m = (X > lo[0]) & (X < hi[0]) & (Y > lo[1]) & (Y < hi[1]) & (Z > lo[2]) & (Z < hi[2])
        chi[m] = c
    return Grid(E[0], E[1], E[2], chi)


# ---- assembly ----------------------------------------------------------------------------
class Core:
    """The facet system of a grid: DOFs, mass matrix K (per unit 1/χ), charge matrix P."""

    def __init__(self, grid, progress=lambda m: None):
        self.grid = g = grid
        nx, ny, nz = g.shape
        occ = g.occ
        d = [np.diff(g.ex), np.diff(g.ey), np.diff(g.ez)]
        cell_id = -np.ones(g.shape, np.int64)
        cell_id[occ] = np.arange(occ.sum())
        self.cell_id = cell_id
        ii, jj, kk = np.nonzero(occ)
        self.cell_lo = np.column_stack([g.ex[ii], g.ey[jj], g.ez[kk]])
        self.cell_hi = np.column_stack([g.ex[ii + 1], g.ey[jj + 1], g.ez[kk + 1]])
        self.cell_chi = g.chi[occ]
        self.cell_ijk = np.column_stack([ii, jj, kk])
        nc = len(ii)
        # DOFs: faces with an occupied cell on either side; per DOF: axis, plane index, (j, k)
        dofs = []                     # (axis, i, j, k, left cell, right cell)
        for a in range(3):
            shp = list(g.shape)
            shp[a] += 1
            pad = [(0, 0)] * 3
            pad[a] = (1, 1)
            o = np.pad(occ, pad)
            cid = np.pad(cell_id, pad, constant_values=-1)
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[a] = slice(0, -1)
            sl_hi[a] = slice(1, None)
            L, R = o[tuple(sl_lo)], o[tuple(sl_hi)]
            cl, cr = cid[tuple(sl_lo)], cid[tuple(sl_hi)]
            f = np.argwhere(L | R)
            for idx in f:
                dofs.append((a, *idx, cl[tuple(idx)], cr[tuple(idx)]))
        D = np.array(dofs, np.int64)
        self.dof = D
        nd = len(D)
        # pieces: cells, then boundary faces
        bnd = np.flatnonzero((D[:, 4] < 0) | (D[:, 5] < 0))
        npc = nc + len(bnd)
        self.n_dof, self.n_pieces = nd, npc
        # charge densities T (dof x piece), mass matrix K (dof x dof, per unit 1/χ)
        Tr, Tc, Tv = [], [], []
        Kr, Kc, Kv = [], [], []
        face_lo = np.zeros((len(bnd), 3))
        face_hi = np.zeros((len(bnd), 3))
        face_axis = np.zeros(len(bnd), np.int64)
        bpos = {int(b): nc + q for q, b in enumerate(bnd)}
        E = [g.ex, g.ey, g.ez]
        for q, (a, i0, i1, i2, cl, cr) in enumerate(D):
            idx = [i0, i1, i2]
            i = idx[a]                                                  # the face's plane on axis a
            dl = d[a][i - 1] if cl >= 0 else None
            dr = d[a][i] if cr >= 0 else None
            if cl >= 0:
                Tr.append(q); Tc.append(cl); Tv.append(-1.0 / dl)        # noqa: E702
            if cr >= 0:
                Tr.append(q); Tc.append(cr); Tv.append(1.0 / dr)          # noqa: E702
            if q in bpos:
                p = bpos[q]
                Tr.append(q); Tc.append(p); Tv.append(1.0 if cr < 0 else -1.0)  # noqa: E702
                lo = np.array([E[b][idx[b]] for b in range(3)])
                hi = np.array([E[b][idx[b] + (0 if b == a else 1)] for b in range(3)])
                face_lo[p - nc], face_hi[p - nc], face_axis[p - nc] = lo, hi, a
        self.T = sp.csr_matrix((Tv, (Tr, Tc)), shape=(nd, npc))
        # mass matrix: in each cell, the two faces of one axis: V/3 each, V/6 between
        V = (self.cell_hi - self.cell_lo).prod(axis=1)
        self.cell_V = V
        face_of = {}
        for q, (a, i0, i1, i2, cl, cr) in enumerate(D):
            if cl >= 0:
                face_of[(cl, a, 1)] = q           # the cell's upper face on axis a
            if cr >= 0:
                face_of[(cr, a, 0)] = q           # the cell's lower face on axis a
        self.cell_faces = np.zeros((nc, 3, 2), np.int64)
        for c in range(nc):
            for a in range(3):
                l, u = face_of[(c, a, 0)], face_of[(c, a, 1)]
                self.cell_faces[c, a] = (l, u)
                Kr += [l, u, l, u]
                Kc += [l, u, u, l]
                Kv += [V[c] / 3, V[c] / 3, V[c] / 6, V[c] / 6]
        self.Kcell = (np.array(Kr), np.array(Kc), np.array(Kv), np.repeat(np.arange(nc), 12))
        self.face_lo, self.face_hi, self.face_axis = face_lo, face_hi, face_axis
        need, have = core_bytes(nd, npc), available_ram()
        if need > MEM_SHARE * have:
            raise ValueError(f"The core mesh is too large for this computer: {nc} cells ({nd} faces) need about "
                             f"{need / 2 ** 30:.1f} GB, {have / 2 ** 30:.1f} GB is free. Use larger core cells.")
        progress(f"Magnetic charge integrals ({nc} cells, {len(bnd)} surface faces)...")
        C = self._coulomb()
        TC = np.asarray(self.T @ C)                    # dof x piece (T is sparse)
        del C
        self.P = np.asarray((self.T @ TC.T).T) / (4 * math.pi)
        del TC
        self.P = (self.P + self.P.T) / 2

    def K(self, chi_cell=None):
        """Mass matrix divided by each cell's χ (dense)."""
        r, c, v, cell = self.Kcell
        chi = self.cell_chi if chi_cell is None else chi_cell
        K = np.zeros((self.n_dof, self.n_dof))
        np.add.at(K, (r, c), v / chi[cell])
        return K

    # -- ∫∫ 1/r between pieces ------------------------------------------------------------
    def _coulomb(self):
        nc = len(self.cell_V)
        nf = len(self.face_axis)
        lo = np.vstack([self.cell_lo, self.face_lo])
        hi = np.vstack([self.cell_hi, self.face_hi])
        kind = np.concatenate([np.full(nc, -1), self.face_axis])       # -1 cell, else face axis
        size = (hi - lo).max(axis=1)
        Q = np.concatenate([self.cell_V, ((self.face_hi - self.face_lo) + np.eye(3)[self.face_axis]).prod(axis=1)])
        cen = (lo + hi) / 2
        n = nc + nf
        C = np.zeros((n, n))
        rows = max(1, int(3e6 // max(n, 1)))

        def work(s):
            e = min(s + rows, n)
            ii, jj = np.meshgrid(np.arange(s, e), np.arange(n), indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            dist = np.linalg.norm(cen[ii] - cen[jj], axis=1)
            far = dist > FAR_Q * np.maximum(size[ii], size[jj])
            out = np.where(far, Q[ii] * Q[jj] / np.maximum(dist, 1e-30), 0.0)
            k = np.flatnonzero(~far)
            if k.size:
                a, b = ii[k], jj[k]
                off = cen[a] - cen[b]
                key = _hash([off[:, 0], off[:, 1], off[:, 2], hi[a, 0] - lo[a, 0], hi[a, 1] - lo[a, 1],
                             hi[a, 2] - lo[a, 2], hi[b, 0] - lo[b, 0], hi[b, 1] - lo[b, 1], hi[b, 2] - lo[b, 2],
                             kind[a].astype(float), kind[b].astype(float)])
                _, u, inv = np.unique(key, return_index=True, return_inverse=True)
                vals = _pair_integral(lo[a[u]], hi[a[u]], kind[a[u]], lo[b[u]], hi[b[u]], kind[b[u]])
                out[k] = vals[inv.ravel()]
            C[s:e] = out.reshape(e - s, n)
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            list(ex.map(work, range(0, n, rows)))
        return (C + C.T) / 2

    # -- coupling to filaments --------------------------------------------------------------
    def field_matrix(self, F):
        """G (dof x B): ∫ W_a · H_k dV for 1 A in each filament."""
        nc = len(self.cell_V)
        B = len(F)
        G = np.zeros((self.n_dof, B))
        eps2 = (0.2235 * (F.w + F.h)) ** 2
        mid = F.c0 + F.u * F.l[:, None] / 2
        lock = __import__("threading").Lock()

        def pts(gq):
            x, w = gq
            P = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
            W = np.einsum("i,j,k->ijk", w, w, w).ravel() / 8
            return P, W
        PA, WA = pts(_G2)
        PB, WB = pts(_G4)

        def work(cells):
            for c in cells:
                lo, hi = self.cell_lo[c], self.cell_hi[c]
                h = hi - lo
                cc = (lo + hi) / 2
                dist = np.linalg.norm(mid - cc, axis=1) - F.l / 2
                near = dist < 2 * h.max()
                rowsG = np.zeros((6, B))
                for sel, P, W in ((~near, PA, WA), (near, PB, WB)):
                    k = np.flatnonzero(sel)
                    if not k.size:
                        continue
                    X = cc + P * h / 2                                    # (q, 3)
                    H = _seg_field(X[None], F.c0[k][:, None], F.u[k][:, None], F.l[k][:, None], eps2[k][:, None])
                    t = (P + 1) / 2                                       # 0..1 in the cell
                    for a in range(3):
                        wl = (1 - t[:, a]) * W * self.cell_V[c]           # lower face's function
                        wu = t[:, a] * W * self.cell_V[c]
                        rowsG[2 * a, k] = H[:, :, a] @ wl
                        rowsG[2 * a + 1, k] = H[:, :, a] @ wu
                with lock:
                    for a in range(3):
                        G[self.cell_faces[c, a, 0]] += rowsG[2 * a]
                        G[self.cell_faces[c, a, 1]] += rowsG[2 * a + 1]
        chunks = np.array_split(np.arange(nc), max(1, THREADS * 4))
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            list(ex.map(work, chunks))
        return G

    def delta_L(self, F, chi_cell=None, progress=lambda m: None):
        """ΔL (B x B) [H] the core adds to the filaments' partial inductances (real χ)."""
        progress("Coil field in the core...")
        G = self.field_matrix(F)
        A = self.K(chi_cell) + self.P
        progress("Core response...")
        X = linalg.cho_solve(linalg.cho_factor((A + A.T) / 2, check_finite=False), G, check_finite=False)
        dL = MU0 * G.T @ X
        return (dL + dL.T) / 2


def _pair_integral(lo_a, hi_a, kind_a, lo_b, hi_b, kind_b):
    """∫_a ∫_b 1/r for piece pairs (unit densities): Gauss points on a (3 per axis), b's exact
    potential."""
    out = np.zeros(len(lo_a))
    # parallel faces: exact (the tape integral of the inductance kernels)
    par = (kind_a >= 0) & (kind_a == kind_b)
    for ax in np.unique(kind_a[par]):
        s = np.flatnonzero(par & (kind_a == ax))
        u, v = [a for a in range(3) if a != ax]
        out[s] = rect_integral(lo_a[s, u], hi_a[s, u], lo_a[s, v], hi_a[s, v], lo_b[s, u], hi_b[s, u],
                               lo_b[s, v], hi_b[s, v], lo_a[s, ax] - lo_b[s, ax])
    # Gauss points per axis on piece a: enough for its aspect ratio and for b's potential, which
    # varies over max(size of b, half the distance) (a long cell next to a small one, or a cell of a
    # graded grid 6:1, read the charge energy 5 % high with a fixed 3 x 3 x 3)
    ext_a = hi_a - lo_a
    cen_a, cen_b = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    scale = np.maximum((hi_b - lo_b).max(axis=1), 0.5 * np.linalg.norm(cen_a - cen_b, axis=1))
    flat = np.where(ext_a > 0, ext_a, np.inf).min(axis=1)
    scale = np.maximum(np.minimum(scale, flat), 1e-30)
    npt = np.where(ext_a > 0, np.clip(np.ceil(2.5 * ext_a / scale[:, None]), 3, 10), 1).astype(int)
    key = npt[:, 0] * 10000 + npt[:, 1] * 100 + npt[:, 2] + (kind_a + 1) * 1000000 + (kind_b + 1) * 10000000
    for kk in np.unique(key[~par]):
        s = np.flatnonzero((key == kk) & ~par)
        ka, kb = int(kind_a[s[0]]), int(kind_b[s[0]])
        n3 = npt[s[0]]
        axes = []
        for a in range(3):
            if n3[a] == 1:
                axes.append((np.zeros(1), np.full(1, 2.0)))
            else:
                axes.append(np.polynomial.legendre.leggauss(int(n3[a])))
        P = np.stack(np.meshgrid(*[ax[0] for ax in axes], indexing="ij"), -1).reshape(-1, 3)
        W = np.einsum("i,j,k->ijk", *[ax[1] for ax in axes]).ravel() / 8
        la, ha = lo_a[s], hi_a[s]
        ext = ha - la
        meas = np.where(ext > 0, ext, 1.0).prod(axis=1)
        X = (la + ha)[:, None, :] / 2 + P[None] * ext[:, None, :] / 2          # (s, q, 3)
        lb, hb = lo_b[s][:, None, :], hi_b[s][:, None, :]
        pot = box_potential(X, lb, hb) if kb < 0 else rect_potential(X, lb, hb, kb)
        out[s] = (pot * W[None]).sum(axis=1) * meas
    return out


def _seg_field(P, A, u, l, eps2):
    """H [A/m per A] at points P of straight filaments from A along unit u, length l (broadcast),
    regularised by eps² (the filament's own size)."""
    d = P - A
    s = np.einsum("...k,...k->...", d, u)
    rp = d - s[..., None] * u
    rho2 = np.einsum("...k,...k->...", rp, rp) + eps2
    t = (l - s) / np.sqrt((l - s) ** 2 + rho2) + s / np.sqrt(s * s + rho2)
    return np.cross(u, rp) * (t / (4 * math.pi * rho2))[..., None]
