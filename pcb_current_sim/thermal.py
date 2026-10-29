"""Steady-state 2D thermal sheet model of the board.

One sheet per copper layer over the board outline.  Lateral conduction comes
from copper (all nets) plus a share of the adjacent dielectric.  Adjacent
sheets are coupled through the dielectric and through the copper barrels of
vias and plated holes (all nets), and the outer sheets lose heat to ambient
per face: h_conv, or with h_auto a local h from natural convection plus
radiation at the local surface temperature.  The thermal grid has its own extent (the
simulated copper plus a margin of board, or the whole board) and a pitch that
is a multiple of the electrical one; heat and temperature are mapped between
the two grids cell by cell.  Cut edges of a cropped board are adiabatic.
The matrix is factorised once; with h_auto the changing surface h is handled
by conjugate gradients preconditioned with that factorisation.
"""
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import raster as rz

MAX_CELLS = 300_000      # thermal cells per layer
RHOC_CU = 3.45e6         # J/m^3K (8960 kg/m^3 x 385 J/kgK)
RHOC_FR4 = 2.0e6         # J/m^3K (1850 kg/m^3 x 1100 J/kgK)
SIGMA = 5.670e-8         # W/m^2K^4


def h_surface(s, rise, L_char):
    """Per-face heat-transfer coefficient [W/m^2K] at a local surface temperature rise [K]:
    laminar natural convection of a vertical plate of height L_char [m], 1.42 (dT/L)^0.25,
    plus grey-body radiation to surroundings at ambient, eps sigma (T^4 - Ta^4) / dT."""
    dT = np.maximum(rise, 0.05)
    Ta = s.T_amb + 273.15
    return 1.42 * (dT / L_char) ** 0.25 + s.emissivity * SIGMA * ((Ta + dT) ** 4 - Ta ** 4) / dT


class ThermalModel:
    def __init__(self, board, grid, thick, s, bbox=None):
        """bbox: thermal extent (x0, y0, x1, y1) in mm; default = the electrical grid."""
        self.grid, self.s, self.layers = grid, s, board.copper_layers
        if bbox is None:
            x0, y0 = grid.x0, grid.y0
            x1, y1 = grid.x0 + grid.nx * grid.h, grid.y0 + grid.ny * grid.h
        else:
            x0, y0, x1, y1 = bbox
            x0, y0 = min(x0, grid.x0), min(y0, grid.y0)
            x1, y1 = max(x1, grid.x0 + grid.nx * grid.h), max(y1, grid.y0 + grid.ny * grid.h)
        cells = (x1 - x0) * (y1 - y0) / grid.h ** 2
        c = max(1, math.ceil(math.sqrt(cells / MAX_CELLS)))
        # align the thermal origin with the electrical grid so each fine cell lies in one coarse cell
        ox = grid.x0 - math.ceil((grid.x0 - x0) / grid.h - 1e-9) * grid.h
        oy = grid.y0 - math.ceil((grid.y0 - y0) / grid.h - 1e-9) * grid.h
        ht = grid.h * c
        self.tg = tg = rz.Grid(ox, oy, ht, max(1, math.ceil((x1 - ox) / ht - 1e-9)),
                               max(1, math.ceil((y1 - oy) / ht - 1e-9)))
        self.c = c
        # fine -> coarse index maps (per column / row)
        xs = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.h
        ys = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.h
        self.ci = np.clip(np.floor((xs - tg.x0) / ht).astype(int), 0, tg.nx - 1)
        self.cj = np.clip(np.floor((ys - tg.y0) / ht).astype(int), 0, tg.ny - 1)
        self.Pc = sp.csr_matrix((np.ones(grid.nx), (self.ci, np.arange(grid.nx))), shape=(tg.nx, grid.nx))
        self.Pr = sp.csr_matrix((np.ones(grid.ny), (self.cj, np.arange(grid.ny))), shape=(tg.ny, grid.ny))
        X, Y = tg.centres()
        fall = rz.coverage_all(board, tg)
        mask = board.outline_mask(X, Y)
        for f in fall.values():
            mask |= f > 0
        self.mask, self.fall = mask, fall
        nm = int(mask.sum())
        idx = np.full(tg.shape, -1, np.int64)
        idx[mask] = np.arange(nm)
        nL, ht = len(self.layers), tg.h * 1e-3
        gaps = board.gaps
        I, J, G, faces = [], [], [], np.zeros(nm * nL)
        for k, L in enumerate(self.layers):
            if nL == 1:
                t_share = board.thickness * 1e-3
            else:
                t_share = ((gaps[k - 1] if k > 0 else 0) + (gaps[k] if k < nL - 1 else 0)) / 2
            K = s.k_cu * thick[L] * fall[L] + s.k_fr4 * t_share
            for sa, sb in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                           ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
                a, b = idx[sa], idx[sb]
                m = (a >= 0) & (b >= 0)
                Ka, Kb = K[sa][m], K[sb][m]
                I.append(a[m] + k * nm); J.append(b[m] + k * nm); G.append(2 * Ka * Kb / (Ka + Kb))
            faces[k * nm:(k + 1) * nm] = (k == 0) + (k == nL - 1)
            if k < nL - 1:
                gv = s.k_fr4 * ht * ht / gaps[k]
                ar = np.arange(nm)
                I.append(ar + k * nm); J.append(ar + (k + 1) * nm); G.append(np.full(nm, gv))
        # copper barrels of vias and plated holes conduct heat between the layers they span
        zs = np.concatenate([[0.0], np.cumsum(gaps)]) if nL > 1 else np.zeros(1)
        holes = [(v.pos, v.drill, v.layers) for v in board.vias]
        holes += [(p.pos, p.drill[0], p.layers) for p in board.pads if p.plated]
        for pos, drill, lays in holes:
            j, i = tg.cell(*pos)
            if not (0 <= j < tg.ny and 0 <= i < tg.nx) or idx[j, i] < 0:
                continue
            ks = sorted(self.layers.index(L) for L in lays if L in self.layers)
            ring = math.pi * s.t_plate * (drill * 1e-3 + s.t_plate)
            for a, b in zip(ks[:-1], ks[1:]):
                I.append([a * nm + idx[j, i]]); J.append([b * nm + idx[j, i]])
                G.append([s.k_cu * ring / (zs[b] - zs[a])])
        I, J, G = np.concatenate(I), np.concatenate(J), np.concatenate(G)
        n = nm * nL
        off = sp.coo_matrix((G, (I, J)), shape=(n, n))
        off = off + off.T
        self.A0 = (sp.diags(np.asarray(off.sum(1)).ravel()) - off).tocsc()
        self.faces, self.area = faces, ht * ht
        self.L_char = min(0.3, max(0.01, math.sqrt(nm * ht * ht)))   # board height for convection
        self.h = np.full(n, float(s.h_conv))
        A = self.A0 + sp.diags(self.faces * self.h * self.area)
        self.A = A.tocsc()
        self.adjacent_gaps = [[g for g in ((gaps[k - 1] if k > 0 else None), (gaps[k] if k < nL - 1 else None))
                               if g is not None] for k in range(nL)]
        if nL == 1:
            self.adjacent_gaps = [[board.thickness * 1e-3]]
        # symmetric minimum-degree ordering: ~3x faster and ~2.5x less fill than COLAMD on grid Laplacians
        self.lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True))
        self.idx, self.nm, self.n = idx, nm, n
        self.T = None         # last coarse result, dict layer -> (nyc, nxc)

    def solve(self, q_fine, point_heat=()):
        """q_fine: layer -> (ny, nx) W per fine cell. Returns layer -> fine T [C]."""
        rhs = self.heat(q_fine, point_heat)
        theta = self.lu.solve(rhs)
        if getattr(self.s, "h_auto", False):
            theta = self._solve_h_auto(rhs, theta)
        return self.to_fine(theta)

    def _solve_h_auto(self, rhs, theta, tol=0.01, max_iter=30):
        """Fixed point on the temperature-dependent surface h; each linear solve is CG
        preconditioned by the factorisation at the initial h (cond <= h_max / h_min)."""
        surf = self.faces > 0
        M = spla.LinearOperator(self.A.shape, self.lu.solve)
        for _ in range(max_iter):
            h = h_surface(self.s, theta[surf], self.L_char)
            change = float(np.max(np.abs(h - self.h[surf]) / self.h[surf]))
            self.h[surf] = h
            self.A = (self.A0 + sp.diags(self.faces * self.h * self.area)).tocsc()
            theta, _ = spla.cg(self.A, rhs, x0=theta, rtol=1e-9, maxiter=200, M=M)
            if change < tol:
                break
        return theta

    def h_mean(self):
        """Area-weighted mean surface h [W/m^2K] (over both faces)."""
        surf = self.faces > 0
        return float(np.sum(self.h[surf] * self.faces[surf]) / np.sum(self.faces[surf]))

    # ---- transient ---------------------------------------------------------
    def capacity(self, thick, t_pulse):
        """Heat capacity per node [J/K]: copper of all nets plus the dielectric a pulse of length
        t_pulse can reach (semi-infinite penetration depth 0.886 sqrt(alpha t), capped at half of
        each adjacent dielectric)."""
        ht = self.tg.h * 1e-3
        dpen = 0.886 * math.sqrt(self.s.k_fr4 / RHOC_FR4 * max(t_pulse, 1e-9))
        C = np.zeros(self.n)
        for k, L in enumerate(self.layers):
            d_eff = sum(min(g / 2, dpen) for g in self.adjacent_gaps[k])
            c = (RHOC_CU * thick[L] * self.fall[L] + RHOC_FR4 * d_eff) * ht * ht
            C[k * self.nm:(k + 1) * self.nm] = c[self.mask]
        return C

    def begin_transient(self, C, dt):
        """Implicit Euler: (C/dt + A) theta_new = C/dt theta + heat."""
        self.C_dt = C / dt
        self.lu_t = spla.splu((self.A + sp.diags(self.C_dt)).tocsc(), permc_spec="MMD_AT_PLUS_A",
                              options=dict(SymmetricMode=True))

    def step(self, theta, heat):
        return self.lu_t.solve(self.C_dt * theta + heat)

    def theta_from(self, grid, maps):
        """Temperature-rise vector sampled (nearest cell) from coarse maps on another grid."""
        X, Y = self.tg.centres()
        jj = np.clip(np.floor((Y - grid.y0) / grid.h).astype(int), 0, grid.ny - 1)
        ii = np.clip(np.floor((X - grid.x0) / grid.h).astype(int), 0, grid.nx - 1)
        out = []
        for L in self.layers:
            a = np.nan_to_num(maps[L][jj, ii] - self.s.T_amb)
            out.append(a[self.mask])
        return np.concatenate(out)

    def heat(self, q_fine, point_heat=()):
        """Right-hand side [W per thermal node] from fine-grid heat and point sources."""
        tg = self.tg
        rhs = np.zeros(self.n)
        for k, L in enumerate(self.layers):
            qc = np.asarray(self.Pr @ (self.Pc @ np.asarray(q_fine[L], float).T).T)
            for x, y, PL, P, *geom in point_heat:
                if PL != L:
                    continue
                w = self._spread(geom[0]) if geom and geom[0] is not None else None
                if w is not None:
                    j0, j1, i0, i1, wt = w
                    qc[j0:j1, i0:i1] += P * wt
                    continue
                j, i = tg.cell(x, y)
                if 0 <= j < tg.ny and 0 <= i < tg.nx:
                    qc[j, i] += P
            rhs[k * self.nm:(k + 1) * self.nm] = qc[self.mask]
        return rhs

    def _spread(self, geom):
        """Normalised weights to spread a source over a pad's area (cached), or None."""
        cache = self.__dict__.setdefault("_spread_cache", {})
        if id(geom) not in cache:
            c = rz.coverage(geom, self.tg, 4)
            if c is not None:
                j0, j1, i0, i1, cov = c
                cov = cov * self.mask[j0:j1, i0:i1]
                c = (j0, j1, i0, i1, cov / cov.sum()) if cov.sum() > 0 else None
            cache[id(geom)] = (geom, c)       # keep geom alive so its id stays unique
        return cache[id(geom)][1]

    def to_fine(self, theta, store=True):
        """Temperature rise vector -> layer -> fine T [C]; also keeps the coarse maps in self.T."""
        tg = self.tg
        coarse, out = {}, {}
        for k, L in enumerate(self.layers):
            Tc = np.full(tg.shape, np.nan)
            Tc[self.mask] = self.s.T_amb + theta[k * self.nm:(k + 1) * self.nm]
            coarse[L] = Tc
            fill = np.where(np.isnan(Tc), self.s.T_amb, Tc)
            out[L] = fill[np.ix_(self.cj, self.ci)].astype(np.float32)
        if store:
            self.T = coarse
        self.last_coarse = coarse
        return out


