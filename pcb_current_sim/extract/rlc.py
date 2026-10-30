"""R, L, C extraction of a trace (any copper path between pads) as an RLC ladder.

1. path.build traces the net's copper between the chosen pads (tracks, arcs, vias,
   stubs; zones become single nodes).
2. Every track piece is sampled every `sample` mm.  At each sample the copper of the
   return nets is cut perpendicular to the track (zones and tracks within ±`window` mm,
   on every layer) and the 2D field solver gives L' and C' for that cross-section.
   Identical cross-sections are solved once.
3. R' = ρ(T) / (w t).  With `dc_solver` the ladder's resistance between port 1 and 2 is
   scaled to the full DC field solution (pads, crowding, vias), so DC R is exact.
4. Vias: barrel R; L = μ0 h/(2π) ln(s/r) with the nearest return via at s, else the
   partial inductance 0.2 h (ln(4h/d) + 1) nH; C from the antipad (Johnson's formula).
   SMD pads add C' (pad width) x pad length; THT pads a via-like C.
5. Chains of track pieces between junctions are cut into equal sections no longer than
   λ/10 at f_max (and `max_section`).  Each section is R + L in series with C/2 to the
   reference at each end (a π ladder); optional G = ωC tanδ at f_max.
"""
import heapq
import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

from .. import linsolve as LS
from .. import raster as rz
from ..board import rot
from . import path as P
from . import xsection as X

MU0 = 4e-7 * math.pi
C_LIGHT = 299792458.0
EPS0 = 8.8541878128e-12
ZONE_CELLS = 60_000          # mesh cells per zone (pitch >= 0.05 mm)
TAP_R = 0.25               # mm, radius of the tap on a pour under a pad that is not a port or element pin
PAIR_TOL = 1e-6           # zone branches weaker than this (relative conductance) are dropped
GND_WORDS = ("GND", "VSS", "0V", "EARTH", "RETURN", "AGND", "DGND", "PGND", "SGND")


@dataclass
class Options:
    f_max: float = 1e9            # Hz: sections are at most λ/10 here; R_ac and G are evaluated here
    max_section: float = 0.0      # mm, extra limit on the section length (0 = none)
    sample: float = 0.5           # mm, cross-section spacing along the tracks
    window: float = 5.0           # mm, how far either side of a track the return copper is searched
    stubs: bool = True            # keep branches to unselected pads (open stubs)
    pad_c: bool = True            # shunt C of the port / stub pads
    dc_solver: bool = True        # scale R to the DC field solution between port 1 and port 2
    dielectric_g: bool = False    # shunt G = ωC tanδ at f_max
    T: float = 20.0               # °C for R
    rho20: float = 1.72e-8
    alpha: float = 0.00393
    t_plate: float = 25e-6        # m, via plating


@dataclass
class Extraction:
    net: str
    ports: list                   # pad ids
    return_nets: list
    opts: Options
    segments: list                # dicts (table rows)
    vias: list                    # dicts
    pads: list                    # dicts (pad C)
    elements: list                # (name, kind R/L/C, node a, node b, value)
    nodes: list                   # names; ports first, then "REF", then internal
    pairs: list                   # per port pair: totals dict
    pieces: list                  # plot data: dict(p0, p1, layer, w, z0, seg)
    markers: list                 # discontinuities: dict(pos, text)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    uuids: list = field(default_factory=list)     # board tracks and vias on the extracted copper
    zones: list = field(default_factory=list)     # dicts (zone models, table rows)
    user: list = field(default_factory=list)      # dicts (user elements, table rows)


def guess_return_nets(board, signal_net=None):
    """GND-like nets; else the net with the most zone area."""
    nets = [n for n in board.nets if n != signal_net]
    g = [n for n in nets if any(w in n.upper().replace("/", "") for w in GND_WORDS)]
    if g:
        return g
    area = {}
    for z in board.zones:
        if z.net and z.net != signal_net:
            area[z.net] = area.get(z.net, 0.0) + z.geom.area()
    return [max(area, key=area.get)] if area else []


class Stack:
    """Copper z positions and dielectric slabs (m, z down, top copper at 0)."""

    def __init__(self, stackup, f):
        self.st = stackup
        self.z = stackup.layer_z()
        self.t = stackup.cu_thickness
        Ls = stackup.copper_layers
        ers = [d.er_tand(f, model=stackup.model)[0] for d in stackup.dielectrics]
        self.tand = [float(d.er_tand(f, model=stackup.model)[1]) for d in stackup.dielectrics]
        self.ers = [float(e) for e in ers]
        self.slabs = []
        for k, d in enumerate(stackup.dielectrics):
            self.slabs.append((self.z[Ls[k]] + self.t[Ls[k]], self.z[Ls[k + 1]], self.ers[k]))
        for k in range(1, len(Ls) - 1):      # inner copper sits in resin: mean of its neighbours
            L = Ls[k]
            self.slabs.append((self.z[L], self.z[L] + self.t[L], (self.ers[k - 1] + self.ers[k]) / 2))

    def near(self, layer):
        """(εr, tanδ) of the dielectric next to a layer (towards the board centre)."""
        k = self.st.copper_layers.index(layer)
        k = min(k, len(self.ers) - 1) if k < len(self.st.copper_layers) / 2 else max(k - 1, 0)
        return (self.ers[k], self.tand[k]) if self.ers else (1.0, 0.0)


class Returns:
    """Return-net copper (zones, tracks) per layer, sampled along lines."""

    PITCH = 0.02          # mm, raster of the return copper (section keys are rounded to this)
    MAX_CELLS = 25_000_000

    def __init__(self, board, nets):
        nets = set(nets)
        self.board = board
        self.masks = {}
        self.prims = {L: [] for L in board.copper_layers}
        for z in board.zones:
            if z.net in nets:
                self.prims[z.layer].append(z.geom)
        for t in board.tracks:
            if t.net in nets:
                self.prims[t.layer].append(t.geom)
        self.vias = [(v.pos, v.drill, set(v.layers)) for v in board.vias if v.net in nets]
        self.vias += [(p.pos, sum(p.drill) / 2, set(p.layers)) for p in board.pads if p.net in nets and p.plated]

    def _mask(self, layer):
        """Return copper of a layer rasterised once (point-in-polygon on big pours is slow)."""
        if layer not in self.masks:
            prims = self.prims.get(layer, [])
            if not prims:
                self.masks[layer] = None
                return None
            b = np.array([g.bbox for g in prims])
            x0, y0, x1, y1 = b[:, 0].min() - 0.1, b[:, 1].min() - 0.1, b[:, 2].max() + 0.1, b[:, 3].max() + 0.1
            h = max(self.PITCH, math.sqrt((x1 - x0) * (y1 - y0) / self.MAX_CELLS))
            nx, ny = int(math.ceil((x1 - x0) / h)), int(math.ceil((y1 - y0) / h))
            m = np.zeros((ny, nx), bool)
            for g in prims:
                gx0, gy0, gx1, gy1 = g.bbox
                i0, i1 = max(0, int((gx0 - x0) / h)), min(nx, int(math.ceil((gx1 - x0) / h)) + 1)
                j0, j1 = max(0, int((gy0 - y0) / h)), min(ny, int(math.ceil((gy1 - y0) / h)) + 1)
                if i1 <= i0 or j1 <= j0:
                    continue
                xs = x0 + (np.arange(i0, i1) + 0.5) * h
                ys = y0 + (np.arange(j0, j1) + 0.5) * h
                if hasattr(g, "lattice"):
                    m[j0:j1, i0:i1] |= g.lattice(xs, ys)
                else:
                    XX, YY = np.meshgrid(xs, ys)
                    m[j0:j1, i0:i1] |= g.inside(XX, YY)
            self.masks[layer] = (m, x0, y0, h)
        return self.masks[layer]

    def inside(self, layer, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        mk = self._mask(layer)
        if mk is None or not X.size:
            return np.zeros(X.shape, bool)
        m, x0, y0, h = mk
        i = np.floor((X - x0) / h).astype(np.int64)
        j = np.floor((Y - y0) / h).astype(np.int64)
        ok = (i >= 0) & (i < m.shape[1]) & (j >= 0) & (j < m.shape[0])
        out = np.zeros(X.shape, bool)
        out[ok] = m[j[ok], i[ok]]
        return out


def _q(x):
    """Round a return-copper edge (mm from the track centre) for the section key: finely near
    the track, coarser further out, where it hardly changes L and C."""
    step = 0.02 if abs(x) < 1.0 else 0.1 if abs(x) < 3.0 else 0.25
    return round(x / step) * step


def _runs(mask, s, ds):
    """(s0, s1) runs of True in mask sampled at s."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(s[i] - ds / 2, s[j] + ds / 2) for i, j in zip(a, b)]


class Extractor:
    def __init__(self, board, stackup, return_nets, opts=None, progress=None, elements=()):
        """elements: elements.Element between pads (R, V = diode, C, L), added to the model."""
        self.board, self.opts = board, opts or Options()
        self.elements = [e for e in elements if e.enabled]
        self.return_nets = list(return_nets)
        self.stack = Stack(stackup, self.opts.f_max)
        self.ret = Returns(board, return_nets)
        self.progress = progress or (lambda m: None)
        self.cache = {}
        self.n_solves = 0
        W = self.opts.window
        self.ds = 0.01
        self.s = np.arange(-W, W + 1e-9, self.ds)

    # ---- cross-sections ------------------------------------------------------
    def section_key(self, p, u, layer, w):
        """Hashable description of the return copper around point p (track direction u)."""
        n = (-u[1], u[0])
        X, Y = p[0] + self.s * n[0], p[1] + self.s * n[1]
        out = []
        for L in self.board.copper_layers:
            m = self.ret.inside(L, X, Y)
            if L == layer:
                m &= np.abs(self.s) > w / 2 + 1e-3
            runs = [(_q(a), _q(b)) for a, b in _runs(m, self.s, self.ds) if b - a >= 0.05]
            if runs:
                out.append((L, tuple(runs)))
        return (layer, round(w, 4), tuple(out))

    def line(self, key):
        """LineParams for a section key (cached), or None without return copper."""
        if key in self.cache:
            return self.cache[key]
        layer, w, rets = key
        st, W = self.stack, self.opts.window * 1e-3
        rects = [X.Rect(-w / 2e3, w / 2e3, st.z[layer], st.z[layer] + st.t[layer], True)]
        for L, runs in rets:
            for a, b in runs:
                rects.append(X.Rect(a * 1e-3, b * 1e-3, st.z[L], st.z[L] + st.t[L]))
        lp = X.solve(X.Section(rects, st.slabs, half_width=W, air=W)) if rets else None
        self.n_solves += 1
        self.cache[key] = lp
        return lp

    def describe(self, key):
        layer, w, rets = key
        if not rets:
            return "none in window"
        out = []
        for L, runs in rets:
            if L == layer:
                gap = min(min(abs(a), abs(b)) for a, b in runs) - w / 2
                out.append(f"coplanar, gap {gap:.2f} mm")
            else:
                under = any(a <= -w / 2 and b >= w / 2 for a, b in runs)
                out.append(L + ("" if under else " (offset)"))
        return ", ".join(out)

    # ---- per-length R ---------------------------------------------------------
    def rho(self):
        o = self.opts
        return o.rho20 * (1 + o.alpha * (o.T - 20.0))

    def r_dc(self, layer, w):
        """Ω/m"""
        return self.rho() / (w * 1e-3 * self.stack.t[layer])

    def r_ac(self, layer, w):
        """Ω/m at f_max: current on the face towards the return (one skin depth deep)."""
        d = math.sqrt(self.rho() / (math.pi * self.opts.f_max * MU0))
        t = self.stack.t[layer]
        return max(self.r_dc(layer, w), self.rho() / (w * 1e-3 * d * (1 - math.exp(-t / d))))

    # ---- vias and pads ----------------------------------------------------------
    def antipad(self, pos, r0, layers):
        """Median distance (mm) from pos to return copper on the given layers (None if > 2 mm)."""
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        found = []
        for L in layers:
            for r in np.arange(r0, 2.0, 0.02):
                if self.ret.inside(L, pos[0] + r * np.cos(ang), pos[1] + r * np.sin(ang)).any():
                    found.append(r)
                    break
        return float(np.median(found)) if found else None

    def johnson_c(self, length_mm, d1, d2, er):
        """Via C [F] (Johnson & Graham): 1.41 εr T D1 / (D2 - D1) pF, dimensions in inches."""
        if d2 is None or d2 <= d1:
            return 0.0
        return 1.41e-12 * er * (length_mm / 25.4) * d1 / (d2 - d1)

    def via(self, k, L1, L2, dz):
        v = self.board.vias[k]
        o = self.opts
        d = v.drill
        R = self.rho() * dz * 1e-3 / (math.pi * o.t_plate * (d * 1e-3 + o.t_plate))
        rv = [(math.dist(pos, v.pos), dr) for pos, dr, ls in self.ret.vias if ls & set(v.layers)]
        s = min(rv)[0] if rv else math.inf
        if s < o.window and s > d:
            L = MU0 * dz * 1e-3 / (2 * math.pi) * math.log(s / (d / 2))
            how = f"return via at {s:.2f} mm"
        else:
            L = 0.2e-9 * dz * (math.log(4 * dz / d) + 1) if dz > 0 else 0.0
            how = "no return via nearby (partial L)"
        er = self.stack.st.er_between(L1, L2) if self.stack.ers else 1.0
        span = self.board.layer_span(v.layers[0], v.layers[-1])
        ra = self.antipad(v.pos, v.size / 2, [L for L in span if self.ret.prims.get(L)])
        C = self.johnson_c(dz, v.size, 2 * ra if ra else None, er)
        return dict(R=R, L=L, C=C, how=how, antipad=ra)

    def pad_c(self, pad):
        """Shunt C [F] of a pad to the return."""
        if pad.plated:
            ra = self.antipad(pad.pos, max(pad.size) / 2, [L for L in pad.layers if self.ret.prims.get(L)])
            er = self.stack.st.er_between(self.board.copper_layers[0], self.board.copper_layers[-1]) \
                if self.stack.ers else 1.0
            return self.johnson_c(self.board.thickness, min(pad.size), 2 * ra if ra else None, er), "THT barrel"
        sx, sy = pad.size
        u = rot(1, 0, pad.angle) if sx >= sy else rot(0, 1, pad.angle)
        key = self.section_key(pad.pos, u, pad.layers[0], min(sx, sy))
        lp = self.line(key)
        return (lp.C * max(sx, sy) * 1e-3 if lp else 0.0), self.describe(key)

    # ---- zones ---------------------------------------------------------------------
    def zone_model(self, zi, terms):
        """Lumped model of a copper pour between the nodes that touch it.

        The zone is meshed on a uniform grid (cut-cell face apertures).  Each link conducts
        σ t a (R) and has the sheet inductance of copper over its return, μ0 h per square
        (h: dielectric to the nearest return copper above / below; both sides in parallel);
        each cell has C = ε0 A Σ εr/h to the return.  Kron reduction of the conductance and of
        the inverse-inductance networks onto the terminals gives an R and an L branch between
        every pair of terminals (exact at DC, and in the high-frequency limit); the cells' C is
        split among the terminals by the harmonic weights of the DC solution."""
        z, st, o = self.board.zones[zi], self.stack, self.opts
        L0 = z.layer
        x0, y0, x1, y1 = z.geom.bbox
        area = z.geom.area()
        h = max(0.05, math.sqrt(max(area, 1e-6) / ZONE_CELLS))
        grid = rz.Grid(x0 - h, y0 - h, h, int(math.ceil((x1 - x0) / h)) + 2, int(math.ceil((y1 - y0) / h)) + 2)
        warnings = []
        cov = np.zeros(grid.shape, np.float32)
        apx = np.zeros((grid.ny, grid.nx + 1), np.float32)
        apy = np.zeros((grid.ny + 1, grid.nx), np.float32)
        res = rz.coverage(z.geom, grid, faces=True)
        if res is not None:
            j0, j1, i0, i1, c, ax, ay = res
            cov[j0:j1, i0:i1], apx[j0:j1, i0:i1 + 1], apy[j0:j1 + 1, i0:i1] = c, ax, ay
        act = cov > 0.02
        X, Y = grid.centres()
        jj, ii = np.nonzero(act)
        n = len(jj)
        cid = np.full(grid.shape, -1, np.int64)
        cid[jj, ii] = np.arange(n)
        xs, ys = X[jj, ii], Y[jj, ii]

        # height above the return, per cell
        k0 = self.board.copper_layers.index(L0)
        ha, hb = np.full(n, np.inf), np.full(n, np.inf)        # above / below
        ea, eb = np.ones(n), np.ones(n)
        for L2 in self.board.copper_layers:
            if L2 == L0 or not self.ret.prims.get(L2):
                continue
            hit = self.ret.inside(L2, xs, ys)
            d = st.st.gap(L0, L2)
            er = st.st.er_between(L0, L2) if st.ers else 1.0
            if self.board.copper_layers.index(L2) < k0:
                m = hit & (d < ha)
                ha[m], ea[m] = d, er
            else:
                m = hit & (d < hb)
                hb[m], eb[m] = d, er
        none = ~np.isfinite(ha) & ~np.isfinite(hb)
        with np.errstate(invalid="ignore", divide="ignore"):
            Lsq = np.where(np.isfinite(ha) & np.isfinite(hb), ha * hb / (ha + hb), np.minimum(ha, hb))
        if none.any():
            Lsq[none] = self.board.thickness * 1e-3
            warnings.append(f"Zone {z.net} on {L0}: {100 * none.mean():.0f} % of it has no return copper above or "
                            f"below; its L there uses h = board thickness and its C is 0.")
        Lsq = MU0 * Lsq                                        # H per square
        A_cell = cov[jj, ii] * (h * 1e-3) ** 2
        Ccell = EPS0 * A_cell * (np.where(np.isfinite(ha), ea / np.where(np.isfinite(ha), ha, 1), 0)
                                 + np.where(np.isfinite(hb), eb / np.where(np.isfinite(hb), hb, 1), 0))

        # links
        I, J, ap = [], [], []
        for (sa, sb), face in ((((slice(None), slice(0, -1)), (slice(None), slice(1, None))), apx[:, 1:-1]),
                               (((slice(0, -1), slice(None)), (slice(1, None), slice(None))), apy[1:-1, :])):
            a, b = cid[sa], cid[sb]
            m = (a >= 0) & (b >= 0) & (face > 0)
            I.append(a[m]); J.append(b[m]); ap.append(face[m])
        I, J, ap = np.concatenate(I), np.concatenate(J), np.concatenate(ap).astype(float)
        G = ap * st.t[L0] / self.rho()
        Gam = ap / ((Lsq[I] + Lsq[J]) / 2)

        # terminals: the cells under each touching pad / via / track end (each cell claimed once).
        # A port or element pad is one node over its whole area (as the DC solver ties it to its
        # electrode); any other pad is plain copper of the pour, tapped near its centre: merging
        # its cells would let current cut across it for free (big THT pads in a pour read R low).
        g = self._graph
        fed = set(g.ports) | {v for e in g.edges if e.kind == "element" for v in (e.a, e.b)}
        owner = np.full(n, -1, np.int64)
        order = sorted(range(len(terms)), key=lambda t: {"pad": 0, "via": 1, "end": 2}[terms[t][1][0]])
        for t in order:
            nd, term = terms[t]
            kind, what = term
            if kind == "pad":
                m = what.geom.inside(xs, ys)
                pos = what.pos
                if nd not in fed:
                    m &= np.hypot(xs - pos[0], ys - pos[1]) <= max(TAP_R, h / 2)
            else:
                pos = None
            if pos is None:
                pn = self._term_pos(nd)
                pos = pn
                m = np.hypot(xs - pos[0], ys - pos[1]) <= max(what, h / 2)
            m &= owner < 0
            if not m.any():
                d = np.hypot(xs - pos[0], ys - pos[1]) + np.where(owner < 0, 0, 1e9)
                m = np.zeros(n, bool)
                m[int(np.argmin(d))] = True
            owner[m] = t
        T = len(terms)
        # node index: terminal t -> t, free cell -> T + rank
        free = np.flatnonzero(owner < 0)
        node = np.where(owner >= 0, owner, 0)
        node[free] = T + np.arange(len(free))
        N = T + len(free)

        def lap(w):
            a, b = node[I], node[J]
            m = a != b
            off = sp.coo_matrix((w[m], (a[m], b[m])), shape=(N, N))
            off = (off + off.T).tocsr()
            return (sp.diags(np.asarray(off.sum(1)).ravel()) - off).tocsr()
        Gl, Gm = lap(G), lap(Gam)
        # drop copper not connected to any terminal
        _, lab = connected_components(abs(Gl) > 0, directed=False)
        live = np.isin(lab, np.unique(lab[:T]))
        keep_i = np.flatnonzero(live[T:]) + T

        def kron(Lp, want_w=False):
            Ltt = Lp[:T][:, :T].toarray()
            if not len(keep_i):
                return Ltt, np.zeros((0, T))
            Lit = Lp[keep_i][:, :T].toarray()
            Wt = LS.factorize(Lp[keep_i][:, keep_i].tocsc()).solve(-Lit)          # harmonic weights
            return Ltt + Lit.T @ Wt, Wt
        Gred, Wg = kron(Gl)
        Mred, _ = kron(Gm)
        Cn = np.zeros(N)
        np.add.at(Cn, node, Ccell)
        Cterm = Cn[:T] + (Wg.T @ Cn[keep_i] if len(keep_i) else 0)

        pairs = []
        gmax = np.max(np.abs(np.diag(Gred))) if T else 0
        for i in range(T):
            for j in range(i + 1, T):
                gij, mij = -Gred[i, j], -Mred[i, j]
                if gij > PAIR_TOL * gmax and mij > 0:
                    pairs.append((i, j, 1 / gij, 1 / mij))
        # worst-case terminal-to-terminal R and L (for the table)
        R_eff = L_eff = 0.0
        if T >= 2:
            Gp, Mp = np.linalg.pinv(Gred), np.linalg.pinv(Mred)
            for i in range(T):
                for j in range(i + 1, T):
                    v = np.zeros(T)
                    v[i], v[j] = 1, -1
                    R_eff, L_eff = max(R_eff, v @ Gp @ v), max(L_eff, v @ Mp @ v)
        size = math.hypot(x1 - x0, y1 - y0)
        er = max(np.max(ea), np.max(eb)) if n else 1.0
        lam10 = C_LIGHT / (o.f_max * math.sqrt(er)) / 10 * 1e3
        if size > lam10:
            warnings.append(f"Zone {z.net} on {L0} spans {size:.0f} mm, more than λ/10 ({lam10:.0f} mm) at f max: its "
                            f"lumped model holds up to about {o.f_max * lam10 / size / 1e6:.0f} MHz.")
        hmin = np.nanmin(np.where(np.isfinite(Lsq / MU0), Lsq / MU0, np.nan)) if n else float("nan")
        note = (f"h to return {hmin * 1e3:.2f} mm" if np.isfinite(hmin) and not none.all() else "no return under it")
        return dict(pairs=pairs, C=list(Cterm), area=area, cells=n, pitch=h, R_eff=R_eff, L_eff=L_eff,
                    note=note, warnings=warnings)

    def _term_pos(self, nd):
        return self._graph.nodes[nd].pos

    # ---- main --------------------------------------------------------------------
    def run(self, port_ids):
        t0 = time.time()
        o, st = self.opts, self.stack
        g = P.build(self.board, port_ids, stubs=o.stubs, layer_z=st.z, elements=self.elements,
                    return_nets=self.return_nets)
        self._graph = g
        warnings = list(g.warnings)
        if not self.return_nets:
            warnings.append("No return net chosen: L and C cannot be extracted (only R).")
        n_z = len({e.a for e in g.edges if e.kind == "zonelink"})
        self.progress(f"Net {', '.join(g.nets)}: {sum(e.kind == 'track' for e in g.edges)} track pieces, "
                      f"{sum(e.kind == 'barrel' for e in g.edges)} via barrels, {n_z} zone(s), "
                      f"{sum(e.kind == 'element' for e in g.edges)} element(s).")

        # sample every track piece
        sub = {}            # edge index -> list of (length mm, key, lp, p0, p1)
        n_tr = sum(e.kind == "track" for e in g.edges)
        done = 0
        for k, e in enumerate(g.edges):
            if e.kind != "track":
                continue
            n = max(1, math.ceil(e.length / o.sample - 1e-9))
            ux, uy = (e.p1[0] - e.p0[0]) / e.length, (e.p1[1] - e.p0[1]) / e.length
            parts = []
            for i in range(n):
                a = (e.p0[0] + (e.p1[0] - e.p0[0]) * i / n, e.p0[1] + (e.p1[1] - e.p0[1]) * i / n)
                b = (e.p0[0] + (e.p1[0] - e.p0[0]) * (i + 1) / n, e.p0[1] + (e.p1[1] - e.p0[1]) * (i + 1) / n)
                key = self.section_key(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), (ux, uy), e.layer, e.width)
                parts.append([e.length / n, key, self.line(key), a, b])
            sub[k] = parts
            done += 1
            if done % 50 == 0:
                self.progress(f"  cross-sections: {done}/{n_tr} pieces, {self.n_solves} field solves")

        def per_len(layer, w, length, lp):
            """(R', L', C', G') per m of one sub-piece; partial L of the piece without a return."""
            Rp = self.r_dc(layer, w)
            if lp is None:
                l, wt = max(length, 1e-3) * 1e-3, (w * 1e-3 + st.t[layer])
                Lp = MU0 / (2 * math.pi) * (math.log(2 * l / wt) + 0.5 + 0.2235 * wt / l)
                return Rp, max(Lp, 0.0), 0.0, 0.0
            er, tand = st.near(layer)
            q = (lp.eeff - 1) / (er - 1) if er > 1 else 0.0
            Gp = 2 * math.pi * o.f_max * lp.C * tand * q * er / lp.eeff if o.dielectric_g else 0.0
            return Rp, lp.L, lp.C, Gp

        # chains of track pieces between junctions
        adj = g.adjacency()
        port_set = set(g.ports)

        def through(nd):
            inc = adj.get(nd, [])
            return (nd not in port_set and g.nodes[nd].kind == "pt" and len(inc) == 2
                    and all(g.edges[k].kind == "track" for _, k in inc))
        used, chains = set(), []
        for k, e in enumerate(g.edges):
            if e.kind != "track" or k in used:
                continue
            # walk back to a chain end, then forward
            cur_n, cur_k = e.a, k
            while through(cur_n):
                nxt = [(v, kk) for v, kk in adj[cur_n] if kk != cur_k][0]
                cur_n, cur_k = nxt
                if cur_k == k:
                    break
            start, seq, nd, kk = cur_n, [], cur_n, cur_k
            while True:
                used.add(kk)
                ee = g.edges[kk]
                fwd = ee.a == nd
                seq.append((kk, fwd))
                nd = ee.b if fwd else ee.a
                if not through(nd):
                    break
                nxt = [(v, k2) for v, k2 in adj[nd] if k2 != kk]
                if not nxt or nxt[0][1] in used:
                    break
                kk = nxt[0][1]
            chains.append((start, nd, seq))

        # netlist
        names = {n: f"P{i + 1}" for i, n in enumerate(g.ports)}
        node_list = [names[n] for n in g.ports] + ["REF"]
        counter = [0]

        taken = set(n.upper() for n in node_list)

        def nname(nd=None):
            """Pad nodes are named after the pad (L1.2 -> L1_2), all others n1, n2, ..."""
            if nd is not None and g.nodes[nd].kind == "ref":
                return "REF"
            if nd is not None and nd in names:
                return names[nd]
            nm = _ident(g.nodes[nd].label) if nd is not None and g.nodes[nd].kind == "pad" else ""
            if not nm or nm.upper() in taken:
                counter[0] += 1
                nm = f"n{counter[0]}"
            taken.add(nm.upper())
            node_list.append(nm)
            if nd is not None:
                names[nd] = nm
            return nm
        elements, shunt_c, shunt_g = [], {}, {}
        used = set()

        def ename(kind, base):
            """SPICE instance names say where an element comes from (R_trk1_2, L_via5, C_L1_2, ...), so
            they cannot be mistaken for the board's reference designators.  SPICE ignores case."""
            nm, k = f"{kind}_{base}", 2
            while nm.upper() in used:
                nm, k = f"{kind}_{base}_{k}", k + 1
            used.add(nm.upper())
            return nm

        def series(a, b, R, L, tag, base):
            mid = nname()
            elements.append([ename("R", base), "R", a, mid, max(R, 1e-6), tag])
            elements.append([ename("L", base), "L", mid, b, max(L, 1e-15), tag])

        segments, pieces, markers = [], [], []
        seg_of_edge = {}
        for ci, (na, nb, seq) in enumerate(chains):
            items = []
            for kk, fwd in seq:
                ps = sub[kk] if fwd else [[l, key, lp, b, a] for l, key, lp, a, b in reversed(sub[kk])]
                e = g.edges[kk]
                for l, key, lp, a, b in ps:
                    items.append((l, key, lp, a, b, e.layer, e.width, e.track))
            # segments: runs of equal cross-section
            segs = []
            for it in items:
                if segs and segs[-1]["key"] == it[1]:
                    s = segs[-1]
                    s["length"] += it[0]
                    s["p1"] = it[4]
                else:
                    segs.append(dict(key=it[1], lp=it[2], length=it[0], p0=it[3], p1=it[4], layer=it[5],
                                     width=it[6]))
            for s in segs:
                Rp, Lp, Cp, Gp = per_len(s["layer"], s["width"], s["length"], s["lp"])
                lp = s["lp"]
                row = dict(chain=ci + 1, kind="track", layer=s["layer"], width_mm=s["width"],
                           length_mm=s["length"], Z0=lp.z0 if lp else float("nan"),
                           eeff=lp.eeff if lp else float("nan"), L_nH_mm=Lp * 1e6, C_pF_mm=Cp * 1e9,
                           R_mOhm_mm=Rp, R_ac_mOhm_mm=self.r_ac(s["layer"], s["width"]),
                           delay_ps=(math.sqrt(Lp * Cp) * s["length"] * 1e-3 * 1e12) if Cp else float("nan"),
                           reference=self.describe(s["key"]),
                           x0=s["p0"][0], y0=s["p0"][1], x1=s["p1"][0], y1=s["p1"][1])
                prev = segments[-1] if segments and segments[-1]["chain"] == ci + 1 else None
                if prev is not None and _similar(prev, row):
                    _merge(prev, row)
                    continue
                if prev is not None:
                    za, zb = prev["Z0"], row["Z0"]
                    if np.isfinite(za) and np.isfinite(zb) and abs(zb - za) > 0.1 * za:
                        markers.append(dict(pos=s["p0"], text=f"{za:.0f} → {zb:.0f} Ω"))
                row["_z_first"], row["_ref_len"] = row["Z0"], row["length_mm"]
                segments.append(row)
            for l, key, lp, a, b, layer, w, tk in items:
                pieces.append(dict(p0=a, p1=b, layer=layer, w=w, z0=lp.z0 if lp else float("nan"),
                                   seg=len(segments)))
            # sections
            ltot = sum(it[0] for it in items)
            ee_max = max([it[2].eeff for it in items if it[2]] or [1.0])
            lsec = C_LIGHT / (o.f_max * math.sqrt(ee_max)) / 10 * 1e3
            if o.max_section > 0:
                lsec = min(lsec, o.max_section)
            nsec = max(1, math.ceil(ltot / lsec - 1e-9))
            bounds = np.linspace(0, ltot, nsec + 1)
            acc = np.zeros((nsec, 4))
            pos = 0.0
            for l, key, lp, a, b, layer, w, tk in items:
                q = np.array(per_len(layer, w, l, lp)) * 1e-3          # per mm
                lo, hi = pos, pos + l
                ov = np.clip(np.minimum(hi, bounds[1:]) - np.maximum(lo, bounds[:-1]), 0, None)
                acc += ov[:, None] * q[None, :]
                pos = hi
            prev = nname(na)
            for i in range(nsec):
                nxt = nname(nb) if i == nsec - 1 else nname()
                R, L, C, G = acc[i]
                series(prev, nxt, R, L, f"chain {ci + 1}", f"trk{ci + 1}_{i + 1}")
                for nd in (prev, nxt):
                    shunt_c[nd] = shunt_c.get(nd, 0.0) + C / 2
                    shunt_g[nd] = shunt_g.get(nd, 0.0) + G / 2
                prev = nxt
            for kk, _ in seq:
                seg_of_edge[kk] = ci + 1

        for row in segments:
            row.pop("_z_first", None)
            row.pop("_ref_len", None)
        vias = []
        for k, e in enumerate(g.edges):
            if e.kind != "barrel":
                continue
            m = self.via(e.via, *e.layers, e.length)
            a, b = nname(e.a), nname(e.b)
            series(a, b, m["R"], m["L"], f"via #{e.via}", f"via{e.via}")
            for nd in (a, b):
                shunt_c[nd] = shunt_c.get(nd, 0.0) + m["C"] / 2
            vias.append(dict(chain="", kind="via", layer=f"{e.layers[0]}→{e.layers[1]}",
                             width_mm=self.board.vias[e.via].drill, length_mm=e.length, R_mOhm=m["R"] * 1e3,
                             L_nH=m["L"] * 1e9, C_pF=m["C"] * 1e12, reference=m["how"],
                             antipad_mm=m["antipad"], x0=e.p0[0], y0=e.p0[1]))

        # zones: meshed sheet over the return, reduced to the nodes that touch it
        zones = []
        zl = {}
        for e in g.edges:
            if e.kind == "zonelink":
                zl.setdefault(e.a, []).append((e.b, e.term))
        for zn, terms in zl.items():
            zi = g.nodes[zn].zone
            self.progress(f"  meshing zone {self.board.zones[zi].net} on {self.board.zones[zi].layer} "
                          f"({len(terms)} terminal(s))...")
            m = self.zone_model(zi, terms)
            names_t = [nname(nd) for nd, _ in terms]
            tag = f"zone {self.board.zones[zi].net} {self.board.zones[zi].layer}"
            for i, j, R, L in m["pairs"]:
                series(names_t[i], names_t[j], R, L, tag, f"pour{len(zones) + 1}_{names_t[i]}_{names_t[j]}")
            er, tand = st.near(self.board.zones[zi].layer)
            for nm, C in zip(names_t, m["C"]):
                shunt_c[nm] = shunt_c.get(nm, 0.0) + C
                if o.dielectric_g:
                    shunt_g[nm] = shunt_g.get(nm, 0.0) + 2 * math.pi * o.f_max * C * tand
            zones.append(dict(item=tag, zone=zi, layer=self.board.zones[zi].layer, terminals=len(terms),
                              area_mm2=m["area"], cells=m["cells"], pitch_mm=m["pitch"], R_mOhm=m["R_eff"] * 1e3,
                              L_nH=m["L_eff"] * 1e9, C_pF=sum(m["C"]) * 1e12, note=m["note"],
                              x0=g.nodes[zn].pos[0], y0=g.nodes[zn].pos[1]))
            warnings += m["warnings"]

        # user elements
        user = []
        for e in g.edges:
            if e.kind != "element":
                continue
            el = e.el
            a, b = nname(e.a), nname(e.b)
            nm = _ident(el.name) or "X"
            tag = f"element {el.name}"
            if el.model == "V":
                elements.append([ename("D", nm), "D", a, b, (el.Vf, el.R), tag])
            elif el.model == "C":
                elements.append([ename("C", nm), "C", a, b, el.C, tag])
            elif el.model == "L":
                elements.append([ename("L", nm), "L", a, b, max(el.L, 1e-15), tag])
            else:
                elements.append([ename("R", nm), "R", a, b, max(el.R, 1e-6), tag])
            user.append(dict(name=el.name, model={"R": "resistor", "V": "diode", "C": "capacitor",
                                                  "L": "inductor"}.get(el.model, el.model),
                             a=el.a, b=el.b, value=el.describe(), nodes=f"{a} – {b}"))

        pads = []
        if o.pad_c:
            used_nodes = {e.a for e in g.edges} | {e.b for e in g.edges} | set(g.ports)
            meshed = [self.board.zones[g.nodes[zn].zone] for zn in zl]
            for nd in used_nodes:
                node = g.nodes[nd]
                if node.kind != "pad":
                    continue
                C, how = self.pad_c(node.pad)
                f = _covered(node.pad, meshed) if not node.pad.plated else 0.0
                if f > 0.01:            # that part of the pad is in a pour's mesh, C included there
                    C *= 1 - f
                    how += f"; {100 * f:.0f} % inside the pour (counted there)"
                nm = nname(nd)
                shunt_c[nm] = shunt_c.get(nm, 0.0) + C
                pads.append(dict(pad=node.label, port=nm if nm.startswith("P") else "", C_pF=C * 1e12,
                                 reference=how, x=node.pos[0], y=node.pos[1]))

        for nm in list(shunt_c):
            if shunt_c[nm] > 0:
                elements.append([ename("C", nm), "C", nm, "REF", shunt_c[nm], ""])
        for nm, G in shunt_g.items():
            if G > 0:
                elements.append([ename("RG", nm), "R", nm, "REF", 1 / G, "dielectric loss"])

        ext = Extraction(g.net, list(port_ids), self.return_nets, o, segments, vias, pads, elements,
                         node_list, [], pieces, markers, warnings)
        ext.zones, ext.user = zones, user
        ext.uuids = sorted({self.board.tracks[e.track].uuid for e in g.edges if e.kind == "track"}
                           | {self.board.vias[e.via].uuid for e in g.edges if e.kind == "barrel"} - {""})

        # DC calibration against the field solution
        r_lad = two_port(ext, "P1", "P2", "R")
        if o.dc_solver and user:
            warnings.append("DC calibration of R is skipped when elements are added (the field solution would "
                            "include them); track R is ρl/(wt), zone R comes from its mesh.")
        elif o.dc_solver:
            self.progress("DC resistance: field solution between port 1 and 2...")
            try:
                r_true = dc_resistance(self.board, port_ids[0], port_ids[1], self.stack.st, o)
                k = r_true / r_lad if r_lad > 0 else 1.0
                for el in elements:
                    if el[1] == "R" and el[5] != "dielectric loss" and not el[5].startswith("element"):
                        el[4] *= k
                ext.summary["R_dc_field_mOhm"] = r_true * 1e3
                ext.summary["R_scale"] = k
            except Exception as ex:      # noqa: BLE001 - calibration is optional
                warnings.append(f"DC field solution failed ({ex}); R is ρl/(wt) only.")
        ext.summary["R_dc_formula_mOhm"] = r_lad * 1e3

        # shunt C to the return: the copper's, and the capacitors you added (kept apart in the tables)
        to_ref = [el for el in elements if el[1] == "C" and "REF" in el[2:4]]
        c_el = float(sum(el[4] for el in to_ref if el[5].startswith("element ")))
        c_cu = float(sum(el[4] for el in to_ref)) - c_el
        # totals per port pair (port 1 to each other port)
        for i, nd in enumerate(g.ports[1:], start=2):
            ks = g.shortest(g.ports[0], nd) or []
            Ltot = Ctot = Rtot = delay = length = 0.0
            z0s = []
            vias_on = 0
            for kk in ks:
                e = g.edges[kk]
                if e.kind == "track":
                    for l, key, lp, a, b in sub[kk]:
                        Rp, Lp, Cp, _ = per_len(e.layer, e.width, l, lp)
                        Ltot += Lp * l * 1e-3
                        Ctot += Cp * l * 1e-3
                        length += l
                        if lp:
                            delay += lp.delay * l * 1e-3
                            z0s.append(lp.z0)
                elif e.kind == "barrel":
                    m = self.via(e.via, *e.layers, e.length)
                    Ltot += m["L"]
                    Ctot += m["C"]
                    delay += math.sqrt(m["L"] * m["C"]) if m["C"] else 0.0
                    vias_on += 1
            # a path through a pour has no single length, Z0 or delay
            pours = len({g.edges[kk].a for kk in ks if g.edges[kk].kind == "zonelink"})
            if pours:
                length = delay = Ltot = float("nan")
                z0s = []
            # R and L of the whole network between the ports (zones, elements, parallel paths);
            # C: everything to the return
            ext.pairs.append(dict(pair=f"P1 → P{i}", a=port_ids[0], b=port_ids[i - 1], length_mm=length,
                                  vias=vias_on, pours=pours, R_mOhm=two_port(ext, "P1", f"P{i}", "R") * 1e3,
                                  L_nH=two_port(ext, "P1", f"P{i}", "L") * 1e9,
                                  C_pF=c_cu * 1e12, C_el_pF=c_el * 1e12,
                                  delay_ps=delay * 1e12,
                                  Z0_min=min(z0s) if z0s else float("nan"),
                                  Z0_max=max(z0s) if z0s else float("nan"),
                                  Z0_avg=math.sqrt(Ltot / Ctot) if Ctot else float("nan")))
        ext.summary.update(
            net=g.net, ports=", ".join(port_ids), return_nets=", ".join(self.return_nets) or "(none)",
            C_total_pF=c_cu * 1e12, C_elements_pF=c_el * 1e12,
            sections=sum(1 for el in elements if el[1] == "L"), field_solves=self.n_solves,
            f_max_Hz=o.f_max, seconds=time.time() - t0)
        if any(s["reference"] == "none in window" for s in segments):
            warnings.append(f"Some track has no return copper within ±{o.window:g} mm: its L is the partial "
                            "inductance of the trace alone and its C is ~0. Check the return nets or widen "
                            "the search window.")
        if markers:
            warnings.append(f"{len(markers)} impedance step(s) above 10 % along the path (marked on the board).")
        self.progress(f"Done in {time.time() - t0:.1f} s ({self.n_solves} field solves).")
        return ext


def _similar(a, b, tol=0.05):
    """Table rows close enough to show as one segment (same layer and width, Z0 within tol)."""
    if (a["layer"], a["width_mm"]) != (b["layer"], b["width_mm"]):
        return False
    za, zb = a["_z_first"], b["Z0"]
    if not np.isfinite(za) or not np.isfinite(zb):
        return not np.isfinite(za) and not np.isfinite(zb)
    return abs(zb - za) <= tol * za


def _merge(a, b):
    """Fold row b into row a, length-weighted."""
    la, lb = a["length_mm"], b["length_mm"]
    for k in ("L_nH_mm", "C_pF_mm", "R_mOhm_mm", "R_ac_mOhm_mm"):
        a[k] = (a[k] * la + b[k] * lb) / (la + lb)
    a["length_mm"] = la + lb
    if a["C_pF_mm"] > 0:
        a["Z0"] = math.sqrt(a["L_nH_mm"] / a["C_pF_mm"] * 1e3)
        a["eeff"] = (C_LIGHT * math.sqrt(a["L_nH_mm"] * 1e-6 * a["C_pF_mm"] * 1e-9)) ** 2
        a["delay_ps"] += b["delay_ps"]
    if lb > a["_ref_len"]:
        a["reference"], a["_ref_len"] = b["reference"], lb
    a["x1"], a["y1"] = b["x1"], b["y1"]


def _covered(pad, zones, n=16):
    """Fraction of an SMD pad's area inside any of the zones on its layer (sampled n x n)."""
    zs = [z for z in zones if z.layer == pad.layers[0]]
    if not zs:
        return 0.0
    r = math.hypot(*pad.size) / 2
    u = np.linspace(-r, r, n)
    X, Y = np.meshgrid(pad.pos[0] + u, pad.pos[1] + u)
    X, Y = X.ravel(), Y.ravel()
    m = np.asarray(pad.geom.inside(X, Y), bool)
    if not m.any():
        return 0.0
    hit = np.zeros(m.sum(), bool)
    for z in zs:
        hit |= np.asarray(z.geom.inside(X[m], Y[m]), bool)
    return float(hit.mean())


def _ident(text):
    """A SPICE-safe identifier from a pad or part name (L1.2 -> L1_2)."""
    return "".join(c if c.isalnum() else "_" for c in text)


def two_port(ext, a, b, what="R"):
    """Resistance ("R": inductors shorted, C open, diodes as their series R) or inductance ("L":
    resistors and diodes shorted, C open; the high-frequency loop L) between nodes a and b of the
    extracted network.  nan when a and b are not joined (e.g. a series capacitor)."""
    idx = {n: k for k, n in enumerate(ext.nodes)}
    if a not in idx or b not in idx:
        return float("nan")
    I, J, W, short = [], [], [], []
    for name, kind, p, q, v, tag in ext.elements:
        if kind == "C" or "REF" in (p, q):
            continue
        if what == "R":
            w = 1 / v if kind == "R" else (1 / max(v[1], 1e-6) if kind == "D" else None)
        else:
            w = 1 / v if kind == "L" else None
        I.append(idx[p])
        J.append(idx[q])
        W.append(w if w is not None else 0.0)
        short.append(w is None)
    if not I:
        return float("nan")
    W = np.array(W)
    big = 1e6 * (W.max() if W.max() > 0 else 1.0)
    W[np.array(short)] = big
    n = len(ext.nodes)
    off = sp.coo_matrix((W, (I, J)), shape=(n, n))
    off = (off + off.T).tocsr()
    _, lab = connected_components(off, directed=False)
    ia, ib = idx[a], idx[b]
    if lab[ia] != lab[ib]:
        return float("nan")
    comp = np.flatnonzero(lab == lab[ia])
    A = (sp.diags(np.asarray(off.sum(1)).ravel()) - off).tocsr()[comp][:, comp]
    pos = {k: i for i, k in enumerate(comp)}
    keep = [i for i in range(len(comp)) if i != pos[ib]]
    rhs = np.zeros(len(comp))
    rhs[pos[ia]] = 1.0
    V = LS.solve(A[keep][:, keep].tocsc(), rhs[keep])
    return float(V[keep.index(pos[ia])])


def ladder_r(ext, a, b):
    return two_port(ext, a, b, "R")


def dc_resistance(board, a, b, stackup, o):
    """Field-solved DC resistance [Ω] between two pads (1 A, thermal off)."""
    from .. import solver as S
    s = S.Settings(thermal=False, magnetic=False, error_estimate=False, auto_resistors=False,
                   rho20=o.rho20, alpha=o.alpha, T_cold=o.T, t_plate=o.t_plate)
    s.thickness = dict(stackup.cu_thickness)
    res = S.simulate(board, [S.Row("rlc", [a], [b], 1.0)], s, progress=lambda m: None, elements=[])
    return res.rows[0]["R_mOhm"] * 1e-3


def main_path(ext, a="P1", b="P2"):
    """Cells along the series path a -> b, for drawing: dicts with kind (R, L, C, D, pour, branch),
    shunt flag, name and value.  The path is the one of least resistance.  A pour's branches are
    drawn as one "pour" cell whose value is (R, L) between the terminals where the path enters and
    leaves it.  Shunt elements to REF follow the node they hang from (dielectric loss is left out);
    series elements leaving the path (stubs) appear as one shunt 'branch' cell per node."""
    adj = {}
    for el in ext.elements:
        nm, kind, p, q, v, tag = el
        if "REF" in (p, q):
            continue
        adj.setdefault(p, []).append((q, el))
        adj.setdefault(q, []).append((p, el))

    def weight(el):
        return 1e-12 + (el[4] if el[1] == "R" else el[4][1] if el[1] == "D" else 0.0)
    dist, prev, heap = {a: 0.0}, {a: None}, [(0.0, a)]
    while heap:
        du, u = heapq.heappop(heap)
        if u == b:
            break
        if du > dist[u]:
            continue
        for v, el in adj.get(u, []):
            dv = du + weight(el)
            if dv < dist.get(v, math.inf):
                dist[v], prev[v] = dv, (u, el)
                heapq.heappush(heap, (dv, v))
    if b not in prev:
        return []
    nodes, series = [b], []
    while nodes[-1] != a:
        u, el = prev[nodes[-1]]
        series.append(el)
        nodes.append(u)
    nodes, series = nodes[::-1], series[::-1]
    on_path = {id(el) for el in series}
    pour_tags = {el[5] for el in series if el[5].startswith("zone ")}
    shunts = {}
    for el in ext.elements:
        nm, kind, p, q, v, tag = el
        if "REF" in (p, q) and tag != "dielectric loss":
            nd = q if p == "REF" else p
            shunts.setdefault(nd, []).append(dict(kind=kind, shunt=True, name=nm, value=v, node=nd))
    cells = []

    def node_cells(nd):
        cells.extend(shunts.get(nd, []))
        off = [el for _, el in adj.get(nd, []) if id(el) not in on_path and el[5] not in pour_tags]
        if off:
            cells.append(dict(kind="branch", shunt=True, name=f"stub at {nd}", value=len(off), node=nd))
    node_cells(nodes[0])
    k = 0
    while k < len(series):
        el = series[k]
        if el[5] in pour_tags:
            j = k
            while j + 1 < len(series) and series[j + 1][5] == el[5]:
                j += 1
            sub = SimpleNamespace(nodes=ext.nodes, elements=[e for e in ext.elements if e[5] == el[5]])
            u, v = nodes[k], nodes[j + 1]
            _, net, layer = el[5].split(" ", 2)
            cells.append(dict(kind="pour", shunt=False, name=f"{net} pour", layer=layer, node=v,
                              value=(two_port(sub, u, v, "R"), two_port(sub, u, v, "L"))))
            node_cells(v)
            k = j + 1
            continue
        name = el[5][len("element "):] if el[5].startswith("element ") else el[0]
        cells.append(dict(kind=el[1], shunt=False, name=name, value=el[4], node=nodes[k + 1]))
        node_cells(nodes[k + 1])
        k += 1
    return cells
