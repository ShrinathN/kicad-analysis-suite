"""DC current-flow network on rasterised copper, with optional electro-thermal coupling.

Nodes: copper cells (per layer), one node per (plated hole, layer), one
super-node per excitation side and one per element pad.  Each row injects +I
at its source super-node and -I at its sink; all rows are solved together.
Lumped elements (resistors, user-defined resistances, constant drops) join
two pad electrodes, possibly on different nets.  Constant-drop elements are
one-way (diodes): solve() iterates their on/off states until consistent.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, dataclass, field, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import raster as rz
from . import elements as EL
from . import linsolve as LS

T_MELT = 1083.0
RESOLVE_K = 20.0        # pulse: full electrical re-solve once copper T has moved this much
CAP_STEP = 1.25         # pulse: dielectric capacity windows (growth factor per window)
PULSE_FRAMES = 20       # pulse: temperature snapshots kept for animation


class SimError(Exception):
    pass


G_SOURCE = 1e6          # S, Norton conductance of a voltage source (10 A -> 10 µV error)
KINDS = {"I": "Current path", "load": "Load (sink)", "V": "Voltage source"}


@dataclass
class Row:
    """One excitation.

    kind "I":    `current` A enters the copper at src and leaves at snk (the external source
                 closes the loop from snk back to src).
    kind "load": a load drawing `current` A out of the copper at src (+) and returning it at snk (-).
    kind "V":    a voltage source holding V(src) - V(snk) = `current` volts (+ at src).
    """
    name: str
    src: list
    snk: list
    current: float          # A, or V for kind "V" (field name kept for saved tables)
    kind: str = "I"

    @property
    def value(self):
        return self.current

    @property
    def unit(self):
        return "V" if self.kind == "V" else "A"

    @property
    def entry_exit(self):
        """(pads where current enters the copper, pads where it leaves) for this row's own loop."""
        return (self.snk, self.src) if self.kind == "load" else (self.src, self.snk)


@dataclass
class Settings:
    pitch: object = "auto"          # "auto" or mm
    rho20: float = 1.72e-8          # ohm*m at 20 C
    alpha: float = 0.00393          # 1/K
    t_plate: float = 25e-6          # barrel plating, m
    thickness: dict = field(default_factory=dict)   # layer -> m (overrides stackup)
    T_cold: float = 20.0            # copper temperature without thermal coupling, C
    thermal: bool = True
    T_amb: float = 25.0
    h_conv: float = 12.0            # W/m^2K per face (convection + linearised radiation); start value if h_auto
    h_auto: bool = True             # h from the local temperature: natural convection + radiation
    emissivity: float = 0.9         # surface emissivity for the radiation part of h_auto
    k_fr4: float = 0.3
    k_cu: float = 390.0
    max_iter: int = 10
    tol: float = 0.5                # K
    error_estimate: bool = True
    max_cells: int = 1_500_000
    auto_resistors: bool = True     # include resistor footprints reached from the rows
    crop: bool = True               # grid only around the simulated copper
    thermal_margin: float = 20.0    # mm of board kept around it for heat spreading
    thermal_pitch: float = 0.3      # mm, thermal cell (0 = as fine as the cell cap allows)
    magnetic: bool = True           # B-field planes and loop inductance
    b_height: float = 1.0           # mm above the top / below the bottom surface for the B maps
    pulse: bool = False             # transient thermal response to a rectangular pulse
    pulse_scale: float = 1.0        # pulse amplitude = table values x this
    pulse_ms: float = 10.0          # pulse length
    pulse_from_steady: bool = False  # start from the steady state at the table values (else ambient)
    pulse_steps: int = 50


_ALL, _LO, _HI = slice(None), slice(0, -1), slice(1, None)
STENCIL = {1: ((_ALL, _LO), (_ALL, _HI)),          # x: (j, i) - (j, i+1)
           0: ((_LO, _ALL), (_HI, _ALL))}          # y: (j, i) - (j+1, i)
OFFSET = {1: (0, 1), 0: (1, 0)}


def solve_spd(A, b):
    """Solve A x = b for an SPD A; b may be a vector or (n, k) columns (result has b's shape)."""
    return LS.solve(A, b)


class Network:
    def __init__(self, board, R, rows, settings, thick, elements=()):
        self.board, self.R, self.rows, self.s, self.t = board, R, rows, settings, thick
        g = R.grid
        self.layers = board.copper_layers
        # cell nodes
        self.idx, n = {}, 0
        for L in self.layers:
            lr = R.layers[L]
            m = (lr.f > 0) & (lr.net >= 0)
            a = np.full(g.shape, -1, np.int64)
            a[m] = np.arange(n, n + m.sum())
            n += int(m.sum())
            self.idx[L] = a
        self.n_cells = n
        # barrel nodes for plated holes of simulated nets
        self.barrels = []        # dicts: hole, nodes {L: node}, rings {L: cell node array}
        nid = {nm: k for k, nm in enumerate(R.nets)}
        for hole in R.holes:
            if hole["net"] not in nid or not hole["layers"]:
                continue
            nodes, rings = {}, {}
            for L in hole["layers"]:
                nodes[L] = n
                n += 1
                cells = rz.ring_cells(R, hole, L, nid[hole["net"]])
                rings[L] = (cells, self.idx[L].ravel()[cells])
            self.barrels.append(dict(hole=hole, nodes=nodes, rings=rings))
        self.hole_barrel = {(b["hole"]["kind"], b["hole"]["index"]): b for b in self.barrels}
        # electrode super-nodes: one per row side and one per element pad
        self.n = n
        self.super, self.elinks = [], []       # elinks: (super, node, g, pad_index, row, side)
        for r, row in enumerate(rows):
            pair = []
            for side, ids in (("src", row.src), ("snk", row.snk)):
                if not ids:
                    raise SimError(f"Row '{row.name}': no {side} pads given.")
                pair.append(self._electrode(ids, f"Row '{row.name}'", (r, side), self.elinks))
            self.super.append(tuple(pair))
        self.elements, self.enodes, self.xlinks = list(elements), [], []
        self.el_on = np.ones(len(self.elements), bool)      # conduction state (diodes may switch off)
        self.el_Vf = np.array([e.Vf if e.model == "V" else 0.0 for e in self.elements])
        self.el_oneway = np.array([e.model == "V" for e in self.elements], bool)
        for k, e in enumerate(self.elements):
            self.enodes.append(tuple(self._electrode([pid], f"Element {e.name}", (("el", k), pin), self.xlinks)
                                     for pin, pid in (("a", e.a), ("b", e.b))))

    def _electrode(self, ids, what, tag, out):
        """New super-node tied to the copper of pads `ids`; returns its node."""
        board, R, thick, rho20 = self.board, self.R, self.t, self.s.rho20
        sn = self.n
        self.n += 1
        for pid in ids:
            pads = [k for k, p in enumerate(board.pads) if p.id == pid]
            if not pads:
                raise SimError(f"{what}: pad {pid} not found on the board.")
            linked = 0
            for k in pads:
                p = board.pads[k]
                b = self.hole_barrel.get(("pad", k))
                if b is not None:
                    for L, node in b["nodes"].items():
                        out.append((sn, node, 1e4 * thick[L] / rho20, k, *tag))
                        linked += 1
                else:
                    for L in p.layers:
                        cells = R.pad_cells.get((k, L), np.zeros(0, np.int64))
                        nodes = self.idx[L].ravel()[cells]
                        nodes = nodes[nodes >= 0]
                        for node in nodes:
                            out.append((sn, int(node), 100 * thick[L] / rho20, k, *tag))
                        linked += nodes.size
            if not linked:
                raise SimError(f"{what}: pad {pid} has no copper on the simulation grid.")
        return sn

    # ------------------------------------------------------------------
    def rho(self, T):
        return self.s.rho20 * (1 + self.s.alpha * (T - 20.0))

    def solve(self, T=None):
        """Operating point. T: dict layer -> (ny, nx) temperature array, or None for T_cold.
        Diode states start from the previous solve and are flipped until no conducting diode
        carries reverse current and no blocking one is forward biased beyond Vf."""
        base = np.zeros(self.n)
        for (src, snk), row in zip(self.super, self.rows):
            v = row.value
            if row.kind == "V":
                base[src] += v * G_SOURCE
                base[snk] -= v * G_SOURCE
            elif row.kind == "load":
                base[src] -= v
                base[snk] += v
            else:
                base[src] += v
                base[snk] -= v
        pairs = [(row, src, snk) for (src, snk), row in zip(self.super, self.rows)]
        seen = set()
        while True:
            asm = self._assemble(T)
            rhs = base.copy()
            for (a, b), G, vf in zip(self.enodes, asm["G_el"], self.el_Vf):
                rhs[a] += G * vf
                rhs[b] -= G * vf
            try:
                V = self._solve_system(asm["A"], asm["off"], rhs[:, None], pairs)[:, 0]
            except SimError as e:
                off = [el.name for el, on in zip(self.elements, self.el_on) if not on]
                if off:
                    raise SimError(f"{e} Blocking diode element(s): {', '.join(off)} (check their direction: "
                                   "pad A is the anode).") from None
                raise
            if not self.el_oneway.any():
                break
            seen.add(self.el_on.tobytes())
            a = np.array([p for p, _ in self.enodes], np.int64)
            b = np.array([q for _, q in self.enodes], np.int64)
            fwd = V[a] - V[b] - self.el_Vf                    # NaN where a side is unexcited
            new = self.el_on.copy()
            new[self.el_oneway & self.el_on & ~(fwd >= -1e-12)] = False
            new[self.el_oneway & ~self.el_on & (fwd > 1e-12)] = True
            if (new == self.el_on).all():
                break
            self.el_on = new
            if new.tobytes() in seen:
                self.R.warnings.append("Diode element states did not settle (oscillating); result may be off.")
                break
        return self._post(V, T, asm)

    def solve_unit_rows(self, T=None):
        """One light Solution per row carrying 1 A around that row's own loop, all other
        sources off (current sources open, voltage sources shorted).  None where the row
        has no closed path by itself (e.g. a voltage source whose loads are current sinks)."""
        asm = self._assemble(T)
        out = [None] * len(self.rows)
        groups = {}                 # matrix variant -> row indices (V rows need their own source removed)
        for r, row in enumerate(self.rows):
            groups.setdefault(r if row.kind == "V" else -1, []).append(r)
        for key, rs in groups.items():
            A, off = asm["A"], asm["off"]
            if key >= 0:
                src, snk = self.super[key]
                gs = asm["Gs"][key]
                D = sp.coo_matrix(([gs, gs, -gs, -gs], ([src, snk, src, snk], [src, snk, snk, src])),
                                  shape=A.shape).tocsr()
                A = (A - D).tocsr()
                off = (off - sp.coo_matrix(([gs, gs], ([src, snk], [snk, src])), shape=A.shape)).tocsr()
                off.eliminate_zeros()
            rhs = np.zeros((self.n, len(rs)))
            pairs = []
            for c, r in enumerate(rs):
                src, snk = self.super[r]
                sgn = -1.0 if self.rows[r].kind == "load" else 1.0
                rhs[src, c], rhs[snk, c] = sgn, -sgn
                pairs.append((self.rows[r], src, snk))
            Vs = self._solve_system(A, off, rhs, pairs, strict=False)
            for c, r in enumerate(rs):
                if Vs is not None and np.isfinite(Vs[self.super[r][0], c]):
                    out[r] = self._post(Vs[:, c], T, dict(asm, Vf_el=0 * self.el_Vf), light=True)
        return out

    # ------------------------------------------------------------------
    def _assemble(self, T):
        R, g, s = self.R, self.R.grid, self.s
        A_i, A_j, A_g = [], [], []
        links = {}

        def T_at(L):
            return T[L] if T is not None else np.full(g.shape, s.T_cold, np.float32)

        for L in self.layers:
            lr, idx, tL = R.layers[L], self.idx[L], self.t[L]
            rho_c = self.rho(T_at(L))
            # cut-cell finite volume: each link conducts through the copper part of the shared face
            for ax in (1, 0):
                sa, sb = STENCIL[ax]
                a, b = idx[sa], idx[sb]
                ap = lr.apx[:, 1:-1] if ax == 1 else lr.apy[1:-1, :]
                m = (a >= 0) & (b >= 0) & (lr.net[sa] == lr.net[sb]) & (ap > 0)
                rho_ab = 0.5 * (rho_c[sa] + rho_c[sb])[m]
                gl = tL / rho_ab * ap[m]
                links[(L, ax)] = (m, a[m], b[m], gl, rho_ab)
                A_i.append(a[m]); A_j.append(b[m]); A_g.append(gl)
        ring_links = []
        barrel_links = []
        for bi, b in enumerate(self.barrels):
            hole = b["hole"]
            for L, (cells, nodes) in b["rings"].items():
                ok = nodes >= 0
                if not ok.any():
                    continue
                rho_c = self.rho(T_at(L).ravel()[cells[ok]])
                gl = 2 * self.t[L] / rho_c
                ring_links.append((bi, L, cells[ok], nodes[ok], gl))
                A_i.append(np.full(ok.sum(), b["nodes"][L])); A_j.append(nodes[ok]); A_g.append(gl)
            Ls = list(b["nodes"])
            jj, ii = g.cell(*hole["pos"])
            jj, ii = min(max(jj, 0), g.ny - 1), min(max(ii, 0), g.nx - 1)
            for L1, L2 in zip(Ls[:-1], Ls[1:]):
                k1, k2 = self.layers.index(L1), self.layers.index(L2)
                Lm = sum(self.board.gaps[k1:k2]) + (self.t[L1] + self.t[L2]) / 2
                Tm = 0.5 * (T_at(L1)[jj, ii] + T_at(L2)[jj, ii])
                d = hole["drill"] * 1e-3
                G = math.pi * s.t_plate * (d + s.t_plate) / (self.rho(Tm) * Lm)
                barrel_links.append((bi, L1, L2, b["nodes"][L1], b["nodes"][L2], G))
                A_i.append(np.array([b["nodes"][L1]])); A_j.append(np.array([b["nodes"][L2]]))
                A_g.append(np.array([G]))
        if self.elinks or self.xlinks:
            e = np.array([(sn, nd, gg) for sn, nd, gg, *_ in self.elinks + self.xlinks])
            A_i.append(e[:, 0].astype(np.int64)); A_j.append(e[:, 1].astype(np.int64)); A_g.append(e[:, 2])
        # DC limit of reactive elements: a capacitor is open, an inductor an ideal link
        G_el = np.array([0.0 if (not on or e.model == "C") else 1 / (EL.R_MIN if e.model == "L" else max(e.R, EL.R_MIN))
                         for e, on in zip(self.elements, self.el_on)])
        on = np.flatnonzero(G_el)
        if on.size:
            A_i.append(np.array([self.enodes[k][0] for k in on])); A_j.append(np.array([self.enodes[k][1] for k in on]))
            A_g.append(G_el[on])
        # voltage sources: Norton equivalent, a stiff conductance between the two terminals
        Gs = [G_SOURCE if row.kind == "V" else 0.0 for row in self.rows]
        for (src, snk), gs in zip(self.super, Gs):
            if gs:
                A_i.append(np.array([src])); A_j.append(np.array([snk])); A_g.append(np.array([gs]))
        I = np.concatenate(A_i).astype(np.int64)
        J = np.concatenate(A_j).astype(np.int64)
        G = np.concatenate(A_g).astype(float)
        n = self.n
        off = sp.coo_matrix((G, (I, J)), shape=(n, n))
        off = (off + off.T).tocsr()
        A = (sp.diags(np.asarray(off.sum(1)).ravel()) - off).tocsr()
        return dict(A=A, off=off, links=links, ring_links=ring_links, barrel_links=barrel_links,
                    G_el=G_el, Vf_el=self.el_Vf, Gs=Gs)

    def _solve_system(self, A, off, rhs, pairs, strict=True):
        """Solve A V = rhs (columns) with one grounded node per excited component.
        pairs: (row, src, snk) that must be connected; strict raises otherwise."""
        ncomp, lab = connected_components(off, directed=False)
        ground = {}
        for row, src, snk in pairs:
            if lab[src] != lab[snk]:
                if not strict:
                    continue
                raise SimError(f"Row '{row.name}': source and sink are not connected by copper or "
                               "resistance elements (open circuit; save and refill zones in KiCad, or add "
                               "the part in the path as an element).")
            ground.setdefault(lab[snk], snk)
        if not ground:
            return None
        live = np.isin(lab, list(ground))
        keep = live.copy()
        keep[list(ground.values())] = False
        V = np.full(rhs.shape, np.nan)
        V[list(ground.values())] = 0.0
        Ak = A[keep][:, keep]
        V[keep] = solve_spd(Ak, rhs[keep])
        return V

    def _post(self, V, T, asm, light=False):
        """Fields and per-item currents from node voltages. light: currents only (no q/V maps)."""
        g, s = self.R.grid, self.s
        h_m = g.h * 1e-3
        links, ring_links, barrel_links = asm["links"], asm["ring_links"], asm["barrel_links"]
        sol = Solution(V=V, T=T)
        for L in self.layers:
            idx = self.idx[L]
            Vg = np.full(g.shape, np.nan)
            m = idx >= 0
            Vg[m] = V[idx[m]]
            q = None if light else np.zeros(g.shape)
            E = {}
            for ax in (1, 0):
                mk, a, b, gl, rho_ab = links[(L, ax)]
                dV = V[a] - V[b]
                ok = ~np.isnan(dV)
                dV = np.where(ok, dV, 0)
                face = np.zeros(mk.shape)
                face[mk] = dV / (rho_ab * h_m)            # A/m^2 across the face
                E[ax] = face
                if not light:
                    P = gl * dV * dV
                    ra, rb = np.nonzero(mk)
                    dj, di = OFFSET[ax]
                    np.add.at(q, (ra, rb), P / 2); np.add.at(q, (ra + dj, rb + di), P / 2)
            Jx = np.zeros(g.shape); Jy = np.zeros(g.shape)
            Jx[:, :-1] += E[1]; Jx[:, 1:] += E[1]
            Jy[:-1, :] += E[0]; Jy[1:, :] += E[0]
            Jx *= 0.5e-6; Jy *= 0.5e-6                     # A/mm^2
            dead = np.isnan(Vg)
            Jx[dead] = np.nan; Jy[dead] = np.nan
            sol.Jx[L], sol.Jy[L] = Jx, Jy
            if not light:
                sol.V_layers[L] = Vg
                sol.q[L] = q
        for bi, L, cells, nodes, gl in ring_links:
            dV = np.nan_to_num(V[self.barrels[bi]["nodes"][L]] - V[nodes])
            if not light:
                np.add.at(sol.q[L].ravel(), cells, gl * dV * dV)
            sol.ring_current.setdefault(bi, {})[L] = float(np.sum(gl * dV))
        for bi, L1, L2, n1, n2, Gb in barrel_links:
            dV = np.nan_to_num(V[n1] - V[n2])
            sol.barrel_links.append((bi, L1, L2, Gb * dV, Gb * dV * dV, dV))
        for (src, snk), row, gs in zip(self.super, self.rows, asm["Gs"]):
            dV = float(V[src] - V[snk])
            sol.row_dV.append(dV)
            sol.row_I.append(gs * (row.value - dV) if row.kind == "V" else row.value)
        if not hasattr(self, "_pk"):
            links = self.elinks + self.xlinks
            keys = [(k, r, side) for _, _, _, k, r, side in links]
            self._pk = list(dict.fromkeys(keys))
            pos = {key: i for i, key in enumerate(self._pk)}
            self._pidx = np.array([pos[key] for key in keys], np.int64)
            self._psn = np.array([l[0] for l in links], np.int64)
            self._pnd = np.array([l[1] for l in links], np.int64)
            self._pg = np.array([l[2] for l in links], float)
        if self._pk:
            I = np.bincount(self._pidx, self._pg * np.nan_to_num(V[self._psn] - V[self._pnd]), len(self._pk))
            sol.pad_current = dict(zip(self._pk, I.tolist()))
        for (a, b), G, vf in zip(self.enodes, asm["G_el"], asm["Vf_el"]):
            dV = float(np.nan_to_num(V[a] - V[b]))
            sol.element_I.append(G * (dV - vf))
            sol.element_dV.append(dV)
        return sol


@dataclass
class Solution:
    V: np.ndarray
    T: object
    V_layers: dict = field(default_factory=dict)
    Jx: dict = field(default_factory=dict)
    Jy: dict = field(default_factory=dict)
    q: dict = field(default_factory=dict)          # W per cell
    row_dV: list = field(default_factory=list)
    row_I: list = field(default_factory=list)          # A delivered by each row's source
    pad_current: dict = field(default_factory=dict)
    ring_current: dict = field(default_factory=dict)
    barrel_links: list = field(default_factory=list)
    element_I: list = field(default_factory=list)      # A, pad a -> pad b
    element_dV: list = field(default_factory=list)     # V_a - V_b

    def J(self, L):
        return np.hypot(self.Jx[L], self.Jy[L])

    def barrel_heat(self, net):
        """[(x, y, layer, W)] point heat sources from barrel resistance."""
        out = []
        for bi, L1, L2, I, P, dV in self.barrel_links:
            pos = net.barrels[bi]["hole"]["pos"]
            out += [(pos[0], pos[1], L1, P / 2), (pos[0], pos[1], L2, P / 2)]
        return out

    def element_heat(self, net):
        """[(x, y, layer, W, pad geometry)]: each element's I^2 R, half spread over each pad's
        area on the component side."""
        board, out = net.board, []
        side = {fp.ref: fp.layer for fp in board.footprints}
        for e, I, dV in zip(net.elements, self.element_I, self.element_dV):
            if not e.heats:
                continue
            for pid in (e.a, e.b):
                p = board.pads_by_id(pid)[0]
                L = side.get(p.ref, p.layers[0])
                if L not in p.layers:
                    L = p.layers[0]
                out.append((p.pos[0], p.pos[1], L, abs(I * dV) / 2, p.geom))
        return out


# ---------------------------------------------------------------------------
def nets_of_rows(board, rows):
    nets = []
    for row in rows:
        rn = set()
        for side in (row.src, row.snk):
            sn = set()
            for pid in side:
                ps = board.pads_by_id(pid)
                if not ps:
                    raise SimError(f"Row '{row.name}': pad {pid} not found on the board.")
                for p in ps:
                    if not p.net:
                        raise SimError(f"Row '{row.name}': pad {pid} has no net.")
                    sn.add(p.net)
            if len(sn) > 1:
                raise SimError(f"Row '{row.name}': pads on one side are on different nets "
                               f"({', '.join(sorted(sn))}); they would be shorted together.")
            rn |= sn
        nets += [n for n in sorted(rn) if n not in nets]
    return nets


def thermal_box(board, box, s, margin=None):
    """Thermal extent: the electrical crop plus thermal_margin of board (None = whole grid)."""
    if box is None:
        return None
    m, (bx0, by0, bx1, by1) = s.thermal_margin if margin is None else margin, board.bbox()
    return (max(box[0] - m, bx0), max(box[1] - m, by0), min(box[2] + m, bx1), min(box[3] + m, by1))


def run_pulse(net, tm, thick, s, base, progress):
    """Rectangular pulse of all rows x pulse_scale for pulse_ms (implicit Euler, coupled rho(T)).

    Between electrical re-solves the current distribution is held and each cell's heat is
    rescaled by rho(T)/rho(T_solved); a full re-solve happens once T has moved by > RESOLVE_K.
    The dielectric heat capacity grows with the thermal penetration depth 0.886 sqrt(alpha t)
    (energy-conserving re-lumping, exact for a constant flux into a semi-infinite slab): the growth
    acts as a heat sink C'(t) phi every step, and the stepping matrix uses the capacity at the start
    of windows over which it grows CAP_STEP x (conservative: fusing times come out a few % short).

    base: (steady ThermalModel, steady Solution) to start from, or None for ambient.  The state is
    theta0 (the starting field) + phi, and only the heat in excess of the starting heat drives phi,
    so an unchanged excitation leaves the temperature exactly where it was."""
    rows0 = net.rows
    net.rows = [replace(r, current=r.current * s.pulse_scale) for r in rows0]
    tp = s.pulse_ms * 1e-3
    n = max(5, int(s.pulse_steps))
    dt = tp / n
    progress(f"Pulse: {s.pulse_scale:g} x table for {s.pulse_ms:g} ms ({n} steps, "
             f"from {'steady state' if base else 'ambient'}, {tm.n:,} thermal nodes)...")
    try:
        if base is not None:
            tm0, sol0 = base
            theta0 = tm.theta_from(tm0.tg, tm0.T)
            T_f = sol0.T
            r0 = tm.heat(sol0.q, sol0.barrel_heat(net) + sol0.element_heat(net))
        else:
            theta0 = np.zeros(tm.n)
            T_f = {L: np.full(tm.grid.shape, s.T_amb, np.float32) for L in tm.layers}
            r0 = np.zeros(tm.n)
        phi = np.zeros(tm.n)
        t_win = dt                                    # current capacity window starts here
        sol = net.solve(T_f)
        T_sol, solves = T_f, 1
        cu = {L: net.R.layers[L].f > 0 for L in tm.layers}
        every = max(1, n // PULSE_FRAMES)
        tm.to_fine(theta0, store=False)
        frames = [(0.0, {L: Tc.astype(np.float32) for L, Tc in tm.last_coarse.items()})]
        t_ms, T_max, fused = [0.0], [max(float(np.nanmax(T_f[L])) for L in tm.layers)], None
        # energy balance with growing capacity:  d(C phi)/dt = heat - A phi,  i.e.
        #   C_w dphi/dt + A phi = heat - C'(t) phi
        # C_w (the stepping matrix) is the capacity at the start of the current window (conservative:
        # it is <= C(t)); the growth sink C'(t) phi is applied every step, so T(t) stays smooth.
        C = tm.capacity(thick, dt)
        C_prev = tm.capacity(thick, 0.0)
        tm.begin_transient(C, dt)
        for k in range(1, n + 1):
            C_now = tm.capacity(thick, k * dt)
            if k * dt > t_win * CAP_STEP ** 2:        # capacity ~ sqrt(t): CAP_STEP x per CAP_STEP^2 in time
                t_win = k * dt
                C = C_now
                tm.begin_transient(C, dt)
            sink = (C_now - C_prev) / dt * phi
            C_prev = C_now
            q = {L: sol.q[L] * net.rho(T_f[L]) / net.rho(T_sol[L]) for L in tm.layers}
            phi = tm.step(phi, tm.heat(q, sol.barrel_heat(net) + sol.element_heat(net)) - r0 - sink)
            T_f = tm.to_fine(theta0 + phi, store=False)
            coarse = tm.last_coarse
            tmax, where = -1e9, None
            for L, Tc in coarse.items():
                j, i = np.unravel_index(np.nanargmax(Tc), Tc.shape)
                if Tc[j, i] > tmax:
                    tmax, where = float(Tc[j, i]), (L, tm.tg.x0 + (i + 0.5) * tm.tg.h, tm.tg.y0 + (j + 0.5) * tm.tg.h)
            t_ms.append(k * dt * 1e3)
            T_max.append(tmax)
            if k % every == 0 or k == n or tmax > T_MELT:
                frames.append((k * dt * 1e3, {L: Tc.astype(np.float32) for L, Tc in coarse.items()}))
            if tmax > T_MELT:
                t0, T0 = t_ms[-2], T_max[-2]
                tf = t0 + (T_MELT - T0) / (tmax - T0) * (t_ms[-1] - t0) if tmax > T0 else t_ms[-1]
                fused = dict(t_ms=tf, layer=where[0], x=where[1], y=where[2])
                progress(f"  copper reaches its melting point after {tf:.3g} ms")
                break
            if max(float(np.max(np.abs(T_f[L] - T_sol[L])[cu[L]], initial=0)) for L in tm.layers) > RESOLVE_K:
                sol = net.solve(T_f)
                T_sol = T_f
                solves += 1
        progress(f"  T max at the end: {T_max[-1]:.1f} °C ({solves} electrical solves)")
    finally:
        net.rows = rows0
    return dict(scale=s.pulse_scale, ms=s.pulse_ms, from_steady=base is not None, t_ms=t_ms,
                T_max=T_max, fused=fused, T_end_max=T_max[-1], where=dict(layer=where[0], x=where[1], y=where[2]),
                coarse=coarse, fine=T_f, frames=frames)


_CACHE = {}     # kind -> (key, board, value): the last raster / thermal model, reused while nothing changed


def cached(kind, board, key, make):
    """(value, hit): reuse the last value of this kind if it was made for the same board object
    and key, else make() it.  Rasters and thermal models depend only on the board and settings,
    not on the excitation, so re-running with new currents skips their setup."""
    hit = _CACHE.get(kind)
    if hit is not None and hit[1] is board and hit[0] == key:
        return hit[2], True
    value = make()
    _CACHE[kind] = (key, board, value)
    return value, False


def simulate(board, rows, settings=None, progress=print, elements=None):
    """elements: list of elements.Element, or None to auto-detect resistors (if enabled)."""
    from .thermal import ThermalModel
    from .results import build_result
    s = settings or Settings()
    t0 = time.time()
    if not rows:
        raise SimError("No excitation rows.")
    warnings = []
    if elements is None:
        elements, warnings = EL.detect(board) if s.auto_resistors else ([], [])
    nets, els, w = EL.active(board, nets_of_rows(board, rows), elements)
    warnings += w
    if els:
        progress(f"{len(els)} resistance element(s): " + ", ".join(e.name for e in els))
    thick = {L: s.thickness.get(L, board.cu_thickness[L]) for L in board.copper_layers}
    box = rz.copper_bbox(board, nets) if s.crop else None
    pitch = rz.auto_pitch(board, nets, s.max_cells, bbox=box) if s.pitch in (None, "auto") else float(s.pitch)
    grid = rz.make_grid(board, pitch, bbox=box)
    progress(f"Rasterising {len(nets)} net(s) at {pitch * 1000:g} µm ({grid.nx} x {grid.ny} x "
             f"{len(board.copper_layers)} layers)...")
    R, hit = cached("raster", board, (tuple(nets), astuple(grid)), lambda: rz.rasterise(board, nets, grid))
    R = replace(R, warnings=list(R.warnings) + warnings)       # the cached raster stays untouched
    net = Network(board, R, rows, s, thick, els)

    # independent work runs in threads (the sparse factorisations release the GIL)
    pool = ThreadPoolExecutor(max_workers=3)
    th_keys = (tuple(sorted(thick.items())), s.k_cu, s.k_fr4, s.h_conv, s.t_plate)

    def thermal_model(kind, bbox, tpitch):
        tm, hit = cached(kind, board, (astuple(grid), bbox, tpitch) + th_keys,
                         lambda: ThermalModel(board, grid, thick, s, bbox=bbox, pitch=tpitch))
        return tm.reset(s)
    f_tm = pool.submit(thermal_model, "thermal", thermal_box(board, box, s), s.thermal_pitch or None) \
        if s.thermal else None
    reach = min(s.thermal_margin, max(2.0, 3e3 * math.sqrt(1.1e-4 * s.pulse_ms * 1e-3)))
    # a short pulse heats a small region (heat spreads ~sqrt(alpha_cu t)): tighter, finer domain
    f_pulse = pool.submit(thermal_model, "pulse", thermal_box(board, box, s, reach) if box else None,
                          min(s.thermal_pitch, reach / 10) if s.thermal_pitch else None) if s.pulse else None

    def coarse_solve():
        g2 = rz.Grid(grid.x0, grid.y0, grid.h * 2, -(-grid.nx // 2), -(-grid.ny // 2))
        try:
            R2, _ = cached("raster2", board, (tuple(nets), astuple(g2)), lambda: rz.rasterise(board, nets, g2))
            return Network(board, replace(R2, warnings=[]), rows, s, thick, els).solve(None), None
        except SimError as e:
            return None, str(e)
    f_coarse = pool.submit(coarse_solve) if s.error_estimate else None

    progress(f"Solving {net.n:,} nodes...")
    try:
        cold = net.solve(None)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    sol, tm, iters, converged, runaway = cold, None, 0, True, False
    if s.thermal:
        tm = f_tm.result()
        progress(f"Electro-thermal loop ({tm.n:,} thermal nodes)...")
        T = {L: np.full(grid.shape, s.T_amb, np.float32) for L in board.copper_layers}
        sol = net.solve(T)
        converged = False
        for iters in range(1, s.max_iter + 1):
            Tn = tm.solve(sol.q, sol.barrel_heat(net) + sol.element_heat(net))
            dT = max(float(np.max(np.abs(Tn[L] - T[L])[R.layers[L].f > 0], initial=0))
                     for L in board.copper_layers)
            T = Tn
            tmax = max(float(T[L].max()) for L in T)
            progress(f"  iteration {iters}: max dT change {dT:.2f} K, T max {tmax:.1f} °C")
            if tmax > T_MELT:
                runaway = True
                break
            sol = net.solve(T)
            if dT < s.tol:
                converged = True
                break
    pulse = None
    if s.pulse:
        tmp = f_pulse.result()
        base = (tm, sol) if (s.pulse_from_steady and tm is not None) else None
        pulse = run_pulse(net, tmp, thick, s, base, progress)
        pulse["tgrid"] = tmp.tg
    coarse = None
    if s.error_estimate:
        progress("Error estimate (solved at 2x pitch alongside)...")
        coarse, err = f_coarse.result()
        if err:
            progress(f"  coarse solve failed ({err}); skipping error estimate")
    mag = None
    if s.magnetic:
        from . import magnetic as MG
        progress("Magnetic field and loop inductance...")
        mg = MG.MomentGrid(grid)
        zL, zb = MG.layer_z(board, thick)
        f_b = pool.submit(MG.b_planes, sol, net, mg, thick, zL, zb, s.b_height)      # alongside the L solve
        units = net.solve_unit_rows(sol.T)
        Lmat, closure = MG.inductance_matrix(net, units, mg, thick, zL)
        mag = dict(grid=mg.g, B=f_b.result(), L=Lmat, closure=closure)
    pool.shutdown(wait=False)
    res = build_result(board, R, net, sol, cold, coarse, tm, rows, s, thick,
                       dict(iterations=iters, converged=converged, runaway=runaway,
                            seconds=time.time() - t0, pulse=pulse), mag)
    progress(f"Done in {time.time() - t0:.1f} s.")
    return res
