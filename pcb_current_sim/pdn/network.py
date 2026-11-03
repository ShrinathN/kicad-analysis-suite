"""The rail's copper as a frequency-domain network over an ideal return, and its impedance.

The power net's copper is meshed as in the Switching Loop tool (`loop.peec.build`: cells with
cut-cell faces, narrow tracks as wires, via barrels, terminals at part pins), but its elements
are local, because the current returns in the plane right under it (v1: the return copper is the
ideal reference):
  - a bar (between two cells) has R_dc · skin(f) and the sheet inductance of copper over its
    return, μ0 h per square (h: dielectric to the nearest return copper above or below; with both,
    the two in parallel);
  - a narrow track has the inductance of a microstrip of its width over h (in air: L' = Z0,air / c);
  - a via barrel has the inductance of a via pair with the nearest return via (μ0 dz / 2π ln(s / r)),
    or its partial self inductance when there is none within 5 mm;
  - every node has the parallel-plate C of its cells to the return, ε0 εr A / h (tracks: the
    microstrip C'), with dielectric loss G = ω C tanδ.
Decaps (their library model + return-side mounting L) and the VRM (R + L) connect terminals to
the reference; loads are current injections.  Y(f) V = I is solved per frequency (sparse LU),
all loads at once.

Plane-pair resonances come out of the mesh (a distributed LC network); the mesh pitch should stay
below about λ/20 at f_max (warned).
"""
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .. import THREADS
from ..extract.rlc import Returns
from ..loop import peec as PE
from ..suite.library import part_from_dict

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
C_LIGHT = 299792458.0
GMIN = 1e-9                  # S to the reference at every node (keeps Y regular without a VRM)
VIA_PAIR_MM = 5.0            # a return via further than this does not pair with a via
NO_RETURN_H = 1.6e-3         # m, h for copper without return copper above or below (warned)


def air_z0(w, h):
    """Hammerstad: Z0 of a microstrip of width w at height h in air (any units, w / h)."""
    u = np.maximum(np.asarray(w, float) / np.asarray(h, float), 1e-6)
    return np.where(u <= 1, 60 * np.log(8 / u + u / 4),
                    120 * np.pi / (u + 1.393 + 0.667 * np.log(u + 1.444)))


def eps_eff(er, w, h):
    return (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 * np.asarray(h, float) / np.asarray(w, float))


def rail_window(board, nets, margin):
    boxes = [z.geom.bbox for z in board.zones if z.net in nets]
    boxes += [t.geom.bbox for t in board.tracks if t.net in nets]
    boxes += [p.geom.bbox for p in board.pads if p.net in nets and p.geom is not None]
    if not boxes:
        raise ValueError(f"No copper on {', '.join(nets)}.")
    b = np.array(boxes)
    return (b[:, 0].min() - margin, b[:, 1].min() - margin, b[:, 2].max() + margin, b[:, 3].max() + margin)


def rail_area(board, nets):
    a = sum(z.geom.area() for z in board.zones if z.net in nets)
    a += sum(t.width * t.length for t in board.tracks if t.net in nets) if board.tracks else 0.0
    return a


@dataclass
class Network:
    mesh: object
    terms: dict                  # terminal name -> node
    inc: object                  # (B x N) sparse incidence, +1 at a, -1 at b
    r_dc: np.ndarray             # Ω per branch
    t_skin: np.ndarray           # m, copper thickness for the skin effect (inf: none)
    L: np.ndarray                # H per branch
    C: np.ndarray                # F per node, to the reference
    tand: np.ndarray             # per node
    h: np.ndarray                # m, per node: height over the return (parallel of both sides)
    pitch: float                 # mm
    warnings: list = field(default_factory=list)

    @property
    def n_nodes(self):
        return len(self.C)

    def y_mesh(self, f):
        w = 2 * math.pi * f
        R = self.r_dc * np.where(np.isfinite(self.t_skin), PE.skin_factor(f, np.where(np.isfinite(self.t_skin),
                                                                                  self.t_skin, 1.0)), 1.0)
        yb = 1.0 / (R + 1j * w * self.L)
        Y = (self.inc.T @ sp.diags(yb) @ self.inc).tocsc()
        return Y + sp.diags(w * self.C * (self.tand + 1j) + GMIN)


def _barrel_L(mesh, ret, k):
    """Via barrel k: pair with the nearest return via / plated hole, else partial self L."""
    x, y = mesh.br_x[k], mesh.br_y[k]
    r = max(mesh.br_w[k], 1e-6)                 # m
    dz = max(mesh.br_len[k], 1e-6)              # m
    d = [math.hypot(px - x, py - y) for (px, py), _dr, _ls in ret.vias]
    s = min(d) * 1e-3 if d else math.inf
    if 2 * r < s < VIA_PAIR_MM * 1e-3:
        return MU0 * dz / (2 * math.pi) * math.log(s / r)
    return max(2e-7 * dz * (math.log(2 * dz / r) - 0.75), 2e-8 * dz)


def build(board, stackup, s, progress=lambda m: None):
    """The rail's network for setup s (terminals: 'C:<ref>' per decap, 'V' for the VRM, 'L:<name>'
    per load)."""
    nets = list(s.rail)
    window = rail_window(board, nets, s.margin)
    area = max((window[2] - window[0]) * (window[3] - window[1]), 1e-6)
    copper = max(rail_area(board, nets), 1.0)
    pitch = s.pitch or max(0.1, math.sqrt(min(copper, area) / max(s.cells, 100)))
    idx = {p.id: k for k, p in enumerate(board.pads)}
    terminals = {}
    for d in s.decaps:
        terminals[f"C:{d.ref}"] = [idx[p] for p in d.pads if p in idx]
    if s.vrm.pads:
        terminals["V"] = [idx[p] for p in s.vrm.pads if p in idx]
    for ld in s.loads:
        terminals[f"L:{ld.name}"] = [idx[p] for p in ld.pads if p in idx]
    terminals = {k: v for k, v in terminals.items() if v}
    progress(f"Meshing {', '.join(nets)} at {pitch:.2f} mm…")
    mesh = PE.build(board, stackup, nets, window, pitch, terminals)
    warnings = list(mesh.warnings)
    N, B = mesh.n_nodes, mesh.n_branches

    # height of every node over the return copper above / below it
    progress("Finding the return under the rail…")
    ret = Returns(board, s.returns)
    layers = mesh.layers
    rlay = [k for k, L in enumerate(layers) if ret.prims.get(L)]
    xs, ys = mesh.node_xy[:, 0], mesh.node_xy[:, 1]
    ha, hb = np.full(N, np.inf), np.full(N, np.inf)
    ea, eb, ta, tb = (np.ones(N) for _ in range(4))
    tand_of = {}
    for k in range(len(layers)):
        for k2 in rlay:
            if k2 == k:
                continue
            on = mesh.node_layer == k
            if not on.any():
                continue
            hit = np.zeros(N, bool)
            hit[on] = ret.inside(layers[k2], xs[on], ys[on])
            g = mesh.gap[k, k2]
            er = stackup.er_between(layers[k], layers[k2]) if stackup.dielectrics else 1.0
            i, j = sorted((k, k2))
            ds = stackup.dielectrics[i:j]
            tdl = sum(d.t * d.tand for d in ds) / max(sum(d.t for d in ds), 1e-12) if ds else 0.0
            tand_of[(k, k2)] = tdl
            if k2 < k:
                m = hit & (g < ha)
                ha[m], ea[m], ta[m] = g, er, tdl
            else:
                m = hit & (g < hb)
                hb[m], eb[m], tb[m] = g, er, tdl
    fa, fb = np.isfinite(ha), np.isfinite(hb)
    none = ~fa & ~fb
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(fa & fb, ha * hb / (ha + hb), np.where(fa, ha, hb))
    if none.any():
        h[none] = NO_RETURN_H
        warnings.append(f"{100 * none.mean():.0f} % of the rail's copper has no return copper above or below it; "
                        "its inductance there uses h = 1.6 mm and it has no C to the return (2-layer boards "
                        "without a plane under the rail are much less accurate).")
    # node C: cells' copper area over each side's return
    cpa = EPS0 * (np.where(fa, ea / np.where(fa, ha, 1), 0) + np.where(fb, eb / np.where(fb, hb, 1), 0))  # F/m²
    wsum = np.where(fa, ea / np.where(fa, ha, 1), 0) + np.where(fb, eb / np.where(fb, hb, 1), 0)
    with np.errstate(invalid="ignore"):
        tand = np.where(wsum > 0, (np.where(fa, ea / np.where(fa, ha, 1) * ta, 0)
                                   + np.where(fb, eb / np.where(fb, hb, 1) * tb, 0)) / np.maximum(wsum, 1e-30), 0)
    A = np.zeros(N)
    hm = mesh.grid.h * 1e-3
    for L in layers:
        cn, cf = mesh.cell_node[L], mesh.cell_f[L]
        m = cn >= 0
        A += np.bincount(cn[m], cf[m] * hm * hm, N)
    C = cpa * A

    # branches
    kind = mesh.br_kind
    a, b = mesh.br_a, mesh.br_b
    hbr = (h[a] + h[b]) / 2
    L = np.zeros(B)
    bars = kind <= 1
    L[bars] = MU0 * hbr[bars] * mesh.br_len[bars] / np.maximum(mesh.br_w[bars], 1e-9)
    wires = kind == 3
    if wires.any():
        z0 = air_z0(mesh.br_w[wires], hbr[wires])
        L[wires] = z0 / C_LIGHT * mesh.br_len[wires]
        er_w = (ea[a[wires]] + ea[b[wires]]) / 2
        cw = eps_eff(np.where(np.isfinite(ha[a[wires]]), er_w, (eb[a[wires]] + eb[b[wires]]) / 2),
                     mesh.br_w[wires], hbr[wires]) / (C_LIGHT * z0) * mesh.br_len[wires]
        cw = np.where(none[a[wires]] & none[b[wires]], 0.0, cw)
        np.add.at(C, a[wires], cw / 2)
        np.add.at(C, b[wires], cw / 2)
    for k in np.flatnonzero(kind == 2):
        L[k] = _barrel_L(mesh, ret, k)
    t_skin = np.where(kind == 2, np.inf, mesh.br_t)
    inc = sp.coo_matrix((np.r_[np.ones(B), -np.ones(B)], (np.r_[np.arange(B), np.arange(B)], np.r_[a, b])),
                        shape=(B, N)).tocsr()
    net = Network(mesh, dict(mesh.terminals), inc, mesh.br_R.copy(), t_skin, L, C, tand, h, pitch, warnings)
    return net


# ---- parts at the terminals ------------------------------------------------------------
def decap_l_ret(board, stackup, s, d):
    """Estimated return-side mounting L of decap d: its return pad's vias down (up) to the nearest
    return plane, paired with the decap's rail pad; 0 when the pad sits on the return plane's layer."""
    if d.l_ret >= 0:
        return d.l_ret, "as entered"
    ret = Returns(board, s.returns)
    pads = {p.id: p for p in board.pads}
    rp = [pads[x] for x in d.ret if x in pads]
    if not rp:
        return 0.0, "no return pad"
    p = rp[0]
    L0 = p.layers[0]
    planes = [L for L in board.copper_layers if L != L0 and ret.prims.get(L) and ret.inside(L, [p.pos[0]], [p.pos[1]])[0]]
    if p.drill and p.plated:
        return 0.0, "through-hole return pin (reaches every layer)"
    if not planes:
        return 0.0, "no return plane under it (return on its own layer)"
    near = min(planes, key=lambda L: stackup.gap(L0, L))
    dz = stackup.gap(L0, near)
    vias = [(math.dist(pos, p.pos), dr) for pos, dr, ls in ret.vias if near in ls]
    if not vias:
        return 0.0, f"no return via found to {near}"
    dist, dr = min(vias)
    rail = [pads[x] for x in d.pads if x in pads]
    sep = math.dist(rail[0].pos, p.pos) if rail else 1.0
    r = max(dr / 2, 0.05)
    s_m = max(dist + sep / 2, 2 * r) * 1e-3
    L = MU0 * dz / (2 * math.pi) * math.log(s_m / (r * 1e-3))
    if dist > 1.0:
        L += MU0 * dz * (dist - max(p.size) / 2) / max(min(p.size), 0.2)     # pad-to-via track over the plane
    return L, f"via {dist:.1f} mm from the pad to {near}"


def vrm_z(v, f):
    return v.r_out + 2j * math.pi * np.asarray(f) * v.l_out


def vrm_y(v, f, direction=1):
    """The regulator's admittance into the rail: its fitted control loop (rising-step fit by default),
    else 1 / (R + jωL)."""
    if v.kind == "loop" and v.model:
        from .regulator import RegulatorModel
        m = RegulatorModel(**v.model)
        lp = m.loop(direction)
        if lp is not None:
            return lp.Y(f)
    return 1.0 / vrm_z(v, f)


def decap_z(d, f, v_rail=0.0, T=25.0, l_ret=0.0):
    p = part_from_dict(d.part) if d.part else None
    if p is None:
        return np.full(np.shape(f), np.inf + 0j)
    return p.Z(f, T=T, vdc=v_rail) + 2j * math.pi * np.asarray(f) * l_ret


@dataclass
class Sweep:
    f: np.ndarray
    loads: list                  # names
    Z: np.ndarray                # (F, P, P) complex, impedance matrix between the loads
    I_dec: dict                  # decap ref -> (F, P) complex current per 1 A at each load
    I_vrm: np.ndarray = None     # (F, P)
    l_ret: dict = field(default_factory=dict)    # decap ref -> (L, how)


def sweep(net, board, stackup, s, f=None, decaps=None, vrm=True, progress=lambda m: None, l_ret=None, direction=1):
    """Z(f) between the loads with the given decaps (default: the enabled ones) and the VRM."""
    f = np.logspace(math.log10(s.f_min), math.log10(s.f_max), s.n_f) if f is None else np.asarray(f, float)
    loads = [ld.name for ld in s.loads if ld.enabled and f"L:{ld.name}" in net.terms]
    if not loads:
        raise ValueError("No load on the rail: add a load (the IC's power pins).")
    ld_nodes = [net.terms[f"L:{n}"] for n in loads]
    caps = [d for d in s.decaps if (d.enabled if decaps is None else d.ref in decaps) and f"C:{d.ref}" in net.terms]
    if l_ret is None:
        l_ret = {d.ref: decap_l_ret(board, stackup, s, d) for d in caps}
    cz = np.array([decap_z(d, f, s.target.v, s.T, l_ret[d.ref][0]) for d in caps]).reshape(len(caps), len(f))
    cn = np.array([net.terms[f"C:{d.ref}"] for d in caps], np.int64)
    use_v = vrm and s.vrm.enabled and "V" in net.terms
    vy = vrm_y(s.vrm, f, direction) if use_v else None
    N, P = net.n_nodes, len(loads)
    rhs = np.zeros((N, P), complex)
    for j, n in enumerate(ld_nodes):
        rhs[n, j] += 1.0
    Z = np.zeros((len(f), P, P), complex)
    I_dec = {d.ref: np.zeros((len(f), P), complex) for d in caps}
    I_v = np.zeros((len(f), P), complex)

    def one(k):
        Y = net.y_mesh(f[k])
        add = np.zeros(N, complex)
        if len(caps):
            yc = 1.0 / cz[:, k]
            np.add.at(add, cn, yc)
        if use_v:
            add[net.terms["V"]] += vy[k]
        Y = (Y + sp.diags(add)).tocsc()
        V = splu(Y, permc_spec="MMD_AT_PLUS_A").solve(rhs)
        Z[k] = V[ld_nodes, :]
        for q, d in enumerate(caps):
            I_dec[d.ref][k] = V[cn[q], :] / cz[q, k]
        if use_v:
            I_v[k] = V[net.terms["V"], :] * vy[k]
    progress(f"Solving {len(f)} frequencies on {N} nodes…")
    with ThreadPoolExecutor(max(1, THREADS)) as ex:
        list(ex.map(one, range(len(f))))
    return Sweep(f, loads, Z, I_dec, I_v if use_v else None, l_ret)
