"""PDN layout checks: what in the layout raises the rail's impedance, with where it is.

From the analysed network and the decap table:
  - rail copper with no return copper above or below it (regions, with area and position);
  - voids, slots and splits in the return under the rail: regions where the nearest return copper is
    much farther away than elsewhere on that layer (the rail's inductance there grows with h);
  - decaps whose return-side mounting L is large (a via far from the pad, no via to the plane),
    or larger than the capacitor's own ESL;
  - loads with no decap close by.
(Which decaps could go is the What-if analysis: current shares alone don't tell, since above the
target band and near anti-resonances every capacitor carries a large share.)
Each finding is an `Issue(level, kind, text, refs, at)`: refs for cross-probing, at = (x, y) mm.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

from . import network as PN

VOID_RATIO = 1.8             # h this many times the layer's typical h: the plane under it is missing
REGION_MM = 2.0              # regions smaller than this (and 3 cells) are antipads around vias: ignored
L_MOUNT_WARN = 0.5e-9        # H, return-side mounting L worth a warning
NEAR_LOAD_MM = 5.0           # a load wants a decap within this distance
SHARE_MIN = 0.03             # report a region adding at least this share of the copper's inductive energy
MAX_REGIONS = 6              # per kind


@dataclass
class Issue:
    level: str                   # "warn" | "info"
    kind: str                    # "no return" | "return void" | "mounting" | "no decap near"
    text: str
    refs: list = field(default_factory=list)
    at: tuple = None             # (x, y) mm, board coordinates


def _regions(net, mask):
    """Connected groups of masked nodes (through the mesh's branches): list of node index arrays."""
    if not mask.any():
        return []
    a, b = net.mesh.br_a, net.mesh.br_b
    keep = mask[a] & mask[b]
    N = net.n_nodes
    g = sp.coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    out = []
    for c in np.unique(lab[mask]):
        out.append(np.flatnonzero(mask & (lab == c)))
    return out


def _describe(net, nodes):
    xy = net.mesh.node_xy[nodes]
    p = net.pitch
    w = xy[:, 0].max() - xy[:, 0].min() + p
    h = xy[:, 1].max() - xy[:, 1].min() + p
    area = float(net.area[nodes].sum()) * 1e6 if net.area is not None else w * h
    k = nodes[np.argmin(np.hypot(*(xy - xy.mean(0)).T))]          # a node of the region near its middle
    at = tuple(float(v) for v in net.mesh.node_xy[k])
    layers = sorted({net.mesh.layers[i] for i in net.mesh.node_layer[nodes]})
    return w, h, area, at, layers


def _big(net, w, h, n):
    return max(w, h) >= max(REGION_MM, 3 * net.pitch) and n >= 3


def regions(net):
    """(no-return regions, void regions) as lists of node arrays, large enough to matter."""
    no_ret = net.no_ret if net.no_ret is not None else np.zeros(net.n_nodes, bool)
    void = np.zeros(net.n_nodes, bool)
    plane = (net.area > 0) if net.area is not None else np.ones(net.n_nodes, bool)
    for k in np.unique(net.mesh.node_layer):
        on = (net.mesh.node_layer == k) & ~no_ret & plane
        if on.sum() < 10:
            continue
        typ = np.median(net.h[on])
        void |= on & (net.h > VOID_RATIO * typ)
    out = []
    for mask in (no_ret & plane, void):
        rs = []
        for nodes in _regions(net, mask):
            w, h, _a, _at, _l = _describe(net, nodes)
            if _big(net, w, h, len(nodes)):
                rs.append(nodes)
        out.append(rs)
    return out


def eval_freq(s, res):
    """Where the planes' inductance matters: the target's upper frequency, within 1 MHz .. f max."""
    return float(min(max(res.f_target, 1e6), s.f_max))


def branch_energy(s, res, l_ret, f):
    """L_b |I_b|² per branch (B, P) with 1 A into each load at f, decaps and the regulator connected."""
    net = res.net
    Y = net.y_mesh(f)
    add = np.zeros(net.n_nodes, complex)
    for d in s.decaps:
        if d.enabled and f"C:{d.ref}" in net.terms:
            add[net.terms[f"C:{d.ref}"]] += 1.0 / PN.decap_z(d, np.array([f]), s.target.v, s.T,
                                                          l_ret.get(d.ref, (0.0, ""))[0])[0]
    if s.vrm.enabled and "V" in net.terms:
        add[net.terms["V"]] += PN.vrm_y(s.vrm, np.array([f]))[0]
    rhs = np.zeros((net.n_nodes, len(res.loads)), complex)
    for j, n in enumerate(res.loads):
        rhs[net.terms[f"L:{n}"], j] = 1.0
    V = splu((Y + sp.diags(add)).tocsc(), permc_spec="MMD_AT_PLUS_A").solve(rhs)
    I = net.y_branch(f)[:, None] * (net.inc @ V)
    return net.L[:, None] * np.abs(I) ** 2


def _typical_h(net):
    """Per layer index: the median height over the return of the rail's plane copper with a return."""
    typ = {}
    for k in np.unique(net.mesh.node_layer):
        on = (net.mesh.node_layer == k) & ~net.no_ret & (net.area > 0)
        if on.sum():
            typ[int(k)] = float(np.median(net.h[on]))
    return typ


def region_shares(net, regs, energy, typ, whole=False):
    """Per region: (share of the copper's inductive energy it adds, for the worst load j, j).  The
    excess of a branch is the part of its L due to h above the layer's typical h; whole=True counts
    all of it (copper with no return under it: its L there is a rough guess, so its whole share is
    what is uncertain / at stake)."""
    if not regs:
        return []
    a, b = net.mesh.br_a, net.mesh.br_b
    if whole:
        excess = (net.mesh.br_kind != 2).astype(float)
    else:
        hl = np.array([typ.get(int(k), np.nan) for k in net.mesh.node_layer[a]])
        hbr = (net.h[a] + net.h[b]) / 2
        excess = np.where((net.mesh.br_kind != 2) & np.isfinite(hl) & (hbr > 0),
                          np.clip(1 - hl / np.where(hbr > 0, hbr, 1), 0, 1), 0.0)
    total = np.maximum(energy.sum(0), 1e-300)
    out = []
    for nodes in regs:
        m = np.zeros(net.n_nodes, bool)
        m[nodes] = True
        br = m[a] | m[b]
        sh = (excess[br, None] * energy[br]).sum(0) / total
        j = int(np.argmax(sh))
        out.append((float(sh[j]), j))
    return out


def check(board, s, res, l_ret):
    """Layout issues of an analysed rail (res: analysis.Result with its network)."""
    net = res.net
    issues = []
    if net is not None and net.area is not None and res.loads:
        none_r, void_r = regions(net)
        f = eval_freq(s, res)
        energy = branch_energy(s, res, l_ret, f) if (none_r or void_r) else None
        typ = _typical_h(net)
        for kind, regs in (("no return", none_r), ("return void", void_r)):
            shares = region_shares(net, regs, energy, typ, whole=(kind == "no return"))
            scored = [(sh, j, n) for (sh, j), n in zip(shares, regs) if sh >= SHARE_MIN]
            scored.sort(key=lambda t: -t[0])
            for sh, j, nodes in scored[:MAX_REGIONS]:
                w, h, area, at, layers = _describe(net, nodes)
                where = f"{w:.1f} × {h:.1f} mm around ({at[0]:.1f}, {at[1]:.1f})"
                cost = (f"adds about {100 * sh:.0f} % to the copper's inductance seen by {res.loads[j]} "
                        f"(at {f / 1e6:.3g} MHz)")
                if kind == "no return":
                    text = (f"Rail copper on {', '.join(layers)} with no return copper above or below it ({where}, "
                            f"{area:.0f} mm²): the current returns somewhere else, so the loop is large there. It "
                            f"holds about {100 * sh:.0f} % of the copper's inductance seen by {res.loads[j]} (at "
                            f"{f / 1e6:.3g} MHz), and that part is a rough estimate. Put return copper under it.")
                else:
                    k = int(net.mesh.node_layer[nodes[0]])
                    ratio = float(np.median(net.h[nodes])) / typ.get(k, np.nan)
                    shape = "slot" if max(w, h) > 4 * min(w, h) else "void"
                    text = (f"The return under the rail on {', '.join(layers)} has a {shape} ({where}): the nearest "
                            f"return there is {ratio:.1f}× farther than elsewhere; it {cost}. Close the gap, or "
                            "route the rail's current around it.")
                issues.append(Issue("warn", kind, text, at=at))
            rest = len(regs) - min(len(scored), MAX_REGIONS)
            if rest:
                what = "areas without return copper" if kind == "no return" else "voids or slots in the return"
                issues.append(Issue("info", kind, f"{rest} more {what} under the rail, each adding less than "
                                                  f"{100 * SHARE_MIN:.0f} % (or not in the top {MAX_REGIONS})."))
    pads = {p.id: p for p in board.pads}
    rows = {r["ref"]: r for r in res.decaps}
    for d in s.decaps:
        if not d.enabled:
            continue
        r = rows.get(d.ref, {})
        at = next((tuple(pads[x].pos) for x in d.pads if x in pads), None)
        lr, how = l_ret.get(d.ref, (0.0, ""))
        esl = r.get("esl", 0.0) or 0.0
        if lr > L_MOUNT_WARN or (esl > 0 and lr > esl):
            more = f", more than the part's own ESL ({esl * 1e9:.2f} nH)" if esl > 0 and lr > esl else ""
            issues.append(Issue("warn", "mounting",
                                f"{d.ref}: return-side mounting L {lr * 1e9:.2f} nH ({how}){more}. A via right at "
                                "the return pad (or two) lowers it.", [d.ref], at))
        elif how.startswith("no return via") or how.startswith("no return plane"):
            issues.append(Issue("warn", "mounting", f"{d.ref}: {how}; its return path is not counted.", [d.ref], at))
    caps = [pads[x].pos for d in s.decaps if d.enabled for x in d.pads if x in pads]
    for ld in s.loads:
        if not ld.enabled:
            continue
        lp = [pads[x].pos for x in ld.pads if x in pads]
        if not lp:
            continue
        dist = min((math.dist(a, b) for a in lp for b in caps), default=math.inf)
        if dist > NEAR_LOAD_MM:
            near = "no decap on the rail" if not math.isfinite(dist) else f"the nearest decap is {dist:.0f} mm away"
            issues.append(Issue("warn", "no decap near",
                                f"{ld.name}: {near}. Its high-frequency current has to come through the planes; a "
                                f"small capacitor within {NEAR_LOAD_MM:.0f} mm of its power pins helps most.",
                                [ld.name], tuple(lp[0])))
    return issues
