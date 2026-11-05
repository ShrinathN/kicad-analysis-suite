"""KiCad copper -> em3d geometry.

The copper of the chosen nets inside a window is meshed by the Switching Loop's PEEC builder
(`loop.peec.build`, every track as a "wire" with its true width): pours and pads become cell
bars, tracks bars along the track, via and plated-hole barrels vertical bars.  Each bar becomes
a 3D segment split into filaments:

* tracks (wires): across the width and through the thickness, graded to the skin depth at the
  discretisation frequency (edge crowding and proximity between tracks come out of the solve);
* pour / pad cell bars: through the thickness only (the cell grid resolves the spread in the
  plane);
* barrels: a ring of BARREL_N plated segments around the hole, through the plating.

Ports are pairs of pad groups (current in at the + pads, out at the - pads).  Links join two pad
groups as one node (an ideal short: e.g. the load's two pins, closing a supply-and-return loop).
Nets without a port are passive conductors: eddy currents in a plane under a coil, or in a
neighbouring trace, are part of the solve.

Board coordinates are mm, y down; the geometry is in metres with z up (z = - depth below the
top copper's surface).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..loop import peec as P
from .geometry import Geometry, SIGMA_CU, division, skin_depth

BARREL_N = 6                          # plated segments around a via barrel
WIRE_PIECE = 2.0                      # mm: longest straight piece of a track (one filament bundle)
MAX_FILS = 12000                      # default filament budget (dense matrices: ~5 GB at 12k)


@dataclass
class Port:
    name: str
    plus: list                        # pad ids ("R1.2") on the + side
    minus: list                       # pad ids on the - side

    def to_dict(self):
        return dict(name=self.name, plus=list(self.plus), minus=list(self.minus))

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("name", "P"), list(d.get("plus", [])), list(d.get("minus", [])))


@dataclass
class Setup:
    nets: list = field(default_factory=list)          # conductors (ports' nets + passive ones)
    ports: list = field(default_factory=list)         # Port
    links: list = field(default_factory=list)         # [[pad ids], [pad ids]]: ideal shorts
    margin: float = 3.0                               # mm around the ports' copper (window)
    window: list = None                               # [x0, y0, x1, y1] mm; None = automatic
    pitch: float = 0.3                                # mm: pour / pad cell size
    f_min: float = 1e3
    f_max: float = 1e9
    f_disc: float = None                              # skin depth of the filaments (default f_max)
    n_width: int = 7                                  # most filaments across a track
    n_height: int = 5                                 # most filaments through a track's copper
    n_pour: int = 3                                   # ... through a pour's or pad's copper (they dominate the count)
    max_fils: int = MAX_FILS
    points: int = 61                                  # frequencies in the results

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["ports"] = [p.to_dict() for p in self.ports]
        return d

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for k in s.__dataclass_fields__:
            if k in d and k != "ports":
                setattr(s, k, d[k])
        s.ports = [Port.from_dict(p) for p in d.get("ports", [])]
        s.links = [[list(a), list(b)] for a, b in d.get("links", [])]
        return s


def pad_index(board):
    return {p.id: k for k, p in enumerate(board.pads)}


def auto_window(board, s):
    """Bounding box (mm) of the conductors' copper connected to the ports' pads, plus the margin:
    tracks, zones, pads and vias of the nets, clipped to what touches the port pads' nets."""
    idx = pad_index(board)
    pads = [board.pads[idx[q]] for p in s.ports for q in p.plus + p.minus if q in idx]
    pads += [board.pads[idx[q]] for a, b in s.links for q in a + b if q in idx]
    if not pads:
        raise ValueError("No port pads on the board.")
    nets = set(s.nets) | {p.net for p in pads}
    xs, ys = [], []
    for t in board.tracks:
        if t.net in nets:
            b = t.geom.bbox
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
    for z in board.zones:
        if z.net in nets:
            b = z.geom.bbox
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
    for p in board.pads:
        if p.net in nets:
            xs.append(p.pos[0])
            ys.append(p.pos[1])
    m = s.margin
    return [min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m]


@dataclass
class Built:
    geom: Geometry
    mesh: object                      # loop.peec.Mesh
    bar_of_seg: np.ndarray            # mesh bar of each segment (-1: none)
    n_fils: int
    window: list
    warnings: list


def build(board, stackup, s, progress=lambda m: None):
    """Geometry of the setup's copper (see the module doc)."""
    if not s.ports:
        raise ValueError("Add a port (two pads or pad groups).")
    idx = pad_index(board)
    terminals = {}

    def term(pads, label):
        miss = [q for q in pads if q not in idx]
        if miss:
            raise ValueError(f"{label}: pad {miss[0]} is not on the board.")
        if not pads:
            raise ValueError(f"{label}: no pads.")
        key = "T" + str(len(terminals))
        terminals[key] = [idx[q] for q in pads]
        return key
    ports = [(p.name, term(p.plus, f"Port {p.name} +"), term(p.minus, f"Port {p.name} -")) for p in s.ports]
    links = [(term(a, "Link"), term(b, "Link")) for a, b in s.links]
    nets = list(dict.fromkeys(list(s.nets) + [board.pads[k].net for v in terminals.values() for k in v]))
    nets = [n for n in nets if n]
    window = list(s.window) if s.window else auto_window(board, s)
    progress("Meshing the copper...")
    mesh = P.build(board, stackup, nets, window, s.pitch, terminals, wire_width=math.inf, piece=WIRE_PIECE)
    warnings = list(mesh.warnings)
    f_disc = s.f_disc or s.f_max
    nw, nh = s.n_width, s.n_height
    while True:
        geom, bar_of_seg, nf = _geometry(mesh, f_disc, nw, nh, min(nh, s.n_pour), ports, links)
        if nf <= s.max_fils or (nw <= 1 and nh <= 1):
            break
        nw, nh = max(1, nw - 2), max(1, nh - 2)
    if (nw, nh) != (s.n_width, s.n_height):
        warnings.append(f"Fewer filaments than asked ({nw} across, {nh} through the copper) to stay within "
                        f"{s.max_fils} filaments: R at the top frequencies reads low. Narrow the window or "
                        f"raise the limit.")
    if nf > s.max_fils:
        raise ValueError(f"{nf} filaments even with one per bar: use a larger cell pitch or a smaller window.")
    return Built(geom, mesh, bar_of_seg, nf, window, warnings)


def _geometry(mesh, f_disc, nw_max, nh_max, np_max, ports, links):
    g = Geometry()
    d = skin_depth(f_disc)
    # one anchor per mesh node; every segment end is its own node joined to the anchor
    zl = {k: mesh.z[L] for k, L in enumerate(mesh.layers)}
    anchors = [g.node(x * 1e-3, y * 1e-3, -zl[int(k)]) for (x, y), k in zip(mesh.node_xy, mesh.node_layer)]
    bar_of_seg, nf = [], 0
    for i in range(mesh.n_branches):
        kind = int(mesh.br_kind[i])
        a, b = anchors[mesh.br_a[i]], anchors[mesh.br_b[i]]
        w, t, ln = float(mesh.br_w[i]), float(mesh.br_t[i]), float(mesh.br_len[i])
        x, y = mesh.br_x[i] * 1e-3, mesh.br_y[i] * 1e-3
        if kind == 2:                                     # barrel: ring of plated segments
            r = w
            z0, z1 = -mesh.br_z0[i], -mesh.br_z1[i]
            nh = division(t, d, 2.0, nh_max)
            for k in range(BARREL_N):
                th = 2 * math.pi * (k + 0.5) / BARREL_N
                cx, cy = x + (r + t / 2) * math.cos(th), y + (r + t / 2) * math.sin(th)
                na, nb = g.node(cx, cy, z0), g.node(cx, cy, z1)
                g.join(a, na)
                g.join(b, nb)
                g.seg(na, nb, 2 * math.pi * (r + t / 2) / BARREL_N, t, wdir=(-math.sin(th), math.cos(th), 0),
                      nw=1, nh=nh)
                bar_of_seg.append(i)
                nf += nh
            continue
        ux, uy = float(mesh.br_ux[i]), float(mesh.br_uy[i])
        z = -float(mesh.br_z0[i])
        pa = (x - ux * ln / 2, y - uy * ln / 2, z)
        pb = (x + ux * ln / 2, y + uy * ln / 2, z)
        na, nb = g.node(*pa), g.node(*pb)
        g.join(a, na)
        g.join(b, nb)
        nwk = division(w, d, 2.0, nw_max) if kind == 3 else 1
        nhk = division(t, d, 2.0, nh_max if kind == 3 else np_max)
        g.seg(na, nb, w, t, nw=nwk, nh=nhk)
        bar_of_seg.append(i)
        nf += nwk * nhk
    T = mesh.terminals
    for name, tp, tm in ports:
        g.port(name, anchors[T[tp]], anchors[T[tm]])
    for ta, tb in links:
        g.join(anchors[T[ta]], anchors[T[tb]])
    return g, np.array(bar_of_seg, int), nf
