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
PITCHES = (0.4, 0.5, 0.7, 1.0, 1.4, 2.0)   # mm: coarser pour cells tried when over the budget


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
    adaptive: bool = True                             # coarser cells away from the ports' copper
    fine_mm: float = None                             # fine cells within this of it (None: 2 x the thickest dielectric, >= 1 mm)
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


def port_nets(board, s):
    """Nets of the ports' and links' pads."""
    idx = pad_index(board)
    pads = [board.pads[idx[q]] for p in s.ports for q in p.plus + p.minus if q in idx]
    pads += [board.pads[idx[q]] for a, b in s.links for q in a + b if q in idx]
    return list(dict.fromkeys(p.net for p in pads if p.net))


def anchor_points(board, s, step=0.25):
    """Points (mm) on the ports' and links' pads and on their nets' tracks: the copper that carries
    the port current for sure.  Pours are left out (a ground pour that is a port's return covers
    the whole board): the window and the fine cells follow these points."""
    idx = pad_index(board)
    nets = set(port_nets(board, s))
    pts = []
    for q in [q for p in s.ports for q in p.plus + p.minus] + [q for a, b in s.links for q in a + b]:
        if q in idx and board.pads[idx[q]].geom is not None:
            bx = board.pads[idx[q]].geom.bbox
            xs, ys = np.meshgrid(np.linspace(bx[0], bx[2], 4), np.linspace(bx[1], bx[3], 4))
            pts.append(np.column_stack([xs.ravel(), ys.ravel()]))
    for t in board.tracks:
        if t.net in nets:
            line = np.array(P._polyline(t))
            for a, b in zip(line[:-1], line[1:]):
                n = max(2, int(math.ceil(np.hypot(*(b - a)) / step)) + 1)
                pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    pts = [q for q in pts if len(q)]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def auto_window(board, s):
    """Bounding box (mm) of the ports' and links' pads and their nets' tracks, plus the margin.
    Pours (of any net) do not widen it: a ground pour over the whole board once made the window
    the whole board (50 000 bars); they are included only inside the window."""
    pts = anchor_points(board, s)
    if not len(pts):
        raise ValueError("No port pads on the board.")
    m = s.margin
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0 - m), float(y0 - m), float(x1 + m), float(y1 + m)]


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
    level = _level(board, stackup, s, window) if s.adaptive else None
    f_disc = s.f_disc or s.f_max
    # fit the filament budget: first a narrower fine band around the ports' copper, then coarser
    # cells everywhere (pads of the path lose accuracy: 1 mm cells read a path's R 8 % low against
    # 0.5 mm), last fewer filaments per track; meshing is fast, so each try is cheap
    d0 = fine_mm(stackup, s)
    tries = [(s.pitch, d0)]
    if s.adaptive:
        d = d0
        while d / 2 >= max(s.pitch, 0.5) - 1e-9:
            d /= 2
            tries.append((s.pitch, d))
    tries += [(p, max(p, tries[-1][1])) for p in PITCHES if p > s.pitch * 1.01]
    for pitch, dfine in tries:
        level = _level(board, stackup, s, window, dfine) if s.adaptive else None
        mesh = P.build(board, stackup, nets, window, pitch, terminals, wire_width=math.inf, piece=WIRE_PIECE,
                       level=level)
        geom, bar_of_seg, nf = _geometry(mesh, f_disc, s.n_width, s.n_height, min(s.n_height, s.n_pour),
                                         ports, links, level)
        if nf <= s.max_fils:
            break
    warnings = list(mesh.warnings)
    nw, nh = s.n_width, s.n_height
    while nf > s.max_fils and (nw > 1 or nh > 1):
        nw, nh = max(1, nw - 2), max(1, nh - 2)
        geom, bar_of_seg, nf = _geometry(mesh, f_disc, nw, nh, min(nh, s.n_pour), ports, links, level)
    if s.adaptive and pitch == s.pitch and dfine < d0 - 1e-9:
        warnings.append(f"Fine pour cells only within {dfine:.2g} mm of the ports' copper (instead of {d0:.2g} mm) "
                        f"to stay within {s.max_fils} filaments.")
    if pitch != s.pitch:
        warnings.append(f"Pour and pad cells of {pitch:g} mm instead of {s.pitch:g} mm to stay within "
                        f"{s.max_fils} filaments (tracks keep their true width). Smaller window or margin, fewer "
                        f"passive nets or a higher limit give finer cells.")
    if (nw, nh) != (s.n_width, s.n_height):
        warnings.append(f"Fewer filaments than asked ({nw} across, {nh} through a track's copper) to stay within "
                        f"{s.max_fils} filaments: R at the top frequencies reads low. Narrow the window or "
                        f"raise the limit.")
    if nf > s.max_fils:
        raise ValueError(
            f"{nf} filaments even with {pitch:g} mm cells and one filament per bar (limit {s.max_fils}). The "
            f"window is {window[2] - window[0]:.0f} x {window[3] - window[1]:.0f} mm. Use a smaller window or "
            f"margin, untick large pours that are not needed, or raise the limit (memory ~{mem_gb(nf):.1f} GB at "
            f"{nf}).")
    out = Built(geom, mesh, bar_of_seg, nf, window, warnings)
    out.pitch, out.fine_mm = pitch, (dfine if s.adaptive else None)
    return out


def fine_mm(stackup, s):
    """Distance (mm) from the ports' copper within which pour cells stay fine: 2 x the thickest
    dielectric between copper layers (the return current spreads over a few heights), >= 1 mm."""
    if s.fine_mm:
        return float(s.fine_mm)
    L = list(stackup.copper_layers)
    gaps = [stackup.gap(a, b) * 1e3 for a, b in zip(L, L[1:])] or [1.6]
    return max(1.0, 2.0 * max(gaps))


def _level(board, stackup, s, window, d1=None):
    """Cell size wanted (1, 2, 4, 8 fine cells) at (X, Y): fine within d1 (default fine_mm) of the
    ports' and links' pads and their nets' tracks, doubling at 2x, 4x that distance."""
    from scipy.spatial import cKDTree
    pts = anchor_points(board, s, step=max(s.pitch / 2, 0.05))
    if not len(pts):
        return None
    tree = cKDTree(pts)
    d1 = d1 or fine_mm(stackup, s)

    def level(X, Y):
        d, _ = tree.query(np.column_stack([np.ravel(X), np.ravel(Y)]))
        return np.where(d < d1, 1, np.where(d < 2 * d1, 2, np.where(d < 4 * d1, 4, 8))).reshape(np.shape(X))
    return level


def mem_gb(n_fils):
    """Peak memory [GB] of the solve for n filaments (dense: L, Qᵀ L, L_q, one factor)."""
    from .system import dense_bytes
    return dense_bytes(n_fils, n_fils) / 2 ** 30


def seconds(n_fils):
    """Rough solve time [s] on this PC (4 threads): the matrices ~n², the factorisations ~n³."""
    return 1e-7 * n_fils ** 2 + 1.2e-11 * n_fils ** 3


def estimate(board, stackup, s):
    """The mesh a run would use, without solving (fast): dict with filaments, window, pitch (the
    cell size after fitting the budget), memory [GB], seconds (rough) and warnings."""
    b = build(board, stackup, s)
    return dict(filaments=b.n_fils, window=b.window, pitch=b.pitch, fine_mm=b.fine_mm, gb=mem_gb(b.n_fils),
                seconds=seconds(b.n_fils), warnings=b.warnings)


def _geometry(mesh, f_disc, nw_max, nh_max, np_max, ports, links, level=None):
    """Segments and filaments of the mesh.  level: the adaptive cell-size function; pour, pad and
    via-barrel copper gets filaments through its thickness only where it is 1 (the fine band
    around the ports' copper), one elsewhere."""
    g = Geometry()
    d = skin_depth(f_disc)

    def near(x, y):
        return level is None or int(np.ravel(level(np.array([x]), np.array([y])))[0]) == 1
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
            nh = division(t, d, 2.0, nh_max) if near(mesh.br_x[i], mesh.br_y[i]) else 1
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
        if kind == 3:
            nhk = division(t, d, 2.0, nh_max)
        else:       # pour / pad cells: through-thickness filaments only in the fine band (near the path)
            nhk = division(t, d, 2.0, np_max) if near(mesh.br_x[i], mesh.br_y[i]) else 1
        g.seg(na, nb, w, t, nw=nwk, nh=nhk)
        bar_of_seg.append(i)
        nf += nwk * nhk
    T = mesh.terminals
    for name, tp, tm in ports:
        g.port(name, anchors[T[tp]], anchors[T[tm]])
    for ta, tb in links:
        g.join(anchors[T[ta]], anchors[T[tb]])
    return g, np.array(bar_of_seg, int), nf
