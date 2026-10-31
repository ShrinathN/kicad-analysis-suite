"""Partial-element equivalent circuit (PEEC) of the copper of a few nets in a window.

Every net is a conductor; none is an ideal reference.  The copper of each layer is cut into
cells on a uniform grid (raster.rasterise: per-net cells and cut-cell face apertures); a current
bar joins the centres of two neighbouring cells of one net (length h, width = face aperture,
thickness = copper), and via / plated-hole barrels are vertical bars between layers.  Each bar
has a DC resistance (skin-corrected at f) and a partial self inductance; every pair of parallel
bars has a partial mutual inductance (perpendicular bars do not couple).  Pads of device pins
("terminals") and via rings are equipotential nodes.

The network is solved in a mesh-current basis (fundamental cycles of a BFS spanning forest), so
the dense matrices are only (B x B) for L and (m x m) per frequency, m ~ number of cells.  A
port is a pair of terminals on one net (current in at one, out at the other); the result is the
port impedance matrix Z(f) = R(f) + jωL(f), mutual couplings between nets included.

Capacitance between nets comes from a potential-coefficient model of the same cells in a
homogeneous dielectric per pair of layers (εr between them; (1 + εr)/2 on an outer layer).

Lengths in the mesh are mm (board units) unless named *_m; electrical values are SI.
"""
import copy
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import integrate, linalg

from .. import THREADS
from .. import raster as rz
from ..board import Capsule

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
KM = MU0 / (4 * math.pi)             # 1e-7 H/m
RHO_CU = 1.72e-8                     # Ω m at 20 °C
MIN_AP = 0.05                        # faces narrower than this fraction of a cell are cut
NEAR = 2.5                           # pairs closer than NEAR x the wider bar use the exact tape integral
WIRE = 1.5                           # tracks narrower than WIRE x pitch are bars along the track ("wires")


# ---- kernels ------------------------------------------------------------------------
def _H(z, d):
    return z * np.arcsinh(z / d) - np.sqrt(z * z + d * d)


def axial(a1, b1, a2, b2, d):
    """∫∫ dx1 dx2 / sqrt((x1 - x2)² + d²) over [a1, b1] x [a2, b2] (two parallel filaments at
    distance d).  Finite for touching collinear filaments (d -> 0) as the log terms cancel."""
    return _H(b1 - a2, d) - _H(b1 - b2, d) - _H(a1 - a2, d) + _H(a1 - b2, d)


def _lg(u, r, q):
    """ln(u + r), r = sqrt(u² + q): stable for u < 0 (= ln q - ln(r - u)); 0 where q = 0 and u <= 0
    (its factor vanishes there)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.log(np.where(u + r > 0, u + r, 1.0))
        neg = np.log(np.where(q > 0, q, 1.0)) - np.log(np.where(r - u > 0, r - u, 1.0))
    return np.where(u >= 0, pos, np.where(q > 0, neg, 0.0))


def _F(x, y, z):
    """Antiderivative with d⁴F/dx²dy² = 1/sqrt(x² + y² + z²) (Hoer & Love; Ruehli 1972)."""
    z = np.abs(z)
    x2, y2, z2 = x * x, y * y, z * z
    r = np.sqrt(x2 + y2 + z2)
    a = 0.5 * x * (y2 - z2) * _lg(x, r, y2 + z2)
    b = 0.5 * y * (x2 - z2) * _lg(y, r, x2 + z2)
    c = -(x2 - 2 * z2 + y2) * r / 6
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(z * r > 0, -x * y * z * np.arctan(x * y / np.where(z * r > 0, z * r, 1.0)), 0.0)
    return a + b + c + d


def rect_integral(xa0, xa1, ya0, ya1, xb0, xb1, yb0, yb1, z):
    """∫∫∫∫ dA dA' / r between two parallel rectangles a and b in planes a distance z apart
    (exact; arrays broadcast).  It gives both the partial mutual inductance of two thin tapes
    carrying current along x, KM / (w_a w_b) x this, and the potential coefficient of two charged
    panels, 1 / (4π ε A_a A_b) x this.  Use it for near pairs: for far ones the 16 terms cancel
    to a small difference of large numbers."""
    tot = 0.0
    for X, sx in ((xa1 - xb0, 1), (xa1 - xb1, -1), (xa0 - xb0, -1), (xa0 - xb1, 1)):
        for Y, sy in ((ya1 - yb0, 1), (ya1 - yb1, -1), (ya0 - yb0, -1), (ya0 - yb1, 1)):
            tot = tot + sx * sy * _F(X, Y, z)
    return tot


def _strip_G(r):
    """Self partial inductance of a thin strip of length l and width w = r l, in units of KM l:
    the double average over the strip of 1/distance (exact, by quadrature)."""
    def f(s):
        return (r - s) * 2.0 * (math.asinh(1.0 / s) - math.sqrt(1.0 + s * s) + s)
    v = integrate.quad(f, 0.0, r, limit=400)[0]
    return 2.0 / r ** 2 * v


_RS = np.logspace(-3, 2.3, 166)
_GS = None


def strip_self(l, w):
    """Self partial inductance [H] of thin strips (arrays): length l, width w (m)."""
    global _GS
    if _GS is None:
        _GS = np.array([_strip_G(r) for r in _RS])
    r = np.clip(w / l, _RS[0], _RS[-1])
    return KM * l * np.interp(np.log(r), np.log(_RS), _GS)


def tube_self(l, r):
    """Self partial inductance [H] of a thin tube (via barrel) of length l, radius r (m)."""
    return KM * axial(0.0, l, 0.0, l, r)


def parallel_block(A0, A1, C, Z, W, T, rows, cols, G, near=NEAR):
    """Partial mutual inductances [H] between bars `rows` and bars `cols` of one orientation.

    A0, A1: axial start / end; C: lateral centre; W: width; T: thickness (m, arrays); Z: layer
    index; G: vertical distance between two layers' facing copper surfaces (m).  At the frequencies
    of interest the current runs on the faces that look at each other (skin depth < copper), so
    that gap, not the distance between copper mid-planes, sets the coupling (on 0.1 mm prepreg with
    35 µm copper the mid-planes would read the plane-pair inductance 35 % high).
    Pairs closer than `near` widths (the diagonal included) use the exact integral over both tapes;
    far pairs the filament formula along their axes."""
    a0, a1, c, z, w = A0[rows, None], A1[rows, None], C[rows, None], Z[rows, None], W[rows, None]
    B0, B1, Cc, Zc, Wc, Tc = A0[cols], A1[cols], C[cols], Z[cols], W[cols], T[cols]
    dc, dz = c - Cc[None, :], G[z[:, 0]][:, Zc]
    d = np.sqrt(dc * dc + dz * dz)
    M = axial(a0, a1, B0[None, :], B1[None, :], np.where(d > 0, d, 1.0))     # d = 0 pairs are near: replaced
    ii, jj = np.nonzero(d < near * np.maximum(w, Wc[None, :]))
    if ii.size:          # exact zero-thickness tapes (current on the copper surface)
        wi, wj = w[ii, 0], Wc[jj]
        zz = dz[ii, jj]
        I = rect_integral(a0[ii, 0], a1[ii, 0], c[ii, 0] - wi / 2, c[ii, 0] + wi / 2,
                          B0[jj], B1[jj], Cc[jj] - wj / 2, Cc[jj] + wj / 2, zz)
        M[ii, jj] = I / (wi * wj)
    return KM * M


# ---- mesh ---------------------------------------------------------------------------
@dataclass
class Mesh:
    grid: object                      # raster.Grid of the window
    layers: list                      # copper layers, stack order
    z: dict                           # layer -> copper mid-plane z [m], top = 0, increasing down
    t: dict                           # layer -> copper thickness [m]
    nets: list                        # net names (index = net id)
    cell_node: dict                   # layer -> (ny, nx) node index per cell, -1 = no copper
    cell_f: dict                      # layer -> (ny, nx) copper fraction
    node_net: np.ndarray
    node_xy: np.ndarray               # (N, 2) mm, mean of the node's cells
    node_layer: np.ndarray            # layer index of the node's first cell
    br_kind: np.ndarray               # 0 x bar, 1 y bar, 2 barrel, 3 wire (narrow track piece, any angle)
    br_a: np.ndarray                  # from node
    br_b: np.ndarray                  # to node
    br_layer: np.ndarray              # layer index (x / y bars); upper layer for barrels
    br_x: np.ndarray                  # centre, mm
    br_y: np.ndarray
    br_len: np.ndarray                # m
    br_w: np.ndarray                  # m: width (bars) or radius (barrels)
    br_t: np.ndarray                  # m: copper (bars) or plating (barrels) thickness
    br_z0: np.ndarray                 # m: bar height, or barrel top
    br_z1: np.ndarray                 # m: barrel bottom (= z0 for bars)
    br_R: np.ndarray                  # Ω at DC
    br_net: np.ndarray
    terminals: dict                   # name -> node
    gap: np.ndarray = None            # (layers x layers) m between facing copper surfaces, 0 on the diagonal
    br_ux: np.ndarray = None          # in-plane direction of each bar (x: 1, 0; y: 0, 1; barrels 0, 0)
    br_uy: np.ndarray = None
    warnings: list = field(default_factory=list)

    @property
    def n_nodes(self):
        return len(self.node_net)

    @property
    def n_branches(self):
        return len(self.br_a)


class _DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        r = a
        while p[r] != r:
            r = p[r]
        while p[a] != r:
            p[a], a = r, p[a]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)
        return min(ra, rb)


def _cells_of(geom, grid, ok, pos):
    """Flat indices of cells whose centre lies in `geom` and where `ok`; else the cell under `pos`
    (or the nearest ok cell within two cells) so that small pads on a coarse grid still connect."""
    j0, j1, i0, i1 = grid.window(geom.bbox, pad=1)
    if j1 > j0 and i1 > i0:
        X, Y = grid.centres(j0, j1, i0, i1)
        m = geom.inside(X, Y) & ok[j0:j1, i0:i1]
        if m.any():
            jj, ii = np.nonzero(m)
            return (jj + j0) * grid.nx + ii + i0
    j, i = grid.cell(*pos)
    best = None
    for dj in range(-2, 3):
        for di in range(-2, 3):
            jj, ii = j + dj, i + di
            if 0 <= jj < grid.ny and 0 <= ii < grid.nx and ok[jj, ii]:
                dist = dj * dj + di * di
                if best is None or dist < best[0]:
                    best = (dist, jj * grid.nx + ii)
    return np.array([best[1]] if best else [], np.int64)


def quadtree_sizes(want):
    """Cell sizes (1, 2 or 4 fine cells) from wanted sizes per fine cell, as aligned blocks: a 4 x 4
    tile is one block if every cell in it may be coarse, else its 2 x 2 sub-tiles likewise."""
    ny, nx = want.shape
    py, px = -ny % 4, -nx % 4
    w = np.pad(want, ((0, py), (0, px)), constant_values=4)
    t4 = w.reshape(w.shape[0] // 4, 4, w.shape[1] // 4, 4).min(axis=(1, 3))
    t2 = w.reshape(w.shape[0] // 2, 2, w.shape[1] // 2, 2).min(axis=(1, 3))
    size = np.where(np.repeat(np.repeat(t2 >= 2, 2, 0), 2, 1), 2, 1)
    size = np.where(np.repeat(np.repeat(t4 >= 4, 4, 0), 4, 1), 4, size)
    return size[:ny, :nx]


def build(board, stackup, nets, window, pitch, terminals, rho=RHO_CU, t_plate=25e-6, level=None):
    """Mesh the copper of `nets` inside `window` (x0, y0, x1, y1 mm) at `pitch` mm.

    terminals: {name: [board pad indices]} - each is one equipotential node (a device pin: all its
    pads, on every layer).  Raises ValueError if a terminal has no copper in the window.
    level: None (uniform) or a function (X, Y) mm -> wanted cell size in fine cells (1, 2 or 4).
    The copper is rasterised at `pitch` everywhere (connectivity is decided there); coarse cells are
    aligned blocks of fine cells of one net, and the bars between two blocks are the sum of their
    shared fine faces, running between the blocks' centres."""
    grid = rz.make_grid(board, pitch, margin=0.0, bbox=window)
    wires = _narrow_tracks(board, nets, window, pitch)
    skip = {k for k, _ in wires}
    rb = copy.copy(board)
    rb.tracks = [t for k, t in enumerate(board.tracks) if k not in skip]
    R = rz.rasterise(rb, nets, grid)
    layers = list(board.copper_layers)
    ztop, tcu = stackup.layer_z(), stackup.cu_thickness
    z = {L: ztop[L] + tcu[L] / 2 for L in layers}
    nx, ny = grid.nx, grid.ny
    cell_node, cell_f, node_cells = {}, {}, []
    n = 0
    for L in layers:
        lr = R.layers[L]
        act = (lr.f > 0) & (lr.net >= 0)
        idx = np.full(grid.shape, -1, np.int64)
        k = int(act.sum())
        idx[act] = np.arange(n, n + k)
        jj, ii = np.nonzero(act)
        node_cells.append((layers.index(L), jj, ii))
        n += k
        cell_node[L], cell_f[L] = idx, lr.f.copy()
    if n == 0:
        raise ValueError("No copper of the loop's nets in the window.")
    lay_of = np.concatenate([np.full(len(jj), k) for k, jj, _ in node_cells])
    cj = np.concatenate([jj for _, jj, _ in node_cells])
    ci = np.concatenate([ii for _, _, ii in node_cells])
    net_of = np.concatenate([R.layers[layers[k]].net[jj, ii] for k, jj, ii in node_cells]).astype(np.int64)
    n_cells = n
    cx = grid.x0 + (ci + 0.5) * grid.h
    cy = grid.y0 + (cj + 0.5) * grid.h
    # blocks: aligned groups of fine cells of one net and layer (all size 1 for a uniform mesh)
    if level is None:
        size = np.ones(grid.shape, np.int64)
    else:
        Xc, Yc = grid.centres()
        size = quadtree_sizes(np.clip(np.asarray(level(Xc, Yc), np.int64), 1, 4))
    sz = size[cj, ci]
    bkey = ((lay_of * 64 + net_of) * 8 + sz) * (grid.ny + 4) * (grid.nx + 4) + (cj // sz) * (grid.nx + 4) + ci // sz
    _, blk_of = np.unique(bkey, return_inverse=True)
    blk_rep = np.full(blk_of.max() + 1, n_cells, np.int64)
    np.minimum.at(blk_rep, blk_of, np.arange(n_cells))
    blk_cx = np.zeros(len(blk_rep))
    blk_cy = np.zeros(len(blk_rep))
    blk_cx[blk_of] = grid.x0 + ((ci // sz) * sz + sz / 2) * grid.h      # geometric block centres
    blk_cy[blk_of] = grid.y0 + ((cj // sz) * sz + sz / 2) * grid.h
    cell_blk = {}
    for k, L in enumerate(layers):
        b_ = np.full(grid.shape, -1, np.int64)
        sel = lay_of == k
        b_[cj[sel], ci[sel]] = blk_of[sel]
        cell_blk[L] = b_
    # narrow tracks: chains of nodes along each track, one bar per piece (<= one cell long)
    wpts, wbars = [], []                  # node positions; (a, b, layer, net, width)
    w_ends = []                           # (node, x, y, layer, net, width, track end?) at polyline vertices
    for k, t in wires:
        L, nid = t.layer, R.nets.index(t.net)
        for (xa, ya), (xb, yb), (va, ta), (vb, tb) in _clip_polyline(_polyline(t), window):
            npc = max(1, int(math.ceil(math.hypot(xb - xa, yb - ya) / max(pitch, t.width))))
            base = n + len(wpts)
            for q in range(npc + 1):
                wpts.append((xa + (xb - xa) * q / npc, ya + (yb - ya) * q / npc, layers.index(L), nid))
            for q in range(npc):
                wbars.append((base + q, base + q + 1, layers.index(L), nid, t.width))
            if va:
                w_ends.append((base, xa, ya, L, nid, t.width, ta))
            if vb:
                w_ends.append((base + npc, xb, yb, L, nid, t.width, tb))
    nw = len(wpts)
    if nw:
        wp = np.array(wpts)
        lay_of = np.concatenate([lay_of, wp[:, 2].astype(np.int64)])
        net_of = np.concatenate([net_of, wp[:, 3].astype(np.int64)])
        cx = np.concatenate([cx, wp[:, 0]])
        cy = np.concatenate([cy, wp[:, 1]])
        n += nw
    dsu = _DSU(n)
    dsu.p[:n_cells] = blk_rep[blk_of]                             # each block starts as one node
    warnings = [w for w in R.warnings if "overlaps" not in w]     # coarse cells straddle clearances
    # a track's ends join the copper under them (pads, pours, wide tracks, via rings); every vertex
    # joins the nearest node of other wires of its net there (consecutive pieces, T-junctions)
    for q in range(n_cells, n):               # wire nodes on copper of their net (pads, rings, pours): merged
        L = layers[lay_of[q]]
        j, i = grid.cell(cx[q], cy[q])
        if 0 <= j < grid.ny and 0 <= i < grid.nx:
            c = cell_node[L][j, i]
            if c >= 0 and R.layers[L].net[j, i] == net_of[q]:
                dsu.union(q, int(c))
    owner = {}
    for q, (a, b, *_rest) in enumerate(wbars):
        owner.setdefault(a, set()).add(q)
        owner.setdefault(b, set()).add(q)
    for node, x, y, L, nid, w, track_end in w_ends:
        if track_end:
            ok = (cell_node[L] >= 0) & (R.layers[L].net == nid)
            cells = _cells_of(Capsule(x, y, x, y, w / 2 + grid.h / 2), grid, ok, (x, y))
            if cells.size:
                c0 = cell_node[L].ravel()[cells]
                dsu.union(node, int(c0[np.argmin(np.hypot(cx[c0] - x, cy[c0] - y))]))
        same = np.flatnonzero((lay_of[n_cells:] == layers.index(L)) & (net_of[n_cells:] == nid)) + n_cells
        same = same[(same != node) & np.array([not (owner.get(int(c), set()) & owner.get(node, set()))
                                                for c in same], bool)]
        if same.size:
            d = np.hypot(cx[same] - x, cy[same] - y)
            j = int(np.argmin(d))
            if d[j] <= w / 2 + pitch / 2 + 1e-6:
                dsu.union(node, int(same[j]))

    # terminals: all cells under the pin's pads, every layer, one node
    term_root = {}
    for name, pads in terminals.items():
        root = None
        for k in pads:
            p = board.pads[k]
            if p.net not in nets or p.geom is None:
                continue
            for L in p.layers:
                if L not in cell_node:
                    continue
                ok = (cell_node[L] >= 0) & (R.layers[L].net == R.nets.index(p.net))
                cells = _cells_of(p.geom, grid, ok, p.pos)
                for c in cell_node[L].ravel()[cells]:
                    root = int(c) if root is None else dsu.union(root, int(c))
        if root is None:
            raise ValueError(f"Terminal {name}: its pads have no copper in the window.")
        term_root[name] = root
    term_cells = {dsu.find(r) for r in term_root.values()}

    # via rings (and plated holes of other parts) per layer: one node each, barrels between layers
    barrels = []          # (node_a, node_b, x, y, z_a, z_b, radius, label)
    holes = [(v.pos, v.size / 2, v.drill / 2, v.layers, v.net, f"via #{k}")
             for k, v in enumerate(board.vias) if v.net in nets]
    holes += [(p.pos, max(p.size) / 2, sum(p.drill) / 4, p.layers, p.net, f"pad {p.id}")
              for k, p in enumerate(board.pads)
              if p.net in nets and p.drill and p.plated and len(p.layers) > 1
              and not any(k in pads for pads in terminals.values())]
    x0, y0, x1, y1 = window
    for pos, rad, rdrill, vlayers, net, label in holes:
        if not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1):
            continue
        nid = R.nets.index(net)
        per_layer = []
        for L in vlayers:
            if L not in cell_node:
                continue
            ok = (cell_node[L] >= 0) & (R.layers[L].net == nid)
            cells = _cells_of(Capsule(*pos, *pos, rad + grid.h / 2), grid, ok, pos)
            root = None
            for c in cell_node[L].ravel()[cells]:
                root = int(c) if root is None else dsu.union(root, int(c))
            if root is not None:
                per_layer.append((L, root))
        for (La, a), (Lb, b) in zip(per_layer, per_layer[1:]):
            barrels.append((a, b, pos[0], pos[1], z[La], z[Lb], max(rdrill, 0.01) * 1e-3, label, layers.index(La)))

    roots = np.array([dsu.find(i) for i in range(n)])
    uniq, node_of = np.unique(roots, return_inverse=True)
    N = len(uniq)
    node_net = np.zeros(N, np.int64)
    node_net[node_of] = net_of
    node_layer = np.zeros(N, np.int64)
    node_layer[node_of[::-1]] = lay_of[::-1]
    cnt = np.bincount(node_of, minlength=N)
    node_xy = np.column_stack([np.bincount(node_of, cx, N) / cnt, np.bincount(node_of, cy, N) / cnt])
    for L in layers:
        m = cell_node[L] >= 0
        cell_node[L][m] = node_of[cell_node[L][m]]

    # bars between neighbouring cells of one net
    h, hm = grid.h, grid.h * 1e-3
    br = {k: [] for k in ("kind", "a", "b", "layer", "x", "y", "len", "w", "t", "z0", "z1", "net", "ux", "uy")}

    def add(kind, a, b, layer, x, y, ln, w, t, z0_, z1_, net):
        for key, v in (("kind", kind), ("a", a), ("b", b), ("layer", layer), ("x", x), ("y", y), ("len", ln),
                       ("w", w), ("t", t), ("z0", z0_), ("z1", z1_), ("net", net)):
            br[key].append(np.broadcast_to(np.asarray(v), np.shape(a)).copy())

    for k, L in enumerate(layers):
        lr, idx, bl = R.layers[L], cell_node[L], cell_blk[L]
        # a partly covered face carries its copper on the side where the neighbouring rows (columns)
        # have copper: the bar is moved there, so that a strip keeps its true width and position
        fn = np.pad(np.where(idx >= 0, lr.f, 0.0), 1)
        for axis in (0, 1):
            if axis == 0:
                A, Bn = idx[:, :-1], idx[:, 1:]
                ap = lr.apx[:, 1:-1]
                na, nb = lr.net[:, :-1], lr.net[:, 1:]
                jj, ii = np.nonzero((A >= 0) & (Bn >= 0) & (na == nb) & (ap >= MIN_AP) & (A != Bn))
                ba, bb = bl[jj, ii], bl[jj, ii + 1]
                side = (fn[jj + 2, ii + 1] + fn[jj + 2, ii + 2]) - (fn[jj, ii + 1] + fn[jj, ii + 2])
                apv = ap[jj, ii]
                lat = grid.y0 + (jj + 0.5) * h + np.sign(side) * (1 - apv) * h / 2
            else:
                A, Bn = idx[:-1, :], idx[1:, :]
                ap = lr.apy[1:-1, :]
                na, nb = lr.net[:-1, :], lr.net[1:, :]
                jj, ii = np.nonzero((A >= 0) & (Bn >= 0) & (na == nb) & (ap >= MIN_AP) & (A != Bn))
                ba, bb = bl[jj, ii], bl[jj + 1, ii]
                side = (fn[jj + 1, ii + 2] + fn[jj + 2, ii + 2]) - (fn[jj + 1, ii] + fn[jj + 2, ii])
                apv = ap[jj, ii]
                lat = grid.x0 + (ii + 0.5) * h + np.sign(side) * (1 - apv) * h / 2
            if not jj.size:
                continue
            # one bar per pair of blocks: width = their shared faces, between the blocks' centres
            a_all, nv_all = A[jj, ii], na[jj, ii]
            pair, inv = np.unique(ba * len(blk_rep) + bb, return_inverse=True)
            wsum = np.bincount(inv, apv * h)
            latm = np.bincount(inv, apv * h * lat) / wsum
            first = np.zeros(len(pair), np.int64)
            first[inv[::-1]] = np.arange(len(inv))[::-1]
            pa, pb = ba[first], bb[first]
            if axis == 0:
                x0_, x1_ = blk_cx[pa], blk_cx[pb]
                xs, ys, ln = (x0_ + x1_) / 2, latm, np.abs(x1_ - x0_)
            else:
                y0_, y1_ = blk_cy[pa], blk_cy[pb]
                xs, ys, ln = latm, (y0_ + y1_) / 2, np.abs(y1_ - y0_)
            a, b = a_all[first], Bn[jj, ii][first]
            add(axis, a, b, k, xs, ys, ln * 1e-3, wsum * 1e-3, tcu[L], z[L], z[L], nv_all[first].astype(np.int64))
            br["ux"].append(np.full(len(a), 1.0 if axis == 0 else 0.0))
            br["uy"].append(np.full(len(a), 0.0 if axis == 0 else 1.0))
    for a, b, k, nid, w in wbars:
        (xa, ya), (xb, yb) = (cx[a], cy[a]), (cx[b], cy[b])
        ln = math.hypot(xb - xa, yb - ya)
        a, b = node_of[a], node_of[b]
        if a != b and ln > 1e-6:
            add(3, np.array([a]), np.array([b]), k, (xa + xb) / 2, (ya + yb) / 2, ln * 1e-3, w * 1e-3,
                tcu[layers[k]], z[layers[k]], z[layers[k]], nid)
            br["ux"].append(np.array([(xb - xa) / ln]))
            br["uy"].append(np.array([(yb - ya) / ln]))
    for a, b, x, y, za, zb, r, label, k in barrels:
        a, b = node_of[a], node_of[b]
        if a != b:
            add(2, np.array([a]), np.array([b]), k, x, y, abs(zb - za), r, t_plate, min(za, zb), max(za, zb),
                node_net[a])
            br["ux"].append(np.zeros(1))
            br["uy"].append(np.zeros(1))
    cat = {key: np.concatenate(v) if v else np.zeros(0) for key, v in br.items()}
    kind = cat["kind"].astype(np.int64)
    Rdc = np.where(kind != 2, rho * cat["len"] / np.maximum(cat["w"] * cat["t"], 1e-18),
                   rho * cat["len"] / (2 * math.pi * np.maximum(cat["w"], 1e-9) * cat["t"]))
    terms = {name: int(node_of[r]) for name, r in term_root.items()}
    m = Mesh(grid, layers, z, dict(tcu), list(R.nets), cell_node, cell_f, node_net, node_xy, node_layer,
             kind, cat["a"].astype(np.int64), cat["b"].astype(np.int64), cat["layer"].astype(np.int64),
             cat["x"], cat["y"], cat["len"], cat["w"], cat["t"], cat["z0"], cat["z1"], Rdc,
             cat["net"].astype(np.int64), terms, warnings=warnings)
    m.gap = np.array([[stackup.gap(La, Lb) if La != Lb else 0.0 for Lb in layers] for La in layers])
    m.br_ux, m.br_uy = cat["ux"], cat["uy"]
    return m


def _polyline(t):
    """Points along a track (segment: its ends; arc: its capsule chain)."""
    g = t.geom
    if hasattr(g, "parts"):
        pts = [(g.parts[0].ax, g.parts[0].ay)] + [(c.bx, c.by) for c in g.parts]
        return pts
    return [tuple(t.start), tuple(t.end)]


def _clip_polyline(pts, box):
    """Pieces of a polyline inside box: ((xa, ya), (xb, yb), (a is a vertex, a is the track's end),
    (same for b)).  Points cut by the box are neither (left open)."""
    x0, y0, x1, y1 = box
    out = []
    last = len(pts) - 2
    for k, ((xa, ya), (xb, yb)) in enumerate(zip(pts, pts[1:])):
        dx, dy = xb - xa, yb - ya
        t0, t1 = 0.0, 1.0
        ok = True
        for p_, q_ in ((-dx, xa - x0), (dx, x1 - xa), (-dy, ya - y0), (dy, y1 - ya)):
            if abs(p_) < 1e-15:
                if q_ < 0:
                    ok = False
                continue
            r = q_ / p_
            if p_ < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
        if not ok or t1 - t0 < 1e-9:
            continue
        out.append(((xa + t0 * dx, ya + t0 * dy), (xa + t1 * dx, ya + t1 * dy),
                    (t0 == 0.0, k == 0 and t0 == 0.0), (t1 == 1.0, k == last and t1 == 1.0)))
    return out


def _narrow_tracks(board, nets, window, pitch):
    """(index, track) of tracks of `nets` touching the window that are narrower than WIRE x pitch
    and not buried in a pour of their own net on their layer (those stay in the raster)."""
    x0, y0, x1, y1 = window
    zones = {}
    for z in board.zones:
        zones.setdefault((z.net, z.layer), []).append(z.geom)
    out = []
    for k, t in enumerate(board.tracks):
        if t.net not in nets or t.width >= WIRE * pitch:
            continue
        bx = t.geom.bbox
        if bx[2] < x0 or bx[0] > x1 or bx[3] < y0 or bx[1] > y1:
            continue
        zs = zones.get((t.net, t.layer), [])
        if zs:
            pts = np.array(_polyline(t))
            mid = (pts[:-1] + pts[1:]) / 2
            P = np.vstack([pts, mid])
            if all(any(bool(g.inside(np.array([x]), np.array([y]))[0]) for g in zs) for x, y in P):
                continue
        out.append((k, t))
    return out


# ---- inductance -----------------------------------------------------------------------
THREAD_BYTES = 64 << 20               # working set of one thread's row block in inductance_matrix
TEMPS = 16                           # float arrays of one block alive at once in parallel_block


def _block_rows(B):
    return int(min(256, max(8, THREAD_BYTES // (8 * TEMPS * max(B, 1)))))


def inductance_matrix(mesh):
    """Dense partial inductance matrix [H] of all branches (B x B).  Upper-triangle row blocks are
    computed in threads (numpy releases the GIL in its array loops), each block sized to keep a
    thread's temporaries near THREAD_BYTES, and mirrored."""
    B = mesh.n_branches
    block = _block_rows(B)
    L = np.zeros((B, B))
    for kind in (0, 1):
        ix = np.flatnonzero(mesh.br_kind == kind)
        if not ix.size:
            continue
        ax = (mesh.br_x if kind == 0 else mesh.br_y)[ix] * 1e-3
        lat = (mesh.br_y if kind == 0 else mesh.br_x)[ix] * 1e-3
        half = mesh.br_len[ix] / 2
        A0, A1 = ax - half, ax + half
        Zc, W, T = mesh.br_layer[ix], mesh.br_w[ix], mesh.br_t[ix]
        Lk = np.zeros((ix.size, ix.size))

        def work(s):
            rows = np.arange(s, min(s + block, ix.size))
            cols = np.arange(s, ix.size)
            blk = parallel_block(A0, A1, lat, Zc, W, T, rows, cols, mesh.gap)
            Lk[rows[0]:rows[-1] + 1, s:] = blk
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            list(ex.map(work, range(0, ix.size, block)))
        Lk = np.triu(Lk) + np.triu(Lk, 1).T
        L[np.ix_(ix, ix)] = Lk
    iw = np.flatnonzero(mesh.br_kind == 3)
    if iw.size:
        ip = np.flatnonzero(mesh.br_kind != 2)

        def wrow(i):
            row = _inplane_row(mesh, i, ip)
            L[i, ip] = row
            L[ip, i] = row
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            list(ex.map(wrow, iw))
        L[iw, iw] = strip_self(mesh.br_len[iw], mesh.br_w[iw])
    iz = np.flatnonzero(mesh.br_kind == 2)
    if iz.size:
        x, y = mesh.br_x[iz] * 1e-3, mesh.br_y[iz] * 1e-3
        r = mesh.br_w[iz]
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        d = np.maximum(d, np.maximum(r[:, None], r[None, :]))       # same axis: GMD of a tube = r
        blk = KM * axial(mesh.br_z0[iz, None], mesh.br_z1[iz, None], mesh.br_z0[None, iz], mesh.br_z1[None, iz], d)
        L[np.ix_(iz, iz)] = blk
    return (L + L.T) / 2


_G6 = np.polynomial.legendre.leggauss(6)


def _inplane_row(mesh, i, J):
    """Partial mutual inductances [H] between in-plane bar i (any direction) and bars J."""
    X, Y = mesh.br_x * 1e-3, mesh.br_y * 1e-3
    ux, uy = mesh.br_ux[i], mesh.br_uy[i]
    UX, UY = mesh.br_ux[J], mesh.br_uy[J]
    li, wi, lj, wj = mesh.br_len[i], mesh.br_w[i], mesh.br_len[J], mesh.br_w[J]
    dot = ux * UX + uy * UY
    cross = ux * UY - uy * UX
    dz = mesh.gap[mesh.br_layer[i], mesh.br_layer[J]]
    rx, ry = X[J] - X[i], Y[J] - Y[i]
    out = np.zeros(len(J))
    par = np.abs(cross) < 1e-6
    if par.any():                     # parallel: the tape formulas in bar i's frame
        k = np.flatnonzero(par)
        sc = rx[k] * ux + ry[k] * uy
        c = -rx[k] * uy + ry[k] * ux
        d = np.sqrt(c * c + dz[k] ** 2)
        far = KM * axial(-li / 2, li / 2, sc - lj[k] / 2, sc + lj[k] / 2, np.where(d > 0, d, 1.0))
        near = d < NEAR * np.maximum(wi, wj[k])
        if near.any():
            q = np.flatnonzero(near)
            I = rect_integral(-li / 2, li / 2, -wi / 2, wi / 2, sc[q] - lj[k][q] / 2, sc[q] + lj[k][q] / 2,
                              c[q] - wj[k][q] / 2, c[q] + wj[k][q] / 2, dz[k][q])
            far[q] = KM * I / (wi * wj[k][q])
        out[k] = np.sign(dot[k]) * far
    k = np.flatnonzero(~par)
    if k.size:                        # not parallel: Neumann's integral (6 x 6 Gauss near, midpoints far)
        D = np.sqrt(rx[k] ** 2 + ry[k] ** 2 + dz[k] ** 2)
        size = np.maximum(np.maximum(li, lj[k]), np.maximum(wi, wj[k]))
        m = KM * dot[k] * li * lj[k] / np.maximum(D, 1e-12)
        near = D < 3 * size
        if near.any():
            q = k[near]
            g, wg = _G6
            g2 = (0.2235 * (wi + wj[q]) / 2) ** 2
            acc = np.zeros(q.size)
            for a in range(6):
                pa = (ux * li / 2 * g[a], uy * li / 2 * g[a])
                for b in range(6):
                    qx = rx[q] + UX[q] * lj[q] / 2 * g[b] - pa[0]
                    qy = ry[q] + UY[q] * lj[q] / 2 * g[b] - pa[1]
                    acc += wg[a] * wg[b] / np.sqrt(qx * qx + qy * qy + dz[q] ** 2 + g2)
            m[near] = KM * dot[q] * li * lj[q] / 4 * acc
        out[k] = m
    return out


def skin_factor(f, t, rho=RHO_CU):
    """R(f)/R_dc of copper of thickness t carrying current on one face (towards its return)."""
    if f <= 0:
        return np.ones_like(t)
    delta = math.sqrt(rho / (math.pi * f * MU0))
    u = t / delta
    return np.where(u < 1e-3, 1.0, u / -np.expm1(-u))


# ---- topology: spanning forest, fundamental cycles, port paths -------------------------
def _forest(N, a, b):
    """BFS spanning forest.  Returns parent node, parent branch (-1 at roots), depth, component."""
    B = len(a)
    ends = np.concatenate([a, b])
    other = np.concatenate([b, a])
    bid = np.concatenate([np.arange(B), np.arange(B)])
    order = np.argsort(ends, kind="stable")
    ptr = np.searchsorted(ends[order], np.arange(N + 1))
    nb, nbr = other[order], bid[order]
    parent = np.full(N, -1)
    pbr = np.full(N, -1)
    depth = np.full(N, -1)
    comp = np.full(N, -1)
    c = 0
    for s in range(N):
        if depth[s] >= 0:
            continue
        depth[s], comp[s] = 0, c
        q = [s]
        while q:
            nxt = []
            for u in q:
                for k in range(ptr[u], ptr[u + 1]):
                    v = nb[k]
                    if depth[v] < 0:
                        depth[v], comp[v], parent[v], pbr[v] = depth[u] + 1, c, u, nbr[k]
                        nxt.append(v)
            q = nxt
        c += 1
    return parent, pbr, depth, comp


def _path(u, v, parent, pbr, depth, a):
    """Branches (index, sign) of the tree path u -> v (sign +1 where the branch runs a -> b along it)."""
    up, down = [], []
    while depth[u] > depth[v]:
        k = pbr[u]
        up.append((k, 1 if a[k] == u else -1))
        u = parent[u]
    while depth[v] > depth[u]:
        k = pbr[v]
        down.append((k, 1 if a[k] == parent[v] else -1))
        v = parent[v]
    while u != v:
        k = pbr[u]
        up.append((k, 1 if a[k] == u else -1))
        u = parent[u]
        k = pbr[v]
        down.append((k, 1 if a[k] == parent[v] else -1))
        v = parent[v]
    return up + down[::-1]


MEM_SHARE = 0.4                      # of the available RAM, at most, for the dense matrices


def available_ram():
    """Available memory [bytes] (Linux /proc/meminfo; 4 GB if unknown)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 4 << 30


def dense_bytes(B, m):
    """Peak memory of Network for B bars and m meshes: L (B²), the threads' row blocks, L·M (B m),
    the mesh matrices and one complex solve (m², twice)."""
    return int(8 * B * B + 8 * B * m + 8 * m * m + 2 * 16 * m * m + THREAD_BYTES * THREADS)


def check_memory(B, m):
    need, have = dense_bytes(B, m), available_ram()
    if need > MEM_SHARE * have:
        raise ValueError(f"The mesh is too large for this computer: {B} current bars need about "
                         f"{need / 2 ** 30:.1f} GB, {have / 2 ** 30:.1f} GB is free. Lower the mesh cells "
                         f"or the margin.")


def components(mesh):
    """Connected piece of copper of each node (an index per node)."""
    return _forest(mesh.n_nodes, mesh.br_a, mesh.br_b)[3]


class Network:
    """The meshed copper with its dense L, ready for port impedances at any frequency.

    ports: list of (terminal, reference terminal) on one net: port current flows into the
    copper at the terminal and out at the reference; port voltage = V(terminal) - V(reference)."""

    def __init__(self, mesh, ports, progress=lambda m: None):
        self.mesh, self.ports = mesh, list(ports)
        a, b = mesh.br_a, mesh.br_b
        N, B = mesh.n_nodes, mesh.n_branches
        parent, pbr, depth, comp = _forest(N, a, b)
        T = mesh.terminals
        for t, r in self.ports:
            if comp[T[t]] != comp[T[r]]:
                raise ValueError(f"{t} and {r} are not joined by copper inside the window (enlarge the margin, "
                                 "or check the net).")
        tree = np.zeros(B, bool)
        tree[pbr[pbr >= 0]] = True
        rows, cols, vals = [], [], []
        cot = np.flatnonzero(~tree)
        for col, e in enumerate(cot):
            rows.append(e)
            cols.append(col)
            vals.append(1.0)
            for k, s in _path(b[e], a[e], parent, pbr, depth, a):
                rows.append(k)
                cols.append(col)
                vals.append(float(s))
        self.M = sp.csc_matrix((vals, (rows, cols)), shape=(B, len(cot)))
        rows, cols, vals = [], [], []
        for col, (t, r) in enumerate(self.ports):
            for k, s in _path(T[t], T[r], parent, pbr, depth, a):
                rows.append(k)
                cols.append(col)
                vals.append(float(s))
        self.P = sp.csc_matrix((vals, (rows, cols)), shape=(B, len(self.ports)))
        check_memory(B, self.M.shape[1])
        progress(f"Partial inductances of {B} bars...")
        self.L = inductance_matrix(mesh)
        progress(f"Mesh matrices ({self.M.shape[1]} meshes)...")
        LM = np.asarray((self.M.T @ self.L).T)                  # B x m
        LP = np.asarray((self.P.T @ self.L).T)                  # B x p
        self.MLM = np.asarray(self.M.T @ LM)
        self.MLP = np.asarray(self.M.T @ LP)
        self.PLP = np.asarray(self.P.T @ LP)
        self.LP = LP
        self.LM = LM

    @property
    def n_meshes(self):
        return self.M.shape[1]

    def _R(self, f):
        m = self.mesh
        return m.br_R * skin_factor(f, m.br_t)

    def solve(self, f, hf=False):
        """(Zp, X): port impedance matrix at f [Hz] and X with branch currents I = (P - M X) u for
        port currents u.  hf=True: the high-frequency limit (R -> 0), Zp = jωL_hf (returns L_hf
        as a real matrix in place of Zp)."""
        M, P = self.M, self.P
        if hf:
            A = self.MLM
            rhs = self.MLP
            X = linalg.solve(A, rhs, assume_a="sym") if A.size else np.zeros((0, P.shape[1]))
            Lp = self.PLP - self.MLP.T @ X
            return (Lp + Lp.T) / 2, X
        w = 2 * math.pi * f
        Rb = self._R(f)
        Rd = sp.diags(Rb)
        MRM = (M.T @ Rd @ M).toarray()
        MRP = (M.T @ Rd @ P).toarray()
        PRP = (P.T @ Rd @ P).toarray()
        A = MRM + 1j * w * self.MLM
        rhs = MRP + 1j * w * self.MLP
        X = linalg.solve(A, rhs, assume_a="sym") if A.size else np.zeros((0, P.shape[1]), complex)
        Zp = PRP + 1j * w * self.PLP - rhs.T @ X
        return (Zp + Zp.T) / 2, X

    def currents(self, X, u):
        """Branch currents [A] for port currents u (from solve())."""
        u = np.asarray(u)
        return self.P @ u - self.M @ (X @ u)

    def energy(self, I):
        """Per-branch share of the magnetic energy 2W = Re(conj(I) . L I) (sums to I* L I)."""
        return np.real(np.conj(I) * (self.L @ I))


# ---- capacitance --------------------------------------------------------------------------
def capacitance(mesh, stackup, max_panels=6000):
    """Maxwell capacitance matrix [F] between the mesh's nets (n_nets x n_nets).

    Each copper cell is a charged square panel; the nets are equipotential.  Potential
    coefficients: exact between near panels, 1/r between far ones, in one homogeneous dielectric.
    Cells are merged 2 x 2, 3 x 3, ... per net and layer when there are more than max_panels."""
    g, layers = mesh.grid, mesh.layers
    c = 1
    while True:
        rows = []
        for k, L in enumerate(layers):
            node = mesh.cell_node[L]
            f = mesh.cell_f[L]
            jj, ii = np.nonzero(node >= 0)
            net = mesh.node_net[node[jj, ii]]
            key = (net * 7 + k) * 10 ** 12 + (jj // c) * 10 ** 6 + ii // c
            uk, inv = np.unique(key, return_inverse=True)
            area = np.bincount(inv, f[jj, ii] * (g.h * 1e-3) ** 2)
            xs = np.bincount(inv, f[jj, ii] * (g.x0 + (ii + 0.5) * g.h)) / np.maximum(np.bincount(inv, f[jj, ii]), 1e-30)
            ys = np.bincount(inv, f[jj, ii] * (g.y0 + (jj + 0.5) * g.h)) / np.maximum(np.bincount(inv, f[jj, ii]), 1e-30)
            nets = np.zeros(len(uk), np.int64)
            nets[inv] = net
            rows.append((np.full(len(uk), k), xs * 1e-3, ys * 1e-3, area, nets))
        n = sum(len(r[0]) for r in rows)
        if n <= max_panels:
            break
        c += 1
    lay = np.concatenate([r[0] for r in rows])
    X = np.concatenate([r[1] for r in rows])
    Y = np.concatenate([r[2] for r in rows])
    A = np.concatenate([r[3] for r in rows])
    NET = np.concatenate([r[4] for r in rows])
    # one εr for the board (series average top to bottom): right for the dominant layer-to-layer
    # capacitance; coplanar coupling on an outer layer (half in air) reads up to 2εr/(1 + εr) high
    er = stackup.er_between(layers[0], layers[-1]) if len(layers) > 1 else 1.0
    a = np.sqrt(A)
    dz = mesh.gap[lay][:, lay]                                   # facing copper surfaces, as for L
    dx, dy = X[:, None] - X[None, :], Y[:, None] - Y[None, :]
    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    Pm = 1.0 / np.maximum(d, 1e-12)
    ii, jj = np.nonzero(d < 3 * np.maximum(a[:, None], a[None, :]))     # near pairs and self: exact squares
    ai, aj = a[ii] / 2, a[jj] / 2
    I = rect_integral(X[ii] - ai, X[ii] + ai, Y[ii] - ai, Y[ii] + ai,
                      X[jj] - aj, X[jj] + aj, Y[jj] - aj, Y[jj] + aj, dz[ii, jj])
    Pm[ii, jj] = I / (A[ii] * A[jj])
    Pm = (Pm + Pm.T) / (2 * 4 * math.pi * EPS0 * er)
    nn = len(mesh.nets)
    S = np.zeros((len(A), nn))
    S[np.arange(len(A)), NET] = 1.0
    Q = linalg.solve(Pm, S, assume_a="pos")
    Cm = S.T @ Q
    return (Cm + Cm.T) / 2
