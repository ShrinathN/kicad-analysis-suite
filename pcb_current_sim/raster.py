"""Coverage-weighted rasterisation of copper onto a uniform grid.

Each cell stores the fraction f in [0, 1] of its area covered by copper,
estimated by S x S supersampling.  Large primitives are only supersampled in
the band of cells straddling their edge.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .board import Capsule

S_SUB = 4
FULL_LIMIT = 2_000_000      # sub-samples per primitive before switching to band mode


@dataclass
class Grid:
    x0: float
    y0: float
    h: float       # mm
    nx: int
    ny: int

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def extent(self):
        """imshow extent (left, right, bottom, top) for a y-down image."""
        return (self.x0, self.x0 + self.nx * self.h, self.y0 + self.ny * self.h, self.y0)

    def centres(self, j0=0, j1=None, i0=0, i1=None):
        j1 = self.ny if j1 is None else j1
        i1 = self.nx if i1 is None else i1
        xs = self.x0 + (np.arange(i0, i1) + 0.5) * self.h
        ys = self.y0 + (np.arange(j0, j1) + 0.5) * self.h
        return np.meshgrid(xs, ys)

    def cell(self, x, y):
        return int(math.floor((y - self.y0) / self.h)), int(math.floor((x - self.x0) / self.h))

    def window(self, bbox, pad=1):
        x0, y0, x1, y1 = bbox
        i0 = max(0, int(math.floor((x0 - self.x0) / self.h)) - pad)
        i1 = min(self.nx, int(math.ceil((x1 - self.x0) / self.h)) + pad)
        j0 = max(0, int(math.floor((y0 - self.y0) / self.h)) - pad)
        j1 = min(self.ny, int(math.ceil((y1 - self.y0) / self.h)) + pad)
        return j0, j1, i0, i1

    def coarsen(self, c):
        return Grid(self.x0, self.y0, self.h * c, -(-self.nx // c), -(-self.ny // c))


def coverage(geom, grid, S=S_SUB, faces=False):
    """Area coverage of one primitive: (j0, j1, i0, i1, cov[, apx, apy]) or None if off-grid.

    With faces=True also return face apertures (copper fraction of each cell
    face): apx[j, k] is the face at x = x0 + (i0 + k) h of row j0 + j, shape
    (rows, cols + 1); apy likewise for horizontal faces, shape (rows + 1, cols).
    """
    j0, j1, i0, i1 = grid.window(geom.bbox)
    if j1 <= j0 or i1 <= i0:
        return None
    h, w, hh = grid.h, i1 - i0, j1 - j0
    sub = (np.arange(S) + 0.5) / S
    xs_sub = grid.x0 + ((np.arange(i0, i1)[:, None] + sub[None, :]).ravel()) * h
    ys_sub = grid.y0 + ((np.arange(j0, j1)[:, None] + sub[None, :]).ravel()) * h
    xs_face = grid.x0 + np.arange(i0, i1 + 1) * h
    ys_face = grid.y0 + np.arange(j0, j1 + 1) * h
    if hasattr(geom, "lattice"):        # polygons (zones): exact supersampling by scanlines
        cov = geom.lattice(xs_sub, ys_sub).reshape(hh, S, w, S).mean(axis=(1, 3), dtype=np.float32)
        if not cov.any():
            return None
        if not faces:
            return j0, j1, i0, i1, cov
        apx = geom.lattice(xs_face, ys_sub).reshape(hh, S, w + 1).mean(axis=1, dtype=np.float32)
        apy = geom.lattice(xs_sub, ys_face).reshape(hh + 1, w, S).mean(axis=2, dtype=np.float32)
        return j0, j1, i0, i1, cov, apx, apy
    if w * hh * S * S <= FULL_LIMIT:
        X, Y = np.meshgrid(xs_sub, ys_sub)
        cov = geom.inside(X, Y).reshape(hh, S, w, S).mean(axis=(1, 3), dtype=np.float32)
        if not cov.any():
            return None
        if not faces:
            return j0, j1, i0, i1, cov
        X, Y = np.meshgrid(xs_face, ys_sub)
        apx = geom.inside(X, Y).reshape(hh, S, w + 1).mean(axis=1, dtype=np.float32)
        X, Y = np.meshgrid(xs_sub, ys_face)
        apy = geom.inside(X, Y).reshape(hh + 1, w, S).mean(axis=2, dtype=np.float32)
        return j0, j1, i0, i1, cov, apx, apy
    # large primitive: centre test, supersample only the band straddling the edge
    Xc, Yc = grid.centres(j0, j1, i0, i1)
    c = geom.inside(Xc, Yc)
    st = np.ones((3, 3), bool)
    band = ndimage.binary_dilation(c, st) & ~ndimage.binary_erosion(c, st, border_value=0)
    cov = c.astype(np.float32)
    jj, ii = np.nonzero(band)
    off = (sub - 0.5) * h
    ox, oy = np.meshgrid(off, off)
    ox, oy = ox.ravel(), oy.ravel()
    for k in range(0, jj.size, 100_000):
        sj, si = jj[k:k + 100_000], ii[k:k + 100_000]
        cov[sj, si] = geom.inside(Xc[sj, si][:, None] + ox[None, :], Yc[sj, si][:, None] + oy[None, :]).mean(axis=1)
    if not cov.any():
        return None
    if not faces:
        return j0, j1, i0, i1, cov
    cp = np.pad(c, 1)
    bp = np.pad(band, 1)
    apx = (cp[1:-1, :-1] & cp[1:-1, 1:]).astype(np.float32)          # (hh, w + 1)
    apy = (cp[:-1, 1:-1] & cp[1:, 1:-1]).astype(np.float32)          # (hh + 1, w)
    for ap, fb, axis in ((apx, bp[1:-1, :-1] | bp[1:-1, 1:], "x"), (apy, bp[:-1, 1:-1] | bp[1:, 1:-1], "y")):
        fj, fi = np.nonzero(fb)
        for k in range(0, fj.size, 100_000):
            sj, si = fj[k:k + 100_000], fi[k:k + 100_000]
            if axis == "x":
                X = np.repeat(xs_face[si][:, None], S, 1)
                Y = grid.y0 + (j0 + sj)[:, None] * h + sub[None, :] * h
            else:
                X = grid.x0 + (i0 + si)[:, None] * h + sub[None, :] * h
                Y = np.repeat(ys_face[sj][:, None], S, 1)
            ap[sj, si] = geom.inside(X, Y).mean(axis=1)
    return j0, j1, i0, i1, cov, apx, apy


@dataclass
class Prim:
    """A conducting primitive, for the per-primitive loss ranking."""
    kind: str      # track | arc | zone | pad | via
    net: str
    layer: str
    label: str
    width: float = 0.0
    length: float = 0.0
    start: tuple = None
    end: tuple = None
    centre: tuple = None


@dataclass
class LayerRaster:
    f: np.ndarray          # coverage of the simulated nets, float32
    net: np.ndarray        # net index (int16), -1 = none
    owner: np.ndarray      # Prim index (int32), -1 = none
    apx: np.ndarray        # vertical-face apertures (ny, nx + 1); column i = left face of cell i
    apy: np.ndarray        # horizontal-face apertures (ny + 1, nx); row j = top face of cell j


@dataclass
class Raster:
    grid: Grid
    nets: list                         # simulated net names (index = net id)
    layers: dict                       # layer -> LayerRaster
    prims: list
    pad_cells: dict = field(default_factory=dict)   # (pad index, layer) -> flat cell indices
    holes: list = field(default_factory=list)       # dicts: kind, ref, net, pos, drill, layers, geom
    warnings: list = field(default_factory=list)


def copper_bbox(board, nets, margin=0.5):
    """Bounding box of all copper of `nets` (plus margin), clipped to the board."""
    nets = set(nets)
    bs = [t.geom.bbox for t in board.tracks if t.net in nets]
    bs += [z.geom.bbox for z in board.zones if z.net in nets]
    bs += [p.geom.bbox for p in board.pads if p.net in nets and p.geom is not None]
    bs += [(v.pos[0] - v.size / 2, v.pos[1] - v.size / 2, v.pos[0] + v.size / 2, v.pos[1] + v.size / 2)
           for v in board.vias if v.net in nets]
    if not bs:
        return None
    b = np.array(bs, float)
    X0, Y0, X1, Y1 = board.bbox()
    return (max(b[:, 0].min() - margin, X0), max(b[:, 1].min() - margin, Y0),
            min(b[:, 2].max() + margin, X1), min(b[:, 3].max() + margin, Y1))


def make_grid(board, pitch, margin=0.5, bbox=None):
    """Grid over `bbox` (default: the whole board) plus margin.

    A cropped grid is snapped to the lattice of the whole-board grid, so
    cropping never changes which copper a cell sees."""
    X0, Y0 = board.bbox()[0] - margin, board.bbox()[1] - margin
    x0, y0, x1, y1 = bbox or board.bbox()
    x0 = X0 + math.floor((x0 - margin - X0) / pitch + 1e-9) * pitch
    y0 = Y0 + math.floor((y0 - margin - Y0) / pitch + 1e-9) * pitch
    nx = int(math.ceil((x1 + margin - x0) / pitch))
    ny = int(math.ceil((y1 + margin - y0) / pitch))
    return Grid(x0, y0, pitch, nx, ny)


def auto_pitch(board, nets, max_cells=1_500_000, max_grid=4_000_000, bbox=None):
    """Pick a grid pitch [mm] from feature size and cell budgets."""
    nets = set(nets)
    widths = [t.width for t in board.tracks if t.net in nets]
    widths += [min(p.size) for p in board.pads if p.net in nets and min(p.size) > 0]
    wmin = min(widths) if widths else 1.0
    area = sum(t.width * t.length for t in board.tracks if t.net in nets)
    area += sum(z.geom.area() for z in board.zones if z.net in nets)
    area += sum(p.size[0] * p.size[1] * len(p.layers) for p in board.pads if p.net in nets)
    x0, y0, x1, y1 = bbox or board.bbox()
    nl = len(board.copper_layers)
    h = max(wmin / 12, math.sqrt(area / max_cells), math.sqrt((x1 - x0) * (y1 - y0) * nl / max_grid), 0.005)
    # round up to a tidy value (1-2-2.5-5 steps)
    e = 10 ** math.floor(math.log10(h))
    for m in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= h - 1e-12:
            return round(m * e, 6)
    return h


def rasterise(board, nets, grid):
    """Rasterise all copper of `nets` on every copper layer."""
    nets = list(nets)
    nid = {n: k for k, n in enumerate(nets)}
    R = Raster(grid, nets, {}, [])
    for L in board.copper_layers:
        R.layers[L] = LayerRaster(np.zeros(grid.shape, np.float32),
                                  np.full(grid.shape, -1, np.int16),
                                  np.full(grid.shape, -1, np.int32),
                                  np.zeros((grid.ny, grid.nx + 1), np.float32),
                                  np.zeros((grid.ny + 1, grid.nx), np.float32))
    best = {L: np.zeros(grid.shape, np.float32) for L in board.copper_layers}

    def paint(geom, layer, net, prim, prefer=False):
        res = coverage(geom, grid, faces=True)
        if res is None:
            return None
        j0, j1, i0, i1, cov, apx, apy = res
        lr, sl = R.layers[layer], (slice(j0, j1), slice(i0, i1))
        np.maximum(lr.f[sl], cov, out=lr.f[sl])
        sx, sy = (slice(j0, j1), slice(i0, i1 + 1)), (slice(j0, j1 + 1), slice(i0, i1))
        np.maximum(lr.apx[sx], apx, out=lr.apx[sx])
        np.maximum(lr.apy[sy], apy, out=lr.apy[sy])
        nsl = lr.net[sl]
        clash = (cov > 0) & (nsl >= 0) & (nsl != nid[net])
        if clash.any():
            R.warnings.append(f"Copper of nets {net} and {nets[int(nsl[clash][0])]} overlaps on {layer}.")
        nsl[cov > 0] = nid[net]
        b = best[layer][sl]
        win = (cov >= b) if prefer else (cov > b)
        win &= cov > 0
        lr.owner[sl][win] = prim
        b[win] = cov[win]
        return res

    def add_prim(**kw):
        R.prims.append(Prim(**kw))
        return len(R.prims) - 1

    for z in board.zones:
        if z.net in nid:
            c = tuple(z.geom.pts.mean(0))
            paint(z.geom, z.layer, z.net, add_prim(kind="zone", net=z.net, layer=z.layer,
                                                   label=f"zone {z.net} {z.layer}", centre=c))
    for k, t in enumerate(board.tracks):
        if t.net in nid:
            mid = ((t.start[0] + t.end[0]) / 2, (t.start[1] + t.end[1]) / 2)
            pid = add_prim(kind="track" if t.kind == "segment" else "arc", net=t.net, layer=t.layer,
                           label=f"{t.kind} #{k} w={t.width:g} L={t.length:.2f}", width=t.width,
                           length=t.length, start=t.start, end=t.end, centre=mid)
            paint(t.geom, t.layer, t.net, pid)
    for k, v in enumerate(board.vias):
        if v.net in nid:
            ring = Capsule(*v.pos, *v.pos, v.size / 2)
            for L in v.layers:
                paint(ring, L, v.net, add_prim(kind="via", net=v.net, layer=L, label=f"via #{k} ring",
                                               centre=v.pos), prefer=True)
    for k, p in enumerate(board.pads):
        if p.net in nid and p.geom is not None:
            for L in p.layers:
                res = paint(p.geom, L, p.net, add_prim(kind="pad", net=p.net, layer=L,
                                                       label=f"pad {p.id} {L}", centre=p.pos), prefer=True)
                if res is not None:
                    j0, j1, i0, i1, cov = res[:5]
                    jj, ii = np.nonzero(cov >= 0.5)
                    R.pad_cells[(k, L)] = (jj + j0) * grid.nx + (ii + i0)

    # drill holes remove copper (any net)
    for k, p in enumerate(board.pads):
        if p.hole_geom is not None:
            R.holes.append(dict(kind="pad", index=k, label=p.id, net=p.net, pos=p.pos,
                                drill=sum(p.drill) / 2, layers=list(p.layers) if p.plated else [],
                                geom=p.hole_geom))
    for k, v in enumerate(board.vias):
        R.holes.append(dict(kind="via", index=k, label=f"via #{k}", net=v.net, pos=v.pos,
                            drill=v.drill, layers=list(v.layers),
                            geom=Capsule(*v.pos, *v.pos, v.drill / 2)))
    for hole in R.holes:
        res = coverage(hole["geom"], grid, faces=True)
        if res is None:
            continue
        j0, j1, i0, i1, cov, apx, apy = res
        for L in board.copper_layers:
            lr, sl = R.layers[L], (slice(j0, j1), slice(i0, i1))
            lr.f[sl] *= 1 - cov
            lr.apx[j0:j1, i0:i1 + 1] *= 1 - apx
            lr.apy[j0:j1 + 1, i0:i1] *= 1 - apy
    for lr in R.layers.values():
        dead = lr.f < 0.02
        lr.f[dead] = 0
        lr.net[dead] = -1
        lr.owner[dead] = -1
    for key in list(R.pad_cells):
        cells = R.pad_cells[key]
        R.pad_cells[key] = cells[R.layers[key[1]].f.ravel()[cells] > 0]
    return R


def ring_cells(R, hole, layer, net_id):
    """Flat indices of copper cells of `net_id` touching the hole wall."""
    g, lr = R.grid, R.layers[layer]
    geom = hole["geom"]
    r = geom.r
    j0, j1, i0, i1 = g.window(geom.bbox, pad=3)
    X, Y = g.centres(j0, j1, i0, i1)
    d = geom.dist(X, Y)
    sub = (lr.f[j0:j1, i0:i1] > 0) & (lr.net[j0:j1, i0:i1] == net_id)
    for band in (1.0, 2.0, 3.0):
        m = sub & (d >= r - 0.5 * g.h) & (d <= r + band * g.h)
        if m.any():
            jj, ii = np.nonzero(m)
            return (jj + j0) * g.nx + (ii + i0)
    return np.zeros(0, np.int64)


def coverage_all(board, grid, sub=0.03, faces=False):
    """Copper fraction of each cell (every net) per layer, for thermal spreading.

    Primitives are OR-ed on a sub-lattice of about `sub` mm and averaged per cell, so copper of
    several primitives in one coarse cell adds up (a per-primitive maximum would not).
    faces=True also returns the copper fraction of the faces between neighbouring cells:
    {layer: (f, apx (ny, nx-1), apy (ny-1, nx))}, apx[j, i] being the face between i and i+1."""
    S = max(2, min(16, math.ceil(grid.h / sub)))
    hs = grid.h / S
    masks = {L: np.zeros((grid.ny * S, grid.nx * S), bool) for L in board.copper_layers}

    def paint(geom, L):
        x0, y0, x1, y1 = geom.bbox
        i0 = max(0, int(math.floor((x0 - grid.x0) / hs)))
        i1 = min(grid.nx * S, int(math.ceil((x1 - grid.x0) / hs)) + 1)
        j0 = max(0, int(math.floor((y0 - grid.y0) / hs)))
        j1 = min(grid.ny * S, int(math.ceil((y1 - grid.y0) / hs)) + 1)
        if i1 <= i0 or j1 <= j0:
            return
        xs = grid.x0 + (np.arange(i0, i1) + 0.5) * hs
        ys = grid.y0 + (np.arange(j0, j1) + 0.5) * hs
        if hasattr(geom, "lattice"):
            m = geom.lattice(xs, ys)
        else:
            X, Y = np.meshgrid(xs, ys)
            m = geom.inside(X, Y)
        masks[L][j0:j1, i0:i1] |= m

    for z in board.zones:
        paint(z.geom, z.layer)
    for t in board.tracks:
        paint(t.geom, t.layer)
    for v in board.vias:
        for L in v.layers:
            paint(Capsule(*v.pos, *v.pos, v.size / 2), L)
    for p in board.pads:
        if p.geom is not None:
            for L in p.layers:
                paint(p.geom, L)
    out = {}
    for L, m in masks.items():
        f = m.reshape(grid.ny, S, grid.nx, S).mean(axis=(1, 3), dtype=np.float32)
        if not faces:
            out[L] = f
            continue
        # a face carries copper where the sub-cells on both sides of it are copper
        fx = m[:, S - 1:-1:S] & m[:, S::S]                   # (ny*S, nx-1)
        fy = m[S - 1:-1:S, :] & m[S::S, :]                   # (ny-1, nx*S)
        apx = fx.reshape(grid.ny, S, grid.nx - 1).mean(axis=1, dtype=np.float32)
        apy = fy.reshape(grid.ny - 1, grid.nx, S).mean(axis=2, dtype=np.float32)
        out[L] = (f, apx, apy)
    return out
