"""Board model parsed from a .kicad_pcb file (KiCad 6-10 formats).

All coordinates are in mm, in KiCad's y-down board frame.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

from . import sexpr
from .sexpr import Q, find, findall, floats, value

COPPER_TYPES = {"signal", "power", "mixed", "jumper"}
DEFAULT_CU = 35e-6       # m
DEFAULT_BOARD = 1.6      # mm


def rot(x, y, deg):
    """Rotate a local offset by a KiCad angle (CCW on screen, y down)."""
    if not deg:
        return x, y
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return x * c + y * s, -x * s + y * c


# ---------------------------------------------------------------- geometry
class Capsule:
    """Segment a-b swept by a disc of radius r (also circles when a == b)."""

    def __init__(self, ax, ay, bx, by, r):
        self.a = (ax, ay)
        self.b = (bx, by)
        self.r = r
        self.bbox = (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r)

    def dist(self, X, Y):
        (ax, ay), (bx, by) = self.a, self.b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return np.hypot(X - ax, Y - ay)
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
        return np.hypot(X - ax - t * dx, Y - ay - t * dy)

    def inside(self, X, Y):
        return self.dist(X, Y) <= self.r


class RoundRect:
    """Rectangle sx x sy centred at c, rotated by `angle`, corner radius rr."""

    def __init__(self, cx, cy, sx, sy, angle=0.0, rr=0.0):
        self.c = (cx, cy)
        self.sx, self.sy, self.angle = sx, sy, angle
        self.rr = min(rr, sx / 2, sy / 2)
        R = math.hypot(sx, sy) / 2
        self.bbox = (cx - R, cy - R, cx + R, cy + R)

    def inside(self, X, Y):
        # rotate the query points into the pad frame (inverse rotation)
        u, v = rot(X - self.c[0], Y - self.c[1], -self.angle)
        hx, hy, r = self.sx / 2 - self.rr, self.sy / 2 - self.rr, self.rr
        qx = np.maximum(np.abs(u) - hx, 0)
        qy = np.maximum(np.abs(v) - hy, 0)
        return qx * qx + qy * qy <= r * r + 1e-15


class Polygon:
    def __init__(self, pts):
        self.pts = np.asarray(pts, float)
        self.path = MplPath(self.pts)
        self.bbox = (*self.pts.min(0), *self.pts.max(0))

    def inside(self, X, Y):
        P = np.column_stack([np.ravel(X), np.ravel(Y)])
        return self.path.contains_points(P).reshape(np.shape(X))

    def lattice(self, xs, ys):
        """Even-odd inside test on the lattice xs (columns, increasing) x ys (rows, increasing) by
        scanlines: (len(ys), len(xs)) bool.  Much faster than contains_points for big zones."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        out = np.zeros((len(ys), len(xs)), bool)
        p = self.pts
        q = np.roll(p, -1, axis=0)
        ymin, ymax = np.minimum(p[:, 1], q[:, 1]), np.maximum(p[:, 1], q[:, 1])
        for r0 in range(0, len(ys), 1024):
            yc = ys[r0:r0 + 1024]
            lo = np.searchsorted(yc, ymin, "left")          # rows with ymin <= y < ymax cross the edge
            n = np.searchsorted(yc, ymax, "left") - lo
            if not n.any():
                continue
            e = np.repeat(np.arange(len(p)), n)
            r = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + lo[e]
            xc = p[e, 0] + (yc[r] - p[e, 1]) * (q[e, 0] - p[e, 0]) / (q[e, 1] - p[e, 1])
            k = np.searchsorted(xs, xc, "left")            # columns left of the crossing flip parity
            w = len(xs) + 1
            diff = np.bincount(r * w, minlength=len(yc) * w) - np.bincount(r * w + k, minlength=len(yc) * w)
            out[r0:r0 + len(yc)] = (np.cumsum(diff.reshape(len(yc), w), axis=1)[:, :-1] & 1).astype(bool)
        return out

    def area(self):
        x, y = self.pts[:, 0], self.pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class Union:
    def __init__(self, parts):
        self.parts = parts
        b = np.array([p.bbox for p in parts])
        self.bbox = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())

    def inside(self, X, Y):
        m = np.zeros(np.shape(X), bool)
        for p in self.parts:
            m |= p.inside(X, Y)
        return m


def arc_points(s, m, e, max_step_deg=5.0):
    """Points along the circular arc through start, mid, end."""
    (x1, y1), (x2, y2), (x3, y3) = s, m, e
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [s, e]
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)
    # sweep from a1 to a3 passing through a2
    sweep = (a3 - a1) % (2 * math.pi)
    if (a2 - a1) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    n = max(2, int(math.ceil(abs(math.degrees(sweep)) / max_step_deg)))
    return [(ux + r * math.cos(a1 + sweep * k / n), uy + r * math.sin(a1 + sweep * k / n))
            for k in range(n + 1)]


def pts_of(node):
    """Points of a (pts (xy ..) (arc ..)) list."""
    out = []
    p = find(node, "pts")
    if p is None:
        return out
    for c in p[1:]:
        if c[0] == "xy":
            out.append((float(c[1]), float(c[2])))
        elif c[0] == "arc":
            out += arc_points(floats(c, "start"), floats(c, "mid"), floats(c, "end"))
    return out


# ------------------------------------------------------------------- items
@dataclass
class Track:
    kind: str          # "segment" | "arc"
    layer: str
    net: str
    width: float
    start: tuple
    end: tuple
    length: float
    geom: object
    uuid: str = ""


@dataclass
class Via:
    pos: tuple
    size: float
    drill: float
    layers: list       # copper layers spanned, in stack order
    net: str
    uuid: str = ""


@dataclass
class Zone:
    net: str
    layer: str
    geom: Polygon


@dataclass
class Pad:
    ref: str
    number: str
    kind: str          # thru_hole | smd | connect | np_thru_hole
    pos: tuple
    angle: float
    shape: str
    size: tuple
    layers: list       # copper layers
    net: str
    geom: object
    drill: tuple = None    # (dx, dy) mm, None for SMD
    hole_geom: object = None

    @property
    def id(self):
        return f"{self.ref}.{self.number}"

    @property
    def plated(self):
        return self.kind == "thru_hole" and self.drill is not None


@dataclass
class Footprint:
    ref: str
    value: str
    lib: str
    pos: tuple
    angle: float
    layer: str             # F.Cu or B.Cu (component side)
    dnp: bool = False
    pads: list = field(default_factory=list)    # indices into Board.pads


@dataclass
class Board:
    path: str
    copper_layers: list
    cu_thickness: dict          # layer -> m
    gaps: list                  # dielectric thickness (m) between layer k and k+1
    thickness: float            # board thickness, mm
    stackup_found: bool
    dielectrics: list = field(default_factory=list)  # per gap: plies [{name, type, material, t (m), er, tand}]
    tracks: list = field(default_factory=list)
    vias: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    pads: list = field(default_factory=list)
    footprints: list = field(default_factory=list)
    outline: list = field(default_factory=list)   # list of Nx2 closed loops
    warnings: list = field(default_factory=list)

    @property
    def nets(self):
        s = {t.net for t in self.tracks} | {p.net for p in self.pads} | {v.net for v in self.vias} \
            | {z.net for z in self.zones}
        return sorted(n for n in s if n)

    def pads_by_id(self, pid):
        return [p for p in self.pads if p.id == pid]

    def pad_ids(self, net=None):
        seen = []
        for p in self.pads:
            if (net is None or p.net == net) and p.id not in seen and p.kind != "np_thru_hole":
                seen.append(p.id)
        return seen

    def bbox(self):
        if self.outline:
            a = np.vstack(self.outline)
            return (*a.min(0), *a.max(0))
        bs = [t.geom.bbox for t in self.tracks] + [p.geom.bbox for p in self.pads if p.geom] \
            + [z.geom.bbox for z in self.zones]
        bs += [(v.pos[0] - v.size / 2, v.pos[1] - v.size / 2, v.pos[0] + v.size / 2, v.pos[1] + v.size / 2)
               for v in self.vias]
        b = np.array(bs)
        return (b[:, 0].min() - 1, b[:, 1].min() - 1, b[:, 2].max() + 1, b[:, 3].max() + 1)

    def outline_mask(self, X, Y):
        """True inside the board outline (even-odd over all loops)."""
        if not self.outline:
            x0, y0, x1, y1 = self.bbox()
            return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        m = np.zeros(np.shape(X), bool)
        P = np.column_stack([np.ravel(X), np.ravel(Y)])
        for loop in self.outline:
            m ^= MplPath(loop).contains_points(P).reshape(np.shape(X))
        return m

    def layer_span(self, a, b):
        i, j = self.copper_layers.index(a), self.copper_layers.index(b)
        if i > j:
            i, j = j, i
        return self.copper_layers[i:j + 1]

    def is_outer(self, layer):
        return layer in (self.copper_layers[0], self.copper_layers[-1])


# ----------------------------------------------------------------- parsing
def _layer_order(name):
    if name == "F.Cu":
        return -1
    if name == "B.Cu":
        return 10_000
    m = re.match(r"In(\d+)\.Cu", name)
    return int(m.group(1)) if m else 5_000


def _expand_layers(names, copper):
    out = []
    for n in names:
        if n in ("*.Cu", "F&B.Cu"):
            cand = copper if n == "*.Cu" else [l for l in ("F.Cu", "B.Cu") if l in copper]
            out += [l for l in cand if l not in out]
        elif n in copper and n not in out:
            out.append(n)
    return sorted(out, key=copper.index)


def _sublayers(layer):
    """Split a stackup layer node into one pseudo-node per ply (the first plus each `addsublayer`)."""
    out, cur = [], [layer[0]]
    for c in layer[1:]:
        if c == "addsublayer":
            out.append(cur)
            cur = [layer[0]]
        elif isinstance(c, list):
            cur.append(c)
    return out + [cur]


def load(path):
    with open(path, encoding="utf-8") as f:
        return parse_board(f.read(), path)


def parse_board(text, path="<string>"):
    root = sexpr.parse(text)
    warnings = []

    # nets table (KiCad <= 9 uses numbered nets)
    net_table = {}
    for n in findall(root, "net"):
        if len(n) >= 3:
            net_table[str(n[1])] = str(n[2])

    def net_of(node):
        c = find(node, "net")
        if c is None or len(c) < 2:
            return ""
        if len(c) >= 3:
            return str(c[2])
        a = c[1]
        if isinstance(a, Q):
            return str(a)
        return net_table.get(str(a), str(a))

    # copper layers
    copper = []
    layers = find(root, "layers")
    for l in (layers[1:] if layers else []):
        if isinstance(l, list) and len(l) >= 3 and str(l[2]) in COPPER_TYPES and str(l[1]).endswith(".Cu"):
            copper.append(str(l[1]))
    copper.sort(key=_layer_order)
    if not copper:
        copper = ["F.Cu", "B.Cu"]

    thickness = float(value(find(root, "general") or ["general"], "thickness", DEFAULT_BOARD))

    # stackup
    cu_t = {l: DEFAULT_CU for l in copper}
    gaps = None
    stack = None
    setup = find(root, "setup")
    if setup is not None:
        stack = find(setup, "stackup")
    plies_by_gap = None
    if stack is not None:
        gaps, plies_by_gap, cur, plies, seen_cu = [], [], 0.0, [], False
        for l in findall(stack, "layer"):
            name, typ = str(l[1]), str(value(l, "type", ""))
            if name in cu_t:
                t = float(value(l, "thickness", 0) or 0)
                if t > 0:
                    cu_t[name] = t * 1e-3
                if seen_cu:
                    gaps.append(cur * 1e-3)
                    plies_by_gap.append(plies)
                cur, plies, seen_cu = 0.0, [], True
            elif typ in ("core", "prepreg") or name.startswith("dielectric"):
                for sub in _sublayers(l):       # KiCad writes extra plies after a bare `addsublayer` atom
                    t = float(value(sub, "thickness", 0) or 0)
                    cur += t
                    er, tand = value(sub, "epsilon_r"), value(sub, "loss_tangent")
                    plies.append(dict(name=name, type=typ, material=str(value(sub, "material", "") or ""),
                                      t=t * 1e-3, er=float(er) if er is not None else None,
                                      tand=float(tand) if tand is not None else None))
        if len(gaps) != len(copper) - 1:
            warnings.append("Stackup dielectric list does not match copper layers; using even split.")
            gaps = plies_by_gap = None
    if gaps is None:
        cu_sum = sum(cu_t.values()) * 1e3
        n = max(len(copper) - 1, 1)
        gaps = [max(thickness - cu_sum, 0.1) / n * 1e-3] * (len(copper) - 1)

    board = Board(path=path, copper_layers=copper, cu_thickness=cu_t, gaps=gaps,
                  thickness=thickness, stackup_found=stack is not None, warnings=warnings,
                  dielectrics=plies_by_gap or [[dict(name="dielectric", type="core", material="", t=g,
                                                     er=None, tand=None)] for g in gaps])

    # tracks
    for s in findall(root, "segment"):
        layer = str(value(s, "layer", ""))
        if layer not in copper:
            continue
        a, b, w = floats(s, "start"), floats(s, "end"), float(value(s, "width"))
        board.tracks.append(Track("segment", layer, net_of(s), w, a, b, math.dist(a, b),
                                  Capsule(*a, *b, w / 2), str(value(s, "uuid", ""))))
    for s in findall(root, "arc"):
        layer = str(value(s, "layer", ""))
        if layer not in copper:
            continue
        a, m, b = floats(s, "start"), floats(s, "mid"), floats(s, "end")
        w = float(value(s, "width"))
        pts = arc_points(a, m, b)
        caps = [Capsule(*pts[k], *pts[k + 1], w / 2) for k in range(len(pts) - 1)]
        L = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
        board.tracks.append(Track("arc", layer, net_of(s), w, a, b, L, Union(caps),
                                  str(value(s, "uuid", ""))))

    # vias
    for v in findall(root, "via"):
        pos = floats(v, "at")
        lay = find(v, "layers")
        names = [str(x) for x in lay[1:]] if lay else ["F.Cu", "B.Cu"]
        names = [n for n in names if n in copper] or [copper[0], copper[-1]]
        span = board.layer_span(names[0], names[-1])
        board.vias.append(Via(pos, float(value(v, "size")), float(value(v, "drill")), span,
                              net_of(v), str(value(v, "uuid", ""))))

    # zones (filled copper only)
    unfilled = 0
    for z in findall(root, "zone"):
        net = net_of(z)
        fps = findall(z, "filled_polygon")
        if not fps and find(z, "keepout") is None:
            unfilled += 1
        for fp in fps:
            layer = str(value(fp, "layer", value(z, "layer", "")))
            pts = pts_of(fp)
            if layer in copper and len(pts) >= 3:
                board.zones.append(Zone(net, layer, Polygon(pts)))
    if unfilled:
        warnings.append(f"{unfilled} zone(s) have no fill: refill zones in KiCad (B) and save.")

    # footprints / pads
    for fp in findall(root, "footprint"):
        fat = find(fp, "at")
        fx, fy = float(fat[1]), float(fat[2])
        fa = float(fat[3]) if len(fat) > 3 and not isinstance(fat[3], list) else 0.0
        ref = None
        for p in findall(fp, "property"):
            if str(p[1]) == "Reference":
                ref = str(p[2])
        if ref is None:
            for t in findall(fp, "fp_text"):
                if str(t[1]) == "reference":
                    ref = str(t[2])
        ref = ref or "?"
        val = ""
        for p in findall(fp, "property"):
            if str(p[1]) == "Value":
                val = str(p[2])
        attr = find(fp, "attr")
        dnp = (attr is not None and "dnp" in [str(x) for x in attr[1:]]) or str(value(fp, "dnp", "no")) == "yes"
        side = str(value(fp, "layer", "F.Cu"))
        foot = Footprint(ref, val, str(fp[1]), (fx, fy), fa, side if side in copper else copper[0], dnp)
        board.footprints.append(foot)
        for p in findall(fp, "pad"):
            number, kind, shape = str(p[1]), str(p[2]), str(p[3])
            pat = find(p, "at")
            px, py = float(pat[1]), float(pat[2])
            pa = float(pat[3]) if len(pat) > 3 and not isinstance(pat[3], list) else fa
            dx, dy = rot(px, py, fa)
            cx, cy = fx + dx, fy + dy
            sx, sy = floats(p, "size", default=(0.0, 0.0))
            lay = find(p, "layers")
            players = _expand_layers([str(x) for x in lay[1:]] if lay else [], copper)
            drill = None
            hole = None
            d = find(p, "drill")
            if d is not None:
                nums = [float(x) for x in d[1:] if not isinstance(x, list) and x != "oval"]
                if nums and nums[0] > 0:
                    ddx = nums[0]
                    ddy = nums[1] if len(nums) > 1 else ddx
                    drill = (ddx, ddy)
                    if abs(ddx - ddy) < 1e-9:
                        hole = Capsule(cx, cy, cx, cy, ddx / 2)
                    else:
                        L = abs(ddx - ddy) / 2
                        ux, uy = rot(L, 0, pa) if ddx > ddy else rot(0, L, pa)
                        hole = Capsule(cx - ux, cy - uy, cx + ux, cy + uy, min(ddx, ddy) / 2)
            geom = _pad_geom(p, shape, cx, cy, sx, sy, pa)
            if kind == "np_thru_hole":
                players = []
            foot.pads.append(len(board.pads))
            board.pads.append(Pad(ref, number, kind, (cx, cy), pa, shape, (sx, sy), players,
                                  net_of(p), geom, drill, hole))

    # board outline
    board.outline = _outline(root)
    if not board.outline:
        warnings.append("No closed Edge.Cuts outline found; using the copper bounding box.")
    return board


def _pad_geom(p, shape, cx, cy, sx, sy, angle):
    if sx <= 0 or sy <= 0:
        return None
    if shape == "circle":
        return Capsule(cx, cy, cx, cy, sx / 2)
    if shape == "oval":
        return RoundRect(cx, cy, sx, sy, angle, min(sx, sy) / 2)
    if shape == "roundrect":
        rr = float(value(p, "roundrect_rratio", 0.25)) * min(sx, sy)
        return RoundRect(cx, cy, sx, sy, angle, rr)
    if shape == "trapezoid":
        ddx, ddy = floats(p, "rect_delta", default=(0.0, 0.0))
        hx, hy, ex, ey = sx / 2, sy / 2, ddx / 2, ddy / 2
        loc = [(-hx - ey, hy + ex), (hx + ey, hy - ex), (hx - ey, -hy + ex), (-hx + ey, -hy - ex)]
        return Polygon([(cx + a, cy + b) for a, b in (rot(x, y, angle) for x, y in loc)])
    if shape == "custom":
        opts = find(p, "options")
        anchor = str(value(opts, "anchor", "rect")) if opts else "rect"
        parts = [Capsule(cx, cy, cx, cy, sx / 2) if anchor == "circle"
                 else RoundRect(cx, cy, sx, sy, angle)]
        prims = find(p, "primitives")
        for g in (prims[1:] if prims else []):
            if g[0] == "gr_poly":
                loc = pts_of(g)
                if len(loc) >= 3:
                    parts.append(Polygon([(cx + a, cy + b) for a, b in (rot(x, y, angle) for x, y in loc)]))
            elif g[0] == "gr_circle":
                c0, e0 = floats(g, "center"), floats(g, "end")
                a, b = rot(*c0, angle)
                parts.append(Capsule(cx + a, cy + b, cx + a, cy + b, math.dist(c0, e0)))
            elif g[0] == "gr_line":
                s0, e0 = floats(g, "start"), floats(g, "end")
                w = float(value(g, "width", value(find(g, "stroke") or ["s"], "width", 0.1)))
                a, b = rot(*s0, angle)
                c, d = rot(*e0, angle)
                parts.append(Capsule(cx + a, cy + b, cx + c, cy + d, w / 2))
        return Union(parts) if len(parts) > 1 else parts[0]
    return RoundRect(cx, cy, sx, sy, angle)   # rect and anything else


def _outline(root):
    chains, loops = [], []
    for g in root[1:]:
        if not isinstance(g, list) or str(value(g, "layer", "")) != "Edge.Cuts":
            continue
        k = g[0]
        if k == "gr_line":
            chains.append([floats(g, "start"), floats(g, "end")])
        elif k == "gr_arc":
            chains.append(arc_points(floats(g, "start"), floats(g, "mid"), floats(g, "end")))
        elif k == "gr_rect":
            (x0, y0), (x1, y1) = floats(g, "start"), floats(g, "end")
            loops.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        elif k == "gr_circle":
            c, e = floats(g, "center"), floats(g, "end")
            r = math.dist(c, e)
            loops.append([(c[0] + r * math.cos(t), c[1] + r * math.sin(t))
                          for t in np.linspace(0, 2 * math.pi, 73)[:-1]])
        elif k == "gr_poly":
            pts = pts_of(g)
            if len(pts) >= 3:
                loops.append(pts)
    # chain open pieces into closed loops
    tol = 1e-3
    chains = [list(c) for c in chains if c and c[0] is not None]
    while chains:
        cur = chains.pop(0)
        grown = True
        while grown and math.dist(cur[0], cur[-1]) > tol:
            grown = False
            for i, c in enumerate(chains):
                if math.dist(cur[-1], c[0]) <= tol:
                    cur += c[1:]
                elif math.dist(cur[-1], c[-1]) <= tol:
                    cur += c[::-1][1:]
                elif math.dist(cur[0], c[-1]) <= tol:
                    cur = c[:-1] + cur
                elif math.dist(cur[0], c[0]) <= tol:
                    cur = c[::-1][:-1] + cur
                else:
                    continue
                chains.pop(i)
                grown = True
                break
        if math.dist(cur[0], cur[-1]) <= tol and len(cur) >= 4:
            loops.append(cur[:-1])
    return [np.asarray(l, float) for l in loops]
