"""Parametric magnetic cores, placed on the board and expanded into boxes (`pcb.CoreBox`).

The solver's grid is a tensor product of axis-aligned cells, so every core is a set of boxes:
round parts (round centre legs, rods, toroids) are staircases of strips with the true area per
strip, and rotations are multiples of 90°.

Shapes (dimensions in mm, letters as in IEC 62317 / maker drawings where they exist):

* ``plate``: a flat plate or ferrite sheet, L x W x T, on the top or bottom side.
* ``E``: an E core pair (``pair`` true) or E + I.  A overall width, B height of one E half,
  C depth, D window height of one half, E window width (between the outer legs' inner faces),
  F centre leg width.  ``round`` true: a round centre leg of diameter F (ER; EQ / PQ / RM
  approximately: their outer legs are shaped, here they are straight).  The legs go through
  slots in the board and the window is centred on the board; ``side`` top / bottom puts the E's
  back (of an E-I) on that side, the I on the other.
  Gaps (mm): ``gap_center`` (ground centre leg) and ``gap_outer`` (outer legs; both = a spacer)
  at the mating plane (E-E: the middle; E-I: the E's leg ends).
* ``rod``: a round rod lying along u, diameter F x length L.
* ``toroid``: OD x ID x H lying flat.
* ``box``: raw board coordinates (lo, hi in `dims`), as `pcb.CoreBox`.

Placement: centre (x, y) on the board [mm, y down] or relative to a footprint (``ref``: x, y are
then the offset from it), ``rot`` 0 / 90 / 180 / 270 (u along board x at 0), ``side`` top /
bottom / through, ``gap_z`` the clearance from the board surface [mm] (solder mask, adhesive).

The material is a grade name from `materials.Library`, or custom μi / f_r (``mu_r``, ``f_r``).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .pcb import CoreBox

SHAPES = ("plate", "E", "rod", "toroid", "box")
SHAPE_NAMES = {"plate": "Plate / ferrite sheet", "E": "E core (E-E, E-I, planar, round leg)",
               "rod": "Rod", "toroid": "Toroid", "box": "Box (board coordinates)"}
SIDES = ("top", "bottom", "through")
STRIP_MM = 0.4                        # staircase strip width for round parts [mm]

DEFAULT_DIMS = {
    "plate": dict(L=10.0, W=10.0, T=0.5),
    "E": dict(A=18.0, B=4.0, C=10.0, D=2.0, E=14.0, F=4.0),          # an E18-like planar pair
    "rod": dict(F=4.0, L=20.0),
    "toroid": dict(OD=10.0, ID=6.0, H=4.0),
    "box": dict(lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0]),
}
DIM_TEXT = {
    "L": "length", "W": "width", "T": "thickness", "A": "overall width", "B": "height of one half",
    "C": "depth", "D": "window height of one half", "E": "window width (between outer legs)",
    "F": "centre leg width / diameter", "OD": "outer diameter", "ID": "inner diameter", "H": "height",
}


@dataclass
class CoreSpec:
    name: str = "core"
    shape: str = "plate"
    dims: dict = field(default_factory=lambda: dict(DEFAULT_DIMS["plate"]))
    material: str = "MnZn power ferrite, μi 2000"     # grade name ("" = custom mu_r / f_r)
    mu_r: float = 2000.0                              # custom material (material "")
    f_r: float = None
    x: float = 0.0
    y: float = 0.0
    ref: str = ""                                     # footprint: x, y are the offset from it
    rot: int = 0
    side: str = "top"
    gap_z: float = 0.05
    pair: bool = True                                 # E: E-E (else E-I)
    round: bool = False                               # E: round centre leg
    gap_center: float = 0.0
    gap_outer: float = 0.0
    enabled: bool = True

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["dims"] = {k: (list(v) if isinstance(v, (list, tuple)) else float(v)) for k, v in self.dims.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        if "lo" in d and "shape" not in d:                       # a saved pcb.CoreBox
            return cls(d.get("name", "core"), "box", dict(lo=list(d["lo"]), hi=list(d["hi"])), "",
                       float(d.get("mu_r", 1000.0)), d.get("f_r"))
        s = cls()
        for k in s.__dataclass_fields__:
            if k in d:
                setattr(s, k, d[k])
        s.dims = {**DEFAULT_DIMS.get(s.shape, {}), **dict(d.get("dims") or {})}
        return s

    def label(self):
        dims = "x".join(f"{v:g}" for k, v in self.dims.items() if not isinstance(v, (list, tuple)))
        kind = SHAPE_NAMES.get(self.shape, self.shape).split(" (")[0]
        if self.shape == "E":
            kind = ("ER" if self.round else "E") + ("-E" if self.pair else "-I")
        return f"{kind} {dims}".strip()


# ---- expansion ---------------------------------------------------------------------------
def material_of(spec, library):
    """(mu_r, f_r, warning) of the spec's material."""
    if spec.material:
        g = library.get(spec.material) if library is not None else None
        if g is not None:
            return float(g.mu_r), g.f_r, None
        return float(spec.mu_r), spec.f_r, f"Core {spec.name}: material '{spec.material}' not in the library, using μ {spec.mu_r:g}."
    return float(spec.mu_r), spec.f_r, None


def board_surfaces(stackup):
    """(z of the top surface, z of the bottom surface) [mm], z up, top copper's surface at 0."""
    lz = stackup.layer_z()
    last = stackup.copper_layers[-1]
    return 0.0, -(lz[last] + stackup.cu(last)) * 1e3


def centre(spec, board):
    """Board (x, y) [mm] of the core's centre."""
    if spec.ref:
        fp = _footprint(board, spec.ref)
        if fp is None:
            raise ValueError(f"Core {spec.name}: footprint {spec.ref} not on the board.")
        return fp[0] + spec.x, fp[1] + spec.y
    return spec.x, spec.y


def _footprint(board, ref):
    """(x, y) [mm] of a footprint: its anchor, else the centre of its pads' bounding box."""
    fp = next((f for f in getattr(board, "footprints", []) if f.ref == ref), None)
    if fp is not None:
        return float(fp.pos[0]), float(fp.pos[1])
    pads = [p for p in board.pads if p.id.split(".")[0] == ref and p.geom is not None]
    if not pads:
        return None
    bb = np.array([p.geom.bbox for p in pads])
    return float((bb[:, 0].min() + bb[:, 2].max()) / 2), float((bb[:, 1].min() + bb[:, 3].max()) / 2)


# ---- slots in the board: where a planar core's legs go ----------------------------------
def slots(board):
    """Cut-outs a core leg can pass through: Edge.Cuts loops inside the board (all loops but the
    largest) and non-plated oval or round holes; [(x0, y0, x1, y1)] mm."""
    out = []
    loops = list(getattr(board, "outline", []) or [])
    if len(loops) > 1:
        def area(l):
            x, y = l[:, 0], l[:, 1]
            return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
        big = int(np.argmax([area(l) for l in loops]))
        out += [(float(l[:, 0].min()), float(l[:, 1].min()), float(l[:, 0].max()), float(l[:, 1].max()))
                for k, l in enumerate(loops) if k != big]
    for p in board.pads:
        if p.kind == "np_thru_hole" and p.hole_geom is not None:
            out.append(tuple(float(v) for v in p.hole_geom.bbox))
    return out


def leg_rects(spec):
    """Local (u, v) rectangles of an E core's legs: centre first, then the outer legs."""
    d = {k: float(v) for k, v in spec.dims.items()}
    A, C, E, F = d["A"], d["C"], d["E"], d["F"]
    leg = (A - E) / 2
    cv = F / 2 if spec.round else C / 2
    return [(-F / 2, -cv, F / 2, cv), (-A / 2, -C / 2, -E / 2, C / 2), (E / 2, -C / 2, A / 2, C / 2)]


def snap(spec, board, tol=0.05):
    """(x, y, rot, note): an E core's centre and rotation that put its centre leg in a slot and
    each outer leg in a slot or outside the board (planar cores' outer legs often pass the board's
    edge in notches).  The centre slot's middle sets the position; ValueError if nothing fits."""
    if spec.shape != "E":
        raise ValueError("Only E cores snap to slots.")
    sl = slots(board)
    if not sl:
        raise ValueError("The board has no slots or cut-outs (Edge.Cuts loops inside the board, or "
                         "non-plated holes).")
    legs = leg_rects(spec)
    inside = getattr(board, "outline_mask", None)

    def fits(r, box):
        return r[0] >= box[0] - tol and r[1] >= box[1] - tol and r[2] <= box[2] + tol and r[3] <= box[3] + tol

    def off_board(r):
        if inside is None or not getattr(board, "outline", None):
            return False
        xs = np.linspace(r[0] + tol, r[2] - tol, 5)
        ys = np.linspace(r[1] + tol, r[3] - tol, 5)
        X, Y = np.meshgrid(xs, ys)
        return not np.any(inside(X, Y))
    best = None
    for box in sl:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for k in range(4):
            placed = []
            for r in legs:
                x, y = _rotate(np.array([r[0], r[2]]), np.array([r[1], r[3]]), k)
                placed.append((cx + x.min(), cy + y.min(), cx + x.max(), cy + y.max()))
            if not fits(placed[0], box):
                continue
            how = []
            for r in placed[1:]:
                if any(fits(r, b) for b in sl):
                    how.append("slot")
                elif off_board(r):
                    how.append("outside the board")
                else:
                    how = None
                    break
            if how is None:
                continue
            score = how.count("slot")
            if best is None or score > best[0]:
                best = (score, cx, cy, 90 * k, how)
    if best is None:
        raise ValueError(f"No slot fits the core's legs ({len(sl)} slots on the board; the centre leg "
                         f"needs {legs[0][2] - legs[0][0]:g} x {legs[0][3] - legs[0][1]:g} mm).")
    _, cx, cy, r, how = best
    return cx, cy, r, f"centre leg in a slot, outer legs: {', '.join(how)}"


def expand(spec, board, stackup, library=None):
    """([CoreBox], warnings): the spec's boxes in board coordinates; gaps are CoreBox with mu_r 1
    listed after the material they cut."""
    warns = []
    if not spec.enabled:
        return [], warns
    mu, fr, w = material_of(spec, library)
    if w:
        warns.append(w)
    if spec.shape == "box":
        return [CoreBox(spec.name, list(spec.dims["lo"]), list(spec.dims["hi"]), mu, fr)], warns
    z_top, z_bot = board_surfaces(stackup)
    d = {k: float(v) for k, v in spec.dims.items()}
    if spec.shape == "plate":
        local = [((-d["L"] / 2, -d["W"] / 2, 0.0), (d["L"] / 2, d["W"] / 2, d["T"]), mu)]
        h = d["T"]
    elif spec.shape == "rod":
        r = d["F"] / 2
        local = [((-d["L"] / 2, a, r + b0), (d["L"] / 2, b, r + b1), mu) for a, b, b0, b1 in _disc_strips(r)]
        h = d["F"]
    elif spec.shape == "toroid":
        local = _toroid(d["OD"] / 2, d["ID"] / 2, d["H"], mu)
        h = d["H"]
    elif spec.shape == "E":
        local, h, wl = _e_core(spec, d, mu, z_top - z_bot)
        warns += [f"Core {spec.name}: {t}" for t in wl]
    else:
        raise ValueError(f"Unknown core shape {spec.shape}.")
    cx, cy = centre(spec, board)
    rot = int(round(spec.rot / 90.0)) % 4
    zc = (z_top + z_bot) / 2
    if spec.shape == "E":                                       # window centred on the board
        flip = -1.0 if spec.side == "top" else 1.0

        def zf(w):
            return zc + flip * (w - h / 2)
    elif spec.side == "top":
        def zf(w):
            return z_top + spec.gap_z + w
    elif spec.side == "bottom":
        def zf(w):
            return z_bot - spec.gap_z - w
    else:                                                       # through a cut-out: centred
        def zf(w):
            return zc + w - h / 2
    out = []
    for lo, hi, m in local:
        x, y = _rotate(np.array([lo[0], hi[0]]), np.array([lo[1], hi[1]]), rot)
        zs = zf(np.array([lo[2], hi[2]]))
        out.append(CoreBox(f"{spec.name}" if m > 1 else f"{spec.name} gap",
                           [cx + x.min(), cy + y.min(), zs.min()], [cx + x.max(), cy + y.max(), zs.max()],
                           m if m > 1 else 1.0, fr if m > 1 else None))
    return out, warns


def _rotate(u, v, k):
    """Board offsets (x, y) of local (u, v) after k quarter turns (u along board x at k = 0,
    along board y at k = 1; boxes stay axis-aligned)."""
    for _ in range(k):
        u, v = -v, u
    return u, v


def _disc_strips(r, step=STRIP_MM):
    """Staircase of a circle of radius r centred at 0: strips (v0, v1, w0, w1) across v with the
    circle's true area per strip (half-height from the exact segment area)."""
    n = max(2, int(math.ceil(2 * r / step)))
    e = np.linspace(-r, r, n + 1)
    out = []
    for a, b in zip(e[:-1], e[1:]):
        def F(t):
            t = np.clip(t, -r, r)
            return t * np.sqrt(r * r - t * t) + r * r * np.arcsin(t / r)   # ∫ 2√(r² - t²) dt
        half = (F(b) - F(a)) / (2 * (b - a))
        out.append((a, b, -half, half))
    return out


def _toroid(ro, ri, h, mu, step=STRIP_MM):
    """Annulus staircase lying flat (z 0..h): strips across u, each one or two boxes."""
    n = max(4, int(math.ceil(2 * ro / step)))
    e = np.linspace(-ro, ro, n + 1)
    out = []

    def chord(r, a, b):                                         # mean half-chord over [a, b]
        def F(t):
            t = np.clip(t, -r, r)
            return t * np.sqrt(r * r - t * t) + r * r * np.arcsin(t / r)
        return (F(b) - F(a)) / (2 * (b - a))
    for a, b in zip(e[:-1], e[1:]):
        co = chord(ro, a, b)
        ci = chord(ri, a, b) if min(abs(a), abs(b)) < ri else 0.0
        if ci <= 1e-9:
            out.append(((a, -co, 0.0), (b, co, h), mu))
        elif co - ci > 1e-9:
            out.append(((a, ci, 0.0), (b, co, h), mu))
            out.append(((a, -co, 0.0), (b, -ci, h), mu))
    return out


def _e_core(spec, d, mu, board_t):
    """Local boxes (u across the legs, v depth, w up from the core's bottom), total height, warnings."""
    A, B, C, D, E, F = (d[k] for k in "ABCDEF")
    warns = []
    if not (0 < F < E < A and 0 < D < B):
        raise ValueError(f"Core {spec.name}: E core dimensions need F < E < A and D < B.")
    leg = (A - E) / 2
    back = B - D
    pair = spec.pair
    H = 2 * B if pair else B + back
    win = 2 * D if pair else D
    if win < board_t:
        warns.append(f"the window ({win:g} mm) is lower than the board ({board_t:.2f} mm).")
    boxes = []
    # lower half: an E with its legs up (the I's plate on top for E-I)
    boxes.append(((-A / 2, -C / 2, 0.0), (A / 2, C / 2, back), mu))                 # back
    for u0 in (-A / 2, A / 2 - leg):                                                # outer legs
        boxes.append(((u0, -C / 2, back), (u0 + leg, C / 2, back + win), mu))
    if spec.round:
        for a, b, w0, w1 in _disc_strips(F / 2):
            boxes.append(((a, w0, back), (b, w1, back + win), mu))
    else:
        boxes.append(((-F / 2, -C / 2, back), (F / 2, C / 2, back + win), mu))
    boxes.append(((-A / 2, -C / 2, back + win), (A / 2, C / 2, H), mu))            # upper back / I
    # gaps at the mating plane: the middle of the window (E-E) or the E's leg ends (E-I)
    zm = back + (D if pair else win)
    for g, us in ((spec.gap_center, [(-F / 2, F / 2)]),
                  (spec.gap_outer, [(-A / 2, -A / 2 + leg), (A / 2 - leg, A / 2)])):
        if g > 0:
            if g >= win:
                raise ValueError(f"Core {spec.name}: a gap of {g:g} mm does not fit the {win:g} mm window.")
            z0 = min(max(zm - g / 2, back), back + win - g) if pair else back + win - g
            for u0, u1 in us:
                half = max(C, F) / 2 if spec.round else C / 2
                boxes.append(((u0, -half, z0), (u1, half, z0 + g), 1.0))
    return boxes, H, warns


def magnetic_path(spec):
    """(A_e [mm²], l_e [mm]) of an E core by the simple rectangular path: centre leg area; mean
    path up the centre leg, along the back's centre line to an outer leg's centre line and down
    (a check of the solved L against N² μ0 μ A_e / l_e, not used by the solver)."""
    d = {k: float(v) for k, v in spec.dims.items()}
    A, B, C, D, E, F = (d[k] for k in "ABCDEF")
    Ae = math.pi * F * F / 4 if spec.round else F * C
    leg, back = (A - E) / 2, B - D
    win = 2 * D if spec.pair else D
    return Ae, 2 * (win + back) + (E + leg)
