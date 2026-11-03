"""Read traces off a plot image (a datasheet's oscilloscope screenshot): calibrate the plot area,
pick a trace by its colour, and get its samples in time and value units.

Pixel coordinates: x to the right, y down (image rows).  The calibration is the plot area's corners
in pixels plus the number of divisions and the units per division (as on a scope screen); values
are measured from the plot area's vertical centre, upwards (only differences matter: the current
step and the voltage deviation).
"""
import math
from dataclasses import dataclass

import numpy as np


def load_image(path):
    """RGB float image (H, W, 3) in 0..1 (PNG, JPEG, ...)."""
    import matplotlib.image as mpimg
    im = mpimg.imread(path)
    if im.dtype != np.float32 and im.dtype != np.float64:
        im = im.astype(np.float64) / 255.0
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    return np.asarray(im[..., :3], float)


def colour_at(img, x, y, r=2):
    """The trace colour under a click: the most saturated pixel near (x, y) (a click lands next to a thin
    line as often as on it; grid and background are grey)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    patch = img[y0:y1, x0:x1].reshape(-1, 3)
    sat = patch.max(1) - patch.min(1)
    return patch[int(np.argmax(sat))]


def _hue(a):
    """Hue in degrees of RGB values (..., 3)."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return np.degrees(np.arctan2(math.sqrt(3) * (g - b), 2 * r - g - b)) % 360


def colour_mask(img, rgb, tol=0.25):
    """Pixels of the trace colour rgb.  A coloured trace is matched by hue (within tol x 90 degrees) among
    pixels that are coloured enough: plain RGB distance confuses the dark antialiased edges of different
    traces (dark blue and dark magenta are close in RGB, far apart in hue).  A black / grey trace is matched
    by RGB distance (within tol)."""
    c = np.asarray(rgb, float)
    sat_c = float(c.max() - c.min())
    if sat_c < 0.15:
        return np.linalg.norm(img - c[None, None, :], axis=2) <= tol
    sat = img.max(axis=2) - img.min(axis=2)
    dh = np.abs((_hue(img) - _hue(c) + 180) % 360 - 180)
    return (dh <= tol * 90) & (sat >= max(0.08, 0.3 * sat_c))


@dataclass
class Calib:
    box: tuple = (0, 0, 1, 1)    # plot area in pixels: x0, y0 (top left), x1, y1 (bottom right)
    nx: int = 10                 # divisions across
    ny: int = 8                  # divisions down
    t_div: float = 1e-4          # s per division

    def t(self, xs):
        x0, _, x1, _ = self.box
        return (np.asarray(xs, float) - x0) / max(x1 - x0, 1e-9) * self.nx * self.t_div

    def div(self, ys):
        """Divisions above the plot area's centre."""
        _, y0, _, y1 = self.box
        return ((y0 + y1) / 2 - np.asarray(ys, float)) / max(y1 - y0, 1e-9) * self.ny


def _runs(rows):
    """Contiguous runs [(first, last)] of sorted row indices (gaps of up to 2 px bridged)."""
    if not rows.size:
        return []
    cut = np.flatnonzero(np.diff(rows) > 3)
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, rows.size - 1]
    return [(int(rows[a]), int(rows[b])) for a, b in zip(starts, ends)]


def trace(img, box, rgb, tol=0.25):
    """Samples (x, y) in pixels of the trace of colour rgb inside box, one per column.

    Each column's pixels of that colour form runs; the trace is followed from the middle of the plot
    outwards, taking in each column the run nearest the previous point (so markers, labels and text of
    the same colour are skipped).  A run much taller than the line is a steep part of the curve: its
    end away from the previous point is taken, so narrow spikes keep their tips."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = sorted((max(0, x0), min(img.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(img.shape[0], y1)))
    sub = img[y0:y1, x0:x1]
    m = colour_mask(sub, rgb, tol)
    runs = [_runs(np.flatnonzero(m[:, c])) for c in range(sub.shape[1])]
    single = [b - a + 1 for r in runs if len(r) == 1 for a, b in r]
    thick = float(np.median(single)) if single else 3.0
    cols = [c for c, r in enumerate(runs) if r]
    if not cols:
        return np.zeros(0), np.zeros(0)
    mid = sub.shape[1] / 2
    start = min((c for c in cols if len(runs[c]) == 1), key=lambda c: abs(c - mid), default=cols[len(cols) // 2])
    y = {}
    a, b = max(runs[start], key=lambda r: r[1] - r[0])
    y[start] = (a + b) / 2
    def gap(r, y_):
        return 0 if r[0] <= y_ <= r[1] else min(abs(r[0] - y_), abs(r[1] - y_))
    far = 0.1 * sub.shape[0]
    for step in (1, -1):
        prev = y[start]
        c = start + step
        while 0 <= c < len(runs):
            if runs[c]:
                a, b = min(runs[c], key=lambda r: gap(r, prev))
                if gap((a, b), prev) > far:
                    # a far jump that the next column doesn't continue is a marker or a label: skip it
                    nxt = runs[c + step] if 0 <= c + step < len(runs) else []
                    if not any(gap(r, (a + b) / 2) <= 2 * thick + 1 for r in nxt):
                        c += step
                        continue
                if b - a + 1 <= 2 * thick + 1:
                    yc = (a + b) / 2
                else:
                    yc = (b - thick / 2) if abs(b - prev) > abs(a - prev) else (a + thick / 2)
                y[c] = prev = yc
            c += step
    cs = np.array(sorted(y))
    return cs + float(x0), np.array([y[c] for c in cs]) + y0


def is_grey(img, tol=0.08):
    """True when the image has no coloured pixels to tell traces apart (black-and-white plots)."""
    sat = img.max(axis=2) - img.min(axis=2)
    return bool(np.percentile(sat, 99.9) < tol)


def from_points(cal, pts, per_div):
    """(t [s], value [units]) of a trace given as points clicked on the image (pixels, any order).

    Sorted by time; points on a vertical edge keep the order they were clicked in and are spread
    by a hair so the trace stays a function of time (the step's direction survives)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 2:
        raise ValueError("Click at least two points on the trace.")
    p = p[np.argsort(p[:, 0], kind="stable")]
    x = p[:, 0].copy()
    eps = 1e-6 * max(x[-1] - x[0], 1.0)
    for k in range(1, len(x)):
        if x[k] <= x[k - 1]:
            x[k] = x[k - 1] + eps
    return cal.t(x), cal.div(p[:, 1]) * per_div


def simplify(xs, ys, max_points=80):
    """A traced curve as a few points (Ramer-Douglas-Peucker: corners, dips and spikes are kept), so it can
    be corrected by hand.  The tolerance grows until at most max_points remain."""
    p = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    if len(p) <= 2:
        return [tuple(q) for q in p]

    def rdp(eps):
        keep = np.zeros(len(p), bool)
        keep[[0, -1]] = True
        stack = [(0, len(p) - 1)]
        while stack:
            a, b = stack.pop()
            if b - a < 2:
                continue
            d = p[b] - p[a]
            n = np.hypot(*d) or 1.0
            q = p[a + 1:b] - p[a]
            dist = np.abs(q[:, 0] * d[1] - q[:, 1] * d[0]) / n
            k = int(np.argmax(dist))
            if dist[k] > eps:
                m = a + 1 + k
                keep[m] = True
                stack += [(a, m), (m, b)]
        return keep
    eps = 1.0
    keep = rdp(eps)
    while keep.sum() > max_points and eps < 1e4:
        eps *= 1.4
        keep = rdp(eps)
    return [tuple(map(float, q)) for q in p[keep]]


def nearest(pts, x, y):
    """Index of the point nearest (x, y), or None."""
    if not pts:
        return None
    p = np.asarray(pts, float)
    return int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))


def extract(img, cal, rgb, per_div, tol=0.25):
    """(t [s], value [units]) of one trace."""
    xs, ys = trace(img, cal.box, rgb, tol)
    if xs.size < 10:
        raise ValueError("Too few pixels of that colour in the plot area: click on the trace itself, or raise "
                         "the colour tolerance.")
    return cal.t(xs), cal.div(ys) * per_div
