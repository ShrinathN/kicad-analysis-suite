"""Ready-made conductor geometries (tests, validation, coil starting points)."""
import math

import numpy as np

from .geometry import Geometry, SIGMA_CU


def square_spiral(turns, outer, w, gap, t, nw=1, nh=1, z=0.0, g=None, name="coil"):
    """Square spiral from the outer corner (0, 0) inwards, side lengths outer, outer, outer - p,
    outer - p, outer - 2p, ... (p = w + gap); port from the outer to the inner end."""
    g = g or Geometry()
    p = w + gap
    lens = [outer] + [outer - (k // 2) * p for k in range(1, 4 * turns)]
    if min(lens) <= w:
        raise ValueError("too many turns for the size")
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    x = y = 0.0
    ids = [g.node(x, y, z)]
    for k, ln in enumerate(lens):
        dx, dy = dirs[k % 4]
        x, y = x + dx * ln, y + dy * ln
        ids.append(g.node(x, y, z))
    for a, b in zip(ids[:-1], ids[1:]):
        g.seg(a, b, w, t, nw=nw, nh=nh)
    g.port(name, ids[0], ids[-1])
    return g


def hairpin(length, w, gap, t, nw=1, nh=1, g=None, name="hairpin"):
    """Go and return strips side by side in one plane, joined at the far end."""
    g = g or Geometry()
    s = w + gap
    a, b, c, d = g.node(0, 0, 0), g.node(length, 0, 0), g.node(length, s, 0), g.node(0, s, 0)
    g.seg(a, b, w, t, nw=nw, nh=nh)
    g.seg(b, c, w, t, nw=max(1, nw // 2 * 2 - 1), nh=nh)
    g.seg(c, d, w, t, nw=nw, nh=nh)
    g.port(name, a, d)
    return g


def polygon_loop(radius, size, n=64, z=0.0, g=None, name="loop", phase=0.3):
    """Circular loop of a square-section wire (side size) as an n-gon; port across a zero gap at
    the first vertex."""
    g = g or Geometry()
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1] + phase
    ids = [g.node(radius * math.cos(a), radius * math.sin(a), z) for a in th]
    end = g.node(radius * math.cos(th[0]), radius * math.sin(th[0]), z)
    for i in range(n):
        g.seg(ids[i], ids[i + 1] if i + 1 < n else end, size, size)
    g.port(name, end, ids[0])
    return g


def round_wire(radius, length, cells=10, sigma=SIGMA_CU, g=None, name="wire"):
    """Straight round wire along x: its section filled with square filaments (2 x cells across
    the diameter), all joined at both ends."""
    g = g or Geometry()
    h = radius / cells
    A, B = g.node(0, 0, 0), g.node(length, 0, 0)
    xs = (np.arange(2 * cells) + 0.5) * h - radius
    for y in xs:
        for z in xs:
            if y * y + z * z <= radius * radius:
                a, b = g.node(0, y, z), g.node(length, y, z)
                g.seg(a, b, h, h, wdir=(0, 1, 0), sigma=sigma)
                g.join(A, a)
                g.join(B, b)
    g.port(name, A, B)
    return g
