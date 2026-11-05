"""Conductor geometry: nodes, rectangular segments and their filaments (FastHenry's model).

A segment is a straight bar between two nodes: width w along its width direction (by default
in the x-y plane, perpendicular to the length; x for a vertical segment) and height h along
the third axis.  It is cut into nw x nh parallel filaments whose sizes grow geometrically by
the ratio rw (rh) from each face towards the middle, as FastHenry does, so a few filaments
resolve the skin depth at the faces.  Every filament of a segment runs between the segment's
two nodes (a node is equipotential across the cross-section).  Nodes joined with `join` are one
electrical node (FastHenry's .equiv).  A port is a pair of nodes: current in at the first, out
at the second, through the conductors.
"""
import math

import numpy as np

from .kernels import Filaments

MU0 = 4e-7 * math.pi
SIGMA_CU = 1 / 1.72e-8               # S/m at 20 °C


def skin_depth(f, sigma=SIGMA_CU, mu_r=1.0):
    """Skin depth [m] at f [Hz]."""
    return math.sqrt(1 / (math.pi * f * MU0 * mu_r * sigma))


def grade(n, r):
    """(sizes, centres) of n slices of a unit span, sizes growing by the ratio r from both edges
    towards the middle (FastHenry's rule); centres measured from the span's middle."""
    if n <= 1:
        return np.array([1.0]), np.array([0.0])
    half = [r ** j for j in range(n // 2)]
    mid = [r ** (n // 2)] if n % 2 else []
    s = np.array(half + mid + half[::-1], float)
    s /= s.sum()
    c = np.cumsum(s) - s / 2 - 0.5
    return s, c


def division(dim, delta, r=2.0, nmax=9):
    """Smallest slice count (1, 3, 5, ...) whose edge slice is no thicker than delta / 2 across a
    span dim graded by r; at most nmax."""
    n = 1
    while n < nmax:
        if grade(n, r)[0][0] * dim <= delta / 2:
            break
        n += 2
    return min(n, nmax)


class Geometry:
    def __init__(self):
        self.xyz = []                 # node positions [m]
        self.names = []
        self.segs = []                # dicts: a, b, w, h, wd, sigma, nw, nh, rw, rh, name
        self.ports = []               # (name, node a, node b)
        self._parent = []

    # -- building --------------------------------------------------------------------
    def node(self, x, y, z, name=None):
        self.xyz.append((float(x), float(y), float(z)))
        self.names.append(name or f"N{len(self.xyz)}")
        self._parent.append(len(self._parent))
        return len(self.xyz) - 1

    def seg(self, a, b, w, h, wdir=None, sigma=SIGMA_CU, nw=1, nh=1, rw=2.0, rh=2.0, name=None):
        pa, pb = np.array(self.xyz[a]), np.array(self.xyz[b])
        if np.linalg.norm(pb - pa) <= 0:
            raise ValueError("a segment needs two distinct node positions")
        self.segs.append(dict(a=a, b=b, w=float(w), h=float(h), wd=None if wdir is None else tuple(wdir),
                              sigma=float(sigma), nw=int(nw), nh=int(nh), rw=float(rw), rh=float(rh),
                              name=name or f"E{len(self.segs) + 1}"))
        return len(self.segs) - 1

    def join(self, *nodes):
        """Make the nodes one electrical node."""
        r = self._find(nodes[0])
        for n in nodes[1:]:
            self._parent[self._find(n)] = r

    def port(self, name, a, b):
        self.ports.append((name, a, b))

    def _find(self, n):
        while self._parent[n] != n:
            self._parent[n] = self._parent[self._parent[n]]
            n = self._parent[n]
        return n

    def discretise(self, f_max, nmax=9, r=2.0):
        """Filament counts for every segment so that the face filaments are no thicker than half
        the skin depth at f_max (at most nmax per side)."""
        for s in self.segs:
            d = skin_depth(f_max, s["sigma"])
            s["nw"], s["nh"] = division(s["w"], d, r, nmax), division(s["h"], d, r, nmax)
            s["rw"] = s["rh"] = r

    # -- derived ---------------------------------------------------------------------
    def frames(self):
        """Per segment: start, unit length direction, length, width and height directions."""
        out = []
        for s in self.segs:
            pa, pb = np.array(self.xyz[s["a"]]), np.array(self.xyz[s["b"]])
            l = float(np.linalg.norm(pb - pa))
            u = (pb - pa) / l
            if s["wd"] is not None:
                wd = np.array(s["wd"], float)
                wd = wd - u * (wd @ u)
            else:
                wd = np.array([-u[1], u[0], 0.0])
                if np.linalg.norm(wd) < 1e-9:
                    wd = np.array([1.0, 0.0, 0.0])
            wd /= np.linalg.norm(wd)
            hd = np.cross(u, wd)
            out.append((pa, u, l, wd, hd))
        return out

    def electrical_nodes(self):
        """(index of each node's electrical node, number of electrical nodes)."""
        roots = np.array([self._find(i) for i in range(len(self.xyz))])
        uniq, idx = np.unique(roots, return_inverse=True)
        return idx, len(uniq)

    def filaments(self):
        """(Filaments, R [Ω] per filament, electrical node a, node b, segment index)."""
        enode, _ = self.electrical_nodes()
        c0, U, Ls, WD, HD, W, H, R, A, Bn, S = ([] for _ in range(11))
        for k, (s, (pa, u, l, wd, hd)) in enumerate(zip(self.segs, self.frames())):
            sw, cw = grade(s["nw"], s["rw"])
            sh, ch = grade(s["nh"], s["rh"])
            for i in range(len(sh)):
                for j in range(len(sw)):
                    w, h = sw[j] * s["w"], sh[i] * s["h"]
                    c0.append(pa + wd * cw[j] * s["w"] + hd * ch[i] * s["h"])
                    U.append(u)
                    Ls.append(l)
                    WD.append(wd)
                    HD.append(hd)
                    W.append(w)
                    H.append(h)
                    R.append(l / (s["sigma"] * w * h))
                    A.append(enode[s["a"]])
                    Bn.append(enode[s["b"]])
                    S.append(k)
        F = Filaments(np.array(c0), np.array(U), np.array(Ls), np.array(WD), np.array(HD), np.array(W), np.array(H))
        return F, np.array(R), np.array(A), np.array(Bn), np.array(S)
