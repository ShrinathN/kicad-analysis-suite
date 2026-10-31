"""Small-signal circuit of a switching cell: the extracted copper (a coupled port block) plus
device models, solved by nodal analysis per frequency.

Nodes are names; "0" is ground.  Elements are two-terminal impedances Z(f) (callables or
constants); a block is the copper's port impedance matrix Zp(f) between node pairs.  zmatrix()
returns the impedance matrix seen at measurement ports (node pairs), for every frequency.
"""
import math

import numpy as np

GMIN = 1e-12          # S from every node to ground, so floating sub-circuits stay solvable
SHORT = 1e-7          # Ω, an ideal short


class Circuit:
    def __init__(self):
        self.names = {}
        self.elements = []      # (a, b, zfun, label)
        self.blocks = []        # (pairs [(a, b)], zfun -> (p, p))
        self.inductors = []     # (label, a, b, L) for energy bookkeeping (package / ESL)

    def node(self, name):
        if name == "0":
            return -1
        if name not in self.names:
            self.names[name] = len(self.names)
        return self.names[name]

    def add(self, a, b, z, label=""):
        """Two-terminal impedance: z a constant (Ω) or a callable f -> Z."""
        self.node(a), self.node(b)
        self.elements.append((a, b, z if callable(z) else (lambda f, v=z: np.full(np.shape(f), v, complex)), label))

    def R(self, a, b, r, label=""):
        self.add(a, b, max(r, SHORT), label)

    def L(self, a, b, l, label="", r=0.0):
        """Inductor (with series r).  Named ones are kept for the energy breakdown."""
        self.add(a, b, lambda f, l=l, r=r: max(r, SHORT) + 2j * np.pi * np.asarray(f) * l, label)
        if label:
            self.inductors.append((label, a, b, l))

    def C(self, a, b, c, label=""):
        if c > 0:
            self.add(a, b, lambda f, c=c: 1 / (2j * np.pi * np.maximum(np.asarray(f, float), 1e-3) * c), label)

    def short(self, a, b, label=""):
        self.R(a, b, SHORT, label)

    def block(self, pairs, zfun):
        for a, b in pairs:
            self.node(a), self.node(b)
        self.blocks.append((list(pairs), zfun))

    # ------------------------------------------------------------------
    def _Y(self, f):
        n = len(self.names)
        Y = np.zeros((n, n), complex)
        idx = lambda s: self.names.get(s, -1) if s != "0" else -1      # noqa: E731
        for a, b, z, _ in self.elements:
            y = 1.0 / complex(z(np.array(f)))
            ia, ib = idx(a), idx(b)
            if ia >= 0:
                Y[ia, ia] += y
            if ib >= 0:
                Y[ib, ib] += y
            if ia >= 0 and ib >= 0:
                Y[ia, ib] -= y
                Y[ib, ia] -= y
        for pairs, zfun in self.blocks:
            Zp = np.asarray(zfun(f), complex)
            Yp = np.linalg.inv(Zp)
            Bm = np.zeros((n, len(pairs)))
            for k, (a, b) in enumerate(pairs):
                if idx(a) >= 0:
                    Bm[idx(a), k] += 1
                if idx(b) >= 0:
                    Bm[idx(b), k] -= 1
            Y += Bm @ Yp @ Bm.T
        # leak to ground: GMIN, or 1e-14 of the stiffest branch, so that a floating island next to
        # an ideal short (1e7 S) stays solvable
        Y[np.diag_indices(n)] += max(GMIN, 1e-14 * float(np.abs(np.diag(Y)).max(initial=0.0)))
        return Y

    def solve(self, f, inject):
        """Node voltages for injected currents {(a, b): I} (I enters the circuit at a, leaves at b)."""
        n = len(self.names)
        rhs = np.zeros(n, complex)
        for (a, b), I in inject.items():
            if a != "0":
                rhs[self.names[a]] += I
            if b != "0":
                rhs[self.names[b]] -= I
        V = np.linalg.solve(self._Y(f), rhs)
        return lambda s: 0.0 if s == "0" else V[self.names[s]]

    def zmatrix(self, freqs, ports):
        """(F, p, p) impedance matrix at measurement ports [(a, b)] over freqs."""
        loose = [x for a, b in ports for x in (a, b) if x != "0" and x not in self.names]
        if loose:
            raise ValueError(f"{', '.join(loose)}: not connected to anything else in the cell's circuit "
                             "(no copper or part joins it to the loop).")
        n = len(self.names)
        out = np.zeros((len(freqs), len(ports), len(ports)), complex)
        Bm = np.zeros((n, len(ports)))
        for k, (a, b) in enumerate(ports):
            if a != "0":
                Bm[self.names[a], k] += 1
            if b != "0":
                Bm[self.names[b], k] -= 1
        for i, f in enumerate(freqs):
            V = np.linalg.solve(self._Y(f), Bm.astype(complex))
            out[i] = Bm.T @ V
        return out


def peak(freqs, z):
    """Resonance of |z(f)|: (f0, Q, |z|max), Q from the half-power width (parabolic refinement of
    the peak in log f).  None if the maximum sits at either end of the sweep."""
    a = np.abs(z)
    k = int(np.argmax(a))
    if k == 0 or k == len(a) - 1:
        return None
    lf = np.log(freqs)
    y0, y1, y2 = np.log(a[k - 1:k + 2])
    den = y0 - 2 * y1 + y2
    dx = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    f0 = math.exp(lf[k] + dx * (lf[k + 1] - lf[k]))
    amax = a[k]
    half = amax / math.sqrt(2)
    lo = k
    while lo > 0 and a[lo] > half:
        lo -= 1
    hi = k
    while hi < len(a) - 1 and a[hi] > half:
        hi += 1
    if a[lo] > half or a[hi] > half:
        return f0, float("nan"), amax

    def cross(i, j):
        t = (half - a[i]) / (a[j] - a[i])
        return math.exp(lf[i] + t * (lf[j] - lf[i]))
    f_lo, f_hi = cross(lo, lo + 1), cross(hi - 1, hi)
    return f0, f0 / (f_hi - f_lo), amax
