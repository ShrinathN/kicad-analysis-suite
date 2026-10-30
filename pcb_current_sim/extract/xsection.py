"""2D quasi-static cross-section solver: per-unit-length C and L of a conductor over its return.

Finite volumes on a graded rectilinear grid solve div(ε grad V) = 0 with the signal
conductor at 1 V, the return conductors at 0 V and a zero-flux outer boundary, so every
field line ends on the return.  One solve with the dielectrics and one in air give

    C' = Q / V,   C0' (air),   L' = 1 / (c² C0'),   Z0 = sqrt(L' / C'),   εeff = C' / C0'

L' is the external (high-frequency) inductance: the return current is assumed to flow
where the field puts it, as it does above a few hundred kHz.  This one model covers
microstrip, stripline, coplanar lines and a pour on the far side of a 2-layer board.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .. import linsolve as LS

EPS0 = 8.8541878128e-12
C0 = 299792458.0


@dataclass
class Rect:
    x0: float          # m, lateral
    x1: float
    z0: float          # m, depth (down)
    z1: float
    signal: bool = False


@dataclass
class Section:
    rects: list                                     # Rect: one or more signal, any number of return
    slabs: list = field(default_factory=list)       # (z0, z1, εr) dielectric layers; air elsewhere
    half_width: float = 5e-3                        # m, lateral extent of the domain (each side)
    air: float = 3e-3                               # m, air above and below the board


@dataclass
class LineParams:
    C: float           # F/m
    C0: float          # F/m, in air
    L: float           # H/m

    @property
    def z0(self):
        return math.sqrt(self.L / self.C)

    @property
    def eeff(self):
        return self.C / self.C0

    @property
    def delay(self):
        """s/m"""
        return math.sqrt(self.L * self.C)


def _fill(a, b, ha, hb, hmax, r=1.25):
    """Grid lines on [a, b]: steps grow from ha at a and hb at b by r, capped at hmax."""
    L = b - a
    if L <= 1.5 * min(ha, hb):
        return np.array([a, b])
    left, right, sa, sb, tl, tr = [], [], ha, hb, 0.0, 0.0
    while tl + tr + min(sa, sb) < L:
        if tl <= tr:
            left.append(sa)
            tl += sa
            sa = min(sa * r, hmax)
        else:
            right.append(sb)
            tr += sb
            sb = min(sb * r, hmax)
    steps = np.array(left + right[::-1])
    steps *= L / steps.sum()
    return a + np.concatenate([[0.0], np.cumsum(steps)])


def _lines(breaks, hmax):
    """breaks: {position: local step}; graded lines through every break."""
    ps = sorted(breaks)
    merged = [ps[0]]
    for p in ps[1:]:
        if p - merged[-1] > 1e-9:
            merged.append(p)
        else:
            breaks[merged[-1]] = min(breaks[merged[-1]], breaks[p])
    out = [np.array([merged[0]])]
    for a, b in zip(merged[:-1], merged[1:]):
        out.append(_fill(a, b, breaks[a], breaks[b], hmax)[1:])
    return np.concatenate(out)


def solve(sec):
    """LineParams for a Section, or None when there is no return conductor."""
    sig = [r for r in sec.rects if r.signal]
    ret = [r for r in sec.rects if not r.signal]
    if not sig or not ret:
        return None
    zs = [z for s in sec.slabs for z in s[:2]] + [z for r in sec.rects for z in (r.z0, r.z1)]
    ztop, zbot = min(zs) - sec.air, max(zs) + sec.air
    X = sec.half_width
    t_sig = min(r.z1 - r.z0 for r in sig)
    w_sig = min(r.x1 - r.x0 for r in sig)
    h0 = max(min(t_sig / 2, w_sig / 12), 1e-6)
    h1 = max(min(r.z1 - r.z0 for r in sec.rects) / 2, h0)
    hmax = max(2 * X, zbot - ztop) / 25
    bx = {-X: hmax, X: hmax}
    bz = {ztop: hmax, zbot: hmax}
    for r in sec.rects:
        h = h0 if r.signal else h1
        for x in (r.x0, r.x1):
            if -X < x < X:
                bx[x] = min(bx.get(x, hmax), h)
        for z in (r.z0, r.z1):
            bz[z] = min(bz.get(z, hmax), h)
    for z0, z1, _ in sec.slabs:
        for z in (z0, z1):
            bz[z] = min(bz.get(z, hmax), max((z1 - z0) / 4, h0))
    xs, zs = _lines(bx, hmax), _lines(bz, hmax)
    nx, nz = len(xs), len(zs)
    dx, dz = np.diff(xs), np.diff(zs)

    # node state: 0 free, 1 signal, 2 return
    state = np.zeros((nz, nx), np.int8)
    tol = 1e-12
    for r in sec.rects:
        mx = (xs >= max(r.x0, -X) - tol) & (xs <= min(r.x1, X) + tol)
        mz = (zs >= r.z0 - tol) & (zs <= r.z1 + tol)
        state[np.ix_(mz, mx)] = 1 if r.signal else 2
    zc = (zs[:-1] + zs[1:]) / 2
    er_row = np.ones(nz - 1)
    for z0, z1, er in sec.slabs:
        er_row[(zc >= z0) & (zc <= z1)] = er

    def params(eps_row):
        e = np.repeat(eps_row[:, None], nx - 1, axis=1)          # cell permittivity (nz-1, nx-1)
        ez = np.zeros((nz + 1, nx - 1))
        ez[1:-1] = e * dz[:, None] / 2
        gx = (ez[:-1] + ez[1:]) / dx[None, :]                      # (nz, nx-1): edge (k, i)-(k, i+1)
        ex = np.zeros((nz - 1, nx + 1))
        ex[:, 1:-1] = e * dx[None, :] / 2
        gz = (ex[:, :-1] + ex[:, 1:]) / dz[:, None]                # (nz-1, nx): edge (k, i)-(k+1, i)
        idx = np.arange(nz * nx).reshape(nz, nx)
        I = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
        J = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
        G = np.concatenate([gx.ravel(), gz.ravel()])
        n = nz * nx
        off = sp.coo_matrix((G, (I, J)), shape=(n, n))
        off = (off + off.T).tocsr()
        A = (sp.diags(np.asarray(off.sum(1)).ravel()) - off).tocsr()
        st = state.ravel()
        free = np.flatnonzero(st == 0)
        V = (st == 1).astype(float)
        if free.size:
            Aff = A[free][:, free].tocsc()
            rhs = -(A[free] @ V)
            V[free] = LS.solve(Aff, rhs)
        cut = (st[I] == 1) != (st[J] == 1)
        a, b = I[cut], J[cut]
        s = np.where(st[a] == 1, 1.0, -1.0)
        return EPS0 * float(np.sum(G[cut] * s * (V[a] - V[b])))

    C = params(er_row)
    Cair = params(np.ones(nz - 1))
    if not (C > 0 and Cair > 0):
        return None
    return LineParams(C, Cair, 1 / (C0 ** 2 * Cair))
