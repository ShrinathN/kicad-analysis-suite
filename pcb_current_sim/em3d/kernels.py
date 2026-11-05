"""Partial inductances between straight filaments of rectangular cross-section.

A filament is a straight bar: start point c0 (centre of its start face), unit direction u,
length l, width w along the unit vector wd and height h along hd (u, wd, hd orthonormal).  The
partial mutual inductance of two bars is μ0/(4π A_i A_j) ∫∫ (u_i · u_j) dV_i dV_j / r.

* Parallel bars (most of a PCB: tracks and pour bars along x or y, vias along z):
  - far apart (centre lines more than NEAR x the larger cross-section apart): two filaments on
    their axes (exact closed form `axial`);
  - nearer: filaments at the geometric mean distance of the two cross-sections (exact closed form
    `log_gmd`); the error is of order (size / length)², a few 1e-4 for bars longer than LONG x
    their cross-section.  NEAR is wide (15) on purpose: centre-line filaments at 3 widths read a
    3-turn spiral's R at 1 GHz 1 % low (proximity hangs on small differences of mutuals);
  - within EXACT sizes and short (not longer than LONG x the cross-section): the exact six-fold
    integral, as the exact four-fold integral over the two
    faces (Hoer & Love, `loop.peec.rect_integral`) integrated over the height difference by Gauss
    quadrature on the pieces of its (piecewise linear) weight.  Self terms come out of the same
    formulas.
* Bars at an angle (diagonal tracks, arcs, coil turns): Neumann's integral, the inner integral
  along bar j in closed form (regularised by the cross-sections' GMD, so touching bars at a
  corner stay finite), the outer one by Gauss quadrature along bar i; far pairs by their
  midpoints.  Perpendicular bars do not couple.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import THREADS
from ..loop.peec import axial, rect_integral

MU0 = 4e-7 * math.pi
KM = MU0 / (4 * math.pi)             # 1e-7 H/m
NEAR = 15.0                          # parallel pairs closer than this x the larger cross-section: GMD
EXACT = 3.0                          # ... closer than this and short: exact six-fold integral
LONG = 30.0                          # bars longer than this x the cross-section count as long
FAR = 4.0                            # angled pairs farther than this x their size: midpoints
PAR = 1e-9                           # 1 - |cos| below this: parallel
_GQ = np.polynomial.legendre.leggauss(6)
_GN = np.polynomial.legendre.leggauss(16)


# ---- closed forms ---------------------------------------------------------------------
def _xlog(p, r2):
    """p · ln(r2), 0 where r2 = 0 (p vanishes faster there)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, p * np.log(np.where(r2 > 0, r2, 1.0)), 0.0)


def _G2(x, y):
    """Antiderivative with d⁴G/dx²dy² = ln sqrt(x² + y²) (exact; derived with sympy, checked)."""
    x2, y2 = x * x, y * y
    r2 = x2 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        axy = np.where(y != 0, np.arctan(x / np.where(y != 0, y, 1.0)), 0.0)
        ayx = np.where(x != 0, np.arctan(y / np.where(x != 0, x, 1.0)), 0.0)
    # x y³ atan(x/y) + x³ y atan(y/x): C² across both axes (|y|³, |x|³), so G is a true
    # antiderivative everywhere (sympy's quadrant form had a kink x³|y| at y = 0)
    return (-_xlog((x2 * x2 - 6 * x2 * y2 + y2 * y2) / 48, r2)
            + (x * y2 * y * axy + x2 * x * y * ayx) / 6 - 25 / 48 * x2 * y2)


def _quad_sum(fun, a0, a1, b0, b1, c0, c1, d0, d1):
    """Σ ± fun(X, Y) over the 16 corner differences of the intervals [a0, a1] - [b0, b1] (X) and
    [c0, c1] - [d0, d1] (Y): the four-fold integral of f(x - x', y - y') when fun is f's double
    antiderivative in each variable."""
    tot = 0.0
    for X, sx in ((a1 - b0, 1), (a1 - b1, -1), (a0 - b0, -1), (a0 - b1, 1)):
        for Y, sy in ((c1 - d0, 1), (c1 - d1, -1), (c0 - d0, -1), (c0 - d1, 1)):
            tot = tot + sx * sy * fun(X, Y)
    return tot


def log_gmd(pi0, pi1, qi0, qi1, pj0, pj1, qj0, qj1):
    """ln of the geometric mean distance between rectangles [pi0, pi1] x [qi0, qi1] and
    [pj0, pj1] x [qj0, qj1] (exact; arrays broadcast).  A square of side s: GMD 0.44705 s."""
    A = (pi1 - pi0) * (qi1 - qi0) * (pj1 - pj0) * (qj1 - qj0)
    return _quad_sum(_G2, pi0, pi1, pj0, pj1, qi0, qi1, qj0, qj1) / A


def bar_integral(l, a0, a1, pi0, pi1, qi0, qi1, pj0, pj1, qj0, qj1):
    """∫ dV_i ∫ dV_j / r for parallel bars: i along [0, l], j along [a0, a1], cross-sections
    [pi0, pi1] x [qi0, qi1] and [pj0, pj1] x [qj0, qj1] (arrays).  The faces' four-fold integral is
    exact; the height difference s = q_i - q_j is integrated with its weight K(s) (the overlap of
    the two height ranges, piecewise linear) by Gauss quadrature on each piece, with a break at
    s = 0 (where the integrand's slope has a log singularity)."""
    e = np.stack([qi0 - qj1, qi0 - qj0, qi1 - qj1, qi1 - qj0])
    lo, hi = e[0], e[3]
    z = np.clip(np.zeros_like(lo), lo, hi)
    pts = np.sort(np.concatenate([e, z[None]]), axis=0)
    g, wg = _GQ
    tot = np.zeros_like(lo)
    for k in range(4):
        s0, s1 = pts[k], pts[k + 1]
        half, mid = (s1 - s0) / 2, (s1 + s0) / 2
        for gk, wk in zip(g, wg):
            s = mid + half * gk
            K = np.clip(np.minimum(qi1, qj1 + s) - np.maximum(qi0, qj0 + s), 0.0, None)
            T = rect_integral(0.0, l, pi0, pi1, a0, a1, pj0, pj1, s)
            tot = tot + wk * half * K * T
    return tot


def _inner(P, A, v, L, eps2):
    """∫ dl / sqrt(|r - P|² + eps²) along the segment from A, direction v, length L (closed form)."""
    d = A - P
    ta = np.einsum("...k,...k->...", d, v)
    rho2 = np.maximum(np.einsum("...k,...k->...", d, d) - ta * ta, 0.0) + eps2
    rho = np.sqrt(np.maximum(rho2, 1e-300))
    return np.arcsinh((ta + L) / rho) - np.arcsinh(ta / rho)


# ---- filament sets ----------------------------------------------------------------------
class Filaments:
    """Arrays of n filaments (see the module doc)."""

    def __init__(self, c0, u, l, wd, hd, w, h):
        self.c0, self.u, self.l = np.asarray(c0, float), np.asarray(u, float), np.asarray(l, float)
        self.wd, self.hd = np.asarray(wd, float), np.asarray(hd, float)
        self.w, self.h = np.asarray(w, float), np.asarray(h, float)

    def __len__(self):
        return len(self.l)


def mutual_pairs(F, i, j):
    """Partial (mutual or self) inductances [H] of filament pairs (i[k], j[k])."""
    i, j = np.asarray(i), np.asarray(j)
    out = np.zeros(i.shape)
    ui, uj = F.u[i], F.u[j]
    dot = np.einsum("ik,ik->i", ui, uj)
    par = np.abs(dot) > 1 - PAR
    if par.any():
        k = np.flatnonzero(par)
        out[k] = _parallel(F, i[k], j[k], np.sign(dot[k]))
    k = np.flatnonzero(~par & (np.abs(dot) > 1e-12))
    if k.size:
        out[k] = _angled(F, i[k], j[k], dot[k])
    return out


def _parallel(F, i, j, sgn):
    li, lj = F.l[i], F.l[j]
    r = F.c0[j] - F.c0[i]
    a = np.einsum("ik,ik->i", r, F.u[i])
    b = a + lj * sgn
    a0, a1 = np.minimum(a, b), np.maximum(a, b)
    dp = np.einsum("ik,ik->i", r, F.wd[i])
    dq = np.einsum("ik,ik->i", r, F.hd[i])
    wi, hi = F.w[i], F.h[i]
    aligned = np.abs(np.einsum("ik,ik->i", F.wd[i], F.wd[j])) >= 0.5
    wj = np.where(aligned, F.w[j], F.h[j])
    hj = np.where(aligned, F.h[j], F.w[j])
    S = np.maximum(np.maximum(wi, hi), np.maximum(wj, hj))
    d = np.hypot(dp, dq)
    M = axial(0.0, li, a0, a1, np.where(d > 0, d, 1.0))
    near = d < NEAR * S
    if near.any():
        q = np.flatnonzero(near)
        rect = (-wi[q] / 2, wi[q] / 2, -hi[q] / 2, hi[q] / 2,
                dp[q] - wj[q] / 2, dp[q] + wj[q] / 2, dq[q] - hj[q] / 2, dq[q] + hj[q] / 2)
        long = (np.minimum(li[q], lj[q]) > LONG * S[q]) | (d[q] >= EXACT * S[q])
        if long.any():
            g = np.flatnonzero(long)
            p = q[g]
            gmd = _gmd_unique(wi[p], hi[p], np.abs(dp[p]), np.abs(dq[p]), wj[p], hj[p])
            M[p] = axial(0.0, li[p], a0[p], a1[p], gmd)
        if (~long).any():
            p = q[np.flatnonzero(~long)]
            M[p] = _exact_unique(li[p], a0[p], a1[p], wi[p], hi[p], dp[p], dq[p], wj[p], hj[p])
    return KM * sgn * M


def _gmd_unique(wi, hi, dp, dq, wj, hj):
    """GMD of centred rectangles wi x hi and wj x hj offset by (dp, dq) >= 0, each distinct
    geometry computed once (it does not depend on the bars' lengths)."""
    _, u, inv = np.unique(_hash([wi, hi, dp, dq, wj, hj]), return_index=True, return_inverse=True)
    g = np.exp(log_gmd(-wi[u] / 2, wi[u] / 2, -hi[u] / 2, hi[u] / 2,
                       dp[u] - wj[u] / 2, dp[u] + wj[u] / 2, dq[u] - hj[u] / 2, dq[u] + hj[u] / 2))
    return g[inv.ravel()]


_MIX = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93,
                 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x94D049BB133111EB, 0xBF58476D1CE4E5B9,
                 0x2545F4914F6CDD1D], dtype=np.uint64)


def _hash(cols):
    """One 64-bit hash per row of the quantised columns (KEY_RES): equal geometry, equal hash;
    a collision between different geometries is ~1e-19 per pair of shapes."""
    h = np.zeros(len(cols[0]), np.uint64)
    with np.errstate(over="ignore"):
        for c, m in zip(cols, _MIX):
            k = np.round(np.asarray(c) / KEY_RES).astype(np.int64).view(np.uint64)
            h = (h ^ (k * m)) * np.uint64(0x9E3779B97F4A7C15)
            h ^= h >> np.uint64(29)
    return h


KEY_RES = 1e-10                      # m: pairs whose relative geometry agrees to this share one exact integral


def _exact_unique(l, a0, a1, wi, hi, dp, dq, wj, hj):
    """Exact six-fold integrals / (A_i A_j) of parallel bar pairs, each distinct relative geometry
    computed once.  Both cross-sections are centred rectangles, so only |dp| and |dq| matter, and
    reversing the axis maps j's span [a0, a1] on [0, l] to [l - a1, l - a0]: the canonical form
    takes the smaller start.  On a pour's cell grid a few hundred shapes cover 10⁵ pairs."""
    dp, dq = np.abs(dp), np.abs(dq)
    flip = (l - a1) < a0
    a0, a1 = np.where(flip, l - a1, a0), np.where(flip, l - a0, a1)
    _, first, inv = np.unique(_hash([l, a0, a1, wi, hi, dp, dq, wj, hj]), return_index=True, return_inverse=True)
    u = first
    I = bar_integral(l[u], a0[u], a1[u], -wi[u] / 2, wi[u] / 2, -hi[u] / 2, hi[u] / 2,
                     dp[u] - wj[u] / 2, dp[u] + wj[u] / 2, dq[u] - hj[u] / 2, dq[u] + hj[u] / 2)
    vals = I / (wi[u] * hi[u] * wj[u] * hj[u])
    return vals[inv.ravel()]


def _angled(F, i, j, dot):
    li, lj = F.l[i], F.l[j]
    mi = F.c0[i] + F.u[i] * li[:, None] / 2
    mj = F.c0[j] + F.u[j] * lj[:, None] / 2
    D = np.linalg.norm(mj - mi, axis=1)
    size = np.maximum(np.maximum(li, lj), np.maximum(np.maximum(F.w[i], F.h[i]), np.maximum(F.w[j], F.h[j])))
    M = dot * li * lj / np.maximum(D, 1e-300)
    near = D < FAR * size
    if near.any():
        q = np.flatnonzero(near)
        ii, jj = i[q], j[q]
        eps2 = (0.2235 * (F.w[ii] + F.h[ii] + F.w[jj] + F.h[jj]) / 2) ** 2
        g, wg = _GN
        acc = np.zeros(q.size)
        for gk, wk in zip(g, wg):
            t = (gk + 1) / 2 * li[q]
            P = F.c0[ii] + F.u[ii] * t[:, None]
            acc += wk * _inner(P, F.c0[jj], F.u[jj], lj[q], eps2)
        M[q] = dot[q] * li[q] / 2 * acc
    return KM * M


def inductance_matrix(F, block_bytes=None):
    """Dense partial inductance matrix [H] of all filaments (n x n), upper-triangle row blocks
    in threads (numpy releases the GIL in its array loops), mirrored.  Blocks are large (a few
    hundred rows): small ones cost more in per-block work than they save (28-row blocks of a
    7400-filament plane took 21 s, 225-row ones 5 s).  block_bytes: temporaries per thread
    (default 1/40 of the free memory, 128-512 MB)."""
    n = len(F)
    L = np.zeros((n, n))
    if block_bytes is None:
        from ..loop.peec import available_ram
        block_bytes = int(min(512 << 20, max(128 << 20, available_ram() // (40 * max(THREADS, 1)) * THREADS)))
    rows = int(min(1024, max(16, block_bytes // (8 * 40 * max(n, 1)))))

    def work(s):
        e = min(s + rows, n)
        ii, jj = np.meshgrid(np.arange(s, e), np.arange(s, n), indexing="ij")
        keep = jj >= ii
        vals = mutual_pairs(F, ii[keep], jj[keep])
        blk = np.zeros(ii.shape)
        blk[keep] = vals
        L[s:e, s:] = blk
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(work, range(0, n, rows)))
    return np.triu(L) + np.triu(L, 1).T
