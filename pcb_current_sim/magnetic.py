"""Magnetic field and loop inductance from the solved current distribution.

Currents are represented as current moments m = J t h^2 (A m) per grid cell and
copper layer (horizontal), as straight filaments for resistance-element bodies
and loop closures, and as vertical filaments for via / plated-hole barrels.

* B field (Biot-Savart) on a plane above the top and below the bottom surface,
  as FFT convolutions of the moments with the Biot-Savart kernels, plus the
  analytic field of finite vertical segments.
* Inductance: Neumann partial-inductance sum  L_mn = mu0/4pi sum m_m . m_n / R
  over the per-row unit current distributions (one row driven with 1 A, all
  other sources off).  The 1/R kernel uses the exact cell-averaged value for
  coincident cells and Gauss quadrature for near neighbours.  Vertical barrel
  segments use Grover's formula for parallel filaments.  Each row's loop is
  closed by a straight filament from the pads where its current leaves the
  copper back to the pads where it enters (the external source / component).

The current distribution is the DC one; at high frequency return currents
crowd under their signal and the loop inductance drops somewhat.
"""
import math

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp

from .raster import Grid

MU0_4PI = 1e-7
SELF_SQUARE = 4 * (math.log(1 + math.sqrt(2)) - (math.sqrt(2) - 1) / 3)   # <1/R> over a unit square pair
MAX_CELLS = 400_000          # moment-grid cells per layer
MARGIN_MM = 3.0              # field-plane margin around the copper


def layer_z(board, thick):
    """Copper layer centre heights [m] (top surface z = 0, up positive) and bottom surface z."""
    z, out = 0.0, {}
    for k, L in enumerate(board.copper_layers):
        out[L] = z - thick[L] / 2
        z -= thick[L] + (board.gaps[k] if k < len(board.gaps) else 0.0)
    return out, z


class MomentGrid:
    """Coarse lattice aligned with the electrical grid (integer factor c), plus a margin."""

    def __init__(self, grid, max_cells=MAX_CELLS, margin_mm=MARGIN_MM):
        c = max(1, math.ceil(math.sqrt(grid.nx * grid.ny / max_cells)))
        h = grid.h * c
        pad = int(math.ceil(margin_mm / h))
        nxc, nyc = -(-grid.nx // c), -(-grid.ny // c)
        self.g = Grid(grid.x0 - pad * h, grid.y0 - pad * h, h, nxc + 2 * pad, nyc + 2 * pad)
        ci = np.arange(grid.nx) // c + pad
        cj = np.arange(grid.ny) // c + pad
        self.Pc = sp.csr_matrix((np.ones(grid.nx), (ci, np.arange(grid.nx))), shape=(self.g.nx, grid.nx))
        self.Pr = sp.csr_matrix((np.ones(grid.ny), (cj, np.arange(grid.ny))), shape=(self.g.ny, grid.ny))
        self.fine = grid

    def aggregate(self, a):
        return np.asarray(self.Pr @ (self.Pc @ np.nan_to_num(np.asarray(a, float)).T).T)

    def deposit_line(self, mx, my, a, b, I):
        """Add a straight current filament a -> b [mm] carrying I [A] to the moment arrays."""
        g = self.g
        L = math.dist(a, b)
        if L == 0 or I == 0:
            return
        n = max(1, int(math.ceil(L / (g.h / 2))))
        t = (np.arange(n) + 0.5) / n
        xs, ys = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        ii = np.floor((xs - g.x0) / g.h).astype(int)
        jj = np.floor((ys - g.y0) / g.h).astype(int)
        ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny)
        np.add.at(mx, (jj[ok], ii[ok]), I * (b[0] - a[0]) * 1e-3 / n)
        np.add.at(my, (jj[ok], ii[ok]), I * (b[1] - a[1]) * 1e-3 / n)


def _offsets(g, dz, near):
    """Offset arrays (dx, dy in m) over (2ny-1, 2nx-1) and a near-cell mask."""
    h = g.h * 1e-3
    di = np.arange(-(g.nx - 1), g.nx)
    dj = np.arange(-(g.ny - 1), g.ny)
    DX, DY = np.meshgrid(di * h, dj * h)
    nearm = (np.abs(DX) <= near * h + 1e-12) & (np.abs(DY) <= near * h + 1e-12) & (abs(dz) < (near + 1) * h)
    return DX, DY, nearm


_GL = np.polynomial.legendre.leggauss(4)


def _cell_avg_inv_r(dx, dy, dz, h):
    """<1/R> between two h x h cells offset by (dx, dy, dz), by 4x4 x 4x4 Gauss quadrature."""
    x, w = _GL
    u = x * h / 2
    U1, V1, U2, V2 = np.meshgrid(u, u, u, u, indexing="ij")
    W = np.einsum("i,j,k,l->ijkl", w, w, w, w) / 16
    R = np.sqrt((dx + U2 - U1) ** 2 + (dy + V2 - V1) ** 2 + dz ** 2)
    return float(np.sum(W / R))


def inv_r_kernel(g, dz, near=3):
    """Cell-averaged 1/R kernel [1/m] for sources and targets on layers dz [m] apart."""
    h = g.h * 1e-3
    DX, DY, nearm = _offsets(g, dz, near)
    with np.errstate(divide="ignore"):
        K = 1.0 / np.sqrt(DX ** 2 + DY ** 2 + dz ** 2)
    for j, i in zip(*np.nonzero(nearm)):
        if DX[j, i] == 0 and DY[j, i] == 0 and dz == 0:
            K[j, i] = SELF_SQUARE / h
        else:
            K[j, i] = _cell_avg_inv_r(DX[j, i], DY[j, i], dz, h)
    return K


class FFTConv:
    """Circular FFT convolution sized so that offsets within the grid never wrap.
    Kernels are given over offsets (2ny-1, 2nx-1) centred on zero offset."""

    def __init__(self, shape):
        self.ny, self.nx = shape
        self.N = (sfft.next_fast_len(2 * self.ny - 1), sfft.next_fast_len(2 * self.nx - 1, real=True))

    def kernel(self, K):
        Kc = np.zeros(self.N)
        Kc[:K.shape[0], :K.shape[1]] = K
        return sfft.rfft2(np.roll(Kc, (-(self.ny - 1), -(self.nx - 1)), axis=(0, 1)))

    def fwd(self, a):
        return sfft.rfft2(a, s=self.N)

    def inv(self, F):
        return sfft.irfft2(F, s=self.N)[:self.ny, :self.nx]


def grover_parallel(a1, a2, b1, b2, d):
    """Mutual partial inductance [H] of parallel filaments along z: [a1, a2] and [b1, b2] (m), spacing d."""
    def F(u):
        return u * math.asinh(u / d) - math.sqrt(u * u + d * d)
    return MU0_4PI * (F(a2 - b1) + F(a1 - b2) - F(a2 - b2) - F(a1 - b1))


# ---------------------------------------------------------------------------
class CurrentSet:
    """Horizontal moments per layer plus vertical segments for one current distribution."""

    def __init__(self, mg, layers):
        g = mg.g
        self.mx = {L: np.zeros(g.shape) for L in layers}
        self.my = {L: np.zeros(g.shape) for L in layers}
        self.vert = []        # (x mm, y mm, z_low m, z_high m, I_up A, radius m)
        self.closure_mm = 0.0


def currents_of(sol, net, mg, thick, zL, closure=None):
    """CurrentSet for a Solution. closure: (from_xy, to_xy, layer, I) external return filament."""
    board, fine = net.board, mg.fine
    cs = CurrentSet(mg, board.copper_layers)
    h_m = fine.h * 1e-3
    for L in board.copper_layers:
        k = 1e6 * thick[L] * h_m * h_m                 # A/mm^2 -> A*m per fine cell
        cs.mx[L] += mg.aggregate(sol.Jx[L]) * k
        cs.my[L] += mg.aggregate(sol.Jy[L]) * k
    side = {fp.ref: fp.layer for fp in board.footprints}
    for e, I in zip(net.elements, sol.element_I):
        pa, pb = board.pads_by_id(e.a)[0], board.pads_by_id(e.b)[0]
        L = side.get(pa.ref, pa.layers[0])
        L = L if L in cs.mx else board.copper_layers[0]
        mg.deposit_line(cs.mx[L], cs.my[L], pa.pos, pb.pos, I)
    if closure is not None:
        a, b, L, I = closure
        mg.deposit_line(cs.mx[L], cs.my[L], a, b, I)
        cs.closure_mm = math.dist(a, b)
    for bi, L1, L2, I, P, dV in sol.barrel_links:
        hole = net.barrels[bi]["hole"]
        if I:
            cs.vert.append((hole["pos"][0], hole["pos"][1], zL[L2], zL[L1], -I,
                            max(hole["drill"] * 1e-3 / 2, 1e-5)))       # +I flows L1 (upper) -> L2
    return cs


def _centroid(board, ids):
    ps = [p for pid in ids for p in board.pads_by_id(pid)]
    return (float(np.mean([p.pos[0] for p in ps])), float(np.mean([p.pos[1] for p in ps]))), ps


def inductance_matrix(net, unit_sols, mg, thick, zL):
    """Partial inductance matrix [H] of the rows' unit loops (None entries for rows without a loop)."""
    board, rows = net.board, net.rows
    sets = []
    for row, sol in zip(rows, unit_sols):
        if sol is None:
            sets.append(None)
            continue
        entry, exit_ = row.entry_exit
        (ea, pa), (xa, px) = _centroid(board, entry), _centroid(board, exit_)
        L = pa[0].layers[0] if pa[0].layers else board.copper_layers[0]
        if L not in (board.copper_layers[0], board.copper_layers[-1]):
            L = board.copper_layers[0]
        sets.append(currents_of(sol, net, mg, thick, zL, closure=(xa, ea, L, 1.0)))
    layers = board.copper_layers
    fc = FFTConv(mg.g.shape)
    kernels = {}
    for a in layers:
        for b in layers:
            dz = round(abs(zL[a] - zL[b]), 9)
            if dz not in kernels:
                kernels[dz] = fc.kernel(inv_r_kernel(mg.g, dz))
    n = len(rows)
    M = np.full((n, n), np.nan)
    live = [r for r in range(n) if sets[r] is not None]
    for m in live:
        F = {b: (fc.fwd(sets[m].mx[b]), fc.fwd(sets[m].my[b])) for b in layers}
        A = {}
        for a in layers:
            Kab = [kernels[round(abs(zL[a] - zL[b]), 9)] for b in layers]
            A[a] = (fc.inv(sum(F[b][0] * K for b, K in zip(layers, Kab))),
                    fc.inv(sum(F[b][1] * K for b, K in zip(layers, Kab))))
        for k in live:
            v = sum(float(np.sum(sets[k].mx[a] * A[a][0] + sets[k].my[a] * A[a][1])) for a in layers)
            v *= MU0_4PI
            for x1, y1, lo1, hi1, I1, r1 in sets[m].vert:
                for x2, y2, lo2, hi2, I2, r2 in sets[k].vert:
                    d = max(math.hypot(x1 - x2, y1 - y2) * 1e-3, r1 if (x1, y1) == (x2, y2) else 0, 1e-6)
                    v += I1 * I2 * grover_parallel(lo1, hi1, lo2, hi2, d)
            M[k, m] = v
    M = 0.5 * (M + M.T)
    return M, [s.closure_mm if s is not None else None for s in sets]


def b_planes(sol, net, mg, thick, zL, z_bottom, height_mm):
    """B field [T] on planes height_mm above the top and below the bottom surface.
    Returns {"above": (Bx, By, Bz), "below": ...} in board axes (x right, y down, z up)."""
    board = net.board
    cs = currents_of(sol, net, mg, thick, zL)
    g = mg.g
    hgt = max(height_mm, 2 * g.h) * 1e-3
    out = {}
    X, Y = g.centres()
    Xm, Ym = X * 1e-3, Y * 1e-3
    fc = FFTConv(g.shape)
    live = [L for L in board.copper_layers if np.any(cs.mx[L]) or np.any(cs.my[L])]
    F = {L: (fc.fwd(cs.mx[L]), fc.fwd(cs.my[L])) for L in live}
    DX, DY, _ = _offsets(g, 1.0, 0)
    for side, zo in (("above", hgt), ("below", z_bottom - hgt)):
        fx = fy = fz = 0
        for L in live:
            dz = zo - zL[L]
            R3 = (DX ** 2 + DY ** 2 + dz ** 2) ** 1.5
            Kz, Kx, Ky = fc.kernel(dz / R3), fc.kernel(DX / R3), fc.kernel(DY / R3)
            Fmx, Fmy = F[L]
            # Biot-Savart B = mu0/4pi m x R / R^3, written in board axes (x right, y down, z up)
            fx = fx - Fmy * Kz
            fy = fy + Fmx * Kz
            fz = fz - (Fmx * Ky - Fmy * Kx)
        if live:
            Bx, By, Bz = (MU0_4PI * fc.inv(f) for f in (fx, fy, fz))
        else:
            Bx, By, Bz = np.zeros(g.shape), np.zeros(g.shape), np.zeros(g.shape)
        for xv, yv, lo, hi, I_up, r in cs.vert:
            rx, ry = Xm - xv * 1e-3, Ym - yv * 1e-3
            rho = np.maximum(np.hypot(rx, ry), r)
            f = MU0_4PI * I_up / rho * ((zo - lo) / np.hypot(rho, zo - lo) - (zo - hi) / np.hypot(rho, zo - hi))
            # +z current: B circulates counter-clockwise seen from above in right-handed (x, -y) axes
            Bx += f * (ry / rho)
            By += f * (-rx / rho)
        out[side] = (Bx, By, Bz)
    return out
