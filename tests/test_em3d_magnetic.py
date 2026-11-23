"""3D extraction, magnetic cores (em3d.magnetic): facet volume-integral method.

References: brute-force charge integrals, the cube's polarisability, the weak-core (Born) limit,
the image method for a plate, the gap reluctance of a gapped ring."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcb_current_sim.em3d import kernels as K             # noqa: E402
from pcb_current_sim.em3d import magnetic as Mg           # noqa: E402
from pcb_current_sim.em3d import shapes as SH             # noqa: E402
from pcb_current_sim.em3d import system as S              # noqa: E402

MU0 = 4e-7 * math.pi
A = np.array


# ---- helpers ---------------------------------------------------------------------------
def _brute(lo_a, hi_a, lo_b, hi_b, kb, n=12):
    """∫_a (potential of b) by n sub-intervals x 4 Gauss points per axis of a."""
    x, w = np.polynomial.legendre.leggauss(4)
    axes = []
    for a in range(3):
        if hi_a[a] == lo_a[a]:
            axes.append((A([lo_a[a]]), A([1.0])))
            continue
        e = np.linspace(lo_a[a], hi_a[a], n + 1)
        hh = (e[1] - e[0]) / 2
        axes.append(((((e[:-1] + e[1:]) / 2)[:, None] + x[None] * hh).ravel(), np.tile(w * hh, n)))
    P = np.stack(np.meshgrid(*[q[0] for q in axes], indexing="ij"), -1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", *[q[1] for q in axes]).ravel()
    pot = Mg.box_potential(P, lo_b, hi_b) if kb < 0 else Mg.rect_potential(P, lo_b, hi_b, kb)
    return float((pot * W).sum())


def _face(axis, at, lo2, hi2):
    o = [a for a in range(3) if a != axis]
    lo, hi = np.zeros(3), np.zeros(3)
    lo[axis] = hi[axis] = at
    lo[o], hi[o] = lo2, hi2
    return lo, hi, axis


def _cell(lo, hi):
    return A(lo, float), A(hi, float), -1


class _Coil:
    """A loop's port inductance with an added partial-inductance matrix (near DC)."""

    def __init__(self, g):
        self.F = g.filaments()[0]
        self.L0 = K.inductance_matrix(self.F)
        self.sy = S.System(g)

    def L(self, dL=0.0):
        sy = self.sy
        QL = np.asarray(sy.Q.T @ (self.L0 + dL))
        Lq = np.asarray((sy.Q.T @ QL.T).T)
        Y = sy.E.T @ np.linalg.solve(sy.Rq + 1j * 2 * np.pi * Lq, sy.E)
        return (1 / Y[0, 0]).imag / (2 * np.pi)


def _box(lo, hi, chi):
    return (A(lo, float) * 1e-3, A(hi, float) * 1e-3, chi)


# ---- tests -----------------------------------------------------------------------------
@pytest.mark.parametrize("a, b", [
    (_cell([0, 0, 0], [1, 1, 1]), _cell([0, 0, 0], [1, 1, 1])),                    # cell self
    (_cell([0, 0, 0], [3, 1, 1]), _cell([0, 0, 0], [3, 1, 1])),                    # long cell self
    (_cell([0, 0, 0], [1, 1, 1]), _cell([1, 0, 0], [2, 1, 1])),                    # face-adjacent
    (_cell([0, 0, 0], [1, 1, 1]), _cell([1, 1, 1], [2, 2, 2])),                    # corner-adjacent
    (_cell([0, 0, 0], [1, 1, .2]), _cell([0, 0, .2], [1, 1, .4])),                 # thin, stacked
    (_cell([0, 0, 0], [1, 1, 1]), _face(2, 1.0, [0, 0], [1, 1])),                  # cell / own face
    (_cell([0, 0, 0], [1, 1, .2]), _face(0, 1.0, [0, 0], [1, .2])),                # thin cell / side
    (_face(2, .2, [0, 0], [1, 1]), _face(0, 1.0, [0, 0], [1, .2])),                # perpendicular, edge
    (_face(2, 1.0, [0, 0], [1, 1]), _face(2, 1.0, [1, 0], [2, 1])),                # coplanar
    (_face(2, .2, [0, 0], [1, 1]), _face(2, 0.0, [0, 0], [1, 1])),                 # parallel
])
def test_charge_integrals_match_brute_force(a, b):
    (la, ha, ka), (lb, hb, kb) = a, b
    code = Mg._pair_integral(la[None], ha[None], A([ka]), lb[None], hb[None], A([kb]))[0]
    assert code == pytest.approx(_brute(la, ha, lb, hb, kb), rel=3e-3)


def test_cube_polarisability_converges_from_below():
    """A χ → ∞ cube in a uniform field: dipole moment / (H0 V) → 3.644 (known value)."""
    out = []
    for n in (1, 2, 4):
        gr = Mg.grid_from_boxes([(A([0, 0, 0.]), A([1, 1, 1.]), 1e6)], 1.0 / n)
        core = Mg.Core(gr)
        g = np.zeros(core.n_dof)                     # ∫ W_a · e_z dV
        for c in range(len(core.cell_V)):
            for q in core.cell_faces[c, 2]:
                g[q] += core.cell_V[c] / 2
        m = np.linalg.solve(core.K() + core.P, g)
        out.append(g @ m)
    assert out[0] == pytest.approx(3.0, rel=1e-3)    # one cell: the sphere-like 1/N, N = 1/3
    assert out[0] < out[1] < out[2] < 3.644
    assert out[2] > 3.5


def test_weak_core_matches_born_limit():
    """χ ≪ 1: ΔL → μ0 χ ∫ |H|² dV over the core (the coupling's normalisation)."""
    co = _Coil(SH.polygon_loop(3e-3, 0.2e-3, n=32, z=0.5e-3))
    lo, hi, chi = A([-4e-3, -4e-3, -1e-3]), A([4e-3, 4e-3, 0.0]), 1e-4
    core = Mg.Core(Mg.grid_from_boxes([(lo, hi, chi)], 0.5e-3))
    dL = core.delta_L(co.F).sum()                    # one filament per segment, all in series
    # reference: ∫ |H|² by 4³ Gauss in each of 0.25 mm sub-cells
    x, w = np.polynomial.legendre.leggauss(4)
    n = [int(round(s / 0.25e-3)) for s in hi - lo]
    pts, wts = [], []
    for a in range(3):
        e = np.linspace(lo[a], hi[a], n[a] + 1)
        hh = (e[1] - e[0]) / 2
        pts.append((((e[:-1] + e[1:]) / 2)[:, None] + x * hh).ravel())
        wts.append(np.tile(w * hh, n[a]))
    P = np.stack(np.meshgrid(*pts, indexing="ij"), -1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", *wts).ravel()
    F = co.F
    H = Mg._seg_field(P[None], F.c0[:, None], F.u[:, None], F.l[:, None], 0.0).sum(axis=0)
    ref = MU0 * chi * float(((H * H).sum(axis=1) * W).sum())
    assert dL < ref                                  # Galerkin: from below
    assert dL == pytest.approx(ref, rel=0.04)


def test_plate_stays_below_the_half_space_bound():
    """Loop over a high-μ plate vs the image method (the half-space is the upper bound).  With a
    regularised coil field (smeared current) the plate read 4 % above it at μr 1e4."""
    R0, wire, h0 = 3e-3, 0.2e-3, 0.5e-3
    g = SH.polygon_loop(R0, wire, n=48, z=h0, name="A")
    SH.polygon_loop(R0, wire, n=48, z=-h0, g=g, name="B")
    Z, _ = S.System(g).direct(1.0)
    m_img = Z[0, 1].imag / (2 * np.pi)
    co = _Coil(SH.polygon_loop(R0, wire, n=48, z=h0))
    L_air = co.L()
    ratio = {}
    for mu in (1e3, 1e4):
        gr = Mg.grid_from_boxes([(A([-6e-3, -6e-3, -1e-3]), A([6e-3, 6e-3, 0.0]), mu - 1)], 0.75e-3)
        dL = co.L(Mg.Core(gr).delta_L(co.F)) - L_air
        ratio[mu] = dL / (m_img * (mu - 1) / (mu + 1))
    assert 0.93 < ratio[1e3] < 1.0
    assert 0.93 < ratio[1e4] < 1.0
    assert ratio[1e4] == pytest.approx(ratio[1e3], rel=0.01)   # saturated: geometry-limited


def _winding(nt=10):
    from pcb_current_sim.em3d import geometry as G
    g, prev, dl, w = G.Geometry(), None, 0.4e-3, 0.2e-3
    ys = np.linspace(3e-3, 7e-3, nt + 1)
    for k in range(nt):
        y0, y1 = ys[k], ys[k + 1]
        pts = [(-dl, y0, -dl), (2e-3 + dl, y0, -dl), (2e-3 + dl, y0, 3e-3 + dl), (-dl, y0, 3e-3 + dl), (-dl, y1, -dl)]
        ids = [g.node(*p) for p in pts]
        if prev is None:
            first = ids[0]
        else:
            g.join(prev, ids[0])
        for p, q in zip(ids[:-1], ids[1:]):
            g.seg(p, q, w, w)
        prev = ids[-1]
    g.port("W", first, prev)
    return g


RING = [((0, 0, 0), (2, 10, 3)), ((8, 0, 0), (10, 10, 3)), ((2, 0, 0), (8, 2, 3)), ((2, 8, 0), (8, 10, 3))]


def test_gapped_ring_follows_the_gap_reluctance():
    """10-turn winding on a 10 x 10 mm square ring (2 x 3 mm legs), μr 1e4, gap g in the far leg:
    L = a/g + b with a = N² μ0 A (the gap) and b the air leakage, independent of g."""
    co = _Coil(_winding())
    Ls, gs = [], A([0.1e-3, 0.2e-3, 0.4e-3])
    for gap in gs:
        boxes = [_box(lo, hi, 9999.0) for lo, hi in RING]
        boxes.append((A([8e-3, 5e-3 - gap / 2, -1e-3]), A([10e-3, 5e-3 + gap / 2, 4e-3]), 0.0))
        Ls.append(co.L(Mg.Core(Mg.grid_from_boxes(boxes, 1e-3)).delta_L(co.F)))
    a, b = np.polyfit(1 / gs, Ls, 1)
    assert a / (MU0 * 100 * 6e-6) == pytest.approx(1.0, abs=0.08)
    assert 0 < b < 8e-6


def test_ungapped_ring_inductance_scales_with_permeability():
    co = _Coil(_winding())
    core = Mg.Core(Mg.grid_from_boxes([_box(lo, hi, 1.0) for lo, hi in RING], 1e-3))
    L = {mu: co.L(core.delta_L(co.F, chi_cell=np.full(len(core.cell_V), mu - 1.0))) for mu in (1e3, 1e4)}
    assert L[1e4] / L[1e3] == pytest.approx(10.0, rel=0.06)
    # magnetic-circuit estimate μ0 μr N² A / l_mean (l_mean 32 mm): the same order
    assert 0.8 < L[1e3] / (MU0 * 1e3 * 100 * 6e-6 / 32e-3) < 1.4


# ---- frequency-dependent (Debye) cores ---------------------------------------------------
def _spiral_over_plate(material, cell=0.75e-3):
    g = SH.square_spiral(3, 8e-3, 0.5e-3, 0.3e-3, 35e-6, nw=3, nh=1, z=0.3e-3)
    gr = Mg.grid_from_boxes([(A([-1.5e-3, -1.5e-3, -1e-3]), A([9.5e-3, 9.5e-3, 0.0]), material)], cell)
    return g, Mg.Core(gr)


def test_debye_material():
    m = Mg.Material("x", 2001.0, 1e6)
    assert m.chi(0) == pytest.approx(2000.0)
    c = m.chi(1e6)                                   # at f_r: μ' halved, μ'' = μ'
    assert c.real == pytest.approx(1000.0) and c.imag == pytest.approx(-1000.0)
    assert Mg.Material("y", 500.0).chi(1e9) == pytest.approx(499.0)


@pytest.mark.parametrize("material", [Mg.Material("static", 2000.0), Mg.Material("MnZn", 2000.0, 1e6)])
def test_core_modes_match_direct_solves(material):
    g, core = _spiral_over_plate(material)
    G = core.field_matrix(g.filaments()[0])
    M = core.modes(G=G)
    assert M.static.all() == (material.f_r is None)
    for f in (0.0, 3e5, 1e6, 1e7, 1e8):
        D = core.delta_L_at(G, f)
        assert np.abs(M.delta_L(f) - D).max() < 1e-6 * np.abs(D).max()


def test_system_with_dispersive_core_direct_and_reduced():
    g, core = _spiral_over_plate(Mg.Material("MnZn", 2000.0, 1e6))
    sy = S.System(g, core=core)
    assert sy.Rq.shape[0] > sy.nq                    # the dispersive modes are unknowns
    red = sy.reduce(1e3, 1e9)
    air = S.System(g)
    F = g.filaments()[0]
    L0, G = K.inductance_matrix(F), core.field_matrix(F)
    for f in (1e3, 1e6, 1e7, 1e9):
        QL = np.asarray(air.Q.T @ (L0 + core.delta_L_at(G, f)))
        Lq = np.asarray((air.Q.T @ QL.T).T)
        Y = air.E.T @ np.linalg.solve(air.Rq + 2j * np.pi * f * Lq, air.E)
        ref = 1 / Y[0, 0]
        assert sy.direct(f)[0][0, 0] == pytest.approx(ref, rel=1e-5)
        assert red.Z(f)[0, 0, 0] == pytest.approx(ref, rel=1e-5)
    Z0, Zr = air.direct(1e6)[0][0, 0], sy.direct(1e6)[0][0, 0]
    assert Zr.imag > 1.5 * Z0.imag and Zr.real > 2 * Z0.real      # more L, and core loss
    I = sy.currents(sy.direct(1e6)[1], [1.0])
    assert I.shape == (len(F),)


def test_closed_core_loss_tangent_follows_the_material():
    """Ungapped high-μ ring: L_core ∝ μ(f), so R_core / (ω L_core) → μ''/μ' = f / f_r (Debye).
    The air-leakage share does not scale with μ, so a μr 1000 ring reads a little lower
    (0.93 at f_r; 0.993 at 1e4, 0.999 at 1e5)."""
    air = S.System(_winding())
    tan = {}
    for mu in (1e3, 1e4):
        mat = Mg.Material("MnZn", mu, 1e6)
        core = Mg.Core(Mg.grid_from_boxes([_box(lo, hi, mat) for lo, hi in RING], 1e-3))
        sy = S.System(_winding(), core=core)
        Ls = (sy.direct(1.0)[0][0, 0] - air.direct(1.0)[0][0, 0]).imag / (2 * np.pi)
        for f in (3e5, 1e6):
            dZ = sy.direct(f)[0][0, 0] - air.direct(f)[0][0, 0]
            tan[mu, f] = dZ.real / dZ.imag
            if mu == 1e4:
                assert abs(dZ) / (2 * np.pi * f) == pytest.approx(Ls * abs(mat.chi(f)) / mat.chi0, rel=0.02)
    assert tan[1e4, 3e5] == pytest.approx(0.3, rel=0.02)
    assert tan[1e4, 1e6] == pytest.approx(1.0, rel=0.02)
    assert 0.85 < tan[1e3, 1e6] < tan[1e4, 1e6]
