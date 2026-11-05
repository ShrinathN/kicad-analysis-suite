"""3D extraction (em3d): filament kernels, skin effect, loops, reduced model, FastHenry."""
import math
import os
import sys

import numpy as np
import pytest
from scipy.special import ellipe, ellipk, jv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcb_current_sim.em3d import fasthenry as FH          # noqa: E402
from pcb_current_sim.em3d import geometry as G            # noqa: E402
from pcb_current_sim.em3d import kernels as K             # noqa: E402
from pcb_current_sim.em3d import shapes as SH             # noqa: E402
from pcb_current_sim.em3d import system as S              # noqa: E402

MU0 = 4e-7 * math.pi


def bar(l, w, h):
    return K.Filaments([[0, 0, 0]], [[1, 0, 0]], [l], [[0, 1, 0]], [[0, 0, 1]], [w], [h])


def both(F, i, j):
    """(exact six-fold, GMD filament) for one pair."""
    old = K.LONG, K.EXACT
    try:
        K.LONG, K.EXACT = 1e30, 1e30
        a = K.mutual_pairs(F, [i], [j])[0]
        K.LONG = 0.0
        b = K.mutual_pairs(F, [i], [j])[0]
    finally:
        K.LONG, K.EXACT = old
    return a, b


def test_gmd_closed_form():
    assert math.exp(K.log_gmd(0, 1, 0, 1, 0, 1, 0, 1)) == pytest.approx(0.44705, rel=1e-4)
    rng = np.random.default_rng(1)
    a, b = rng.random((400000, 2)), rng.random((400000, 2)) + [1.5, 0]
    mc = math.exp(np.mean(np.log(np.linalg.norm(a - b, axis=1))))
    assert math.exp(K.log_gmd(0, 1, 0, 1, 1.5, 2.5, 0, 1)) == pytest.approx(mc, rel=2e-3)


@pytest.mark.parametrize("l", [1e-3, 1e-2, 1e-1])
def test_bar_self_inductance_matches_grover(l):
    w, h = 1e-4, 3.5e-5
    ex, gm = both(bar(l, w, h), 0, 0)
    grover = 2e-7 * l * (math.log(2 * l / (w + h)) + 0.5 + 0.2235 * (w + h) / l)
    assert ex == pytest.approx(grover, rel=2e-3)
    if l >= 1e-2:
        assert gm == pytest.approx(ex, rel=5e-4)


def test_near_pairs_exact_and_gmd_agree():
    for c in ([0, 2e-4, 0], [0, 0, 4e-5], [1e-2, 0, 0]):
        F = K.Filaments([[0, 0, 0], c], [[1, 0, 0]] * 2, [1e-2] * 2, [[0, 1, 0]] * 2, [[0, 0, 1]] * 2,
                        [1e-4] * 2, [3.5e-5] * 2)
        ex, gm = both(F, 0, 1)
        assert gm == pytest.approx(ex, rel=1e-3)
    # an angled pair approaching parallel joins the parallel formula
    a = 1e-4
    F = K.Filaments([[0, 0, 0], [0, 1e-3, 0]], [[1, 0, 0], [math.cos(a), math.sin(a), 0]], [1e-2] * 2,
                    [[0, 1, 0], [-math.sin(a), math.cos(a), 0]], [[0, 0, 1]] * 2, [1e-5] * 2, [1e-5] * 2)
    ang = K._angled(F, np.array([0]), np.array([1]), np.array([math.cos(a)]))[0]
    par = K._parallel(F, np.array([0]), np.array([1]), np.array([1.0]))[0]
    assert ang == pytest.approx(par, rel=1e-3)


def test_round_wire_skin_effect_matches_bessel():
    r, l = 0.5e-3, 0.1
    g = SH.round_wire(r, l, cells=10)
    s = S.System(g)
    area = len(g.segs) * (r / 10) ** 2
    rdc = l / (G.SIGMA_CU * area)
    for x in (1.0, 2.5, 5.0):
        d = r / x
        f = 1 / (math.pi * MU0 * G.SIGMA_CU * d * d)
        Z, _ = s.direct(f)
        k = (1 - 1j) / d
        bessel = (k * r / 2 * jv(0, k * r) / jv(1, k * r)).real
        assert Z[0, 0].real / rdc == pytest.approx(bessel, rel=3e-3)
    Z, _ = s.direct(1.0)
    l_dc = MU0 * l / (2 * math.pi) * (math.log(2 * l / r) - 0.75)
    assert Z[0, 0].imag / (2 * math.pi) == pytest.approx(l_dc, rel=1e-3)
    assert Z[0, 0].real == pytest.approx(rdc, rel=1e-9)


def test_coaxial_loops_match_maxwell():
    R, dz, a = 10e-3, 5e-3, 0.2e-3
    g = SH.polygon_loop(R, a, name="A")
    SH.polygon_loop(R, a, z=dz, g=g, name="B")
    Z, _ = S.System(g).direct(1e3)
    w = 2 * math.pi * 1e3
    k2 = 4 * R * R / (4 * R * R + dz * dz)
    k = math.sqrt(k2)
    M = MU0 * R * ((2 / k - k) * ellipk(k2) - 2 / k * ellipe(k2))
    L = MU0 * R * (math.log(8 * R / (0.44705 * a)) - 2)
    assert Z[0, 1].imag / w == pytest.approx(M, rel=3e-3)
    assert Z[0, 0].imag / w == pytest.approx(L, rel=3e-3)


def test_reduced_model_matches_direct_solve():
    g = SH.square_spiral(3, 10e-3, 0.5e-3, 0.3e-3, 35e-6, nw=5, nh=3)
    SH.hairpin(8e-3, 0.5e-3, 0.3e-3, 35e-6, nw=3, nh=3, g=g, name="near")
    s = S.System(g)
    m = s.reduce(1e3, 1e9)
    for f in (0.0 + 1e-3, 1e4, 3e6, 1e8, 1e9):
        Zd, _ = s.direct(f)
        Zr = m.Z(f)[0]
        assert np.abs(Zr - Zd).max() / np.abs(Zd).max() < 1e-4
    # passive: every mode's time constant is >= 0 and R(f) of the coil grows with f
    assert (m.tau >= 0).all()
    R = m.Z(np.geomspace(1e3, 1e9, 30))[:, 0, 0].real
    assert (np.diff(R) > -1e-9 * R[0]).all()


def test_discretisation_follows_skin_depth():
    assert G.division(35e-6, G.skin_depth(1e4)) == 1
    n = G.division(35e-6, G.skin_depth(1e8))
    assert G.grade(n, 2.0)[0][0] * 35e-6 <= G.skin_depth(1e8) / 2
    s, c = G.grade(5, 2.0)
    assert s.sum() == pytest.approx(1.0) and s[0] < s[1] < s[2] and c[2] == pytest.approx(0.0)


def test_unconnected_port_is_an_error():
    g = G.Geometry()
    a, b, c, d = g.node(0, 0, 0), g.node(1e-2, 0, 0), g.node(0, 1e-3, 0), g.node(1e-2, 1e-3, 0)
    g.seg(a, b, 1e-4, 1e-4)
    g.seg(c, d, 1e-4, 1e-4)
    g.port("P", a, d)
    with pytest.raises(ValueError, match="not joined"):
        S.System(g)


@pytest.mark.skipif(FH.find() is None, reason="FastHenry not installed")
def test_matches_fasthenry():
    for g, r_tol in ((SH.square_spiral(3, 10e-3, 0.5e-3, 0.3e-3, 35e-6, nw=9, nh=5), 0.006),
                     (SH.hairpin(20e-3, 1e-3, 0.2e-3, 35e-6, nw=7, nh=3), 0.003)):
        m = S.System(g).reduce(1e3, 1e9)
        f, Zf = FH.run(g, 1e3, 1e9, 1)
        Z = m.Z(f)
        np.testing.assert_allclose(Z[:, 0, 0].real, Zf[:, 0, 0].real, rtol=r_tol)
        np.testing.assert_allclose(Z[:, 0, 0].imag, Zf[:, 0, 0].imag, rtol=3e-3)
