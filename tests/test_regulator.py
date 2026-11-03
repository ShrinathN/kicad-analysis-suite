import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pcb_current_sim.pdn import regulator as RG   # noqa: E402


def record(loop, C=44e-6, esr=3e-3, noise=0.0, seed=1):
    """A synthetic load-transient plot: 1 -> 2 A at 0.5 ms, back at 1.5 ms (5 µs edges), 2 ms long."""
    t = np.linspace(0, 2e-3, 2000)
    i = 1 + np.clip((t - 0.5e-3) / 5e-6, 0, 1) - np.clip((t - 1.5e-3) / 5e-6, 0, 1)
    v = RG.response(loop, t, i, C, esr)
    v = v + noise * np.random.default_rng(seed).standard_normal(len(t)) + 0.05     # AC-coupled: any offset
    return t, i, v


def test_loop_response_is_the_rc_limit_and_crossover():
    lp = RG.Loop("cot", K=50.0, f_z=0.0, f_p=1e9)                # a stiff resistive source 20 mΩ
    t, i, v = record(lp)
    assert v[1400] - v[400] == pytest.approx(-1 / 50.0, rel=0.02)   # droop 1 A x 1/K
    lp2 = RG.Loop("current", K=10.0, f_z=3e3, f_p=300e3)
    assert lp2.f_cross(44e-6) == pytest.approx(10 / (2 * math.pi * 44e-6), rel=0.15)
    assert lp2.f_cross(440e-6) < lp2.f_cross(44e-6) / 5          # 10x the C: a slower loop


@pytest.mark.parametrize("kind,truth", [("current", dict(K=8.0, f_z=4e3, f_p=150e3)),
                                         ("ldo", dict(K=30.0, f_p=60e3))])
def test_fit_recovers_the_loop(kind, truth):
    fixed = RG.TYPES[kind][2]
    lp = RG.Loop(kind, **{**fixed, **truth})
    t, i, v = record(lp, noise=1e-3)
    fits = RG.fit(t, i, t, v, 44e-6, 3e-3, kind)
    assert [f.direction for f in fits] == [1, -1]
    for f in fits:
        assert f.rms < 0.1 * f.peak
        for k, x in truth.items():
            assert getattr(f.loop, k) == pytest.approx(x, rel=0.25), k
        # what matters for the PDN: the same Z with another capacitance
        z_true = 1 / (lp.Y(1e4) + 2j * math.pi * 1e4 * 200e-6)
        z_fit = 1 / (f.loop.Y(1e4) + 2j * math.pi * 1e4 * 200e-6)
        assert abs(z_fit) == pytest.approx(abs(z_true), rel=0.1)


def test_boost_rhp_zero():
    assert RG.rhp_zero(2.4, 3.3, 1.0, 1e-6) == pytest.approx(3.3 * (2.4 / 3.3) ** 2 / (2 * math.pi * 1e-6))
    assert RG.rhp_zero(5, 3.3, 1, 1e-6) == 0.0


def test_digitise_scope_image():
    from pcb_current_sim.pdn import digitise as DG
    H, W = 420, 1000
    img = np.full((H, W, 3), 0.15)                       # dark scope screen
    img[::40, :, :] = 0.4                                 # grey grid lines
    box = (50, 10, 950, 410)                              # 10 x 8 divisions of 90 x 50 px
    cal = DG.Calib(box, 10, 8, 200e-6)
    xs = np.arange(50, 950)
    t_true = (xs - 50) / 900 * 10 * 200e-6
    v_true = 0.12 * np.exp(-((t_true - 0.6e-3) / 50e-6) ** 2)    # a 120 mV bump, 100 mV/div
    rows = np.round((410 + 10) / 2 - v_true / 0.1 * 50).astype(int)
    for x, r in zip(xs, rows):
        img[r - 1:r + 2, x] = (0.9, 0.2, 0.8)              # magenta, 3 px thick
    rgb = DG.colour_at(img, 300, rows[250] + 1)
    t, v = DG.extract(img, cal, rgb, 0.1)
    assert len(t) > 800
    assert v.max() == pytest.approx(0.12, abs=0.004) and t[np.argmax(v)] == pytest.approx(0.6e-3, abs=5e-6)


def test_fit_from_points_clicked_by_hand():
    """A black-and-white plot read by clicking: a few dozen points, steps as two points on one vertical,
    rounded to half a pixel, in no particular order.  The loop must come out as from the full trace."""
    from pcb_current_sim.pdn import digitise as DG
    truth = RG.Loop("current", K=8.0, f_z=4e3, f_0=0.0, f_p=150e3)
    t, i, v = record(truth)
    v = v - 0.05
    cal = DG.Calib((0, 0, 1000, 400), 10, 8, 200e-6)             # 2 ms across, 8 divisions of 50 px
    px = lambda tt: tt / 2e-3 * 1000                             # noqa: E731
    py = lambda val, per: 200 - val / per * 50                  # noqa: E731
    cur = [(0, 1), (0.5e-3, 1), (0.5e-3, 2), (1.5e-3, 2), (1.5e-3, 1), (2e-3, 1)]
    pi = [(round(2 * px(a)) / 2, round(2 * py(b, 0.5)) / 2) for a, b in cur]
    after = np.array([-60, 0, 3, 6, 10, 15, 20, 30, 45, 60, 80, 100, 130, 170, 220, 300, 400, 490]) * 1e-6
    tv = np.concatenate([0.5e-3 + after, 1.5e-3 + after, [0.0, 2e-3 - 1e-6]])
    vv = np.interp(tv, t, v)
    pv = [(round(2 * px(a)) / 2, round(2 * py(b, 0.02)) / 2) for a, b in zip(tv, vv)]
    np.random.default_rng(3).shuffle(pv)                         # clicked in any order
    ti, ii = DG.from_points(cal, pi, 0.5)
    assert np.all(np.diff(ti) > 0) and ii[2] == pytest.approx(2.0) and ii[1] == pytest.approx(1.0)
    tv2, v2 = DG.from_points(cal, pv, 0.02)
    fits = RG.fit(ti, ii, tv2, v2, 44e-6, 3e-3, "current")
    assert [f.direction for f in fits] == [1, -1]
    for f in fits:
        assert f.rms < 0.1 * f.peak
        z_true = 1 / (truth.Y(1e4) + 2j * math.pi * 1e4 * 200e-6)
        z_fit = 1 / (f.loop.Y(1e4) + 2j * math.pi * 1e4 * 200e-6)
        assert abs(z_fit) == pytest.approx(abs(z_true), rel=0.15)
    assert DG.nearest(pv, *pv[4]) == 4 and DG.nearest([], 0, 0) is None
    with pytest.raises(ValueError):
        DG.from_points(cal, pi[:1], 0.5)
    grey = np.full((50, 60, 3), 0.9)
    grey[20, :] = 0.0
    assert DG.is_grey(grey)
    grey[30, :] = (1.0, 0.0, 0.0)
    assert not DG.is_grey(grey)


def test_ringing_recovery_auto_type_fitted_c_and_window():
    t, i, v = record(RG.Loop("resonant", K=6.0, f_z=3e3, f_p=1e9, f_n=60e3, q=2.0), noise=1e-3)
    assert not RG.stable(RG.Loop("resonant", K=6.0, f_z=3e3, f_p=1e9, f_n=25e3, q=4.0), 44e-6, 3e-3)
    fits = RG.fit(t, i, t, v, 44e-6, 3e-3, "auto")
    for f in fits:
        assert f.loop.kind == "resonant" and f.rms < 0.03 * f.peak
        assert f.loop.f_n == pytest.approx(60e3, rel=0.1) and f.loop.q == pytest.approx(2.0, rel=0.2)
        assert dict(f.tried)["current"] > f.rms
    # the datasheet said 44 µF, the plot was taken with 30 µF: fitted, the loop comes out right
    truth = RG.Loop("current", K=8, f_z=4e3, f_p=150e3)
    t, i, v = record(truth, C=30e-6)
    f = RG.fit(t, i, t, v, 44e-6, 3e-3, "current", fit_c=True)[0]
    assert f.c == pytest.approx(30e-6, rel=0.05) and f.loop.K == pytest.approx(8, rel=0.1) and not f.at_limit
    # fit only 100 µs after each edge
    fw = RG.fit(t, i, t, v, 30e-6, 3e-3, "current", window=100e-6, weight="peak")
    assert all(f.t[-1] - f.t[0] < 0.25e-3 for f in fw)
    assert all(f.loop.K == pytest.approx(8, rel=0.15) for f in fw)


def test_colour_by_hue_and_simplify():
    from pcb_current_sim.pdn import digitise as DG
    img = np.ones((40, 50, 3))
    img[10, :] = (0.12, 0.14, 0.44)              # dark blue edge pixels
    img[30, :] = (0.29, 0.02, 0.28)              # dark magenta
    img[20, :] = (0.0, 0.0, 0.0)                 # a black trace
    m = DG.colour_mask(img, (0.73, 0.12, 0.70), 0.3)
    assert m[30].all() and not m[10].any() and not m[20].any()       # RGB distance would take row 10 too
    assert DG.colour_mask(img, (0.05, 0.05, 0.05), 0.3)[20].all()
    xs = np.arange(200.0)
    ys = np.where(xs < 100, 10.0, 50.0) + np.where((xs > 140) & (xs < 150), -20.0, 0.0)
    p = DG.simplify(xs, ys)
    assert len(p) < 20 and (141.0, 30.0) in p and (99.0, 10.0) in p      # the step corners and the dip kept
