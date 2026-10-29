import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pcb_current_sim.matching import analysis as A        # noqa: E402
from pcb_current_sim.matching import microstrip as MS     # noqa: E402
from pcb_current_sim.matching import model as M           # noqa: E402
from pcb_current_sim.matching import touchstone as TS     # noqa: E402
from pcb_current_sim.suite.library import Part            # noqa: E402

F0 = 100e6


def lmatch(tol=0.0, q=0.0):
    """50 Ω source to a 10 Ω load at 100 MHz: shunt C (25 Ω) at the source, series L (20 Ω) to the load."""
    s = M.Setup(f_start=50e6, f_stop=150e6, n_f=101, load=M.Port(r=10.0),
                bands=[M.Band("b", 95e6, 105e6, rl_min=15)])
    C = 1 / (2 * math.pi * F0 * 25)
    L = 20 / (2 * math.pi * F0)
    s.elements = [M.new_element(s, "C", True, Part("C", "c", C, tol)),
                  M.new_element(s, "L", False, Part("L", "l", L, tol, q=q, f_q=F0))]
    return s


def test_lmatch_is_matched_and_lossless():
    nom = A.nominal(lmatch())
    k = np.argmin(abs(nom["f"] - F0))
    assert abs(nom["gam"][k]) < 1e-9 and nom["gt"][k] == pytest.approx(1.0)
    assert nom["band"][0]["rl_worst"] > 20 and nom["ok"]


def test_lossy_inductor_gives_dissipative_loss():
    nom = A.nominal(lmatch(q=20))
    k = np.argmin(abs(nom["f"] - F0))
    # series R = X_L / Q = 1 Ω in a 10 Ω loop: about 10 % of the power, 0.41 dB
    assert nom["diss"][k] == pytest.approx(-10 * math.log10(10 / 11), rel=0.05)


def test_quarter_wave_transformer_and_stubs():
    s = M.Setup(f_start=0.5e9, f_stop=1.5e9, n_f=101, load=M.Port(r=100.0))
    tl = M.new_element(s, "TL", f0=1e9)
    tl.z0 = math.sqrt(50 * 100)
    s.elements = [tl]
    nom = A.nominal(s)
    assert abs(nom["gam"][np.argmin(abs(nom["f"] - 1e9))]) < 1e-9
    # a quarter-wave shorted stub is an open at f0, a quarter-wave open stub a short
    for kind, expect in (("stub_short", 0.0), ("stub_open", 1.0)):
        st = M.new_element(s, kind, f0=1e9)
        s2 = M.Setup(f_start=0.5e9, f_stop=1.5e9, n_f=101, elements=[st])
        g = A.nominal(s2)["gam"][50]
        assert abs(g) == pytest.approx(expect, abs=1e-6)


def test_microstrip_50_ohm_on_fr4():
    z0, eeff = MS.z0_eeff(3.0e-3, 1.6e-3, 4.4)
    assert z0 == pytest.approx(50.5, abs=1.5) and 3.2 < eeff < 3.5
    assert MS.z0_eeff(MS.width_for(50, 1.6e-3, 4.4), 1.6e-3, 4.4)[0] == pytest.approx(50, rel=1e-6)


def test_touchstone_formats_and_s2abcd(tmp_path):
    p = tmp_path / "thru.s2p"
    p.write_text("! thru\n# MHz S DB R 50\n100 -100 0 0 0 0 0 -100 0\n200 -100 0 0 0 0 0 -100 0\n")
    ts = TS.load(str(p))
    assert ts.f[1] == 200e6 and abs(ts.S[0, 1, 0]) == pytest.approx(1)      # S21 = 1
    a, b, c, d = TS.s2abcd(ts.at([150e6]), 50)
    assert abs(a[0] - 1) < 1e-3 and abs(b[0]) < 0.1 and abs(d[0] - 1) < 1e-3
    q = tmp_path / "ant.s1p"
    q.write_text("# GHz S RI R 50\n1 0.2 0.1\n2 -0.2 0.0\n")
    port = M.Port(kind="s1p", path=str(q))
    g = (port.Z([1e9]) - 50) / (port.Z([1e9]) + 50)
    assert g[0] == pytest.approx(0.2 + 0.1j)
    with pytest.raises(ValueError):
        TS.parse("# GHz Z RI R 50\n1 1 1\n", 1)


def test_s2p_element_reproduces_a_series_inductor(tmp_path):
    f = np.linspace(1e8, 1e9, 10)
    zl = 2j * math.pi * f * 10e-9
    s11 = zl / (zl + 100)                      # series Z between two 50 Ω ports
    s21 = 100 / (zl + 100)
    lines = ["# Hz S RI R 50"] + [f"{fi} {a.real} {a.imag} {b.real} {b.imag} {b.real} {b.imag} {a.real} {a.imag}"
                                  for fi, a, b in zip(f, s11, s21)]
    p = tmp_path / "l.s2p"
    p.write_text("\n".join(lines))
    s = M.Setup(f_start=1e8, f_stop=1e9, n_f=10, load=M.Port(r=20.0))
    s.elements = [M.Element("S2P", s2p=str(p), name="X1")]
    ref = M.Setup(f_start=1e8, f_stop=1e9, n_f=10, load=M.Port(r=20.0))
    ref.elements = [M.new_element(ref, "L", False, Part("L", "l", 10e-9, 0))]
    assert np.allclose(A.nominal(s)["gam"], A.nominal(ref)["gam"], atol=1e-9)


def test_corners_mc_yield_and_worst_case():
    s = lmatch(tol=10)
    s.mc_n, s.seed = 4000, 3
    r = A.run(s)
    assert r.corners.n == 4 and r.mc.n == 4000
    assert 0 < r.mc.yield_() < 1                                   # 10 % parts fail a 15 dB spec sometimes
    w = r.worst[0]
    assert w["margin"] <= min(r.corners.band[0]["rl_worst"].min(), r.mc.band[0]["rl_worst"].min()) - 15 + 1e-6
    assert all(v.lo - 1e-18 <= x <= v.hi + 1e-18 for v, x in zip(r.vars, w["x"]))
    g = A.run(s.__class__(**{**s.__dict__, "dist": "gauss"}), do_worst=False)
    assert g.mc.yield_() > r.mc.yield_()                            # Gaussian parts sit nearer nominal
    assert np.all(g.mc.X == g.mc.X.clip([v.lo for v in r.vars], [v.hi for v in r.vars]))   # truncated
    assert r.oat.shape == (2, 1, 2, 2)
    tot = np.nansum(r.share[:, 0, 0])          # small here: at a perfect match worst RL is V-shaped, not linear
    assert 0 <= tot <= 1 + 1e-9


def test_worst_case_finds_interior_minimum():
    """With a resonant response the worst point need not be a corner: a matched L-network with a
    narrow band and generous tolerances, where both corners of one variable over-shoot. The search
    must do at least as well as the corners."""
    s = lmatch(tol=20)
    s.bands = [M.Band("b", 99e6, 101e6, rl_min=40)]
    r = A.run(s, do_mc=False)
    assert r.worst[0]["margin"] <= r.corners.band[0]["rl_worst"].min() - 40 + 1e-6


def test_temperature_variable_moves_x7r():
    s = lmatch()
    c = s.elements[0]
    p = c.get_part()
    p.dielectric, p.tol = "X7R", 0
    from dataclasses import asdict
    c.part = asdict(p)
    s.temp, s.T_min, s.T_max = True, -55, 125
    vs = A.variables(s)
    assert [v.attr for v in vs] == ["T"]
    r = A.run(s, do_mc=False, do_worst=False)
    assert r.corners.n == 2 and r.corners.band[0]["rl_worst"].max() < r.nominal["band"][0]["rl_worst"]


def test_setup_roundtrip_and_bom():
    s = lmatch(tol=5)
    s.elements.append(M.new_element(s, "C", True, Part("C", "c2", 1e-12, 5)))
    s.elements[-1].part["value"] = s.elements[0].part["value"]
    s.elements[-1].part["name"] = s.elements[0].part["name"]
    d = s.to_dict()
    s2 = M.Setup.from_dict(d)
    assert s2.to_dict() == d and s2.elements[0].get_part().value == s.elements[0].get_part().value
    b = A.bom(s)
    assert [r["qty"] for r in b] == [2, 1] and b[0]["refs"] == "C1, C2"
