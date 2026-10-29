import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pcb_current_sim import board as B          # noqa: E402
from pcb_current_sim import scenario as SC      # noqa: E402
from pcb_current_sim import solver as S         # noqa: E402
from pcb_current_sim.suite import stackup as ST  # noqa: E402
from pcb_current_sim.suite.library import Library, Part, parse_si  # noqa: E402
from pcb_current_sim.suite.session import Session  # noqa: E402

FOUR_LAYER = """(kicad_pcb (version 20260206) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "prepreg") (thickness 0.1) (material "FR4") (epsilon_r 4.4) (loss_tangent 0.02)
       addsublayer (thickness 0.1) (material "FR408") (epsilon_r 3.7) (loss_tangent 0.01))
    (layer "In1.Cu" (type "copper") (thickness 0.0175))
    (layer "dielectric 2" (type "core") (thickness 1.065) (material "FR4") (epsilon_r 4.6) (loss_tangent 0.02))
    (layer "In2.Cu" (type "copper") (thickness 0.0175))
    (layer "dielectric 3" (type "prepreg") (thickness 0.2) (material "FR4"))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start 0 0) (end 10 10) (layer "Edge.Cuts"))
)"""


def test_stackup_plies_and_sublayers():
    b = B.parse_board(FOUR_LAYER)
    assert b.copper_layers == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    assert b.gaps[0] == pytest.approx(0.2e-3)           # both plies of dielectric 1 counted
    st = ST.from_board(b)
    d1 = st.dielectrics[0]
    assert d1.er == pytest.approx(0.2 / (0.1 / 4.4 + 0.1 / 3.7))  # series plates
    assert d1.material == "FR4/FR408"
    assert st.dielectrics[2].er == ST.DEFAULT_ER        # no epsilon_r in the file
    assert st.gap("F.Cu", "In2.Cu") == pytest.approx(0.2e-3 + 17.5e-6 + 1.065e-3)


def test_djordjevic_sarkar_fit_and_causality():
    d = ST.Dielectric("d", "FR4", 1e-3, 4.3, 0.02, f_ref=1e9)
    er, td = d.er_tand(1e9)
    assert er == pytest.approx(4.3) and td == pytest.approx(0.02)
    f = np.logspace(6, 10, 50)
    er, td = d.er_tand(f)
    assert np.all(np.diff(er) < 0)                      # εr falls with f (Kramers-Kronig)
    assert np.all((td > 0.015) & (td < 0.025))          # tanδ nearly flat over the band
    assert d.er_tand(1e6, model="const")[0] == pytest.approx(4.3)


def test_scenario_v1_migrates_and_sections_are_kept(tmp_path):
    p = tmp_path / "b.current_sim.json"
    p.write_text(json.dumps({"rows": [{"name": "r", "src": ["J1.1"], "snk": ["J2.1"], "current": 2.0}],
                             "settings": {"thermal": False, "thickness": {"F.Cu": 5e-5}}}))
    rows, s, els = SC.load_scenario(str(p))
    assert rows[0].current == 2.0 and not s.thermal and s.thickness == {"F.Cu": 5e-5}
    SC.save_section(str(p), "matching", {"x": 1})
    s.thickness = {"F.Cu": 6e-5}
    SC.save_scenario(str(p), rows, s)
    d = json.loads(p.read_text())
    assert d["version"] == 2 and d["matching"] == {"x": 1}
    assert d["stackup"]["cu_thickness"] == {"F.Cu": 6e-5} and "thickness" not in d["dc"]["settings"]
    assert SC.load_scenario(str(p))[1].thickness == {"F.Cu": 6e-5}


def test_session_stackup_overrides_persist(tmp_path):
    bp = tmp_path / "four.kicad_pcb"
    bp.write_text(FOUR_LAYER)
    s = Session(library_path=str(tmp_path / "lib.json"))
    s.load_board(str(bp))
    seen = []
    s.listen(seen.append)
    s.stackup.dielectrics[1].er, s.stackup.dielectrics[1].from_board = 4.0, False
    s.set_stackup(s.stackup)
    s.set_cu_override("F.Cu", 50e-6)
    SC.save_scenario(s.scenario_path, [], S.Settings(thickness={"F.Cu": 50e-6}))   # DC save keeps εr edits
    s2 = Session(library_path=str(tmp_path / "lib.json"))
    s2.load_board(str(bp))
    assert s2.stackup.dielectrics[1].er == 4.0 and s2.stackup.dielectrics[0].from_board
    assert s2.stackup.cu("F.Cu") == 50e-6 and seen == ["stackup", "stackup"]


@pytest.mark.parametrize("text,v", [("4.7n", 4.7e-9), ("4n7", 4.7e-9), ("100 nH", 100e-9), ("2u2", 2.2e-6),
                                    ("0R1", 0.1), ("10k", 1e4), ("1.5pF", 1.5e-12), ("2.4G", 2.4e9),
                                    ("1e-9", 1e-9), ("47", 47.0), ("10 mΩ", 0.01)])
def test_parse_si(text, v):
    assert parse_si(text) == pytest.approx(v)


def test_part_models():
    f0 = 3e9
    L = Part("L", "l", 10e-9, srf=f0, q=30, f_q=500e6, dcr=0.1)
    f = np.linspace(1e9, 5e9, 4001)
    assert f[np.argmax(np.abs(L.Z(f)))] == pytest.approx(f0, rel=2e-3)       # parallel resonance at SRF
    z = Part("L", "l", 10e-9, q=30, f_q=500e6, dcr=0.1).Z(500e6)
    assert z.imag / z.real == pytest.approx(30)                               # Q fitted at f_q
    C = Part("C", "c", 100e-9, esr=0.02, srf=20e6)
    assert f_min(C) == pytest.approx(20e6, rel=2e-3) and abs(C.Z(20e6)) == pytest.approx(0.02, rel=1e-3)
    X7R = Part("C", "x", 1e-6, dielectric="X7R", v_rated=25, bias_loss=0.4)
    assert X7R.Z(1e3, vdc=25).imag == pytest.approx(C_Z(1e-6 * 0.6, 1e3).imag, rel=1e-6)
    S = np.array([[9e-9], [10e-9], [11e-9]])                 # MC samples broadcast against f
    assert L.Z(np.array([1e8, 2e8]), value=S).shape == (3, 2)
    assert Part("C", "c", 1e-12, tol=0.1e-12, tol_unit="abs").tol_abs() == 0.1e-12
    for p in Library("/nonexistent/lib.json").parts:                          # built-in tolerances are sane
        assert 0 < p.tol_abs() < 0.35 * p.value, p.name


def f_min(part):
    f = np.linspace(5e6, 50e6, 9001)
    return f[np.argmin(np.abs(part.Z(f)))]


def C_Z(c, f):
    return 1 / (2j * math.pi * f * c)


def test_library_roundtrip(tmp_path):
    p = tmp_path / "sub" / "lib.json"
    lib = Library(str(p))
    n0 = len(lib.parts)
    lib.put(Part("L", "LQW15AN10NJ", 10e-9, 5, dcr=0.1))
    lib.put(Part("L", "LQW15AN10NJ", 10e-9, 2, dcr=0.1))        # same name replaces
    lib.save()
    lib2 = Library(str(p))
    assert len(lib2.parts) == n0 + 1 and lib2.get("LQW15AN10NJ").tol == 2
    lib2.remove("generic 0402 RF 10n")                          # built-ins cannot be removed
    assert lib2.get("generic 0402 RF 10n") is not None
    p.write_text("{broken")
    assert Library(str(p)).error and len(Library(str(p)).parts) == n0


def test_external_finds_flatpak_app_from_sandbox(monkeypatch):
    from pcb_current_sim.suite import external as X
    calls = []

    def fake_run(argv, timeout=5):
        calls.append(argv)
        if argv[-2:] == ["info", "org.paraview.ParaView"]:
            return "Ref: app/org.paraview.ParaView/x86_64/stable\nVersion: 6.1.0\nInstallation: system\n"
        return None
    monkeypatch.setattr(X, "_run", fake_run)
    monkeypatch.setattr(X.shutil, "which", lambda exe: None)
    monkeypatch.setattr(X, "in_flatpak", lambda: True)
    monkeypatch.setattr(X, "host_reachable", lambda: True)
    pv = next(f for f in X.detect()[0] if f.label == "ParaView")
    assert pv.where == "host (flatpak)" and "6.1.0" in pv.path
    assert pv.argv == ["flatpak-spawn", "--host", "flatpak", "run", "org.paraview.ParaView"]
    monkeypatch.setattr(X, "host_reachable", lambda: False)             # sandbox without host access
    found, notes = X.detect()
    assert next(f for f in found if f.label == "ParaView").where == "unknown" and "unknown" in notes[1]


def test_fmt_like_keeps_the_value_prefix():
    from pcb_current_sim.suite.library import fmt_like
    assert fmt_like(0.1e-12, 1e-12, "F") == "0.1 pF" and fmt_like(0.3e-9, 4.7e-9, "H") == "0.3 nH"
