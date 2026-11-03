import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pcb_current_sim.pdn import model as PM      # noqa: E402
from pcb_current_sim.pdn import network as PN    # noqa: E402
from pcb_current_sim.suite import stackup as ST  # noqa: E402
from test_loop import board, fp, via, zone       # noqa: E402

EPS0 = 8.8541878128e-12
A_MM, B_MM, H_MM, ER = 40.0, 30.0, 1.51, 4.5


def plane_pair(decap=False, vrm=False):
    """VCC plane 40 x 30 mm on F.Cu over a GND plane on B.Cu (1.51 mm, εr 4.5); load U1 at a corner;
    optionally C1 (1 µF) at the right edge with a GND via, and a regulator output pad L1."""
    items = (zone("VCC", "F.Cu", 0, 0, A_MM, B_MM) + zone("GND", "B.Cu", -3, -3, A_MM + 4, B_MM + 3)
             + fp("U1", [("1", (1, 1), (1, 1), "VCC")], "ic"))
    if decap:
        items += fp("C1", [("1", (39.5, 15), (1, 1), "VCC"), ("2", (41.5, 15), (1, 1), "GND")], "1u") \
            + via((41.5, 15), "GND")
    if vrm:
        items += fp("L1", [("1", (39.5, 25), (1.5, 1.5), "VCC"), ("2", (42.5, 25), (1.5, 1.5), "SW")], "4u7") \
            + fp("U2", [("1", (42.5, 28), (0.6, 0.6), "SW")], "buck")
    return board(items, h=H_MM, er=ER, outline=(-4, -4, A_MM + 5, B_MM + 4))


def setup_for(b, **kw):
    s = PM.detect(b, "VCC")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_detect_rail_parts():
    b = plane_pair(decap=True, vrm=True)
    assert PM.rail_candidates(b)[0] == "VCC"
    s = PM.detect(b)
    assert s.rail == ["VCC"] and s.returns == ["GND"]
    assert [d.ref for d in s.decaps] == ["C1"] and s.decaps[0].pads == ["C1.1"] and s.decaps[0].ret == ["C1.2"]
    assert s.decaps[0].part["value"] == pytest.approx(1e-6)
    assert s.vrm.ref == "L1" and s.vrm.pads == ["L1.1"]
    assert [ld.name for ld in s.loads] == ["U1"]                  # U2 is not on the rail


def test_bare_plane_capacitance_and_first_resonance():
    b = plane_pair()
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e6, f_max=3e9, n_f=241)
    net = PN.build(b, st, s)
    r = PN.sweep(net, b, st, s)
    z = np.abs(r.Z[:, 0, 0])
    c_plate = EPS0 * ER * (A_MM * B_MM + 0.0) * 1e-6 / (H_MM * 1e-3)
    assert z[0] == pytest.approx(1 / (2 * math.pi * 1e6 * c_plate), rel=0.05)
    f10 = 299792458.0 / (2 * A_MM * 1e-3 * math.sqrt(ER))         # 1.77 GHz, the first cavity mode
    band = (r.f > 1.2e9) & (r.f < 2.5e9)
    f_peak = r.f[band][np.argmax(z[band])]
    assert f_peak == pytest.approx(f10, rel=0.05)


def test_decap_series_resonance_and_vrm():
    b = plane_pair(decap=True, vrm=True)
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e3, f_max=1e9, n_f=181)
    s.decaps[0].part.update(esr=0.01, esl=0.5e-9, dielectric="C0G", bias_loss=0.0)
    s.vrm.r_out, s.vrm.l_out = 5e-3, 20e-9
    net = PN.build(b, st, s)
    l_ret, how = PN.decap_l_ret(b, st, s, s.decaps[0])
    assert 0.2e-9 < l_ret < 2e-9 and "via" in how
    bare = PN.sweep(net, b, st, s, decaps=[], vrm=False)
    cap = PN.sweep(net, b, st, s, vrm=False)
    full = PN.sweep(net, b, st, s)
    zc = np.abs(cap.Z[:, 0, 0])
    assert zc[0] == pytest.approx(1 / (2 * math.pi * 1e3 * 1e-6), rel=0.03)      # the 1 µF dominates
    k = int(np.argmin(zc[cap.f < 1e8]))
    f0_part = 1 / (2 * math.pi * math.sqrt(0.5e-9 * 1e-6))                         # 7.1 MHz without mounting
    # mounting and plane L lower it: corner load -> edge cap 41 mm apart is about 6 nH of plane
    # (spreading over a quarter plane at the corner, a half plane at the edge: μ0 h / θ ln(d / r))
    assert 1.2e6 < cap.f[k] < f0_part
    assert 0.01 <= zc[k] < 0.03                                                    # ESR + copper
    zf = np.abs(full.Z[:, 0, 0])
    from pcb_current_sim.extract import rlc as X
    r_cu = X.dc_resistance(b, "U1.1", "L1.1", st, X.Options())                     # the DC solver
    assert full.Z[0, 0, 0].real == pytest.approx(5e-3 + r_cu, rel=0.05)            # the VRM + copper at low f
    assert np.abs(bare.Z[0, 0, 0]) > 1e3                                           # the bare plane: pF
    # the decap carries the load current near its resonance, the VRM at low f
    assert abs(full.I_vrm[0, 0]) == pytest.approx(1.0, rel=0.02)
    assert abs(cap.I_dec["C1"][k, 0]) == pytest.approx(1.0, rel=0.1)


def test_analysis_target_decap_table_and_what_if():
    from pcb_current_sim.pdn import analysis as PA
    b = plane_pair(decap=True, vrm=True)
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e3, f_max=3e8, n_f=91)
    s.target.v, s.target.ripple, s.target.i_step, s.target.t_rise = 3.3, 5.0, 1.0, 10e-9   # 165 mΩ to 35 MHz
    r = PA.run(b, st, s)
    assert r.z_target == pytest.approx(0.165) and r.f_target == pytest.approx(35e6)
    c = r.checks[0]
    assert c["load"] == "U1" and c["ok"] == (c["ratio"] <= 1) and c["worst_f"] <= 35e6
    row = r.decaps[0]
    assert row["ref"] == "C1" and 1e6 < row["f_srf"] < 1e7 and row["dist_mm"] == pytest.approx(
        math.hypot(39.5 - 1, 15 - 1), abs=0.01)
    assert row["band"][0] < row["f_srf"] < row["band"][1]
    w = PA.what_if(b, st, s, r, n_f=21)
    assert w["C1"]["rise"] > 2                  # the only decap: removing it hurts
    v = PA.node_voltages(b, st, s, r, 1e6)
    assert v.shape == (r.nodes,) and v.max() == pytest.approx(np.abs(r.Z[np.argmin(abs(r.f - 1e6)), 0, 0]),
                                                              rel=0.3)


def test_touchstone_export_round_trip_and_terminations(tmp_path):
    """The complete-rail file reads back as the analysed Z; the bare-board file with the decap and the
    regulator connected at their ports gives the same Z at the load."""
    from pcb_current_sim.matching import touchstone as TS
    from pcb_current_sim.pdn import analysis as PA
    b = plane_pair(decap=True, vrm=True)
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e3, f_max=3e8, n_f=41)
    r = PA.run(b, st, s)
    p1, names = PA.touchstone(b, st, s, r, str(tmp_path / "rail.s2p"), "loads", z0=50)
    assert p1.endswith(".s1p") and names == ["U1"]
    t = TS.load(p1)
    z1 = 50 * (1 + t.S[:, 0, 0]) / (1 - t.S[:, 0, 0])
    assert np.allclose(t.f, r.f) and np.allclose(z1, r.Z[:, 0, 0], rtol=1e-6)
    p3, names = PA.touchstone(b, st, s, r, str(tmp_path / "bare.snp"), "board", z0=1)
    assert p3.endswith(".s3p") and [n.split()[0] for n in names] == ["U1", "C1", "regulator"]
    t = TS.load(p3)
    eye = np.eye(3)
    Zb = np.linalg.solve((eye - t.S).transpose(0, 2, 1), (eye + t.S).transpose(0, 2, 1)).transpose(0, 2, 1)
    d = s.decaps[0]
    zt = np.stack([PN.decap_z(d, t.f, s.target.v, s.T, PN.decap_l_ret(b, st, s, d)[0]),
                   1 / PN.vrm_y(s.vrm, t.f)], axis=1)                   # (F, 2) terminations
    q = [1, 2]
    zl = Zb[:, 0, 0] - np.einsum("fi,fij,fj->f", Zb[:, 0, q],
                                 np.linalg.inv(Zb[:, 1:, 1:] + zt[:, :, None] * np.eye(2)), Zb[:, q, 0])
    assert np.allclose(zl, r.Z[:, 0, 0], rtol=1e-4)


def test_fitted_regulator_loop_in_the_pdn():
    from dataclasses import asdict
    from pcb_current_sim.extract import rlc as X
    from pcb_current_sim.pdn import regulator as RG
    b = plane_pair(decap=True, vrm=True)
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e3, f_max=1e8, n_f=61)
    lp = RG.Loop("cot", K=50.0, f_z=0.0, f_0=0.0, f_p=1e6)                 # a 20 mΩ source, 1 MHz pole
    s.vrm.kind, s.vrm.model = "loop", RG.RegulatorModel(kind="cot", up=asdict(lp)).to_dict()
    net = PN.build(b, st, s)
    r = PN.sweep(net, b, st, s, decaps=[])
    r_cu = X.dc_resistance(b, "U1.1", "L1.1", st, X.Options())
    assert r.Z[0, 0, 0].real == pytest.approx(0.02 + r_cu, rel=0.05)
    k = np.argmin(abs(r.f - 1e6))                                          # above its pole the loop gives way
    assert abs(r.Z[k, 0, 0]) > abs(r.Z[0, 0, 0]) * 1.2


def test_load_steps_on_the_board():
    from dataclasses import asdict
    from pcb_current_sim.pdn import analysis as PA
    from pcb_current_sim.pdn import regulator as RG
    b = plane_pair(decap=True, vrm=True)
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e3, f_max=3e8, n_f=41)
    up = RG.Loop("current", K=6.0, f_z=8e3, f_0=0.0, f_p=50e3)
    dn = RG.Loop("current", K=3.0, f_z=8e3, f_0=0.0, f_p=50e3)        # slower on the way down
    s.vrm.kind, s.vrm.model = "loop", RG.RegulatorModel(up=asdict(up), down=asdict(dn)).to_dict()
    s.decaps[0].part.update(value=47e-6, esr=3e-3, esl=0.5e-9, dielectric="C0G", bias_loss=0.0)
    ld = s.loads[0]
    ld.i_step, ld.delay, ld.t_rise, ld.width, ld.t_fall = 1.0, 50e-6, 1e-6, 300e-6, 1e-6
    s.t_stop = 800e-6
    r = PA.run(b, st, s)
    w, c = r.waves, r.wave_checks[0]
    assert w is not None and w.fit_err < 0.05
    # the lumped test circuit (loop ∥ 47 µF) predicts the dip; the board adds a little copper
    t = np.linspace(0, 800e-6, 8001)
    i = np.clip((t - 50e-6) / 1e-6, 0, 1)
    v_ref = RG.response(up, t, i, 47e-6, 3e-3)
    assert c["dip"] == pytest.approx(v_ref.min(), rel=0.1)
    assert c["peak"] > -c["dip"]                                        # the falling edge's slower loop
    assert abs(w.v["U1"][-1]) < 0.1 * abs(c["dip"])                     # settled again


def four_layer(items, outline=(-4, -4, 45, 34)):
    """F.Cu / In1.Cu (0.2 mm) / In2.Cu (1.1 mm) / B.Cu (0.2 mm), εr 4.5."""
    from pcb_current_sim import board as B
    x0, y0, x1, y1 = outline
    lay = lambda n: f'(layer "{n}" (type "copper") (thickness 0.035))'
    die = lambda k, t: f'(layer "dielectric {k}" (type "core") (thickness {t}) (epsilon_r 4.5) (loss_tangent 0.02))'
    return B.parse_board(f"""(kicad_pcb (version 20260206) (generator "test")
  (general (thickness 1.64))
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup {lay("F.Cu")} {die(1, 0.2)} {lay("In1.Cu")} {die(2, 1.1)} {lay("In2.Cu")} {die(3, 0.2)}
    {lay("B.Cu")}))
  (gr_rect (start {x0} {y0}) (end {x1} {y1}) (layer "Edge.Cuts"))
  {items}
)""")


def test_layout_checks_find_a_slot_in_the_return_plane():
    """VCC on F.Cu over GND on In1 (0.2 mm) with a 1 mm slot across the middle, GND also on B.Cu:
    over the slot the nearest return is B.Cu, ~7x farther. U1 and its decap are on either side, so the
    rail's current crosses the slot."""
    from pcb_current_sim.pdn import analysis as PA
    from pcb_current_sim.pdn import layout as LA
    items = (zone("VCC", "F.Cu", 0, 0, A_MM, B_MM)
             + zone("GND", "In1.Cu", -3, -3, 19.5, B_MM + 3) + zone("GND", "In1.Cu", 20.5, -3, A_MM + 3, B_MM + 3)
             + zone("GND", "B.Cu", -3, -3, A_MM + 3, B_MM + 3)
             + fp("U1", [("1", (1, 15), (1, 1), "VCC")], "ic")
             + fp("C1", [("1", (38, 15), (1, 1), "VCC"), ("2", (38, 17), (1, 1), "GND")], "100n")
             + via((38, 17), "GND"))
    b = four_layer(items)
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e5, f_max=1e8, n_f=11, transient=False)
    r = PA.run(b, st, s)
    _none, voids = LA.regions(r.net)
    assert len(voids) == 1
    v = [i for i in r.layout if i.kind == "return void"]
    assert len(v) == 1 and "slot" in v[0].text and 19 < v[0].at[0] < 21
    assert not [i for i in r.layout if i.kind in ("no return", "mounting")]


def test_layout_checks_no_return_far_decap_mounting():
    """GND only under x < 25: the right part of VCC has no return; U1 at the far left has its only
    decaps 35 mm away, and their return pads have no plane under them."""
    from pcb_current_sim.pdn import analysis as PA
    items = (zone("VCC", "F.Cu", 0, 0, A_MM, B_MM) + zone("GND", "B.Cu", -3, -3, 25, B_MM + 3)
             + fp("U1", [("1", (1, 15), (1, 1), "VCC")], "ic")
             + fp("C1", [("1", (36, 15), (1, 1), "VCC"), ("2", (38, 15), (1, 1), "GND")], "10u")
             + fp("C2", [("1", (36, 20), (1, 1), "VCC"), ("2", (38, 20), (1, 1), "GND")], "10n"))
    b = board(items, h=H_MM, er=ER, outline=(-4, -4, A_MM + 5, B_MM + 4))
    st = ST.from_board(b)
    s = setup_for(b, f_min=1e4, f_max=3e8, n_f=41, transient=False)
    r = PA.run(b, st, s)
    kinds = {}
    for i in r.layout:
        kinds.setdefault(i.kind, []).append(i)
    assert len(kinds["no return"]) == 1 and kinds["no return"][0].at[0] > 25 and kinds["no return"][0].level == "warn"
    assert kinds["no decap near"][0].refs == ["U1"]
    assert {i.refs[0] for i in kinds["mounting"]} == {"C1", "C2"}      # return pads over no plane, no via
    assert "return void" not in kinds


def test_saved_regulator_fits_by_part_number(tmp_path):
    """A fit saved under the IC's part number is found on a board where that IC sits on the regulator
    output part's switch node, and used when the rail has no fit yet."""
    from pcb_current_sim.pdn import regstore as RS
    b = plane_pair(decap=True, vrm=True)                     # U2 "buck" on SW, the inductor L1's other net
    s = setup_for(b)
    path = RS.path_for(str(tmp_path / "library.json"))
    model = dict(part="", kind="current", c_test=22e-6, up=dict(kind="current", K=5.0, f_z=9e3, f_p=60e3))
    RS.save(path, "BUCK", model)
    RS.save(path, "other", dict(model, c_test=1e-6))
    models = RS.load(path)
    assert set(models) == {"BUCK", "other"} and models["BUCK"]["part"] == "BUCK"
    assert RS.on_board(b, s, models) == [("BUCK", "U2")]
    assert RS.apply(b, s, models) == "BUCK" and s.vrm.kind == "loop" and s.vrm.model["c_test"] == 22e-6
    assert RS.apply(b, s, models) == ""                     # a rail with a fit keeps it
    RS.save(path, "buck", dict(model, c_test=47e-6))         # same part, other case: replaced
    assert set(RS.load(path)) == {"buck", "other"} and RS.load(path)["buck"]["c_test"] == 47e-6
    RS.remove(path, "other")
    assert set(RS.load(path)) == {"buck"}
