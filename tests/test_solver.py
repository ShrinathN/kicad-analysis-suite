import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pcb_current_sim import board as B          # noqa: E402
from pcb_current_sim import solver as S         # noqa: E402
from pcb_current_sim import results as RS       # noqa: E402
from pcb_current_sim.thermal import ThermalModel  # noqa: E402
from pcb_current_sim import raster as rz        # noqa: E402

RS_SHEET = 1.72e-8 / 35e-6       # ohm/square, 1 oz


def make_board(items, outline=(-5, -5, 30, 30)):
    x0, y0, x1, y1 = outline
    return B.parse_board(f"""(kicad_pcb (version 20260206) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "core") (thickness 1.51))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start {x0} {y0}) (end {x1} {y1}) (layer "Edge.Cuts"))
  {items}
)""")


def seg(a, b, w, net, layer="F.Cu"):
    return f'(segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "{layer}") (net "{net}"))'


def smd(ref, pos, net, size=(1, 1), layer="F.Cu", shape="rect", angle=0):
    return (f'(footprint "t" (layer "{layer}") (at {pos[0]} {pos[1]} {angle}) (property "Reference" "{ref}")'
            f' (pad "1" smd {shape} (at 0 0 {angle}) (size {size[0]} {size[1]}) (layers "{layer}") (net "{net}")))')


def via(pos, net, size=0.8, drill=0.4):
    return f'(via (at {pos[0]} {pos[1]}) (size {size}) (drill {drill}) (layers "F.Cu" "B.Cu") (net "{net}"))'


def cold(pitch=0.05, **kw):
    kw.setdefault("magnetic", False)
    return S.Settings(pitch=pitch, thermal=False, error_estimate=False, **kw)


def run(board, rows, s):
    return S.simulate(board, rows, s, progress=lambda m: None)


def test_straight_track_resistance():
    b = make_board(seg((0, 0), (20, 0), 1, "N") + smd("A", (0, 0), "N") + smd("C", (20, 0), "N"))
    r = run(b, [S.Row("r", ["A.1"], ["C.1"], 1.0)], cold())
    expect = 19 * RS_SHEET * 1e3                  # pad inner edges are 19 mm apart
    assert r.rows[0]["R_mOhm"] == pytest.approx(expect, rel=0.02)


def test_rotated_footprint_pad_position_and_shape():
    txt = ('(footprint "t" (layer "F.Cu") (at 10 5 90) (property "Reference" "U1")'
           ' (pad "1" smd rect (at 2 0 90) (size 2 1) (layers "F.Cu") (net "N")))')
    p = make_board(txt).pads[0]
    assert p.pos == pytest.approx((10, 3))       # +x local -> up on screen after +90 deg
    assert p.geom.inside(np.array([10.0]), np.array([3.9]))[0]
    assert not p.geom.inside(np.array([10.9]), np.array([3.0]))[0]


def test_arc_track_resistance():
    arc = '(arc (start -10 0) (mid 0 -10) (end 10 0) (width 1) (layer "F.Cu") (net "N"))'
    b = make_board(arc + smd("A", (-10, 0), "N") + smd("C", (10, 0), "N"), outline=(-15, -15, 15, 5))
    assert b.tracks[0].length == pytest.approx(math.pi * 10, rel=1e-3)
    r = run(b, [S.Row("r", ["A.1"], ["C.1"], 1.0)], cold())
    expect = RS_SHEET * math.pi / math.log(10.5 / 9.5) * 1e3 - RS_SHEET * 1e3   # annulus minus pad overlap
    assert r.rows[0]["R_mOhm"] == pytest.approx(expect, rel=0.02)


def test_via_between_layers():
    items = (seg((0, 0), (10, 0), 1, "N") + seg((10, 0), (20, 0), 1, "N", "B.Cu") + via((10, 0), "N")
             + smd("A", (0, 0), "N") + smd("C", (20, 0), "N", layer="B.Cu"))
    b = make_board(items)
    r = run(b, [S.Row("r", ["A.1"], ["C.1"], 1.0)], cold(0.025))
    L = 1.51e-3 + 35e-6
    barrel = 1.72e-8 * L / (math.pi * 25e-6 * (0.4e-3 + 25e-6))
    # current enters the barrel at the hole wall (r = 0.2 mm), so each track is
    # between 9.3 mm (pad edge to wall) and 9.5 mm (pad edge to via centre) long
    lo, hi = 2 * 9.3 * RS_SHEET + barrel, 2 * 9.5 * RS_SHEET + barrel
    R = r.rows[0]["R_mOhm"] * 1e-3
    assert lo < R < hi * 1.03
    assert r.vias[0]["I_A"] == pytest.approx(1.0, rel=1e-3)


def test_parallel_vias_share_current():
    items = (seg((0, 0), (10, 1), 1, "N") + seg((0, 0), (10, -1), 1, "N")
             + seg((10, 1), (20, 0), 1, "N", "B.Cu") + seg((10, -1), (20, 0), 1, "N", "B.Cu")
             + via((10, 1), "N") + via((10, -1), "N")
             + smd("A", (0, 0), "N", (2, 3)) + smd("C", (20, 0), "N", (2, 3), layer="B.Cu"))
    r = run(make_board(items), [S.Row("r", ["A.1"], ["C.1"], 2.0)], cold())
    vs = [v for v in r.vias if v["kind"] == "via"]
    assert len(vs) == 2
    for v in vs:
        assert v["I_A"] == pytest.approx(1.0, rel=0.02)
        assert v["share_pct"] == pytest.approx(50, abs=1)
    assert vs[0]["group"] == vs[1]["group"]


def test_shared_return_carries_sum():
    items = (seg((0, 0), (10, 0), 1, "GND") + seg((10, 0), (20, 5), 1, "GND") + seg((10, 0), (20, -5), 1, "GND")
             + smd("G", (0, 0), "GND") + smd("L1", (20, 5), "GND") + smd("L2", (20, -5), "GND"))
    r = run(make_board(items, (-5, -10, 25, 10)),
            [S.Row("a", ["L1.1"], ["G.1"], 1.0), S.Row("b", ["L2.1"], ["G.1"], 2.0)], cold())
    shared = [g for g in r.segments if g["kind"] == "track" and g["x"] == pytest.approx(5)][0]
    assert shared["I_A"] == pytest.approx(3.0, rel=0.03)
    # row b pushes twice the current through its branch: larger drop than row a
    assert r.rows[1]["dV_mV"] > r.rows[0]["dV_mV"]


def test_unconnected_nets_error():
    b = make_board(seg((0, 0), (10, 0), 1, "A") + smd("P", (0, 0), "A") + smd("Q", (10, 0), "B"))
    with pytest.raises(S.SimError, match="not connected"):
        run(b, [S.Row("bad", ["P.1"], ["Q.1"], 1.0)], cold())
    b = make_board(smd("P", (0, 0), "A") + smd("Q", (0, 5), "B") + smd("C", (10, 0), "A"))
    with pytest.raises(S.SimError, match="one side are on different nets"):
        run(b, [S.Row("bad", ["P.1", "Q.1"], ["C.1"], 1.0)], cold())


def test_open_circuit_error():
    b = make_board(smd("P", (0, 0), "A") + smd("Q", (10, 0), "A"))
    with pytest.raises(S.SimError, match="not connected"):
        run(b, [S.Row("open", ["P.1"], ["Q.1"], 1.0)], cold())


def test_diagonal_track_coverage_weighting():
    """A 45 deg track at a coarse grid matches the same track axis-aligned on a fine grid."""
    L = 20.0
    d = L / math.sqrt(2)
    diag = make_board(seg((0, 0), (d, d), 1, "N") + smd("A", (0, 0), "N", (1, 1), shape="circle")
                      + smd("C", (d, d), "N", (1, 1), shape="circle"))
    straight = make_board(seg((0, 0), (L, 0), 1, "N") + smd("A", (0, 0), "N", (1, 1), shape="circle")
                          + smd("C", (L, 0), "N", (1, 1), shape="circle"))
    rows = [S.Row("r", ["A.1"], ["C.1"], 1.0)]
    r_ref = run(straight, rows, cold(0.02)).rows[0]["R_mOhm"]
    assert run(diag, rows, cold(0.125)).rows[0]["R_mOhm"] == pytest.approx(r_ref, rel=0.02)   # 8 cells/width
    assert run(diag, rows, cold(0.02)).rows[0]["R_mOhm"] == pytest.approx(r_ref, rel=0.005)


def test_thermal_uniform_heating_matches_analytic():
    b = make_board("", outline=(0, 0, 20, 20))
    s = S.Settings(h_conv=12.0, h_auto=False, k_fr4=0.3, T_amb=25.0)
    g = rz.make_grid(b, 0.5, margin=0)
    tm = ThermalModel(b, g, {"F.Cu": 35e-6, "B.Cu": 35e-6}, s)
    P = 0.5
    q = {"F.Cu": np.full(g.shape, P / (g.nx * g.ny)), "B.Cu": np.zeros(g.shape)}
    T = tm.solve(q)
    hc, gv = 12.0, 0.3 / 1.51e-3
    qa = P / (20e-3 * 20e-3)
    T1 = qa * (gv + hc) / (hc * (2 * gv + hc))
    assert float(np.mean(T["F.Cu"])) - 25 == pytest.approx(T1, rel=0.01)


def test_ipc_and_onderdonk():
    A = 1.0 * 0.035                         # 1 mm x 1 oz, mm^2
    assert RS.ipc2221(A, 10, True) == pytest.approx(2.39, rel=0.01)
    I1 = RS.onderdonk_current(A, 1.0, 25)
    assert I1 == pytest.approx(10.1, rel=0.01)
    assert RS.onderdonk_time(A, I1, 25) == pytest.approx(1.0, rel=1e-9)


def test_reference_board_matches_previous_script():
    path = os.path.join(os.path.dirname(__file__), "..", "test.kicad_pcb")
    if not os.path.exists(path):
        pytest.skip("reference board not present")
    b = B.load(path)
    rows = [S.Row("sig", ["J1.2"], ["J2.2"], 10), S.Row("gnd", ["J2.1"], ["J1.1"], 10)]
    r = run(b, rows, S.Settings(thermal=False))
    assert r.rows[0]["R_mOhm"] == pytest.approx(27.3, rel=0.02)
    assert r.rows[1]["R_mOhm"] == pytest.approx(20.2, rel=0.02)


# ---- resistance elements ----------------------------------------------------
from pcb_current_sim import elements as EL     # noqa: E402


def part(ref, value, pads, lib="Resistor_SMD:R_1206_3216Metric", size=(1, 1)):
    """Footprint at the origin with pads [(number, (x, y), net)] in board coordinates."""
    ps = "".join(f' (pad "{n}" smd rect (at {x} {y}) (size {size[0]} {size[1]}) (layers "F.Cu") (net "{net}"))'
                 for n, (x, y), net in pads)
    return (f'(footprint "{lib}" (layer "F.Cu") (at 0 0) (property "Reference" "{ref}")'
            f' (property "Value" "{value}"){ps})')


def series_board(value="0R1"):
    return make_board(seg((0, 0), (9, 0), 1, "N1") + seg((11, 0), (20, 0), 1, "N2")
                      + smd("A", (0, 0), "N1") + smd("C", (20, 0), "N2")
                      + part("R1", value, [("1", (9, 0), "N1"), ("2", (11, 0), "N2")]))


def test_parse_values_and_ratings():
    for t, v in (("10k", 1e4), ("4k7", 4700), ("0R1", 0.1), ("R005", 0.005), ("5m", 0.005),
                 ("1M", 1e6), ("22 Ohm", 22), ("10k 1%", 1e4), ("0", 0), ("2.2k", 2200)):
        assert EL.parse_ohms(t) == pytest.approx(v)
    assert EL.parse_ohms("R_Small") is None
    assert EL.guess_rating("Resistor_SMD:R_2512_6332Metric") == 1.0
    assert EL.guess_rating("Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal") == 0.25


def test_series_resistor_across_nets():
    b = series_board()
    r = run(b, [S.Row("r", ["A.1"], ["C.1"], 1.0)], cold())
    copper = 2 * 8 * RS_SHEET                        # two 8 mm tracks between pad edges
    assert r.rows[0]["R_mOhm"] == pytest.approx((0.1 + copper) * 1e3, rel=0.01)
    e = r.elements[0]
    assert e["name"] == "R1" and e["source"] == "auto"
    assert e["I_A"] == pytest.approx(1.0, rel=1e-6)
    assert e["P_W"] == pytest.approx(0.1, rel=1e-6)
    assert e["P_rated_W"] == 0.25


def test_parallel_resistors_share_by_conductance():
    items = (seg((0, 0), (8, 0), 1, "N1") + seg((12, 0), (20, 0), 1, "N2")
             + seg((8, -2), (8, 2), 1, "N1") + seg((12, -2), (12, 2), 1, "N2")
             + smd("A", (0, 0), "N1") + smd("C", (20, 0), "N2")
             + part("R1", "1", [("1", (8, -2), "N1"), ("2", (12, -2), "N2")])
             + part("R2", "2", [("1", (8, 2), "N1"), ("2", (12, 2), "N2")]))
    r = run(make_board(items), [S.Row("r", ["A.1"], ["C.1"], 3.0)], cold())
    I = {e["name"]: e["I_A"] for e in r.elements}
    assert I["R1"] == pytest.approx(2.0, rel=1e-3)
    assert I["R2"] == pytest.approx(1.0, rel=1e-3)


def test_user_element_on_multipad_part_and_pruning():
    items = (seg((0, 0), (9, 0), 1, "IN") + seg((11, 0), (20, 0), 1, "OUT")
             + smd("A", (0, 0), "IN") + smd("C", (20, 0), "OUT")
             + part("U1", "FET", [("1", (9, 0), "IN"), ("2", (10, 2), "GATE"), ("3", (11, 0), "OUT")],
                    lib="Package_TO_SOT_SMD:SOT-23")
             + part("R5", "10k", [("1", (10, 4), "GATE"), ("2", (12, 4), "PULL")]))   # dead end
    b = make_board(items)
    rows = [S.Row("r", ["A.1"], ["C.1"], 2.0)]
    with pytest.raises(S.SimError, match="not connected"):
        run(b, rows, cold())
    fet = EL.Element("U1", "U1.1", "U1.3", 0.05)
    auto, _ = EL.detect(b)
    nets, used, _ = EL.active(b, ["IN", "OUT"], EL.merge(auto, [fet]))
    assert [e.name for e in used] == ["U1"] and set(nets) == {"IN", "OUT"}
    r = S.simulate(b, rows, cold(), progress=lambda m: None, elements=EL.merge(auto, [fet]))
    assert r.elements[0]["P_W"] == pytest.approx(4 * 0.05, rel=1e-6)


def test_crop_matches_full_board():
    b = make_board(seg((0, 0), (10, 0), 1, "N") + smd("A", (0, 0), "N") + smd("C", (10, 0), "N"),
                   outline=(-5, -30, 60, 30))
    rows = [S.Row("r", ["A.1"], ["C.1"], 1.0)]
    full = run(b, rows, cold(crop=False))
    crop = run(b, rows, cold(crop=True))
    assert crop.grid.nx * crop.grid.ny < full.grid.nx * full.grid.ny / 20
    assert crop.rows[0]["R_mOhm"] == pytest.approx(full.rows[0]["R_mOhm"], rel=1e-6)


def test_element_heats_board():
    b = series_board("1")
    rows = [S.Row("r", ["A.1"], ["C.1"], 0.5)]
    hot = run(b, rows, S.Settings(pitch=0.1, error_estimate=False))
    assert hot.elements[0]["P_W"] == pytest.approx(0.25, rel=0.02)
    t_el = hot.sample("F.Cu", 9, 0)["T"]
    t_far = hot.sample("F.Cu", 0, 0)["T"]
    assert t_el > t_far + 5                       # 0.25 W into the pads: clearly hotter there


# ---- voltage sources and loads ------------------------------------------------
def supply_board():
    """J1 (VIN, GND) -> 20 mm tracks -> J2 (VIN, GND); 1 mm wide, 1 oz, F.Cu."""
    return make_board(seg((0, 0), (20, 0), 1, "VIN") + seg((0, 3), (20, 3), 1, "GND")
                      + smd("J1", (0, 0), "VIN") + smd("J1G", (0, 3), "GND")
                      + smd("J2", (20, 0), "VIN") + smd("J2G", (20, 3), "GND"), (-5, -5, 25, 8))


def test_voltage_source_with_resistive_load():
    b = supply_board()
    rows = [S.Row("vin", ["J1.1"], ["J1G.1"], 12.0, kind="V")]
    load = EL.Element("RL", "J2.1", "J2G.1", 2.4)
    r = S.simulate(b, rows, cold(), progress=lambda m: None, elements=[load])
    path = 2 * 19 * RS_SHEET
    I = 12.0 / (2.4 + path)
    assert r.rows[0]["I_A"] == pytest.approx(I, rel=1e-3)
    assert r.rows[0]["dV_mV"] == pytest.approx(12e3, rel=1e-5)
    assert r.elements[0]["V_mV"] == pytest.approx(I * 2.4e3, rel=1e-3)


def test_load_sees_supply_minus_ir_drop():
    b = supply_board()
    rows = [S.Row("vin", ["J1.1"], ["J1G.1"], 5.0, kind="V"), S.Row("mcu", ["J2.1"], ["J2G.1"], 2.0, kind="load")]
    r = run(b, rows, cold())
    drop = 2.0 * 2 * 19 * RS_SHEET
    assert r.rows[1]["dV_mV"] == pytest.approx((5.0 - drop) * 1e3, rel=1e-4)
    assert r.rows[0]["I_A"] == pytest.approx(2.0, rel=1e-4)


# ---- magnetic field and inductance -----------------------------------------
MU0 = 4e-7 * math.pi


def test_b_field_above_strip():
    w, l, d = 1.0, 40.0, 1.0
    b = make_board(seg((0, 0), (l, 0), w, "N") + smd("A", (0, 0), "N") + smd("C", (l, 0), "N"), (-5, -10, 45, 10))
    r = run(b, [S.Row("r", ["A.1"], ["C.1"], 1.0)], cold(magnetic=True))
    s = r.sample("F.Cu", l / 2, 0)
    expect = MU0 / (math.pi * w * 1e-3) * math.atan(w / (2 * (d + 0.0175))) * 1e6       # infinite strip
    assert s["By"] == pytest.approx(expect, rel=0.01)       # +x current: field points to +y (down) above it
    assert abs(s["Bz"]) < 0.05 * expect
    below = r.sample("B.Cu", l / 2, 0)                       # 1 mm under the bottom surface: 2.5625 mm away
    expect_b = MU0 / (math.pi * w * 1e-3) * math.atan(w / (2 * (1.58 + d - 0.0175))) * 1e6
    assert below["By"] == pytest.approx(-expect_b, rel=0.01)


def test_hairpin_loop_inductance_matches_grover():
    l2, sp_, w = 30.0, 3.0, 1.0
    items = (seg((0, 0), (l2, 0), w, "N") + seg((l2, 0), (l2, sp_), w, "N") + seg((l2, sp_), (0, sp_), w, "N")
             + smd("A", (0, 0), "N") + smd("C", (0, sp_), "N"))
    r = run(make_board(items, (-5, -10, 40, 15)), [S.Row("loop", ["A.1"], ["C.1"], 1.0)], cold(magnetic=True))
    L = r.inductance[0]["L_nH"] * 1e-9

    def Lp(l, d):
        l, d = l * 1e-3, d * 1e-3
        return MU0 * l / (2 * math.pi) * (math.asinh(l / d) - math.sqrt(1 + (d / l) ** 2) + d / l)

    gmd = 0.2235 * (w + 0.035)
    rect = 2 * (Lp(l2, gmd) - Lp(l2, sp_)) + 2 * (Lp(sp_, gmd) - Lp(sp_, l2))
    assert L == pytest.approx(rect, rel=0.08)


def test_grover_parallel_filaments():
    from pcb_current_sim.magnetic import grover_parallel
    l, d = 0.01, 0.001
    expect = MU0 * l / (2 * math.pi) * (math.log(l / d + math.sqrt(1 + (l / d) ** 2)) - math.sqrt(1 + (d / l) ** 2)
                                        + d / l)
    assert grover_parallel(0, l, 0, l, d) == pytest.approx(expect, rel=1e-9)


# ---- pulse / fusing -----------------------------------------------------------
def test_adiabatic_pulse_matches_onderdonk(monkeypatch):
    from pcb_current_sim import thermal as TH
    monkeypatch.setattr(TH, "RHOC_FR4", 1.0)            # no dielectric heat capacity: adiabatic copper
    b = make_board(seg((0, 0), (30, 0), 1, "N") + smd("A", (0, 0), "N", (2, 2)) + smd("C", (30, 0), "N", (2, 2)),
                   (-5, -5, 35, 5))
    I10 = RS.onderdonk_current(0.035, 0.01, 25)
    s = cold(0.1, pulse=True, pulse_ms=20, pulse_steps=200, k_fr4=1e-6)
    r = run(b, [S.Row("p", ["A.1"], ["C.1"], I10)], s)
    assert r.pulse["fused"]["t_ms"] == pytest.approx(10.0, rel=0.05)


def test_pulse_from_steady_state_at_same_current_stays_put():
    b = make_board(seg((0, 0), (20, 0), 0.5, "N") + smd("A", (0, 0), "N") + smd("C", (20, 0), "N"), (-5, -5, 25, 5))
    s = S.Settings(pitch=0.1, error_estimate=False, magnetic=False, pulse=True, pulse_ms=50, pulse_steps=20,
                   pulse_from_steady=True)
    r = run(b, [S.Row("p", ["A.1"], ["C.1"], 2.0)], s)
    assert r.pulse["T_max"][-1] == pytest.approx(r.summary["T_max"], abs=1.0)
    s.pulse_scale = 3.0
    r3 = run(b, [S.Row("p", ["A.1"], ["C.1"], 2.0)], s)
    assert r3.pulse["T_max"][-1] > r.summary["T_max"] + 5


def test_3d_scene_has_layers_vias_and_quantity_menu(tmp_path):
    from pcb_current_sim import export
    items = (seg((0, 0), (10, 0), 1, "N") + seg((10, 0), (20, 0), 1, "N", "B.Cu") + via((10, 0), "N")
             + smd("A", (0, 0), "N") + smd("C", (20, 0), "N", layer="B.Cu"))
    r = run(make_board(items), [S.Row("r", ["A.1"], ["C.1"], 1.0)], cold(0.1, magnetic=True))
    fig = export.figure_3d(r)
    kinds = [t.type for t in fig.data]
    assert kinds.count("surface") >= 2 and "scatter3d" in kinds
    menu = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert menu[0].startswith("|J|") and any(m.startswith("B normal") for m in menu)
    z = {t.name: float(np.nanmax(t.z)) for t in fig.data if t.type == "surface" and t.name in ("F.Cu", "B.Cu")}
    assert z["F.Cu"] > z["B.Cu"]
    assert export.export_html_3d(r, str(tmp_path / "s.html"), js="cdn")


def _read_vti_cell(path, name):
    """Minimal reader for our own .vti output: (origin, spacing, extent, array) of one cell array."""
    import base64
    import xml.etree.ElementTree as ET
    import zlib
    root = ET.parse(path).getroot()
    img = root.find("ImageData")
    arr = [a for a in img.iter("DataArray") if a.get("Name") == name][0]
    head = np.frombuffer(base64.b64decode(arr.text[:24]), np.uint32)          # 1 block: 16 bytes -> 24 chars
    assert head[0] == 1
    raw = zlib.decompress(base64.b64decode(arr.text[24:]))
    assert len(raw) == head[1]
    data = np.frombuffer(raw, {"Float32": np.float32, "Int32": np.int32}[arr.get("type")])
    ext = [int(v) for v in img.get("WholeExtent").split()]
    return ([float(v) for v in img.get("Origin").split()], float(img.get("Spacing").split()[0]), ext,
            data.reshape(ext[3], ext[1], -1).squeeze())


def test_paraview_export_roundtrip(tmp_path):
    import xml.etree.ElementTree as ET
    from pcb_current_sim.vtk_export import export_vtk
    items = (seg((0, 0), (10, 0), 1, "N") + seg((10, 0), (20, 0), 1, "N", "B.Cu") + via((10, 0), "N")
             + smd("A", (0, 0), "N") + smd("C", (20, 0), "N", layer="B.Cu"))
    s = S.Settings(pitch=0.1, error_estimate=False, pulse=True, pulse_ms=5, pulse_steps=10)
    r = run(make_board(items), [S.Row("r", ["A.1"], ["C.1"], 2.0)], s)
    files = export_vtk(r, str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["board.vtm", "pulse.pvd"]
    vtm = ET.parse(tmp_path / "board.vtm").getroot()
    names = {b.get("name") for b in vtm.iter("Block")}
    assert {"copper", "thermal", "magnetic", "geometry"} <= names
    origin, h, ext, J = _read_vti_cell(tmp_path / "data" / "copper_F_Cu.vti", "J_A_per_mm2")
    g = r.grid
    assert (ext[1], ext[3]) == (g.nx, g.ny) and h == pytest.approx(g.h)
    assert origin[2] == pytest.approx(r.summary["layer_z_mm"]["F.Cu"])
    assert origin[1] == pytest.approx(-(g.y0 + g.ny * g.h))          # y flipped to point up
    np.testing.assert_allclose(J[::-1], r.fields["F.Cu"]["J"], equal_nan=True)   # rows reversed (y up)
    _, _, _, Jv = _read_vti_cell(tmp_path / "data" / "copper_F_Cu.vti", "J_vector")
    np.testing.assert_allclose(Jv[::-1][..., 1], -np.asarray(r.fields["F.Cu"]["Jy"]), equal_nan=True)
    pvd = ET.parse(tmp_path / "pulse.pvd").getroot()
    steps = sorted({float(d.get("timestep")) for d in pvd.iter("DataSet")})
    assert len(steps) == len(r.pulse_frames) and steps[-1] == pytest.approx(5e-3)
    for d in pvd.iter("DataSet"):
        assert (tmp_path / d.get("file")).exists()


def test_polygon_scanline_matches_contains_points():
    rng = np.random.default_rng(3)
    for _ in range(10):
        p = B.Polygon(rng.uniform(-1, 1, (int(rng.integers(3, 40)), 2)))
        xs, ys = np.linspace(-1.1, 1.1, 211) + 1e-7, np.linspace(-1.1, 1.1, 173) + 3e-7
        X, Y = np.meshgrid(xs, ys)
        assert np.array_equal(p.lattice(xs, ys), p.inside(X, Y))


def test_auto_h_uniform_heating_matches_balance():
    from pcb_current_sim.thermal import h_surface
    b = make_board("", outline=(0, 0, 40, 40))
    s = S.Settings(h_auto=True, T_amb=25.0)
    g = rz.make_grid(b, 0.5, margin=0)
    tm = ThermalModel(b, g, {"F.Cu": 35e-6, "B.Cu": 35e-6}, s)
    P = 2.0
    q = {L: np.full(g.shape, P / 2 / (g.nx * g.ny)) for L in ("F.Cu", "B.Cu")}
    rise = float(np.mean(tm.solve(q)["F.Cu"])) - 25
    A = 40e-3 * 40e-3
    hh = float(h_surface(s, np.array([rise]), tm.L_char)[0])
    assert P == pytest.approx(2 * A * hh * rise, rel=0.01)          # both faces at h(rise)
    assert 10 < hh < 20


def test_via_barrels_carry_heat_between_layers():
    b0 = make_board("", outline=(0, 0, 20, 20))
    b1 = make_board("".join(via((10 + dx, 10 + dy), "N") for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)),
                    outline=(0, 0, 20, 20))
    s = S.Settings(h_conv=12.0, h_auto=False)
    g = rz.make_grid(b0, 0.25, margin=0)
    dT = []
    for b in (b0, b1):
        tm = ThermalModel(b, g, {"F.Cu": 35e-6, "B.Cu": 35e-6}, s)
        q = {L: np.zeros(g.shape) for L in ("F.Cu", "B.Cu")}
        T = tm.solve(q, [(10, 10, "F.Cu", 0.5)])
        j, i = g.cell(10, 10)
        dT.append(T["F.Cu"][j, i] - T["B.Cu"][j, i])
    assert dT[1] < 0.5 * dT[0]


def test_element_heat_spreads_over_pads():
    b = make_board(seg((0, 0), (10, 0), 1, "N") + smd("A", (0, 0), "N", (2, 2)) + smd("C", (10, 0), "N", (2, 2)))
    s = S.Settings(h_conv=12.0, h_auto=False)
    g = rz.make_grid(b, 0.1)
    tm = ThermalModel(b, g, {"F.Cu": 35e-6, "B.Cu": 35e-6}, s)
    q = {L: np.zeros(g.shape) for L in ("F.Cu", "B.Cu")}
    pad = b.pads_by_id("A.1")[0]
    Tp = tm.solve(q, [(0, 0, "F.Cu", 0.5)])["F.Cu"].max()
    Ta = tm.solve(q, [(0, 0, "F.Cu", 0.5, pad.geom)])["F.Cu"].max()
    assert Ta < Tp
    assert float(np.sum(tm.heat(q, [(0, 0, "F.Cu", 0.5, pad.geom)]))) == pytest.approx(0.5)
