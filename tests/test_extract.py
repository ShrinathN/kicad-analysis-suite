import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcb_current_sim import board as B                 # noqa: E402
from pcb_current_sim.extract import rlc as X            # noqa: E402
from pcb_current_sim.extract import spice as SP         # noqa: E402
from pcb_current_sim.extract import xsection as XS      # noqa: E402
from pcb_current_sim.matching import microstrip as MS   # noqa: E402
from pcb_current_sim.suite import stackup as ST         # noqa: E402

ER, H = 4.5, 1.51e-3
C_LIGHT = 299792458.0


def make_board(items, outline=(-5, -10, 65, 10)):
    x0, y0, x1, y1 = outline
    return B.parse_board(f"""(kicad_pcb (version 20260206) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "core") (thickness 1.51) (epsilon_r {ER}) (loss_tangent 0.02))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start {x0} {y0}) (end {x1} {y1}) (layer "Edge.Cuts"))
  {items}
)""")


def seg(a, b, w, net, layer="F.Cu"):
    return f'(segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "{layer}") (net "{net}"))'


def smd(ref, pos, net, size=(1, 1), layer="F.Cu"):
    return (f'(footprint "t" (layer "{layer}") (at {pos[0]} {pos[1]}) (property "Reference" "{ref}")'
            f' (pad "1" smd rect (at 0 0) (size {size[0]} {size[1]}) (layers "{layer}") (net "{net}")))')


def via(pos, net, size=0.8, drill=0.4):
    return f'(via (at {pos[0]} {pos[1]}) (size {size}) (drill {drill}) (layers "F.Cu" "B.Cu") (net "{net}"))'


def zone(net, layer, x0, y0, x1, y1):
    pts = " ".join(f"(xy {x} {y})" for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return (f'(zone (net "{net}") (layer "{layer}") (filled_polygon (layer "{layer}") (pts {pts})))')


def extract(b, ports, ret=("GND",), **kw):
    kw.setdefault("dc_solver", False)
    return X.Extractor(b, ST.from_board(b), list(ret), X.Options(**kw)).run(ports)


def test_microstrip_cross_section_matches_hammerstad():
    w, t = 2.9e-3, 1e-6
    sec = XS.Section([XS.Rect(-w / 2, w / 2, -t, 0, True), XS.Rect(-1, 1, H, H + 35e-6)], [(0, H, ER)],
                     half_width=15e-3, air=15e-3)
    p = XS.solve(sec)
    z0, ee = MS.z0_eeff(w, H, ER)
    assert p.z0 == pytest.approx(z0, rel=0.015)
    assert p.eeff == pytest.approx(ee, rel=0.015)


def test_microstrip_over_plane():
    L = 50.0
    b = make_board(seg((0, 0), (L, 0), 2.9, "SIG") + smd("J1", (0, 0), "SIG") + smd("J2", (L, 0), "SIG")
                   + zone("GND", "B.Cu", -5, -10, 65, 10))
    e = extract(b, ["J1.1", "J2.1"], window=8.0, sample=5.0, f_max=1e9)
    z0, ee = MS.z0_eeff(2.9e-3, H, ER)
    pr = e.pairs[0]
    assert pr["length_mm"] == pytest.approx(L)
    # 35 µm copper lowers Z0 slightly against the zero-thickness formula
    assert pr["Z0_avg"] == pytest.approx(z0, rel=0.04)
    assert pr["delay_ps"] == pytest.approx(L * 1e-3 * math.sqrt(ee) / C_LIGHT * 1e12, rel=0.02)
    assert e.segments[0]["reference"] == "B.Cu"
    assert pr["R_mOhm"] == pytest.approx(1.72e-8 * L / (2.9e-3 * 35e-6), rel=0.01)   # ρl/(wt): Ω/m·mm = mΩ
    # λ/10 at 1 GHz on this line is ~16 mm: 50 mm needs 4 sections
    assert e.summary["sections"] == 4
    net = SP.write(e)
    assert ".subckt TRACE_SIG_J1_1_J2_1 P1 P2 REF" in net and net.rstrip().endswith(".ends TRACE_SIG_J1_1_J2_1")


def test_dc_solver_calibrates_r():
    b = make_board(seg((0, 0), (20, 0), 0.5, "SIG") + smd("J1", (0, 0), "SIG") + smd("J2", (20, 0), "SIG")
                   + zone("GND", "B.Cu", -5, -10, 65, 10))
    e = extract(b, ["J1.1", "J2.1"], dc_solver=True, sample=2.0)
    assert e.pairs[0]["R_mOhm"] == pytest.approx(e.summary["R_dc_field_mOhm"], rel=1e-6)
    # the field solution sees the pads (wider than the track): a little below ρl/(wt)
    assert 0.85 < e.summary["R_scale"] < 1.0


def test_via_path_and_stub():
    items = (seg((0, 0), (10, 0), 0.3, "SIG") + via((10, 0), "SIG") + seg((10, 0), (30, 0), 0.3, "SIG", "B.Cu")
             + seg((20, 0), (20, 5), 0.3, "SIG", "B.Cu")                 # T-junction stub to TP1
             + smd("J1", (0, 0), "SIG") + smd("J2", (30, 0), "SIG", layer="B.Cu")
             + smd("TP1", (20, 5), "SIG", layer="B.Cu") + via((12, 2), "GND")
             + zone("GND", "F.Cu", 12, -10, 65, 10) + zone("GND", "B.Cu", -5, -10, 8, 10))
    b = make_board(items)
    full = extract(b, ["J1.1", "J2.1"], sample=1.0)
    bare = extract(b, ["J1.1", "J2.1"], sample=1.0, stubs=False)
    assert len(full.vias) == 1 and "return via at" in full.vias[0]["reference"]
    assert full.vias[0]["L_nH"] > 0
    assert full.pairs[0]["length_mm"] == pytest.approx(30.0)
    assert full.summary["C_total_pF"] > bare.summary["C_total_pF"] + 0.1         # the 5 mm stub and its pad
    assert any(p["pad"] == "TP1.1" for p in full.pads)


def test_coplanar_and_missing_return():
    items = (seg((0, 0), (20, 0), 0.5, "SIG") + smd("J1", (0, 0), "SIG") + smd("J2", (20, 0), "SIG")
             + zone("GND", "F.Cu", -5, 0.45, 25, 5) + zone("GND", "F.Cu", -5, -5, 25, -0.45))
    b = make_board(items)
    e = extract(b, ["J1.1", "J2.1"], sample=2.0)
    assert e.segments[0]["reference"].startswith("coplanar")
    assert 40 < e.segments[0]["Z0"] < 110
    e2 = extract(b, ["J1.1", "J2.1"], ret=(), sample=2.0)
    assert e2.segments[0]["reference"] == "none in window" and e2.pairs[0]["L_nH"] > 0
    assert any("No return net" in w for w in e2.warnings)


def test_errors_and_spice_format():
    b = make_board(seg((0, 0), (10, 0), 0.3, "SIG") + smd("J1", (0, 0), "SIG") + smd("J2", (20, 0), "SIG")
                   + smd("J3", (5, 5), "OTHER"))
    with pytest.raises(ValueError, match="No copper path"):
        extract(b, ["J1.1", "J2.1"])
    with pytest.raises(ValueError, match="no element joins"):
        extract(b, ["J1.1", "J3.1"])
    assert SP.fmt(1.5e6) == "1.5Meg" and SP.fmt(2.2e-12) == "2.2p" and SP.fmt(0.047) == "47m"


def test_inner_layer_stripline():
    b = B.parse_board("""(kicad_pcb (version 20260206) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "prepreg") (thickness 0.2) (epsilon_r 4.2))
    (layer "In1.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 2" (type "core") (thickness 0.3) (epsilon_r 4.2))
    (layer "In2.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 3" (type "prepreg") (thickness 0.9) (epsilon_r 4.2))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start -5 -10) (end 35 10) (layer "Edge.Cuts"))
  """ + seg((0, 0), (30, 0), 0.15, "SIG", "In1.Cu") + via((0, 0), "SIG") + via((30, 0), "SIG")
        + smd("J1", (0, 0), "SIG") + smd("J2", (30, 0), "SIG")
        + zone("GND", "F.Cu", 2, -10, 28, 10) + zone("GND", "In2.Cu", -5, -10, 35, 10) + ")")
    e = X.Extractor(b, ST.from_board(b), ["GND"], X.Options(sample=5.0, dc_solver=False)).run(["J1.1", "J2.1"])
    s = [r for r in e.segments if r["length_mm"] > 20][0]
    assert s["reference"] == "F.Cu, In2.Cu"
    assert s["eeff"] == pytest.approx(4.2, rel=0.01)           # fully embedded: εeff = εr
    # offset stripline (IPC-2141A): Z0 = 80/sqrt(εr) ln(1.9 (2h+t)/(0.8w+t)) (1 - h/(4 (h+c+t))), h = 0.2 mm
    h, c, w, t = 0.2, 0.3, 0.15, 0.035
    ipc = 80 / math.sqrt(4.2) * math.log(1.9 * (2 * h + t) / (0.8 * w + t)) * (1 - h / (4 * (h + c + t)))
    assert s["Z0"] == pytest.approx(ipc, rel=0.12), s["Z0"]
    assert len(e.vias) == 6                                     # 3 barrel pieces per via (F to B)


def _wheeler(W, b, t, er):
    """Symmetric stripline with a thick strip (Wheeler; ~1 % accurate)."""
    x = t / b
    m = 2 / (1 + 2 / 3 * x / (1 - x))
    dW = (b - t) * x / (math.pi * (1 - x)) * (1 - 0.5 * math.log((x / (2 - x)) ** 2
                                                                 + (0.0796 * x / (W / b + 1.1 * x)) ** m))
    k = (b - t) / (W + dW)
    return 30 / math.sqrt(er) * math.log(1 + 4 / math.pi * k * (8 / math.pi * k
                                                                  + math.sqrt((8 / math.pi * k) ** 2 + 6.27)))


@pytest.mark.parametrize("w", [0.15, 0.3])
def test_symmetric_stripline_matches_wheeler(w):
    b = B.parse_board("""(kicad_pcb (version 20260206) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "prepreg") (thickness 0.25) (epsilon_r 4.2))
    (layer "In1.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 2" (type "core") (thickness 0.25) (epsilon_r 4.2))
    (layer "In2.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 3" (type "prepreg") (thickness 0.9) (epsilon_r 4.2))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start -5 -10) (end 35 10) (layer "Edge.Cuts"))
  """ + seg((0, 0), (30, 0), w, "SIG", "In1.Cu") + via((0, 0), "SIG") + via((30, 0), "SIG")
        + smd("J1", (0, 0), "SIG") + smd("J2", (30, 0), "SIG")
        + zone("GND", "F.Cu", -5, -10, 35, 10) + zone("GND", "In2.Cu", -5, -10, 35, 10) + ")")
    e = X.Extractor(b, ST.from_board(b), ["GND"], X.Options(sample=10.0, dc_solver=False)).run(["J1.1", "J2.1"])
    s = max(e.segments, key=lambda r: r["length_mm"])
    assert s["Z0"] == pytest.approx(_wheeler(w, 0.535, 0.035, 4.2), rel=0.025)


@pytest.mark.skipif(not __import__("shutil").which("ngspice"), reason="ngspice not installed")
def test_subcircuit_runs_in_ngspice(tmp_path):
    """A matched 50 mm microstrip: the step at the far end arrives after the line delay."""
    import subprocess
    w = 2.75          # ~50 Ω on this stackup (35 µm copper)
    b = make_board(seg((0, 0), (50, 0), w, "SIG") + smd("J1", (0, 0), "SIG", size=(w, w))
                   + smd("J2", (50, 0), "SIG", size=(w, w)) + zone("GND", "B.Cu", -5, -10, 65, 10))
    e = extract(b, ["J1.1", "J2.1"], f_max=10e9, sample=5.0, pad_c=False)
    z0, delay = e.pairs[0]["Z0_avg"], e.pairs[0]["delay_ps"] * 1e-12
    lib = SP.write(e, name="LINE")
    assert lib.isascii()
    deck = (lib + f"VS in 0 PULSE(0 1 0 20p 20p 10n 20n)\nRS in p1 {z0:.3f}\nX1 p1 p2 0 LINE\nRL p2 0 {z0:.3f}\n"
            ".tran 1p 2n\n.measure tran t50 WHEN v(p2)=0.25 RISE=1\n.end\n")
    (tmp_path / "deck.cir").write_text(deck)
    out = subprocess.run(["ngspice", "-b", str(tmp_path / "deck.cir")], capture_output=True, text=True, timeout=60)
    line = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("t50")]
    assert line, out.stdout + out.stderr
    t50 = float(line[0].split("=")[1].split()[0])
    assert t50 - 10e-12 == pytest.approx(delay, rel=0.1)       # 50 % of the 0.5 V step, after the 20 ps edge's half


# ---- copper pours and user elements ------------------------------------------------
from pcb_current_sim import elements as EL       # noqa: E402

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12


def test_zone_between_pads_over_plane():
    """A 20 x 4 mm pour joined to 1 x 4 mm pads at both ends, GND plane 1.51 mm below."""
    b = make_board(zone("SIG", "F.Cu", 0, -2, 20, 2) + smd("J1", (0.5, 0), "SIG", size=(1, 4))
                   + smd("J2", (19.5, 0), "SIG", size=(1, 4)) + zone("GND", "B.Cu", -5, -10, 65, 10))
    e = extract(b, ["J1.1", "J2.1"], dc_solver=False, pad_c=False)
    z = e.zones[0]
    assert z["terminals"] == 2
    assert z["C_pF"] == pytest.approx(EPS0 * ER * 80e-6 / H * 1e12, rel=0.03)        # ε A / h
    assert e.pairs[0]["L_nH"] == pytest.approx(MU0 * H * 18 / 4 * 1e9, rel=0.02)    # μ0 h l / W between pad edges
    r_true = X.dc_resistance(b, "J1.1", "J2.1", ST.from_board(b), X.Options())
    assert e.pairs[0]["R_mOhm"] == pytest.approx(r_true * 1e3, rel=0.03)            # the mesh matches the DC solver
    assert "zone SIG F.Cu" in SP.write(e)


def test_zone_with_track_and_via_terminals():
    """A track feeds a pour; a via in the pour drops to a track on B.Cu: one path through the zone."""
    items = (seg((0, 0), (5, 0), 0.3, "SIG") + zone("SIG", "F.Cu", 5, -3, 15, 3) + via((14, 0), "SIG")
             + seg((14, 0), (25, 0), 0.3, "SIG", "B.Cu") + smd("J1", (0, 0), "SIG")
             + smd("J2", (25, 0), "SIG", layer="B.Cu") + zone("GND", "B.Cu", -5, 2, 65, 10))
    b = make_board(items)
    e = extract(b, ["J1.1", "J2.1"], sample=1.0)
    assert len(e.zones) == 1 and e.zones[0]["terminals"] == 2
    assert len(e.vias) == 1
    assert math.isfinite(e.pairs[0]["L_nH"]) and e.pairs[0]["R_mOhm"] > 0


def test_elements_join_nets_and_reach_ref():
    items = (seg((0, 0), (10, 0), 0.5, "A") + seg((12, 0), (22, 0), 0.5, "B")
             + smd("J1", (0, 0), "A") + smd("R1", (10, 0), "A") + smd("R1B", (12, 0), "B")
             + smd("J2", (22, 0), "B") + smd("C1", (22, 2), "GND") + zone("GND", "B.Cu", -5, -10, 65, 10))
    b = make_board(items)
    els = [EL.Element("R1", "R1.1", "R1B.1", 10.0), EL.Element("C1", "J2.1", "C1.1", 0.0, model="C", C=100e-9),
           EL.Element("L1", "R1.1", "R1B.1", 0.0, model="L", L=1e-6, enabled=False)]
    st = ST.from_board(b)
    e = X.Extractor(b, st, ["GND"], X.Options(sample=2.0), elements=els).run(["J1.1", "J2.1"])
    r_tracks = 1.72e-8 * 20e-3 / (0.5e-3 * 35e-6)
    assert e.pairs[0]["R_mOhm"] == pytest.approx((10 + r_tracks) * 1e3, rel=0.01)
    caps = [el for el in e.elements if el[0] == "C_C1"]
    assert caps and "REF" in caps[0][2:4] and caps[0][4] == pytest.approx(100e-9)
    assert any("DC calibration of R is skipped" in w for w in e.warnings)
    els[2].enabled, els[0].enabled = True, False                     # now an inductor instead of the resistor
    e2 = X.Extractor(b, st, ["GND"], X.Options(sample=2.0), elements=els).run(["J1.1", "J2.1"])
    assert e2.pairs[0]["L_nH"] == pytest.approx(1000 + e.pairs[0]["L_nH"], rel=0.01)


@pytest.mark.skipif(not __import__("shutil").which("ngspice"), reason="ngspice not installed")
def test_diode_element_in_ngspice(tmp_path):
    import subprocess
    items = (seg((0, 0), (10, 0), 1, "A") + seg((12, 0), (22, 0), 1, "B") + smd("J1", (0, 0), "A")
             + smd("D1", (10, 0), "A") + smd("D1K", (12, 0), "B") + smd("J2", (22, 0), "B")
             + zone("GND", "B.Cu", -5, -10, 65, 10))
    b = make_board(items)
    d = EL.Element("D1", "D1.1", "D1K.1", 0.05, model="V", Vf=0.7)
    e = X.Extractor(b, ST.from_board(b), ["GND"], X.Options(sample=5.0), elements=[d]).run(["J1.1", "J2.1"])
    lib = SP.write(e, name="T")
    assert "DVF" in lib and lib.isascii()
    deck = lib + "VS in 0 5\nRS in p1 10\nX1 p1 p2 0 T\nRL p2 0 33\n.op\n.print op v(p1) v(p2)\n.control\nop\n" \
        "print v(p1)-v(p2) v(p2)\n.endc\n.end\n"
    (tmp_path / "d.cir").write_text(deck)
    out = subprocess.run(["ngspice", "-b", str(tmp_path / "d.cir")], capture_output=True, text=True, timeout=60)
    vals = [ln for ln in out.stdout.splitlines() if "v(p1)-v(p2)" in ln.replace(" ", "") or ln.strip().startswith("v(p1)-v(p2)")]
    drop = float(vals[0].split("=")[1]) if vals else None
    I = 4.3 / 43
    assert drop == pytest.approx(0.7 + 0.05 * I + 0.015, abs=0.02), out.stdout[-800:]
