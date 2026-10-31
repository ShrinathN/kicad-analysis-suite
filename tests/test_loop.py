"""Switching Loop: the PEEC extraction against closed forms and the other solvers."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcb_current_sim import board as B                    # noqa: E402
from pcb_current_sim.extract import rlc as X              # noqa: E402
from pcb_current_sim.extract import xsection as XS        # noqa: E402
from pcb_current_sim.loop import peec as P                # noqa: E402
from pcb_current_sim.suite import stackup as ST           # noqa: E402

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12


def board(items, h=1.51, er=4.5, outline=(-12, -12, 40, 12)):
    return B.parse_board(board_text(items, h, er, outline))


def board_text(items, h=1.51, er=4.5, outline=(-12, -12, 40, 12)):
    x0, y0, x1, y1 = outline
    return (f"""(kicad_pcb (version 20260206) (generator "test")
  (general (thickness {h + 0.07}))
  (layers (0 "F.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "core") (thickness {h}) (epsilon_r {er}) (loss_tangent 0.02))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start {x0} {y0}) (end {x1} {y1}) (layer "Edge.Cuts"))
  {items}
)""")


def seg(a, b, w, net, layer="F.Cu"):
    return f'(segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "{layer}") (net "{net}"))'


def smd(ref, pos, net, size=(1, 1), layer="F.Cu"):
    return (f'(footprint "t" (layer "{layer}") (at {pos[0]} {pos[1]}) (property "Reference" "{ref}")'
            f' (pad "1" smd rect (at 0 0) (size {size[0]} {size[1]}) (layers "{layer}") (net "{net}")))')


def via(pos, net, size=0.6, drill=0.3):
    return f'(via (at {pos[0]} {pos[1]}) (size {size}) (drill {drill}) (layers "F.Cu" "B.Cu") (net "{net}"))'


def zone(net, layer, x0, y0, x1, y1):
    pts = " ".join(f"(xy {x} {y})" for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return f'(zone (net "{net}") (layer "{layer}") (filled_polygon (layer "{layer}") (pts {pts})))'


def terms(b, *refs):
    return {r: [k for k, p in enumerate(b.pads) if p.ref == r] for r in refs}


def port_L(b, nets, window, pitch, t, r, hf=True, f=1.0):
    m = P.build(b, ST.from_board(b), nets, window, pitch, terms(b, t, r))
    n = P.Network(m, [(t, r)])
    if hf:
        return n.solve(0, hf=True)[0][0, 0], m, n
    Z = n.solve(f)[0][0, 0]
    return Z, m, n


# ---- kernels -------------------------------------------------------------------------
def brute_rect(xa, ya, xb, yb, z, n=40):
    g, w = np.polynomial.legendre.leggauss(n)

    def pts(a):
        return (a[0] + a[1]) / 2 + (a[1] - a[0]) / 2 * g, (a[1] - a[0]) / 2 * w
    X1, W1 = pts(xa)
    Y1, V1 = pts(ya)
    X2, W2 = pts(xb)
    Y2, V2 = pts(yb)
    r = np.sqrt((X1[:, None, None, None] - X2[None, None, :, None]) ** 2
                + (Y1[None, :, None, None] - Y2[None, None, None, :]) ** 2 + z * z)
    return float(np.einsum("i,j,k,l,ijkl->", W1, V1, W2, V2, 1 / r))


@pytest.mark.parametrize("case", [((0, 1), (0, 1), (0, 1), (0, 1), 0.5), ((0, 1), (0, 0.3), (2, 3), (0.5, 0.8), 0.2),
                                  ((0, 1), (0, 1), (0.3, 1.4), (-0.2, 0.6), 0.1), ((0, 2), (0, 0.5), (2, 4), (0, 0.5), 0.3)])
def test_rect_integral_matches_quadrature(case):
    (xa, ya, xb, yb, z) = case
    exact = P.rect_integral(xa[0], xa[1], ya[0], ya[1], xb[0], xb[1], yb[0], yb[1], z)
    assert exact == pytest.approx(brute_rect(xa, ya, xb, yb, z), rel=2e-4)
    assert P.rect_integral(0, 1, 0, 1, 0, 1, 0, 1, 0.0) == pytest.approx(2.97320, abs=1e-4)   # square panel


def test_strip_self_matches_rosa():
    for l, w in ((20e-3, 1e-3), (10e-3, 0.1e-3)):
        rosa = MU0 * l / (2 * math.pi) * (math.log(2 * l / w) + 0.5 + 0.2235 * w / l)
        assert P.strip_self(np.array(l), np.array(w)) == pytest.approx(rosa, rel=2e-3)
        exact = P.KM * P.rect_integral(0, l, 0, w, 0, l, 0, w, 0.0) / w ** 2
        assert exact == pytest.approx(rosa, rel=2e-3)


# ---- meshed copper ------------------------------------------------------------------
def plane_pair(l, W, h, margin, pitch):
    vias = "".join(via((l - 0.3, y), "A", 0.4, 0.2) for y in np.linspace(-W / 2 + 0.3, W / 2 - 0.3, 5))
    items = (zone("A", "F.Cu", 0, -W / 2, l, W / 2) + zone("A", "B.Cu", -1, -W / 2 - margin, l + 1, W / 2 + margin)
             + vias + smd("J1", (0.15, 0), "A", size=(0.3, W)) + smd("J0", (0.15, 0), "A", size=(0.3, W), layer="B.Cu"))
    b = board(items, h=h)
    return b, (-1, -W / 2 - margin - 0.01, l + 1, W / 2 + margin + 0.01)


def per_mm(fn, l1, l2):
    return (fn(l2) - fn(l1)) / (l2 - l1)


def test_wide_strip_over_plane_matches_microstrip():
    """A 3 mm strip 0.1 mm over a wide plane: L' from Hammerstad (thin strip, air) is 37.3 pH/mm.
    The current sits on the facing copper surfaces, so L' follows the dielectric gap."""
    def L(l):
        b, win = plane_pair(l, 3.0, 0.1, 3.0, 0.3)
        return port_L(b, ["A"], win, 0.3, "J1", "J0")[0]
    u = 30.0
    z0 = 376.73 / (2 * math.pi) * math.log((6 + (2 * math.pi - 6) * math.exp(-(30.666 / u) ** 0.7528)) / u
                                           + math.sqrt(1 + 4 / u ** 2))
    assert per_mm(L, 3, 6) * 1e3 == pytest.approx(z0 / 299792458.0, rel=0.03)


def test_uniform_current_matches_kernel():
    """At DC the current is uniform in two equal plates: the mesh must give the continuous result."""
    W, h = 3.0, 0.1

    def L(l):
        b, win = plane_pair(l, W, h, 0.0, 0.25)
        Z = port_L(b, ["A"], win, 0.25, "J1", "J0", hf=False, f=1.0)[0]
        return Z.imag / (2 * math.pi)
    Wm, hm = W * 1e-3, h * 1e-3

    def uni(l):
        s11 = P.rect_integral(0, l, 0, Wm, 0, l, 0, Wm, 0.0)
        m12 = P.rect_integral(0, l, 0, Wm, 0, l, 0, Wm, hm)
        return 2 * P.KM * (s11 - m12) / Wm ** 2
    assert per_mm(L, 3, 6) == pytest.approx(per_mm(uni, 3e-3, 6e-3) * 1e-3, rel=0.01)


def test_strip_over_plane_matches_field_solver():
    """1 mm trace 1.51 mm over a plane, shorted by a via: L' within 6 % of the 2D field solver
    (the mesh has 2-4 bars across the trace, so edge crowding is only partly resolved)."""
    w, h = 1.0, 1.51e-3

    def L(l):
        items = (seg((0, 0), (l, 0), w, "A") + smd("J1", (0, 0), "A", size=(w, w)) + via((l, 0), "A", 0.8, 0.4)
                 + zone("A", "B.Cu", -4, -5, l + 4, 5) + smd("J0", (0, 0), "A", size=(w, w), layer="B.Cu"))
        b = board(items)
        return port_L(b, ["A"], (-4, -5, l + 4, 5), 0.4, "J1", "J0")[0]
    sec = XS.Section([XS.Rect(-w / 2 * 1e-3, w / 2 * 1e-3, -35e-6, 0, True), XS.Rect(-0.02, 0.02, h, h + 35e-6)],
                     [(0, h, 4.5)], 0.03, 0.01)
    assert per_mm(L, 6, 12) == pytest.approx(XS.solve(sec).L * 1e-3, rel=0.06)


def test_via_pair_loop():
    """Two vias 2 mm apart joined on B.Cu, port between their top pads: barrels (tube self, mutual)
    and the closing trace, against a partial-inductance hand estimate."""
    s, l, r = 2e-3, 1.545e-3, 0.15e-3

    def M(ln, d):
        return MU0 / (2 * math.pi) * ln * (math.asinh(ln / d) - math.sqrt(1 + (d / ln) ** 2) + d / ln)
    trace = 1.3e-3                                     # between the via rings
    est = 2 * M(l, r) - 2 * M(l, s) + MU0 * trace / (2 * math.pi) * (math.log(2 * trace / 0.3e-3) + 0.5)
    items = (via((0, 0), "A") + via((2, 0), "A") + seg((0, 0), (2, 0), 0.3, "A", "B.Cu")
             + smd("J1", (0, 0), "A", (0.6, 0.6)) + smd("J2", (2, 0), "A", (0.6, 0.6)))
    L, m, _ = port_L(board(items), ["A"], (-2, -2, 4, 2), 0.1, "J1", "J2")
    assert (m.br_kind == 2).sum() == 2
    assert L == pytest.approx(est, rel=0.12)


def test_resistance_matches_dc_solver():
    items = (seg((0, 0), (8, 0), 0.5, "A") + seg((8, 0), (8, 5), 0.5, "A") + smd("J1", (0, 0), "A")
             + smd("J2", (8, 5), "A"))
    b = board(items)
    Z, m, _ = port_L(b, ["A"], (-2, -2, 10, 7), 0.1, "J1", "J2", hf=False, f=1.0)
    r_dc = X.dc_resistance(b, "J1.1", "J2.1", ST.from_board(b), X.Options())
    assert Z.real == pytest.approx(r_dc, rel=0.03)


def test_plate_capacitance():
    """10 x 10 mm plate 0.2 mm over a 20 x 20 mm plane: ε A / h plus a little fringing."""
    items = (zone("A", "F.Cu", 0, 0, 10, 10) + zone("G", "B.Cu", -5, -5, 15, 15) + smd("J1", (5, 5), "A")
             + smd("J0", (5, 5), "G", layer="B.Cu"))
    b = board(items, h=0.2, outline=(-6, -6, 16, 16))
    m = P.build(b, ST.from_board(b), ["A", "G"], (-5, -5, 15, 15), 0.4, terms(b, "J1", "J0"))
    C = -P.capacitance(m, ST.from_board(b))[0, 1]
    plate = EPS0 * 4.5 * 100e-6 / 0.2e-3
    assert plate < C < 1.2 * plate


def test_terminal_outside_window_and_broken_copper():
    items = seg((0, 0), (10, 0), 0.5, "A") + smd("J1", (0, 0), "A") + smd("J2", (10, 0), "A")
    b = board(items)
    with pytest.raises(ValueError, match="no copper in the window"):
        P.build(b, ST.from_board(b), ["A"], (-2, -2, 5, 2), 0.2, terms(b, "J1", "J2"))
    b2 = board(seg((0, 0), (4, 0), 0.5, "A") + seg((6, 0), (10, 0), 0.5, "A") + smd("J1", (0, 0), "A")
               + smd("J2", (10, 0), "A"))
    m = P.build(b2, ST.from_board(b2), ["A"], (-2, -2, 12, 2), 0.2, terms(b2, "J1", "J2"))
    with pytest.raises(ValueError, match="not joined by copper"):
        P.Network(m, [("J1", "J2")])


# ---- a whole cell -----------------------------------------------------------------------
def fp(ref, pads, value="x"):
    """Footprint with pads [(number, (x, y), (w, h), net)] on F.Cu."""
    ps = " ".join(f'(pad "{n}" smd rect (at {x} {y}) (size {w} {h}) (layers "F.Cu") (net "{net}"))'
                  for n, (x, y), (w, h), net in pads)
    return f'(footprint "t" (layer "F.Cu") (at 0 0) (property "Reference" "{ref}") (property "Value" "{value}") {ps})'


def buck_cell(text=False):
    """Q1 (VIN -> SW) and Q2 (SW -> GND) side by side, C1 across VIN-GND below them, a GND plane,
    gate traces to a driver U1 (HO returns to SW, LO to GND), L1 on the switch node.
    text=True returns the board file (sim/demo/buck_demo.kicad_pcb is this board)."""
    items = (fp("Q1", [("1", (0, 0), (2, 2), "VIN"), ("2", (3, 0), (2, 2), "SW"), ("3", (1.5, 2.2), (0.6, 0.6), "G1")])
             + fp("Q2", [("1", (6, 0), (2, 2), "SW"), ("2", (9, 0), (2, 2), "GND"), ("3", (7.5, 2.2), (0.6, 0.6), "G2")])
             + fp("C1", [("1", (0, -4), (1.2, 1.4), "VIN"), ("2", (2.5, -4), (1.2, 1.4), "GND")], "10u")
             + fp("L1", [("1", (4.5, 4), (2, 2), "SW"), ("2", (4.5, 9), (2, 2), "VOUT")], "4u7")
             + fp("U1", [("1", (1.5, 7), (0.6, 0.6), "G1"), ("2", (3, 7), (0.6, 0.6), "SW"),
                         ("3", (7.5, 7), (0.6, 0.6), "G2"), ("4", (9, 7), (0.6, 0.6), "GND")], "driver")
             + zone("VIN", "F.Cu", -1, -5, 1, 1) + zone("SW", "F.Cu", 2, -1, 7, 1)
             + zone("GND", "F.Cu", 1.9, -4.7, 3.1, -3.3) + zone("GND", "B.Cu", -3, -7, 12, 9)
             + via((2.5, -4.3), "GND") + via((9.5, -0.5), "GND") + via((9, 7.3), "GND")
             + seg((1.5, 2.2), (1.5, 7), 0.3, "G1") + seg((3, 1), (3, 7), 0.3, "SW")
             + seg((7.5, 2.2), (7.5, 7), 0.3, "G2") + seg((9, 1), (9, 7), 0.3, "GND")
             + seg((4.5, 1), (4.5, 4), 1.0, "SW"))
    return board(items, outline=(-5, -9, 14, 12)) if not text else board_text(items, outline=(-5, -9, 14, 12))


def test_detect_buck_cell():
    from pcb_current_sim.loop import model as M
    b = buck_cell()
    assert M.switch_nets(b)[0] == "SW"
    s = M.detect(b)
    roles = {d.ref: (d.role, d.kind) for d in s.devices}
    assert roles["Q1"] == ("upper", "fet") and roles["Q2"] == ("lower", "fet") and roles["C1"] == ("cap", "cap")
    assert (s.rail_net, s.ret_net) == ("VIN", "GND")
    assert s.device("Q1").nets == {"D": "VIN", "S": "SW", "G": "G1"}
    drives = {g.fet: (g.out, g.ret) for g in s.drives}
    assert drives["Q1"] == (["U1", "G1"], ["U1", "SW"]) and drives["Q2"] == (["U1", "G2"], ["U1", "GND"])
    s2 = M.Setup.from_dict(s.to_dict())
    assert s2.to_dict() == s.to_dict()


def test_cell_analysis_consistency():
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = buck_cell()
    s = M.detect(b)
    s.cells = 2500
    r = A.run(b, ST.from_board(b), s)
    assert len(r.commutations) == 2 and len(r.gates) == 2
    for c in r.commutations:
        assert 0.3 < c["L_board_nH"] < c["L_total_nH"] < 30
        assert sum(v for _, v in c["breakdown"]) == pytest.approx(c["L_total_nH"], rel=0.02)
        # ringing: the off FET's Coss (+ board C) against the loop L
        off = s.device("Q2" if c["off"] == "lower" else "Q1").fet()
        f_lc = 1 / (2 * math.pi * math.sqrt(c["L_total_nH"] * 1e-9 * off.coss_at(s.v_rail)))
        assert c["f_ring"] == pytest.approx(f_lc, rel=0.15)
        assert c["Q"] > 1
    for g in r.gates:
        assert 1 < g["L_gate_nH"] < 30 and abs(g["L_cs_nH"]) < g["L_gate_nH"]
    assert "LOOP_COPPER" in r.spice and ".ends" in r.spice


def test_narrow_diagonal_track_as_wires():
    """A 0.3 mm diagonal track on a coarse grid becomes bars along the track: R and L agree with a
    fine raster of the same track (and with Rosa's formula)."""
    items = seg((0, 0), (8, 8), 0.3, "A") + smd("J1", (0, 0), "A", (0.6, 0.6)) + smd("J2", (8, 8), "A", (0.6, 0.6))
    b = board(items)
    Zw, mw, _ = port_L(b, ["A"], (-1, -1, 9, 9), 0.25, "J1", "J2", hf=False)
    Zr, mr, _ = port_L(b, ["A"], (-1, -1, 9, 9), 0.1, "J1", "J2", hf=False)
    assert (mw.br_kind == 3).sum() > 10 and (mr.br_kind == 3).sum() == 0
    assert Zw.imag == pytest.approx(Zr.imag, rel=0.05)
    assert Zw.real == pytest.approx(Zr.real, rel=0.08)
    l = math.hypot(8, 8) - 0.6
    rosa = MU0 * l * 1e-3 / (2 * math.pi) * (math.log(2 * l / 0.3) + 0.5)
    assert Zw.imag / (2 * math.pi) == pytest.approx(rosa, rel=0.05)


def test_circuit_floating_island_next_to_shorts():
    """A part left hanging on nets nothing else reaches (a gate resistor of a FET that is switched
    off) must not make the circuit singular next to ideal shorts."""
    from pcb_current_sim.loop import circuit as CI
    c = CI.Circuit()
    c.R("a", "0", 1.0)
    c.short("a", "b")
    c.short("x", "y")                 # island: no path to ground
    Z = c.zmatrix([1e6], [("b", "0")])
    assert Z[0, 0, 0].real == pytest.approx(1.0, rel=1e-6)


def test_gate_resistor_without_its_drive_is_left_out():
    """A gate resistor whose FET is switched off (its drive drops out) must not stay in the circuit
    hanging on nets nothing else reaches: that made the matrix singular on a real board."""
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = buck_cell()
    s = M.detect(b)
    s.cells = 1500
    s.devices.append(M.Device("U1", "gate_r", "res", {"1": "G2", "2": "GND"}, dict(r=10.0, ls=0.5e-9)))
    s.drives = [g for g in s.drives if g.fet == "Q1"]          # Q2's drive gone, as when Q2 is off
    r = A.run(b, ST.from_board(b), s)
    assert len(r.commutations) == 2 and [g["fet"] for g in r.gates] == ["Q1"]
    assert "U1:G2" not in r.terminals


def test_quadtree_sizes_are_aligned_blocks():
    want = np.full((9, 10), 4)
    want[0, 0] = 1
    want[5, 9] = 2
    size = P.quadtree_sizes(want)
    assert size[0, 0] == 1 and size[1, 1] == 1                 # the 2 x 2 holding the fine cell
    assert (size[2:4, 0:4] == 2).all() and size[0, 2] == 2       # the rest of its 4 x 4 tile at 2
    assert (size[4:8, 8:10] == 2).all() and size[8, 0] == 4      # partial tile at the edge


def test_adaptive_mesh_matches_uniform_fine():
    """Fine cells only near a 1 mm trace over a plane, 2x / 4x further out: the same L' as a uniform
    fine mesh with well under half the nodes."""
    def run(l, pitch, level):
        items = (seg((0, 0), (l, 0), 1.0, "A") + smd("J1", (0, 0), "A") + via((l, 0), "A", 0.8, 0.4)
                 + zone("A", "B.Cu", -4, -6, l + 4, 6) + smd("J0", (0, 0), "A", layer="B.Cu"))
        b = board(items)
        m = P.build(b, ST.from_board(b), ["A"], (-4, -6, l + 4, 6), pitch, terms(b, "J1", "J0"), level=level)
        return P.Network(m, [("J1", "J0")]).solve(0, hf=True)[0][0, 0], m.n_nodes
    near = lambda X, Y: np.where(np.abs(Y) < 1.5, 1, np.where(np.abs(Y) < 3.5, 2, 4))   # noqa: E731
    (ua, _), (ub, nu) = run(6, 0.2, None), run(12, 0.2, None)
    (aa, _), (ab, na) = run(6, 0.2, near), run(12, 0.2, near)
    assert na < 0.5 * nu
    assert (ab - aa) == pytest.approx(ub - ua, rel=0.03)


def test_display_map_follows_the_copper():
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = buck_cell()
    s = M.detect(b)
    s.cells = 1500
    r = A.run(b, ST.from_board(b), s)
    assert r.pitch_min > 0 and r.display["grid"].h <= 0.05
    img = r.display["data"]["F.Cu"][:, :, r.display["keys"].index((0, "J"))]
    g = r.display["grid"]
    j, i = g.cell(4.5, 0.0)                       # middle of the switch-node pour: carries current
    assert np.isfinite(img[j, i]) and img[j, i] > 0
    j, i = g.cell(4.5, -2.5)                      # between the pours on F.Cu: no copper there
    assert np.isnan(img[j, i])
    e = r.maps[0]["e"]
    assert e.sum() == pytest.approx(sum(v for lab, v in r.commutations[0]["breakdown"]
                                        if "package" not in lab and "ESL" not in lab and "driver" not in lab),
                                    rel=0.05)


def kelvin_cell(text=False):
    """The buck cell with a Kelvin-source pad on Q2 (pad 4, GND net): the driver's return trace goes
    to it instead of to the power source pad."""
    items = (fp("Q1", [("1", (0, 0), (2, 2), "VIN"), ("2", (3, 0), (2, 2), "SW"), ("3", (1.5, 2.2), (0.6, 0.6), "G1")])
             + fp("Q2", [("1", (6, 0), (2, 2), "SW"), ("2", (9, 0), (2, 2), "GND"), ("3", (7.5, 2.2), (0.6, 0.6), "G2"),
                         ("4", (8.6, 2.2), (0.6, 0.6), "GND")])
             + fp("C1", [("1", (0, -4), (1.2, 1.4), "VIN"), ("2", (2.5, -4), (1.2, 1.4), "GND")], "10u")
             + fp("L1", [("1", (4.5, 4), (2, 2), "SW"), ("2", (4.5, 9), (2, 2), "VOUT")], "4u7")
             + fp("U1", [("1", (1.5, 7), (0.6, 0.6), "G1"), ("2", (3, 7), (0.6, 0.6), "SW"),
                         ("3", (7.5, 7), (0.6, 0.6), "G2"), ("4", (8.6, 7), (0.6, 0.6), "GND")], "driver")
             + zone("VIN", "F.Cu", -1, -5, 1, 1) + zone("SW", "F.Cu", 2, -1, 7, 1)
             + zone("GND", "F.Cu", 1.9, -4.7, 3.1, -3.3) + zone("GND", "B.Cu", -3, -7, 12, 9)
             + via((2.5, -4.3), "GND") + via((9.5, -0.5), "GND")
             + seg((1.5, 2.2), (1.5, 7), 0.3, "G1") + seg((3, 1), (3, 7), 0.3, "SW")
             + seg((7.5, 2.2), (7.5, 7), 0.3, "G2") + seg((8.6, 2.2), (8.6, 7), 0.3, "GND")
             + seg((4.5, 1), (4.5, 4), 1.0, "SW"))
    return board_text(items, outline=(-5, -9, 14, 12)) if text else board(items, outline=(-5, -9, 14, 12))


def test_kelvin_source_pin_lowers_common_source_inductance():
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = kelvin_cell()
    s = M.detect(b)
    s.cells = 1500
    q2 = s.device("Q2")
    assert q2.nets == {"D": "SW", "S": "GND", "G": "G2"}
    pins = q2.pins(b)
    assert sorted(b.pads[k].number for k in pins["S"]) == ["2", "4"]       # merged by net
    lcs = {}
    for kel in ([], ["4"]):
        q2.kelvin = kel
        r = A.run(b, ST.from_board(b), s)
        lcs[bool(kel)] = next(g for g in r.gates if g["fet"] == "Q2")["L_cs_nH"]
    q2.kelvin = ["4"]
    pins = q2.pins(b)
    assert [b.pads[k].number for k in pins["S"]] == ["2"] and [b.pads[k].number for k in pins["KS"]] == ["4"]
    assert lcs[True] < lcs[False] - 0.1          # the package's source inductance is no longer shared


def buck_ic(text=False):
    """A buck converter IC U1 with both switches inside: VIN (2 pins), SW (2 pins), PGND (2 pins), FB,
    an input cap C1 next to it, L1 on SW, a GND plane."""
    items = (fp("U1", [("1", (0, 0), (0.6, 0.4), "VIN"), ("2", (0, 0.65), (0.6, 0.4), "VIN"),
                       ("3", (0, 1.3), (0.6, 0.4), "SW"), ("4", (0, 1.95), (0.6, 0.4), "SW"),
                       ("5", (2.6, 1.95), (0.6, 0.4), "PGND"), ("6", (2.6, 1.3), (0.6, 0.4), "PGND"),
                       ("7", (2.6, 0.65), (0.6, 0.4), "FB"), ("8", (2.6, 0), (0.6, 0.4), "EN")], "TPS62xxx")
             + fp("C1", [("1", (-1.5, -1.5), (0.8, 0.9), "VIN"), ("2", (0.2, -1.5), (0.8, 0.9), "PGND")], "10u")
             + fp("L1", [("1", (-2.5, 2.5), (1.5, 1.5), "SW"), ("2", (-2.5, 6), (1.5, 1.5), "VOUT")], "1u")
             + zone("VIN", "F.Cu", -2, -2, 0.3, 0.9) + zone("SW", "F.Cu", -3.3, 1.0, 0.3, 3.3)
             + zone("PGND", "F.Cu", 0.6, -2, 3, 2.3) + zone("PGND", "B.Cu", -5, -4, 5, 8)
             + via((1.2, 0), "PGND") + via((1.2, 1.6), "PGND"))
    return board_text(items, outline=(-6, -5, 6, 9)) if text else board(items, outline=(-6, -5, 6, 9))


def test_converter_ic_with_internal_switches():
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = buck_ic()
    assert "SW" in M.switch_nets(b)
    s = M.detect(b, "SW")
    assert (s.rail_net, s.ret_net) == ("VIN", "PGND")
    hs, ls = s.by_role("upper")[0], s.by_role("lower")[0]
    assert hs.internal and ls.internal and hs.ref == ls.ref == "U1" and hs.key != ls.key
    assert hs.nets == {"D": "VIN", "S": "SW"} and ls.nets == {"D": "SW", "S": "PGND"}
    assert not s.drives and "internal" in s.label().lower()
    s.cells, s.i_switch = 1200, 2.0
    for d in (hs, ls):
        d.model.update(t_rise=2e-9, t_fall=1e-9)
    r = A.run(b, ST.from_board(b), s)
    assert len(r.commutations) == 2 and not r.gates
    for c in r.commutations:
        assert 0.2 < c["L_total_nH"] < 10 and math.isfinite(c["f_ring"])
        assert c["didt_A_ns"] == pytest.approx(2.0)                    # 2 A in 1 ns (the off switch's t_f)
        assert c["V_over"] == pytest.approx(c["L_total_nH"] * 2.0, rel=1e-6)
    assert not any("No gate drive" in w for w in r.warnings)


def boost_ic(text=False):
    """A synchronous boost IC: VIN (supply, input cap C1), SW (inductor L1 from VIN), VOUT (internal
    high-side switch SW -> VOUT, output cap C2), PGND; GND plane."""
    items = (fp("U1", [("1", (0, 0), (0.6, 0.4), "VIN"), ("2", (0, 0.65), (0.6, 0.4), "SW"),
                       ("3", (0, 1.3), (0.6, 0.4), "SW"), ("4", (2.6, 1.3), (0.6, 0.4), "VOUT"),
                       ("5", (2.6, 0.65), (0.6, 0.4), "VOUT"), ("6", (2.6, 0), (0.6, 0.4), "PGND")], "TPS61xxx")
             + fp("C1", [("1", (-1.5, -1.5), (0.8, 0.9), "VIN"), ("2", (0.2, -1.5), (0.8, 0.9), "PGND")], "10u")
             + fp("C2", [("1", (4.2, 1.0), (0.8, 0.9), "VOUT"), ("2", (4.2, -0.8), (0.8, 0.9), "PGND")], "22u")
             + fp("L1", [("1", (-2.5, 1.0), (1.5, 1.5), "SW"), ("2", (-2.5, -2.5), (1.5, 1.5), "VIN")], "2u2")
             + zone("SW", "F.Cu", -3.3, 0.4, 0.3, 1.6) + zone("VOUT", "F.Cu", 2.3, 0.4, 4.7, 1.6)
             + zone("PGND", "F.Cu", 2.3, -1.4, 4.7, 0.2) + zone("PGND", "B.Cu", -5, -4, 6, 4)
             + via((3.5, -0.6), "PGND") + via((0.2, -1.9), "PGND"))
    return board_text(items, outline=(-6, -5, 7, 5)) if text else board(items, outline=(-6, -5, 7, 5))


def test_sync_boost_ic_rail_is_the_output():
    """The hot loop of a boost IC closes through its VOUT pin and the output cap, not the input: the
    rail is the IC power net that is not the inductor's far end."""
    from pcb_current_sim.loop import model as M
    b = boost_ic()
    s = M.detect(b, "SW")
    assert (s.rail_net, s.ret_net) == ("VOUT", "PGND")
    hs = s.by_role("upper")[0]
    assert hs.internal and hs.nets == {"D": "VOUT", "S": "SW"}
    assert [d.ref for d in s.by_role("cap")] == ["C2"]
    # editing the rail by hand moves the switch pins and searches the caps again
    M.set_cell_nets(b, s, rail="VIN")
    assert s.by_role("upper")[0].nets["D"] == "VIN" and [d.ref for d in s.by_role("cap")] == ["C1"]
    M.set_cell_nets(b, s, rail="VOUT")
    assert s.by_role("upper")[0].nets["D"] == "VOUT" and [d.ref for d in s.by_role("cap")] == ["C2"]


def boost_ic_linked(text=False):
    """A boost IC whose VOUT pins (net VO) reach the output cap C2 (on VOUT) only through a 0 Ω option
    resistor R5; R6 is a 0 Ω bypass from VO to VIN; FB has a cap footprint C3 that is not fitted."""
    dnp = lambda t: t.replace('(property "Reference"', '(attr smd dnp) (property "Reference"', 1)  # noqa: E731
    items = (fp("U1", [("1", (0, 0), (0.6, 0.4), "VIN"), ("2", (0, 0.65), (0.6, 0.4), "SW"),
                       ("3", (0, 1.3), (0.6, 0.4), "SW"), ("4", (2.6, 1.3), (0.6, 0.4), "VO"),
                       ("5", (2.6, 0.65), (0.6, 0.4), "VO"), ("6", (2.6, 0), (0.6, 0.4), "PGND"),
                       ("7", (1.3, 2.2), (0.4, 0.4), "Net-(U1-FB)")], "TPS61xxx")
             + fp("C1", [("1", (-1.5, -1.5), (0.8, 0.9), "VIN"), ("2", (0.2, -1.5), (0.8, 0.9), "PGND")], "10u")
             + fp("R5", [("1", (4.0, 1.0), (0.6, 0.5), "VO"), ("2", (5.0, 1.0), (0.6, 0.5), "VOUT")], "0R")
             + fp("R6", [("1", (3.0, 3.0), (0.6, 0.5), "VO"), ("2", (2.0, 3.0), (0.6, 0.5), "VIN")], "0R")
             + fp("C2", [("1", (6.2, 1.0), (0.8, 0.9), "VOUT"), ("2", (6.2, -0.8), (0.8, 0.9), "PGND")], "22u")
             + dnp(fp("C3", [("1", (1.3, 3.2), (0.5, 0.5), "Net-(U1-FB)"), ("2", (1.3, 4.2), (0.5, 0.5), "PGND")],
                      "10p"))
             + fp("L1", [("1", (-2.5, 1.0), (1.5, 1.5), "SW"), ("2", (-2.5, -2.5), (1.5, 1.5), "VIN")], "2u2")
             + zone("SW", "F.Cu", -3.3, 0.4, 0.3, 1.6) + zone("VO", "F.Cu", 2.3, 0.4, 4.3, 1.6)
             + zone("VOUT", "F.Cu", 4.7, 0.4, 6.7, 1.6)
             + zone("PGND", "F.Cu", 2.3, -1.4, 6.7, 0.2) + zone("PGND", "B.Cu", -5, -4, 8, 5)
             + via((3.5, -0.6), "PGND") + via((0.2, -1.9), "PGND"))
    return board_text(items, outline=(-6, -5, 9, 6)) if text else board(items, outline=(-6, -5, 9, 6))


def test_rail_reaching_its_caps_through_a_zero_ohm_link():
    """The output cap behind a 0 Ω option resistor is in the hot loop (R5 becomes a rail link); the
    bypass 0 Ω to the input is not followed, an unfitted FB cap does not make FB look like a rail, and
    caps that are not across the rail give a clear error instead of a crash."""
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = boost_ic_linked()
    s = M.detect(b, "SW")
    assert (s.rail_net, s.ret_net) == ("VO", "PGND")
    assert [d.ref for d in s.by_role("link")] == ["R5"]
    assert [d.ref for d in s.by_role("cap")] == ["C2"]
    s.cells, s.adaptive, s.v_rail = 600, False, 5.0
    res = A.run(b, ST.from_board(b), s)
    assert any(name.startswith("R5") for name, _ in res.commutations[0]["breakdown"])
    # a stale cell: the input cap as the hot-loop cap
    s.devices = [d for d in s.devices if d.kind != "cap"] + [M.make_device(b, s, "C1", "cap")]
    with pytest.raises(ValueError, match="not across the cell's rail"):
        A.run(b, ST.from_board(b), s)


def test_fixed_capacitance_without_voltage_scaling():
    from pcb_current_sim.loop import model as M
    m = M.FetModel(coss=200e-12, crss=20e-12, v_ref=0.0)
    assert m.coss_at(5.0) == pytest.approx(200e-12) and m.coss_at(30.0) == pytest.approx(200e-12)
    m.v_ref = 10.0
    assert m.coss_at(40.0) == pytest.approx(100e-12, rel=1e-6)          # 1/√V


def chain_cell(text=False):
    """The buck cell with Q2's gate driven through two series resistors: U1 pin 3 (DRV2) -> R2 ->
    N1 -> R1 -> G2 (Q2's gate)."""
    items = (fp("Q1", [("1", (0, 0), (2, 2), "VIN"), ("2", (3, 0), (2, 2), "SW"), ("3", (1.5, 2.2), (0.6, 0.6), "G1")])
             + fp("Q2", [("1", (6, 0), (2, 2), "SW"), ("2", (9, 0), (2, 2), "GND"), ("3", (7.5, 2.2), (0.6, 0.6), "G2")])
             + fp("C1", [("1", (0, -4), (1.2, 1.4), "VIN"), ("2", (2.5, -4), (1.2, 1.4), "GND")], "10u")
             + fp("L1", [("1", (4.5, 4), (2, 2), "SW"), ("2", (4.5, 9), (2, 2), "VOUT")], "4u7")
             + fp("R1", [("1", (7.5, 3.5), (0.6, 0.5), "G2"), ("2", (7.5, 4.5), (0.6, 0.5), "N1")], "4R7")
             + fp("R2", [("1", (7.5, 5.5), (0.6, 0.5), "N1"), ("2", (7.5, 6.3), (0.6, 0.5), "DRV2")], "2R2")
             + fp("U1", [("1", (1.5, 7), (0.6, 0.6), "G1"), ("2", (3, 7), (0.6, 0.6), "SW"),
                         ("3", (7.5, 7), (0.6, 0.6), "DRV2"), ("4", (9, 7), (0.6, 0.6), "GND")], "driver")
             + zone("VIN", "F.Cu", -1, -5, 1, 1) + zone("SW", "F.Cu", 2, -1, 7, 1)
             + zone("GND", "F.Cu", 1.9, -4.7, 3.1, -3.3) + zone("GND", "B.Cu", -3, -7, 12, 9)
             + via((2.5, -4.3), "GND") + via((9.5, -0.5), "GND") + via((9, 7.3), "GND")
             + seg((1.5, 2.2), (1.5, 7), 0.3, "G1") + seg((3, 1), (3, 7), 0.3, "SW")
             + seg((7.5, 2.2), (7.5, 3.5), 0.3, "G2") + seg((7.5, 4.5), (7.5, 5.5), 0.3, "N1")
             + seg((7.5, 6.3), (7.5, 7), 0.3, "DRV2") + seg((9, 1), (9, 7), 0.3, "GND")
             + seg((4.5, 1), (4.5, 4), 1.0, "SW"))
    return board_text(items, outline=(-5, -9, 14, 12)) if text else board(items, outline=(-5, -9, 14, 12))


def test_gate_drive_through_a_chain_of_series_parts():
    from pcb_current_sim.loop import analysis as A
    from pcb_current_sim.loop import model as M
    b = chain_cell()
    s = M.detect(b)
    g2 = next(g for g in s.drives if g.fet == "Q2")
    assert g2.out == ["U1", "DRV2"] and g2.ret == ["U1", "GND"] and g2.chain == ["R1", "R2"]
    assert s.device("R1").model["r"] == pytest.approx(4.7) and s.device("R2").model["r"] == pytest.approx(2.2)
    s.cells = 1500
    r = A.run(b, ST.from_board(b), s)
    rg = next(x for x in r.gates if x["fet"] == "Q2")["R_gate"]
    m = s.device("Q2").fet()
    assert rg == pytest.approx(4.7 + 2.2 + g2.rout + m.rg, rel=0.03)     # the series parts add up
    # a pad-to-pad element (as in DC & Thermal) as a further series part: 10 Ω from U1 pin 3 to R2 pin 2
    # would short nothing (same net), so put it across R1 instead: 4.7 Ω parallel 10 Ω
    el = M.element_part(b, "Rx", "R1.1", "R1.2", 10.0)
    s.devices.append(el)
    g2.chain.append("Rx")
    assert el.pins(b) == {"1": [k for k, p in enumerate(b.pads) if p.id == "R1.1"],
                          "2": [k for k, p in enumerate(b.pads) if p.id == "R1.2"]}
    r = A.run(b, ST.from_board(b), s)
    rg2 = next(x for x in r.gates if x["fet"] == "Q2")["R_gate"]
    assert rg2 == pytest.approx(4.7 * 10 / 14.7 + 2.2 + g2.rout + m.rg, rel=0.03)
    # older setups with one gate resistor still load
    d = s.to_dict()
    d["drives"] = [dict(fet="Q2", out=["U1", "DRV2"], ret=["U1", "GND"], r_ext="R1", rout=2.0)]
    assert M.Setup.from_dict(d).drives[0].chain == ["R1"]


def test_readiness_checks_and_plain_verdict():
    """Checks list what is missing (with fixes); a cell from another board is flagged as stale; the
    verdict names the biggest contributors."""
    from pcb_current_sim.loop import advice, analysis as A, checks as CK, model as M
    b = buck_cell()
    s = M.detect(b)
    issues = CK.check(b, s)
    assert not [i for i in issues if i.level == "error"]
    assert {i.fix for i in issues} >= {"edit:Q1", "edit:Q2", "opt:i_switch"}
    stale = M.Setup.from_dict(s.to_dict())
    stale.devices[0].ref = "Q9"                          # a part that is no longer on the board
    assert any(i.level == "error" and i.fix == "detect" for i in CK.check(b, stale))
    s.cells = 1500
    r = A.run(b, ST.from_board(b), s)
    text = "\n".join(advice.verdict(r))
    assert "hot loop is" in text and "C1 ESL" in text and "Gate of Q1" in text


def test_transient_solver_against_an_analytic_rlc_step():
    """Series R-L-C driven by a 1 V step: the capacitor voltage of the underdamped response."""
    from pcb_current_sim.loop import transient as TR
    R, L, C = 2.0, 10e-9, 1e-9
    c = TR.Circuit()
    c.V("in", "0", lambda t: 1.0 if t > 0 else 0.0, 1e-6)
    c.L("in", "m", L, r=R)
    c.C("m", "0", C)
    t, xs = c.run(200e-9, 0.02e-9)
    v = c.v(xs, "m")
    a = R / (2 * L)
    w0 = 1 / math.sqrt(L * C)
    wd = math.sqrt(w0 ** 2 - a ** 2)
    ref = 1 - np.exp(-a * t) * (np.cos(wd * t) + a / wd * np.sin(wd * t))
    assert np.max(np.abs(v[5:] - ref[5:])) < 0.01


def test_switching_transient_of_the_buck_cell():
    """The edges ring at the frequency analysis' ringing frequency, with about its Q; the turn-off
    overshoot is about L_loop · di/dt; the switch node settles at the rail plus a diode drop."""
    from pcb_current_sim.loop import analysis as A, model as M, transient as TR
    b = buck_cell()
    s = M.detect(b)
    s.cells, s.i_switch, s.v_rail = 1500, 10.0, 12.0
    r = A.run(b, ST.from_board(b), s)
    edges, summ = TR.run(b, s, r)
    off = next(e for e in edges if e.kind == "off").metrics
    on = next(e for e in edges if e.kind == "on").metrics
    c = r.commutations[0]
    assert off["f_ring"] == pytest.approx(c["f_ring"], rel=0.06)
    assert off["Q_ring"] == pytest.approx(c["Q"], rel=0.3)
    assert off["overshoot"] == pytest.approx(c["L_total_nH"] * off["didt_A_ns"], rel=0.15)
    e_off = next(e for e in edges if e.kind == "off")
    assert e_off.v_sw[-1] == pytest.approx(12.0 + 0.8, abs=1.0)
    assert on["i_rr_peak"] > 0 and summ["P_sw_W"] > 0 and off["vgs_other_peak"] < off["vth_other"] + 5


def test_ngspice_matches_the_built_in_simulator_and_runs_vendor_models():
    """The same edge in ngspice (the built-in device equations as SPICE elements) agrees with the
    built-in solver; a FET with a vendor subcircuit runs in ngspice."""
    from pcb_current_sim.loop import analysis as A, model as M, spice_tran as SP, transient as TR
    if SP.available() is None:
        pytest.skip("no ngspice")
    b = buck_cell()
    s = M.detect(b)
    s.cells, s.i_switch, s.v_rail = 1500, 10.0, 12.0
    r = A.run(b, ST.from_board(b), s)
    out = {}
    for sim in ("builtin", "ngspice"):
        s.simulator = sim
        out[sim] = {e.kind: e.metrics for e in TR.run(b, s, r)[0]}
    for kind in ("on", "off"):
        a, n = out["builtin"][kind], out["ngspice"][kind]
        assert n["V_peak"] == pytest.approx(a["V_peak"], rel=0.05)
        assert n["f_ring"] == pytest.approx(a["f_ring"], rel=0.03)
        assert n["E_uJ"] == pytest.approx(a["E_uJ"], rel=0.2)
    lib = os.path.join(os.path.dirname(__file__), "data", "testfet.lib")
    assert SP.subckts(lib) == {"TESTFET": ["1", "2", "3"]} and SP.guess_pins(["1", "2", "3"]) == "D G S"
    q1 = s.device("Q1")
    q1.model.update(spice_file=lib, spice_subckt="TESTFET", spice_pins="D G S", spice_pkg=True)
    s.simulator = "auto"
    assert TR.simulator(b, s) == "ngspice"
    edges, summ = TR.run(b, s, r)
    off = next(e for e in edges if e.kind == "off").metrics
    assert summ["simulator"] == "ngspice" and off["V_peak"] > 12 and math.isfinite(off["f_ring"])


def test_saved_switch_models_apply_by_part_number(tmp_path):
    """A model saved under a part number is applied to every part with that value (case-insensitive),
    an IC's switches under '<value> high-side'; the generic flag goes."""
    from pcb_current_sim.loop import model as M, saved as SV
    store = str(tmp_path / "switch_models.json")
    b = buck_ic()
    s = M.detect(b, "SW")
    hs = s.by_role("upper")[0]
    key = SV.key_of(b, hs)
    assert key.endswith("high-side")
    SV.save(store, key.upper(), "fet", dict(rds_on=0.05, t_rise=3e-9, t_fall=2e-9, coss=80e-12, v_ref=0.0))
    done = SV.apply(b, s, SV.load(store))
    assert done == [hs.key]
    assert hs.model["rds_on"] == 0.05 and hs.model["t_fall"] == 2e-9 and hs.model["assumed"] is False
    assert s.by_role("lower")[0].model.get("assumed")           # the low side was not saved
