"""3D extraction of board copper: tracks, spiral coils, a track over a plane, SPICE models."""
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from test_loop import board, seg, smd, zone                  # noqa: E402
from pcb_current_sim.em3d import analysis as A            # noqa: E402
from pcb_current_sim.em3d import pcb                      # noqa: E402
from pcb_current_sim.em3d import shapes as SH             # noqa: E402
from pcb_current_sim.em3d import spice                    # noqa: E402
from pcb_current_sim.em3d import system as S              # noqa: E402
from pcb_current_sim.suite import stackup as ST           # noqa: E402

W2 = 2 * math.pi


def run(items, ports, nets, **kw):
    b = board(items)
    s = pcb.Setup(nets=nets, ports=[pcb.Port(*p) for p in ports], **kw)
    return A.run(b, ST.from_board(b), s)


def spiral_points(turns, outer, w, gap):
    p = w + gap
    lens = [outer] + [outer - (k // 2) * p for k in range(1, 4 * turns)]
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    x = y = 0.0
    pts = [(0.0, 0.0)]
    for k, ln in enumerate(lens):
        dx, dy = dirs[k % 4]
        x, y = x + dx * ln, y + dy * ln
        pts.append((x, y))
    return pts


def coil_items(pts, w, net="C", pad=0.25):
    return ("".join(seg(a, b, w, net) for a, b in zip(pts[:-1], pts[1:]))
            + smd("P1", pts[0], net, (pad, pad)) + smd("P2", pts[-1], net, (pad, pad)))


def test_straight_track_r_and_l():
    r = run(seg((0, 0), (20, 0), 0.5, "A") + smd("P1", (0, 0), "A", (0.6, 0.6)) + smd("P2", (20, 0), "A", (0.6, 0.6)),
            [("t", ["P1.1"], ["P2.1"])], ["A"], f_max=1e8, pitch=0.2)
    R, L = r.dc()
    l, w, t = 20e-3, 0.5e-3, 35e-6
    assert R[0] == pytest.approx(1.72e-8 * l / (w * t), rel=1e-3)
    assert L[0, 0] == pytest.approx(2e-7 * l * (math.log(2 * l / (w + t)) + 0.5 + 0.2235 * (w + t) / l), rel=2e-3)
    assert r.R[-1, 0] > 2 * R[0]                       # skin effect at 100 MHz


def test_board_spiral_equals_the_analytic_geometry():
    """Tracks drawn in KiCad give the same coil as the hand-built segments, and no corner joins
    the next turn (a track end once fell back to the nearest copper 2 cells away)."""
    turns, outer, w, gap = 3, 10.0, 0.5, 0.3
    r = run(coil_items(spiral_points(turns, outer, w, gap), w), [("coil", ["P1.1"], ["P2.1"])], ["C"],
            f_max=1e8, pitch=0.2, n_height=7)
    g = SH.square_spiral(turns, outer * 1e-3, w * 1e-3, gap * 1e-3, 35e-6)
    g.discretise(1e8, nmax=7)
    m = S.System(g).reduce(1e3, 1e8)
    for f in (1e3, 1e6, 1e8):
        np.testing.assert_allclose(r.at(f)[0, 0], m.Z(f)[0, 0, 0], rtol=3e-3)


def test_track_over_plane_matches_microstrip():
    """A 0.5 mm track 1.51 mm over a B.Cu pour, shorted to it at the far end (a link): the loop L
    is the microstrip's (Hammerstad, thin strip) to within the ends' few %."""
    items = (seg((0, 0), (20, 0), 0.5, "A") + smd("P1", (0, 0), "A", (0.6, 0.6)) + zone("G", "B.Cu", -3, -6, 23, 6)
             + smd("G1", (0, 1.5), "G", (0.6, 0.6), layer="B.Cu") + smd("R1", (20, 0), "A", (0.6, 0.6))
             + smd("R2", (20, 0), "G", (0.6, 0.6), layer="B.Cu"))
    r = run(items, [("ms", ["P1.1"], ["G1.1"])], ["A", "G"], f_max=1e8, pitch=0.5, links=[[["R1.1"], ["R2.1"]]])
    u = 0.5 / 1.51
    z0 = 376.73 / W2 * math.log((6 + (W2 - 6) * math.exp(-(30.666 / u) ** 0.7528)) / u + math.sqrt(1 + 4 / u ** 2))
    L_ms = z0 / 299792458.0 * 0.02
    assert r.L[-1, 0, 0] == pytest.approx(L_ms, rel=0.05)
    assert r.L[0, 0, 0] > r.L[-1, 0, 0]                 # the DC return spreads over the plane


def two_coils():
    pts = spiral_points(2, 6.0, 0.4, 0.3)
    items = coil_items(pts, 0.4, "A")
    pts2 = [(x + 8.0, y) for x, y in pts]
    items += ("".join(seg(a, b, 0.4, "B") for a, b in zip(pts2[:-1], pts2[1:]))
              + smd("Q1", pts2[0], "B", (0.25, 0.25)) + smd("Q2", pts2[-1], "B", (0.25, 0.25)))
    return run(items, [("A", ["P1.1"], ["P2.1"]), ("B", ["Q1.1"], ["Q2.1"])], ["A", "B"], f_max=1e8, pitch=0.2,
               n_width=3, n_height=3)


def test_coupled_coils_reciprocal():
    r = two_coils()
    Z = r.at(1e6)
    assert Z[0, 1] == pytest.approx(Z[1, 0], rel=1e-9)
    assert -0.2 < r.k[20, 0, 1] < -0.005                # side by side in one plane: A's flux returns through B
    assert r.Q[30, 0] > 1


def ngspice_ac(deck, probes, f):
    """Complex node voltages (rows: frequencies) from ngspice AC runs, one point per run."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "o")
        text = deck + "\n.control\n"
        for k, x in enumerate(f):
            text += f"ac lin 1 {x:.6e} {x:.6e}\nwrdata {out}{k} " + " ".join(probes) + "\n"
        text += ".endc\n.end\n"
        p = os.path.join(d, "t.cir")
        with open(p, "w") as fh:
            fh.write(text)
        subprocess.run([shutil.which("ngspice"), "-b", p], cwd=d, capture_output=True, timeout=120, check=True)
        rows = []
        for k in range(len(f)):
            with open(f"{out}{k}") as fh:
                vals = [float(v) for v in fh.read().split()]
            # wrdata: per probe a (frequency, real, imaginary) triple
            rows.append([complex(vals[3 * q + 1], vals[3 * q + 2]) for q in range(len(probes))])
    return np.array(rows)


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not installed")
def test_spice_models_match_in_ngspice():
    turns, outer, w, gap = 3, 10.0, 0.5, 0.3
    r1 = run(coil_items(spiral_points(turns, outer, w, gap), w), [("coil", ["P1.1"], ["P2.1"])], ["C"],
             f_max=1e8, pitch=0.2, n_width=5, n_height=5)
    f = np.array([0.1, 1e4, 1e6, 3e7, 1e8])
    deck = spice.foster(r1, "COIL") + "XU a 0 COIL\nI1 0 a AC 1\n"
    v = ngspice_ac(deck, ["v(a)"], f)[:, 0]
    np.testing.assert_allclose(v, r1.model.Z(f)[:, 0, 0], rtol=1e-4)
    r2 = two_coils()
    for text in (spice.foster(r2, "TWO"), spice.fixed(r2, 1e6, "TWO")):
        deck = text + "XU a 0 b 0 TWO\nI1 0 a AC 1\nRb b 0 1e12\n"
        v = ngspice_ac(deck, ["v(a)", "v(b)"], np.array([1e6]))[0]
        Z = r2.at(1e6)
        assert v[0] == pytest.approx(Z[0, 0], rel=1e-3)
        assert v[1] == pytest.approx(Z[1, 0], rel=2e-2)      # fixed: off-diagonal R left out


def test_coarse_cells_do_not_short_coil_turns():
    """On cells coarser than the turn spacing, a pad's edge cell or a corner's search reached the next
    turn (a 4-turn coil read 33 mΩ / 24 nH instead of 145 mΩ / 227 nH): tracks join copper only where
    they touch it."""
    pts = spiral_points(4, 12.0, 0.5, 0.3)
    items = ("".join(seg(a, b, 0.5, "C") for a, b in zip(pts[:-1], pts[1:]))
             + smd("L1", pts[0], "C", (0.8, 0.8)) + smd("L2", pts[-1], "C", (0.5, 0.5)))
    out = []
    for pitch in (0.2, 0.6, 1.0):
        r = run(items, [("c", ["L1.1"], ["L2.1"])], ["C"], f_min=1e3, f_max=1e6, pitch=pitch, n_width=1, n_height=1)
        out.append(r.dc())
    for R, L in out[1:]:
        assert R[0] == pytest.approx(out[0][0][0], rel=1e-3)
        assert L[0, 0] == pytest.approx(out[0][1][0, 0], rel=1e-3)
    assert out[0][0][0] == pytest.approx(1.72e-8 * 147.2e-3 / (0.5e-3 * 35e-6), rel=0.03)
