"""3D extraction: magnetic material library, parametric cores and their placement on the board."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from test_em3d_pcb import coil_items, spiral_points       # noqa: E402
from test_loop import board                               # noqa: E402
from pcb_current_sim.em3d import analysis as A            # noqa: E402
from pcb_current_sim.em3d import cores as CR              # noqa: E402
from pcb_current_sim.em3d import materials as MT          # noqa: E402
from pcb_current_sim.em3d import pcb                      # noqa: E402
from pcb_current_sim.suite import stackup as ST           # noqa: E402

MU0 = 4e-7 * math.pi


def _vol(boxes):
    return sum(float(np.prod(np.subtract(b.hi, b.lo))) for b in boxes if b.mu_r > 1)


def test_material_library_and_user_grades(tmp_path):
    lib = MT.Library(str(tmp_path / "materials.json"))
    assert len(lib.grades) == len(MT.BUILTIN) and all(g.builtin for g in lib.grades)
    for g in MT.BUILTIN:                                  # Snoek-like: μi f_r within one class's band
        if g.kind in ("MnZn ferrite", "NiZn ferrite"):
            assert 1e9 < g.mu_r * g.f_r < 1e11
    lib.save(MT.Grade("my 3C-like", 2300.0, 2.5e6, "MnZn ferrite"))
    lib.save(MT.Grade(MT.BUILTIN[0].name, 1800.0, 2e6))   # a user grade replaces a built-in name
    lib2 = MT.Library(str(tmp_path / "materials.json"))
    assert lib2.get("MY 3c-LIKE").mu_r == 2300.0
    assert lib2.get(MT.BUILTIN[0].name).mu_r == 1800.0 and len(lib2.grades) == len(MT.BUILTIN) + 1
    lib2.remove("my 3C-like")
    assert MT.Library(str(tmp_path / "materials.json")).get("my 3C-like") is None
    m = MT.Grade("x", 1001.0, 1e6).mu(1e6)
    assert m.real == pytest.approx(501.0) and m.imag == pytest.approx(-500.0)


@pytest.mark.parametrize("with_mu2", [True, False])
def test_debye_fit_recovers_the_parameters(with_mu2):
    f = np.geomspace(1e4, 3e7, 25)
    true = MT.Grade("t", 2300.0, 2.5e6)
    mu = true.mu(f)
    mu_i, f_r, rms = MT.fit_debye(f, mu.real, -mu.imag if with_mu2 else None)
    assert mu_i == pytest.approx(2300.0, rel=1e-4) and f_r == pytest.approx(2.5e6, rel=1e-4) and rms < 1e-6


def test_staircases_keep_the_true_areas():
    r = 2.0
    strips = CR._disc_strips(r)
    assert sum((b - a) * (w1 - w0) for a, b, w0, w1 in strips) == pytest.approx(math.pi * r * r, rel=1e-12)
    tor = CR._toroid(5.0, 3.0, 1.0, 100.0)
    area = sum((hi[0] - lo[0]) * (hi[1] - lo[1]) for lo, hi, _ in tor)
    assert area == pytest.approx(math.pi * (25 - 9), rel=1e-9)


def _flat_board():
    b = board(coil_items(spiral_points(3, 12.0, 0.5, 0.3), 0.5))
    return b, ST.from_board(b)


def test_e_core_boxes_volume_rotation_and_placement():
    b, st = _flat_board()
    z_top, z_bot = CR.board_surfaces(st)
    d = dict(A=18.0, B=2.4, C=5.0, D=1.2, E=14.0, F=4.0)
    spec = CR.CoreSpec("E", "E", d, "", 2000.0, x=6.0, y=6.0)
    boxes, w = CR.expand(spec, b, st)
    leg = (18 - 14) / 2
    assert _vol(boxes) == pytest.approx(2 * 18 * 5 * 1.2 + (2 * leg + 4) * 5 * 2.4, rel=1e-12)
    lo = np.min([x.lo for x in boxes], axis=0)
    hi = np.max([x.hi for x in boxes], axis=0)
    assert (lo[2] + hi[2]) / 2 == pytest.approx((z_top + z_bot) / 2)       # window centred on the board
    assert hi[0] - lo[0] == pytest.approx(18) and hi[1] - lo[1] == pytest.approx(5)
    rot = CR.expand(CR.CoreSpec("E", "E", d, "", 2000.0, x=6.0, y=6.0, rot=90), b, st)[0]
    lo_r, hi_r = np.min([x.lo for x in rot], axis=0), np.max([x.hi for x in rot], axis=0)
    assert hi_r[0] - lo_r[0] == pytest.approx(5) and hi_r[1] - lo_r[1] == pytest.approx(18)
    # E-I: the I's plate is the back's thickness, the window D; gaps follow the material boxes
    ei = CR.CoreSpec("E", "E", dict(d, D=1.8), "", 2000.0, pair=False, gap_center=0.3, gap_outer=0.3)
    bx, w = CR.expand(ei, b, st)
    assert not w
    assert [x.mu_r > 1 for x in bx] == [True] * 5 + [False] * 3
    tall = CR.CoreSpec("E", "E", dict(d, D=0.5), "", 2000.0)
    assert any("window" in t for t in CR.expand(tall, b, st)[1])
    with pytest.raises(ValueError):
        CR.expand(CR.CoreSpec("E", "E", dict(d, F=15.0), "", 2000.0), b, st)


def test_plate_sides_footprint_reference_and_round_trip():
    b, st = _flat_board()
    z_top, z_bot = CR.board_surfaces(st)
    top = CR.expand(CR.CoreSpec("s", "plate", dict(L=4.0, W=2.0, T=0.5), "", 100.0, gap_z=0.1), b, st)[0][0]
    assert top.lo[2] == pytest.approx(z_top + 0.1) and top.hi[2] == pytest.approx(z_top + 0.6)
    bot = CR.expand(CR.CoreSpec("s", "plate", dict(L=4.0, W=2.0, T=0.5), "", 100.0, side="bottom",
                                gap_z=0.1), b, st)[0][0]
    assert bot.hi[2] == pytest.approx(z_bot - 0.1) and bot.lo[2] == pytest.approx(z_bot - 0.6)
    ref = CR.expand(CR.CoreSpec("s", "plate", dict(L=4.0, W=2.0, T=0.5), "", 100.0, ref="P1", x=1.0), b, st)[0][0]
    assert (ref.lo[0] + ref.hi[0]) / 2 == pytest.approx(1.0) and (ref.lo[1] + ref.hi[1]) / 2 == pytest.approx(0.0)
    with pytest.raises(ValueError):
        CR.expand(CR.CoreSpec("s", "plate", ref="U99"), b, st)
    lib = MT.Library(None)
    g = CR.expand(CR.CoreSpec("s", "plate", material=MT.BUILTIN[4].name), b, st, lib)[0][0]
    assert g.mu_r == MT.BUILTIN[4].mu_r and g.f_r == MT.BUILTIN[4].f_r
    _, w = CR.expand(CR.CoreSpec("s", "plate", material="no such grade", mu_r=55.0), b, st, lib)
    assert w and "not in the library" in w[0]
    spec = CR.CoreSpec("E", "E", dict(A=18.0, B=2.4, C=5.0, D=1.2, E=14.0, F=4.0), "", 2000.0, round=True,
                       gap_center=0.2, rot=270, side="top")
    s2 = pcb.Setup.from_dict(pcb.Setup(cores=[spec]).to_dict())
    assert s2.cores[0].to_dict() == spec.to_dict()


def test_planar_e_core_around_a_board_spiral():
    """3-turn KiCad spiral around the centre leg of a planar E-E pair through the board: ungapped
    near the rectangular-path estimate; gapped above N² μ0 A_e / (g + l_e/μ) by the gap's
    fringing plus the winding's own air inductance (1.33 at 0.2 mm: fringing ~1.14 and ~0.2 µH)."""
    b, st = _flat_board()
    port = [pcb.Port("coil", ["P1.1"], ["P2.1"])]
    d = dict(A=18.0, B=2.4, C=5.0, D=1.2, E=14.0, F=4.0)
    out = {}
    for g in (0.0, 0.2):
        spec = CR.CoreSpec("E", "E", d, "", 2000.0, x=6.0, y=6.0, gap_center=g)
        s = pcb.Setup(nets=["C"], ports=port, f_min=1e3, f_max=1e7, pitch=0.4, n_width=2, n_height=1,
                      cores=[spec], core_cell=1.0)
        r = A.run(b, st, s, materials=MT.Library(None))
        assert not r.warnings
        Ae, le = CR.magnetic_path(spec)
        out[g] = r.dc()[1][0, 0] / (9 * MU0 * Ae * 1e-6 / ((g + le / 2000.0) * 1e-3))
    assert 0.75 < out[0.0] < 1.0
    assert 1.15 < out[0.2] < 1.5


def _rect(x0, y0, x1, y1):
    return f'(gr_rect (start {x0} {y0}) (end {x1} {y1}) (layer "Edge.Cuts"))'


E18 = dict(A=18.0, B=2.4, C=5.0, D=1.2, E=14.0, F=4.0)


def test_e_core_snaps_to_board_slots():
    coil = coil_items(spiral_points(3, 12.0, 0.5, 0.3), 0.5)
    spec = CR.CoreSpec("E", "E", dict(E18), "", 2000.0)
    # three slots drawn on the board: centre 4.4 x 5.4 at (6, 6), outer legs' at x = -2 and 14
    b = board(coil + _rect(3.8, 3.3, 8.2, 8.7) + _rect(-3.2, 3.3, -0.8, 8.7) + _rect(12.8, 3.3, 15.2, 8.7))
    assert len(CR.slots(b)) == 3
    x, y, rot, note = CR.snap(spec, b)
    assert (x, y) == pytest.approx((6.0, 6.0)) and rot in (0, 180) and note.count("slot") == 3
    # the same slots drawn in a footprint turned by 90°: the core turns with them
    fp = ('(footprint "core" (layer "F.Cu") (at 6 6 90) (property "Reference" "T1")'
          ' (fp_rect (start -2.2 -2.7) (end 2.2 2.7) (layer "Edge.Cuts"))'
          ' (fp_rect (start -9.2 -2.7) (end -6.8 2.7) (layer "Edge.Cuts"))'
          ' (fp_rect (start 6.8 -2.7) (end 9.2 2.7) (layer "Edge.Cuts")))')
    b2 = board(coil + fp)
    x, y, rot, _ = CR.snap(spec, b2)
    assert (x, y) == pytest.approx((6.0, 6.0)) and rot in (90, 270)
    boxes, _ = CR.expand(CR.CoreSpec("E", "E", dict(E18), "", 2000.0, ref="T1", rot=rot), b2, ST.from_board(b2))
    lo = np.min([q.lo for q in boxes], axis=0)
    hi = np.max([q.hi for q in boxes], axis=0)
    assert hi[1] - lo[1] == pytest.approx(18.0)                       # along board y
    # outer legs past the board's edges: only the centre slot is needed
    b3 = board(coil + _rect(3.8, 3.3, 8.2, 8.7), outline=(-0.5, -12, 12.5, 12))
    _, _, _, note = CR.snap(spec, b3)
    assert note.count("outside the board") == 2
    with pytest.raises(ValueError, match="no slots"):
        CR.snap(spec, board(coil))
    with pytest.raises(ValueError, match="No slot fits"):
        CR.snap(CR.CoreSpec("E", "E", dict(E18, F=6.0), "", 2000.0), b)
