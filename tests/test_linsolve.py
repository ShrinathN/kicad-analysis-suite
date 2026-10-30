import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcb_current_sim import linsolve as LS            # noqa: E402
from pcb_current_sim import raster as rz              # noqa: E402
from pcb_current_sim import solver as S               # noqa: E402
from test_solver import make_board, seg, smd         # noqa: E402


def laplacian(n=40, shift=1e-3):
    d = sp.diags([-1, 2, -1], [-1, 0, 1], (n, n))
    return (sp.kronsum(d, d) + shift * sp.eye(n * n)).tocsc()


def test_factor_solves_vectors_columns_and_complex():
    A = laplacian()
    b = np.random.default_rng(0).random((A.shape[0], 3))
    f = LS.factorize(A)
    x = f.solve(b)
    assert x.shape == b.shape and np.abs(A @ x - b).max() < 1e-9
    assert f.solve(b[:, 0]).shape == (A.shape[0],)
    Ac = (A + 1j * 0.1 * sp.eye(A.shape[0])).tocsc()           # complex symmetric (future AC solves)
    xc = LS.factorize(Ac, spd=False).solve(b[:, 0] + 0j)
    assert np.abs(Ac @ xc - b[:, 0]).max() < 1e-9
    assert LS.factorize(sp.csc_matrix((0, 0))).solve(np.zeros(0)).shape == (0,)


def test_forced_superlu_and_cg_fallback(monkeypatch):
    monkeypatch.setenv(LS.ENV, "superlu")
    assert LS.backend() == "superlu"
    A = laplacian(20)
    b = np.ones(A.shape[0])
    assert np.allclose(LS.CGFactor(A).solve(b), LS.factorize(A).solve(b), rtol=1e-6)


def test_coverage_all_adds_primitives_in_one_cell():
    """Two tracks that each cover a third of a coarse cell give two thirds, not the maximum."""
    b = make_board(seg((0, 0.15), (3, 0.15), 0.3, "A") + seg((0, 0.75), (3, 0.75), 0.3, "B"), (-1, -1, 4, 2))
    g = rz.Grid(0.0, 0.0, 0.9, 3, 1)
    f = rz.coverage_all(b, g)["F.Cu"]
    assert f[0, 1] == pytest.approx(2 / 3, abs=0.05)


def hot(**kw):
    return S.Settings(pitch=0.1, error_estimate=False, magnetic=False, **kw)


def board_two_tracks():
    return make_board(seg((0, 0), (20, 0), 0.5, "N") + smd("A", (0, 0), "N") + smd("C", (20, 0), "N"))


def test_cache_reuses_setup_only_when_nothing_relevant_changed():
    b = board_two_tracks()
    rows = [S.Row("r", ["A.1"], ["C.1"], 2.0)]
    s = hot()
    S._CACHE.clear()
    r1 = S.simulate(b, rows, s, progress=lambda m: None)
    tm1 = S._CACHE["thermal"][2]
    r2 = S.simulate(b, rows, s, progress=lambda m: None)
    assert S._CACHE["thermal"][2] is tm1                          # reused
    assert r2.summary["T_max"] == pytest.approx(r1.summary["T_max"], abs=1e-9)
    s.T_amb = 35.0                                                # used after construction: reset(s) picks it up
    r3 = S.simulate(b, rows, s, progress=lambda m: None)
    assert S._CACHE["thermal"][2] is tm1
    assert r3.summary["T_max"] > r1.summary["T_max"] + 5
    s.h_conv = 20.0                                               # builds the matrix: a new model
    S.simulate(b, rows, s, progress=lambda m: None)
    assert S._CACHE["thermal"][2] is not tm1
    S.simulate(board_two_tracks(), rows, s, progress=lambda m: None)   # a re-parsed board is a new board
    assert S._CACHE["raster"][1] is not b


def test_thermal_pitch_setting():
    b = board_two_tracks()
    rows = [S.Row("r", ["A.1"], ["C.1"], 3.0)]
    fine = S.simulate(b, rows, hot(thermal_pitch=0), progress=lambda m: None)
    coarse = S.simulate(b, rows, hot(thermal_pitch=0.3), progress=lambda m: None)
    rise_f, rise_c = fine.summary["T_max"] - 25, coarse.summary["T_max"] - 25
    assert rise_c == pytest.approx(rise_f, rel=0.05)
    assert S._CACHE["thermal"][2].tg.h == pytest.approx(0.3)
