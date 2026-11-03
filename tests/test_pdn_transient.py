import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pcb_current_sim.pdn import regulator as RG   # noqa: E402
from pcb_current_sim.pdn import transient as TR   # noqa: E402


def z_test(lp, f, C=44e-6, esr=3e-3, esl=0.5e-9):
    s = 2j * math.pi * f
    return 1 / (lp.Y(f) + 1 / (esr + 1 / (s * C) + s * esl))


def test_vector_fit_of_regulator_and_capacitor():
    lp = RG.Loop("current", K=6.0, f_z=8e3, f_0=0.0, f_p=50e3)
    f = np.logspace(1, 9, 200)
    z = z_test(lp, f)
    rat = TR.vectfit(f, z, 12)
    assert rat.err < 0.01
    assert np.all(rat.poles.real < 0)


def test_ramp_response_matches_time_simulation():
    lp = RG.Loop("current", K=6.0, f_z=8e3, f_0=0.0, f_p=50e3)
    f = np.logspace(1, 9, 200)
    rat = TR.vectfit(f, z_test(lp, f, esl=0.0), 12)
    t = np.linspace(0, 400e-6, 4001)
    tr = 5e-6
    i = np.clip((t - 50e-6) / tr, 0, 1)
    v_ref = RG.response(lp, t, i, 44e-6, 3e-3)                     # lsim of the same circuit
    v = -rat.ramp(t - 50e-6, tr)
    assert np.max(np.abs(v - v_ref)) < 0.02 * np.max(np.abs(v_ref))


def test_pulses_and_superposition():
    from pcb_current_sim.pdn.model import Load
    a = Load("U1", [], i_step=1.0, delay=10e-6, t_rise=1e-6, width=50e-6, t_fall=1e-6)
    b = Load("U2", [], i_step=0.5, delay=100e-6, t_rise=1e-6, width=-1, t_fall=1e-6)    # stays on
    assert np.allclose([e[:2] for e in TR.pulse_edges(a, 1e-3)], [(10e-6, 1.0), (61e-6, -1.0)])
    assert len(TR.pulse_edges(b, 1e-3)) == 1
    c = Load("U3", [], i_step=1.0, delay=0, t_rise=1e-7, width=1e-6, t_fall=1e-7, period=4e-6)
    assert len(TR.pulse_edges(c, 20e-6)) == 10
    # two loads on a lumped rail: the parts add up to the total, each load's own part is its step x Z
    lp = RG.Loop("cot", K=50.0, f_z=0.0, f_0=0.0, f_p=1e6)
    f = np.logspace(1, 9, 160)
    z = z_test(lp, f)
    Z = np.zeros((len(f), 2, 2), complex)
    Z[:] = z[:, None, None]                                         # both loads on one node
    fits = TR.fit_matrix(f, Z)
    w = TR.simulate([a, b], fits, fits, 400e-6, ["U1", "U2"])
    assert np.allclose(w.v["U1"], w.parts[("U1", "U1")] + w.parts[("U1", "U2")])
    k = np.argmin(abs(w.t - 300e-6))
    assert w.v["U1"][k] == pytest.approx(-0.5 / 50, rel=0.05)      # U2's 0.5 A held: droop 0.5 A x 20 mΩ
    assert w.i["U2"][k] == pytest.approx(0.5)
