"""RC snubber designer: a lumped model of the hard turn-off, to choose R and C before checking
them on the board (Analyse with the snubber in the cell).

The model (inductive turn-off: the voltage has risen to the rail, then the current falls):
  the supply V, the loop's L and R, and across the switch its capacitance C_par (C_oss / C_j plus the
  board's), the snubber (R_s + ESL + C_s), and the switch channel whose current falls from I to 0
  at di/dt.  States: loop current i, switch voltage v, snubber current i_s, snubber cap voltage v_s.
      L di/dt   = V - R i - v
      C dv/dt   = i - i_ch(t) - i_s
      ESL di_s/dt = v - R_s i_s - v_s
      C_s dv_s/dt = i_s
Starting point (classic): C_s = 3 C_par (lowers the ringing frequency by about half), R_s = √(L / C_par)
(the ring's characteristic impedance: critical damping of the original ring).  Its loss is C_s V² f_sw
(charged and discharged through R_s once per cycle each way, whatever R_s is).
"""
import math

import numpy as np


def suggest(L, C_par, k=3.0):
    """(R_s, C_s) starting point."""
    return math.sqrt(L / C_par), k * C_par


def response(L, R, C_par, V, I, didt, snub=None, t_end=None, n=4000):
    """(t, v) across the switch at turn-off.  snub = (R_s, C_s, ESL) or None."""
    f0 = 1 / (2 * math.pi * math.sqrt(L * C_par))
    t_f = I / max(didt, 1e-3)
    if t_end is None:
        t_end = t_f + 12 / f0
    h = t_end / n
    if snub:
        rs, cs, esl = snub
        esl = max(esl, 1e-13)
        A = np.array([[-R / L, -1 / L, 0, 0],
                      [1 / C_par, 0, -1 / C_par, 0],
                      [0, 1 / esl, -rs / esl, -1 / esl],
                      [0, 0, 1 / cs, 0]])
        B = np.array([V / L, -1 / C_par, 0, 0])     # x' = A x + B · [1, i_ch]
        x = np.array([I, V, 0.0, V])
    else:
        A = np.array([[-R / L, -1 / L], [1 / C_par, 0]])
        B = np.array([V / L, -1 / C_par])
        x = np.array([I, V])
    Bv, Bi = np.zeros(len(x)), np.zeros(len(x))
    Bv[0], Bi[1] = B[0], B[1]
    ich = lambda t: I * min(max(1 - t / t_f, 0.0), 1.0)          # noqa: E731
    E = np.eye(len(x))
    Mi = np.linalg.inv(E - h / 2 * A)
    Mp = E + h / 2 * A
    t = np.arange(n + 1) * h
    v = np.empty(n + 1)
    v[0] = x[1]
    for k in range(n):
        u = Bv + Bi * (ich(t[k]) + ich(t[k + 1])) / 2
        x = Mi @ (Mp @ x + h * u)
        v[k + 1] = x[1]
    return t, v


def metrics(t, v, V):
    """Peak, overshoot over V, and how fast the ring dies: time until |v - V| stays below 5 % of V."""
    pk = float(np.max(v))
    dev = np.abs(v - V) > 0.05 * V
    k = np.nonzero(dev)[0]
    settle = float(t[k[-1]]) if k.size else 0.0
    return dict(V_peak=pk, overshoot=pk - V, t_settle=settle)


def sweep(L, R, C_par, V, I, didt, C_s, esl, rs_values, n=1500):
    """[(R_s, overshoot, t_settle)] for snubber resistances rs_values at C_s (all at once)."""
    t0, _ = response(L, R, C_par, V, I, didt, n=n)
    h = t0[-1] / n
    t_f = I / max(didt, 1e-3)
    esl = max(esl, 1e-13)
    rs = np.asarray(rs_values, float)
    A = np.zeros((len(rs), 4, 4))
    A[:, 0, 0], A[:, 0, 1] = -R / L, -1 / L
    A[:, 1, 0], A[:, 1, 2] = 1 / C_par, -1 / C_par
    A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = 1 / esl, -rs / esl, -1 / esl
    A[:, 3, 2] = 1 / C_s
    E = np.eye(4)
    Mi = np.linalg.inv(E - h / 2 * A)
    Mp = E + h / 2 * A
    T = np.einsum("kij,kjl->kil", Mi, Mp)
    x = np.tile([I, V, 0.0, V], (len(rs), 1))
    ich = lambda t: I * min(max(1 - t / t_f, 0.0), 1.0)          # noqa: E731
    v = np.empty((n + 1, len(rs)))
    v[0] = V
    for k in range(n):
        u = np.array([V / L, -(ich(k * h) + ich((k + 1) * h)) / 2 / C_par, 0, 0])
        x = np.einsum("kij,kj->ki", T, x) + h * (Mi @ u)
        v[k + 1] = x[:, 1]
    out = []
    for j, r in enumerate(rs):
        m = metrics(t0, v[:, j], V)
        out.append((float(r), m["overshoot"], m["t_settle"]))
    return out
