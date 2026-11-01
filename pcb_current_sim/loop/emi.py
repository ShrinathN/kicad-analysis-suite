"""EMI indicator of a switching cell, from its two simulated edges (an estimate, not a compliance test).

One switching period is put together from the turn-on edge, the on-time, the turn-off edge and the
off-time (at the duty cycle of the hard-switched switch); its Fourier series gives the harmonics of
  - the switch-node voltage (against the return),
  - the hot-loop current (the hard-switched switch's current, without its mean).
From those:
  radiated   the hot loop as a small magnetic loop: E = η0 k² |m| I / (4π r) with |m| / I the loop's
             effective area from the frequency analysis (the magnetic moment of its extracted HF
             currents), doubled for the ground reflection of a test site (+6 dB), worst direction;
  conducted  common mode: the switch node drives the LISNs (50 Ω ∥ 50 Ω) through its capacitance to
             the chassis / earth (given by the user; heatsinks, mounting holes, cable shields).
A receiver of resolution bandwidth RBW adds the harmonics within ±RBW/2 (worst case: in phase), so a
switching frequency below the RBW reads higher than its single harmonics.
Left out: differential-mode conducted noise (it depends on the input filter), cable and common-mode
radiation of attached wires (often the largest above 30 MHz), detector weighting (QP / average
read lower than this peak sum), and the near field.
"""
import math

import numpy as np

ETA0 = 376.73
C0 = 299792458.0
MAX_SAMPLES = 1 << 22
R_LISN_CM = 25.0            # Ω, the two 50 Ω LISNs in parallel for common-mode current

# limits: [(f_start, f_end, dB_start, dB_end)], log-linear in f between start and end
CONDUCTED = {   # CISPR 32, AC mains, quasi-peak and average [dBµV]
    "b": dict(qp=[(150e3, 500e3, 66, 56), (500e3, 5e6, 56, 56), (5e6, 30e6, 60, 60)],
              av=[(150e3, 500e3, 56, 46), (500e3, 5e6, 46, 46), (5e6, 30e6, 50, 50)]),
    "a": dict(qp=[(150e3, 500e3, 79, 79), (500e3, 30e6, 73, 73)],
              av=[(150e3, 500e3, 66, 66), (500e3, 30e6, 60, 60)]),
}
RADIATED_10M = {  # CISPR 32, quasi-peak at 10 m [dBµV/m]; 3 m is 10 dB higher
    "b": [(30e6, 230e6, 30, 30), (230e6, 1e9, 37, 37)],
    "a": [(30e6, 230e6, 40, 40), (230e6, 1e9, 47, 47)],
}
RBW_CONDUCTED = 9e3
RBW_RADIATED = 120e3


def limit_line(bands, f):
    """Limit [dB] at frequencies f (NaN outside the bands)."""
    f = np.asarray(f, float)
    out = np.full(f.shape, np.nan)
    for f0, f1, a, b in bands:
        m = (f >= f0) & (f <= f1)
        out[m] = a + (b - a) * np.log10(f[m] / f0) / math.log10(f1 / f0)
    return out


def limits(key):
    """(class letter, distance [m], conducted dict, radiated bands) of an EMI_LIMITS key ('b10')."""
    cls, dist = key[0], float(key[1:])
    shift = {10.0: 0.0, 3.0: 10.0}.get(dist, 20 * math.log10(10.0 / dist))     # the standard's 3 m table is +10 dB
    rad = [(f0, f1, a + shift, b + shift) for f0, f1, a, b in RADIATED_10M[cls]]
    return cls, dist, CONDUCTED[cls], rad


def period(edges, f_sw, duty):
    """(t, v_node, i_loop) over one switching period on a uniform grid: turn-on edge, on-time,
    turn-off edge, off-time.  An edge longer than its part of the period is cut short."""
    on = next(e for e in edges if e.kind == "on")
    off = next(e for e in edges if e.kind == "off")
    from .transient import T0
    T = 1.0 / f_sw
    h = min(float(on.t[1] - on.t[0]), float(off.t[1] - off.t[0]))
    if T / h > MAX_SAMPLES:
        h = T / MAX_SAMPLES
    n = int(round(T / h))
    h = T / n
    duty = min(max(duty, 0.01), 0.99)
    k_on = int(round(duty * n))

    def piece(e, n_len):
        tt = T0 - 0.2e-9 + h * np.arange(n_len)
        v = np.interp(tt, e.t, e.v_node)          # holds the last value after the simulated window
        i = np.interp(tt, e.t, e.i_loop)
        return v, i
    v1, i1 = piece(on, k_on)
    v2, i2 = piece(off, n - k_on)
    return h * np.arange(n), np.concatenate([v1, v2]), np.concatenate([i1, i2])


def harmonics(x, h):
    """(f_n, peak amplitude) of the periodic signal x sampled at step h, n >= 1."""
    X = np.fft.rfft(x) / len(x)
    a = 2 * np.abs(X[1:])
    f = np.arange(1, len(X)) / (len(x) * h)
    return f, a


def receiver(f, a, rbw):
    """What a receiver of bandwidth rbw reads at each harmonic: the sum of the harmonics within
    ±rbw/2 (in phase, worst case)."""
    if len(f) < 2 or f[0] >= rbw:
        return a
    cs = np.concatenate([[0.0], np.cumsum(a)])
    lo = np.searchsorted(f, f - rbw / 2, "left")
    hi = np.searchsorted(f, f + rbw / 2, "right")
    return cs[hi] - cs[lo]


def db_uv(a):
    """Peak amplitude [V] -> dBµV of its RMS (as a receiver calibrated for sines reads it)."""
    return 20 * np.log10(np.maximum(a / math.sqrt(2), 1e-12) / 1e-6)


def envelope(f, v, duty, f_sw, t_r):
    """Trapezoid bound of the harmonics: 2 V d below 1/(π d T), -20 dB/dec to 1/(π t_r), then -40."""
    tau = duty / f_sw
    return 2 * v * duty * np.minimum(1, 1 / (math.pi * f * tau)) * np.minimum(1, 1 / (math.pi * f * max(t_r, 1e-12)))


def loop_area(result, s):
    """Effective area [mm²] of the hot loop the simulated switch commutates (its turn-off)."""
    d = s.device(result.edges[0].switch) if result.edges else None
    c = next((c for c in result.commutations if d is not None and c["off"] == d.role), None) \
        or next((c for c in result.commutations if c["active"]), result.commutations[0])
    return c.get("A_loop_mm2", float("nan"))


def analyse(result, s):
    """The spectra and their margins to the limit.  None when the edges were not simulated."""
    edges = result.edges
    if not edges or not any(e.v_node is not None for e in edges):
        return None
    area = loop_area(result, s) * 1e-6           # m² per A
    t, v, i = period(edges, s.f_sw, s.duty)
    h = t[1] - t[0]
    fv, av = harmonics(v, h)
    _, ai = harmonics(i - i.mean(), h)
    cls, dist, cond, rad = limits(s.emi_limit)
    out = dict(t=t, v=v, i=i, f=fv, v_amp=av, i_amp=ai, area_mm2=area * 1e6, limit=s.emi_limit, distance=dist,
               f_max=fv[-1])
    tr = [e.metrics.get("t_rise_ns", float("nan")) for e in edges]
    tr = [x for x in tr if math.isfinite(x)]
    out["t_rise"] = (min(tr) if tr else 1.0) * 1e-9
    out["envelope"] = envelope(fv, s.v_rail, min(max(s.duty, 0.01), 0.99), s.f_sw, out["t_rise"])
    # radiated: small loop, worst direction, +6 dB ground reflection
    k = 2 * math.pi * fv / C0
    e_far = 2 * ETA0 * k ** 2 * area * ai / (4 * math.pi * dist)
    m = (fv >= 30e6) & (fv <= 1e9)
    rad_db = db_uv(receiver(fv, e_far, RBW_RADIATED))
    out["rad_f"], out["rad_db"] = fv[m], rad_db[m]
    out["rad_limit"] = limit_line(rad, fv[m])
    out.update(_margin(out["rad_f"], out["rad_db"], out["rad_limit"], "rad"))
    # conducted common mode through the switch node's capacitance to the chassis
    mc = (fv >= 150e3) & (fv <= 30e6)
    out["cond_f"] = fv[mc]
    out["cond_qp"] = limit_line(cond["qp"], fv[mc])
    out["cond_av"] = limit_line(cond["av"], fv[mc])
    if s.c_cm > 0:
        w = 2 * math.pi * fv[mc]
        vl = av[mc] * R_LISN_CM * w * s.c_cm / np.sqrt(1 + (R_LISN_CM * w * s.c_cm) ** 2)
        out["cond_db"] = db_uv(receiver(fv[mc], vl, RBW_CONDUCTED))
        out.update(_margin(out["cond_f"], out["cond_db"], out["cond_qp"], "cond"))
    else:
        out["cond_db"] = None
    return out


def _margin(f, level, limit, key):
    """Worst margin (limit - level, dB; negative = over) and its frequency."""
    ok = np.isfinite(limit)
    if not ok.any():
        return {f"{key}_margin": float("nan"), f"{key}_at": float("nan")}
    d = np.where(ok, limit - level, np.inf)
    k = int(np.argmin(d))
    return {f"{key}_margin": float(d[k]), f"{key}_at": float(f[k])}
