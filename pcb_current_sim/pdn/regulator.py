"""A regulator's control loop as seen by the rail, fitted to its datasheet's load-transient plot.

The regulator is a current into its output pin that reacts to the rail voltage through its loop:
an admittance to the reference

    Y(s) = K (s + ωz) / (s + ω0) · (1 + s / ωz2) / (1 + s / ωp) / (1 + s / (Q ωn) + s² / ωn²) · (1 − s / ωr)

  K   high-frequency transconductance [S]: with the output capacitance C the loop crosses over at
      about f_c = K / (2π C), so the model reacts to a different C as the real loop does;
  ωz  compensation zero (ωz = 0: no integrator action, see ω0);
  ω0  integrator leak: 0 = an ideal integrator (no DC error); with ωz = 0 the DC output
      resistance is 1 / K (droop, as a constant-on-time converter or an LDO without much gain);
  ωp  high-frequency pole (the current loop / the compensation's pole / the LDO's error amplifier);
  ωr  a boost's right-half-plane zero (0 = none): R_load (1 − D)² / L from the operating point;
  ωz2 a second zero (type-III compensation; 0 = none);
  ωn, Q  a resonant pole pair (0 = none): a recovery that rings for several cycles.

Control types fix some of these and fit the rest (`TYPES`).  The fit simulates the datasheet's test
circuit (the regulator with the datasheet's output capacitance C_test, ESR) driven by the digitised
current step and matches the digitised output deviation, edge by edge (rising and falling steps
are fitted separately: real regulators answer them differently).
"""
import math
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import optimize, signal

TWO_PI = 2 * math.pi
W_REF = TWO_PI * 1e5            # rad/s: time and frequency are scaled by this in the simulation

# key: (label, free parameters, fixed values)
TYPES = {
    "current": ("Current-mode PWM (most buck / boost converters)", ("K", "f_z", "f_p"), dict(f_0=0.0)),
    "voltage": ("Voltage-mode PWM", ("K", "f_z", "f_p"), dict(f_0=0.0)),
    "cot": ("Constant on-time / hysteretic", ("K", "f_p"), dict(f_z=0.0, f_0=0.0)),
    "ldo": ("Linear regulator (LDO)", ("K", "f_p"), dict(f_z=0.0, f_0=0.0)),
    "resonant": ("Ringing recovery (current-mode with a resonant pole pair)", ("K", "f_z", "f_n", "q"),
                 dict(f_0=0.0, f_p=1e9)),
    "general": ("General: two zeros, a pole and a resonant pair (type III)", ("K", "f_z", "f_z2", "f_p", "f_n", "q"),
                dict(f_0=0.0)),
}
AUTO = "auto"
AUTO_LABEL = "Try every type and keep the best"
START = dict(K=1.0, f_z=1e3, f_z2=1e4, f_p=1e5, f_n=1e5, q=1.0)        # replaced per edge (see fit_edge)
LO = dict(K=1e-4, f_z=10.0, f_z2=10.0, f_p=1e2, f_n=1e2, q=0.1)
HI = dict(K=1e5, f_z=1e7, f_z2=1e8, f_p=1e9, f_n=1e8, q=30.0)


@dataclass
class Loop:
    kind: str = "current"
    K: float = 10.0              # S
    f_z: float = 5e3             # Hz
    f_0: float = 0.0             # Hz
    f_p: float = 200e3           # Hz
    f_rhp: float = 0.0           # Hz (boost), 0 = none
    f_z2: float = 0.0            # Hz, second zero, 0 = none
    f_n: float = 0.0             # Hz, resonant pole pair, 0 = none
    q: float = 0.7

    def Y(self, f):
        """Admittance into the rail [S] at frequency f (array)."""
        s = 2j * math.pi * np.asarray(f, float)
        if self.f_z == 0 and self.f_0 == 0:
            y = self.K + 0 * s
        else:
            y = self.K * (s + TWO_PI * self.f_z) / (s + TWO_PI * self.f_0)
        y = y / (1 + s / (TWO_PI * self.f_p))
        if self.f_z2 > 0:
            y = y * (1 + s / (TWO_PI * self.f_z2))
        if self.f_n > 0:
            wn = TWO_PI * self.f_n
            y = y / (1 + s / (self.q * wn) + (s / wn) ** 2)
        if self.f_rhp > 0:
            y = y * (1 - s / (TWO_PI * self.f_rhp))
        return y

    def poly(self):
        """(num, den) of Y in the scaled variable s' = s / W_REF (numpy poly1d coefficient lists)."""
        wz, w0, wp = (TWO_PI * x / W_REF for x in (self.f_z, self.f_0, self.f_p))
        if self.f_z == 0 and self.f_0 == 0:
            num, den = np.array([self.K]), np.array([1.0])
        else:
            num, den = self.K * np.array([1.0, wz]), np.array([1.0, w0])
        den = np.polymul(den, [1 / wp, 1.0])
        if self.f_z2 > 0:
            num = np.polymul(num, [W_REF / (TWO_PI * self.f_z2), 1.0])
        if self.f_n > 0:
            wn = TWO_PI * self.f_n / W_REF
            den = np.polymul(den, [1 / wn ** 2, 1 / (self.q * wn), 1.0])
        if self.f_rhp > 0:
            num = np.polymul(num, [-W_REF / (TWO_PI * self.f_rhp), 1.0])
        return num, den

    def f_cross(self, C):
        """Loop crossover with output capacitance C: |Y| = ωC."""
        f = np.logspace(1, 8, 701)
        d = np.abs(self.Y(f)) - TWO_PI * f * C
        k = np.flatnonzero(np.diff(np.sign(d)) != 0)
        return float(f[k[0]]) if k.size else math.nan


def rhp_zero(v_in, v_out, i_out, L):
    """A boost's right-half-plane zero [Hz]: R_load (1 − D)² / (2π L), D = 1 − V_in / V_out."""
    if min(v_in, v_out, i_out, L) <= 0 or v_out <= v_in:
        return 0.0
    d1 = v_in / v_out
    return (v_out / i_out) * d1 * d1 / (TWO_PI * L)


def _total(loop, C, esr):
    """(num, den) of the admittance regulator ∥ (C + ESR) in s' = s / W_REF."""
    nY, dY = loop.poly()
    Cs = C * W_REF                                  # Y_C = s' Cs / (1 + s' Cs ESR)
    nC, dC = np.array([Cs, 0.0]), np.array([Cs * esr, 1.0])
    return np.polyadd(np.polymul(nY, dC), np.polymul(nC, dY)), np.polymul(dY, dC)


def stable(loop, C, esr=0.0):
    """True when the regulator with output capacitance C is a stable loop (the poles of Z = 1 / Y_total,
    i.e. the zeros of Y_total, in the left half plane)."""
    num = np.trim_zeros(_total(loop, C, esr)[0], "f")
    r = np.roots(num) if len(num) > 1 else np.zeros(0)
    return bool(np.all(r.real < 0))


def response(loop, t, i, C, esr=0.0):
    """Output deviation v(t) of the test circuit (regulator ∥ C + ESR) to the load current i(t)
    (uniformly sampled; i(t) - i(0) drives it)."""
    num, den = _total(loop, C, esr)
    num = np.trim_zeros(num, "f")
    sys_ = signal.lti(-den, num)                     # V = −Z I, Z = den / num
    tau = (np.asarray(t, float) - t[0]) * W_REF
    u = np.asarray(i, float) - i[0]
    try:
        _, v, _ = signal.lsim(sys_, u, tau)
    except ValueError:                               # improper (ESR = 0 with a pole-free Y): add a tiny ESR
        return response(loop, t, i, C, max(esr, 1e-6))
    return v


def edges(t, i):
    """Rising / falling steps of a current trace: [(direction +1 / -1, index of the 50 % crossing)]."""
    i = np.asarray(i, float)
    lo, hi = np.percentile(i, 5), np.percentile(i, 95)
    if hi - lo <= 0:
        return []
    mid = (lo + hi) / 2
    above = i > mid
    k = np.flatnonzero(above[1:] != above[:-1])
    out, last = [], -10**9
    for j in k:
        if j - last > max(3, len(i) // 50):        # one crossing per edge (noise on the threshold)
            out.append((1 if above[j + 1] else -1, int(j)))
            last = j
    return out


def segments(t, i):
    """(direction, slice) per edge: from a little before it to the middle of the gap to the next one."""
    e = edges(t, i)
    out = []
    for q, (d, j) in enumerate(e):
        a = (e[q - 1][1] + j) // 2 if q > 0 else 0
        b = (j + e[q + 1][1]) // 2 if q + 1 < len(e) else len(i)
        out.append((d, slice(a, b)))
    return out


@dataclass
class Fit:
    direction: int               # +1 rising load step, -1 falling
    loop: Loop
    rms: float                   # V, residual
    peak: float                  # V, measured peak deviation
    t: list = field(default_factory=list)          # the edge's samples (for plots)
    v_meas: list = field(default_factory=list)
    v_fit: list = field(default_factory=list)
    c: float = 0.0               # the test capacitance used (fitted when asked)
    esr: float = 0.0
    tried: list = field(default_factory=list)      # 'auto': [(kind, rms)] of every type tried
    at_limit: list = field(default_factory=list)   # parameters that ended on a bound (the model is being stretched)


def _baseline(v, n):
    return float(np.median(v[: max(3, n)]))


def fit_edge(t, i, v, C, esr=0.0, kind="current", f_rhp=0.0, direction=1, fit_c=False, weight="even"):
    """Fit the loop's free parameters to one edge (t, i, v uniformly sampled, v the deviation).

    kind 'auto' tries every type and keeps the best (a type with more parameters only when it is clearly
    better); fit_c also fits the test capacitance (within ±50 %) and its ESR (datasheets rarely give the
    capacitance after DC-bias derating); weight 'peak' makes the residual count more where the deviation
    is large, so the dip / overshoot is matched first."""
    if kind == AUTO:
        fits = []
        for k in TYPES:
            try:
                fits.append(fit_edge(t, i, v, C, esr, k, f_rhp, direction, fit_c, weight))
            except ValueError:
                pass
        if not fits:
            raise ValueError("The fit failed: check the traces and the test capacitance.")
        best = None
        for f in sorted(fits, key=lambda f: len(TYPES[f.loop.kind][1])):      # simplest first
            if best is None or f.rms < 0.85 * best.rms:
                best = f
        best.tried = [(f.loop.kind, f.rms) for f in fits]
        return best
    t, i, v = (np.asarray(x, float) for x in (t, i, v))
    j = edges(t, i)
    j0 = j[0][1] if j else len(t) // 10
    pre = max(3, int(0.8 * j0))
    v = v - _baseline(v, pre)
    i = i.copy()
    i[:pre] = np.median(i[:pre])          # the load is steady before the step (digitising noise there drives
    #                                       the model as a false step)
    di = abs(np.percentile(i, 95) - np.percentile(i, 5))
    dv = max(np.max(np.abs(v)), 1e-6)
    w = 1 + 3 * np.abs(v) / dv if weight == "peak" else np.ones_like(v)
    label, free, fixed = TYPES[kind]
    K0 = max(di / dv, 1e-3)
    fc0 = K0 / (TWO_PI * C)
    n_c = 2 if fit_c else 0
    starts = []
    for kmul in (1.0, 3.0, 0.4):
        x = dict(START, K=K0 * kmul, f_z=fc0 / 5, f_p=fc0 * 4, f_z2=fc0 * 2, f_n=fc0 * 1.5, q=1.5)
        x0 = list(np.log([x[p] for p in free]))
        if fit_c:
            x0 += [0.0, math.log(max(esr, 1e-3))]
        starts.append(np.array(x0))
    lo = list(np.log([LO[p] for p in free])) + ([math.log(0.5), math.log(1e-4)] if fit_c else [])
    hi = list(np.log([HI[p] for p in free])) + ([math.log(1.5), math.log(0.2)] if fit_c else [])
    lo, hi = np.array(lo), np.array(hi)

    def make(x):
        d = dict(fixed)
        d.update({p: float(math.exp(q)) for p, q in zip(free, x)})
        return Loop(kind=kind, f_rhp=f_rhp, **d)

    def cap(x):
        return (C * math.exp(x[-2]), math.exp(x[-1])) if fit_c else (C, esr)

    def resid(x):
        try:
            lp = make(x)
            if not stable(lp, *cap(x)):
                return np.full_like(v, 1e3)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = (response(lp, t, i, *cap(x)) - v) / dv * w
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            return np.full_like(v, 1e3)
        return np.where(np.isfinite(r), r, 1e3)
    best = None
    for x0 in starts:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = optimize.least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                                           x_scale=1.0, max_nfev=150 * len(x0))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None or not stable(make(best.x), *cap(best.x)):
        raise ValueError("No stable loop of this type fits this edge: check the traces and the test capacitance.")
    lp = make(best.x)
    c, e = cap(best.x)
    vf = response(lp, t, i, c, e)
    names = list(free) + (["C", "ESR"] if fit_c else [])
    edge = [nm for nm, x, a, b in zip(names, best.x, lo, hi) if min(x - a, b - x) < 0.02 * (b - a)]
    return Fit(direction, lp, float(np.sqrt(np.mean((vf - v) ** 2))), float(np.max(np.abs(v))),
               list(t), list(v), list(vf), c, e, at_limit=edge)


def resample(t, y, n=None):
    """Uniform resampling of a digitised trace (sorted by t)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    o = np.argsort(t, kind="stable")
    t, y = t[o], y[o]
    n = n or min(max(len(t), 200), 2000)
    tu = np.linspace(t[0], t[-1], n)
    return tu, np.interp(tu, t, y)


def fit(ti, i, tv, v, C, esr=0.0, kind="current", f_rhp=0.0, step=None, window=None, fit_c=False,
        weight="even"):
    """Fit every edge of a load-transient record (current trace ti, i and voltage trace tv, v, each
    with its own samples): [Fit] in time order.  window [s]: fit only this long after each edge
    (e.g. to leave out a light-load burst pattern later on); see fit_edge for the rest."""
    ti, i, tv, v = (np.asarray(x, float) for x in (ti, i, tv, v))
    a, b = max(ti.min(), tv.min()), min(ti.max(), tv.max())
    if b <= a:
        raise ValueError("The current and voltage traces do not overlap in time.")
    n = min(max(len(ti), len(tv), 400), 3000)
    t = np.linspace(a, b, n)
    oi, ov = np.argsort(ti, kind="stable"), np.argsort(tv, kind="stable")
    i = np.interp(t, ti[oi], i[oi])
    v = np.interp(t, tv[ov], v[ov])
    if step:                     # the current step as stated (the datasheet's caption), not as read
        lo, hi = np.percentile(i, 5), np.percentile(i, 95)
        if hi - lo > 0:
            i = lo + (i - lo) * step / (hi - lo)
    out, errs = [], []
    dt = t[1] - t[0]
    segs = list(zip(segments(t, i), edges(t, i)))

    def size(sl, j):                        # the step's size: median after the edge minus median before
        return abs(np.median(i[j + 1:sl.stop]) - np.median(i[sl.start:j + 1])) if sl.start <= j < sl.stop - 1 else 0
    big = max((size(sl, j) for (_, sl), (_, j) in segs), default=0.0)
    for (d, sl), (_, j) in segs:
        if size(sl, j) < 0.3 * big or j < 0.02 * len(t) or j > 0.98 * len(t):
            continue                        # a glitch on the current trace, or an edge with no record around it
        if window and window > 0:
            n = int(math.ceil(window / dt))
            sl = slice(max(sl.start, j - n), min(sl.stop, j + 1 + n))       # as long before the edge (the baseline)
        if sl.stop - sl.start < 20:
            continue
        try:
            out.append(fit_edge(t[sl], i[sl], v[sl], C, esr, kind, f_rhp, d, fit_c, weight))
        except ValueError as e:
            errs.append(str(e))
    if not out:
        raise ValueError(errs[0] if errs else "No load step found in the current trace.")
    return out


@dataclass
class RegulatorModel:
    """A fitted regulator, stored with the PDN setup (and by part number for other boards)."""
    part: str = ""
    kind: str = "current"
    c_test: float = 44e-6        # F, the datasheet's output capacitance for the plot
    esr_test: float = 0.003      # Ω
    f_rhp: float = 0.0
    up: dict = field(default_factory=dict)       # Loop as a dict (rising load step)
    down: dict = field(default_factory=dict)     # falling step
    note: str = ""
    source: dict = field(default_factory=dict)   # how it was read: image path, plot area, scales, colours

    def loop(self, direction=1):
        d = self.up if direction > 0 or not self.down else self.down
        return Loop(**d) if d else None

    def to_dict(self):
        return asdict(self)
