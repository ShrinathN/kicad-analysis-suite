"""Tolerance analyses of a matching network: corners, Monte Carlo, bounded worst case, sensitivity, yield.

Every tolerance (part value, line Z0 / width / εr, optionally temperature) is one
variable with a [lo, hi] range. Responses are computed for many samples at once
(chunks of samples × frequencies) and reduced to per-band metrics:
  worst return loss in the band (min over f) and worst insertion loss (max over f).
"""
import itertools
import os
from dataclasses import dataclass, field

import numpy as np

from ..suite.library import fmt_like, fmt_si
from . import model as M
from . import touchstone as TS

MAX_CORNER_VARS = 14            # 2^14 = 16384 corners
CLOUD = 200                     # sample curves kept for the Smith chart
CHUNK = 1_000_000               # samples × frequencies per evaluation chunk


@dataclass
class Var:
    label: str
    elem: int                   # element index, -1 for temperature
    attr: str                   # "value", "z0", "w", "er", "T"
    nom: float
    lo: float
    hi: float
    unit: str = ""

    def text(self, v):
        if self.attr == "T":
            return f"{v:.0f} °C"
        if self.attr == "w":
            return f"{v * 1e3:.4g} mm"
        if self.attr == "er":
            return f"{v:.3f}"
        return fmt_like(v, self.nom, self.unit)


def variables(setup):
    out = []
    for i, e in enumerate(setup.elements):
        p = e.get_part()
        if p is not None:
            t = p.tol_abs()
            if t > 0:
                out.append(Var(e.name, i, "value", p.value, max(p.value - t, 1e-3 * p.value), p.value + t, p.unit))
        elif e.kind != "S2P":
            if e.line == "ms":
                if e.w_tol > 0:
                    out.append(Var(f"{e.name} w", i, "w", e.w, max(e.w - e.w_tol, 0.05 * e.w), e.w + e.w_tol, "m"))
                if e.er_tol > 0:
                    out.append(Var(f"{e.name} εr", i, "er", e.er, max(e.er * (1 - e.er_tol / 100), 1.0),
                                   e.er * (1 + e.er_tol / 100)))
            elif e.z0_tol > 0:
                out.append(Var(f"{e.name} Z0", i, "z0", e.z0, e.z0 * (1 - e.z0_tol / 100),
                               e.z0 * (1 + e.z0_tol / 100), "Ω"))
    if setup.temp:
        out.append(Var("T", -1, "T", M.T_NOM, setup.T_min, setup.T_max, "°C"))
    return out


def sample(vars_, n, dist, rng):
    lo = np.array([v.lo for v in vars_])
    hi = np.array([v.hi for v in vars_])
    u = rng.random((n, len(vars_)))
    if dist == "gauss":
        from scipy.stats import truncnorm
        z = truncnorm.rvs(-3, 3, size=(n, len(vars_)), random_state=rng) / 3      # in [-1, 1]
        g = (lo + hi) / 2 + z * (hi - lo) / 2
        is_t = np.array([v.attr == "T" for v in vars_])
        u = np.where(is_t, lo + u * (hi - lo), g)                              # temperature stays uniform
        return u
    return lo + u * (hi - lo)


def corners(vars_):
    return np.array(list(itertools.product(*[(v.lo, v.hi) for v in vars_])), float).reshape(-1, len(vars_))


def nominal_x(vars_):
    return np.array([[v.nom for v in vars_]])


# ---------------------------------------------------------------- results
@dataclass
class Stats:
    """Responses of a set of samples (Monte Carlo or corners)."""
    X: np.ndarray
    rl: np.ndarray              # (S, F) float32, dB
    il: np.ndarray
    diss: np.ndarray
    zin: np.ndarray             # (≤CLOUD, F) complex64, for the Smith chart
    band: list                  # per band: dict(rl_worst, il_worst, ok) arrays (S,)
    ok: np.ndarray              # all bands pass, (S,)

    @property
    def n(self):
        return len(self.X)

    def yield_(self, b=None):
        return float(np.mean(self.ok if b is None else self.band[b]["ok"])) if self.n else float("nan")

    def envelope(self, q, key="rl"):
        a = getattr(self, key)
        return np.percentile(a, q, axis=0)


@dataclass
class Result:
    f: np.ndarray
    vars: list
    nominal: dict               # gam, gt, zin, rl, il, diss, vswr (F,); band: list of dicts
    mc: Stats = None
    corners: Stats = None
    worst: list = field(default_factory=list)     # dict(band, metric, value, margin, x)
    oat: np.ndarray = None      # (k, nbands, 2 metrics, 2 lo/hi): worst RL / IL with one variable at lo / hi
    share: np.ndarray = None    # (k, nbands, 2 metrics): fraction of MC variance explained linearly
    warnings: list = field(default_factory=list)


def band_masks(setup, f):
    return [(f >= b.f1) & (f <= b.f2) for b in setup.bands]


def band_metrics(setup, masks, rl, il):
    out = []
    ok_all = np.ones(rl.shape[0], bool)
    for b, m in zip(setup.bands, masks):
        if not m.any():
            d = dict(rl_worst=np.full(rl.shape[0], np.nan), il_worst=np.full(rl.shape[0], np.nan),
                     ok=np.ones(rl.shape[0], bool))
        else:
            rw, iw = rl[:, m].min(1), il[:, m].max(1)
            ok = np.ones(rl.shape[0], bool)
            if b.rl_req() > 0:
                ok &= rw >= b.rl_req()
            if b.il_max > 0:
                ok &= iw <= b.il_max
            d = dict(rl_worst=rw, il_worst=iw, ok=ok)
        out.append(d)
        ok_all &= d["ok"]
    return out, ok_all


def evaluate(setup, f, vars_, X, masks, progress=None, cancel=None, what=""):
    S, F = len(X), len(f)
    rl = np.empty((S, F), np.float32)
    il = np.empty((S, F), np.float32)
    diss = np.empty((S, F), np.float32)
    keep = min(S, CLOUD)
    zin_keep = np.empty((keep, F), np.complex64)
    step = max(1, CHUNK // max(F, 1))
    for a in range(0, S, step):
        if cancel is not None and cancel():
            raise InterruptedError
        b = min(S, a + step)
        gam, gt, zin = M.response(setup, f, vars_, X[a:b])
        r, i, d, _ = M.metrics_db(gam, gt)
        rl[a:b], il[a:b], diss[a:b] = r, i, d
        if a < keep:
            zin_keep[a:min(b, keep)] = zin[:min(b, keep) - a]
        if progress:
            progress(f"{what}: {b} / {S}")
    band, ok = band_metrics(setup, masks, rl, il)
    return Stats(X, rl, il, diss, zin_keep, band, ok)


def check(setup):
    """Problems that stop a run (raised) and warnings."""
    warn = []
    if setup.f_stop <= setup.f_start or setup.f_start <= 0:
        raise ValueError("The sweep needs 0 < f_start < f_stop.")
    for port, nm in ((setup.source, "Source"), (setup.load, "Load")):
        if port.kind == "s1p":
            if not os.path.exists(port.path):
                raise ValueError(f"{nm}: file not found: {port.path or '(none)'}")
            ts = TS.load(port.path)
            if ts.nports != 1:
                raise ValueError(f"{nm}: {os.path.basename(port.path)} is not a one-port file")
            if not ts.covers(setup.f_start, setup.f_stop):
                warn.append(f"{nm}: {os.path.basename(port.path)} covers {fmt_si(ts.f[0], 'Hz')}–"
                            f"{fmt_si(ts.f[-1], 'Hz')}; held constant outside")
        elif port.r <= 0 and port.kind != "parallel RLC":
            raise ValueError(f"{nm}: R must be > 0")
    for e in setup.elements:
        if e.kind == "S2P":
            if not os.path.exists(e.s2p):
                raise ValueError(f"{e.name}: file not found: {e.s2p or '(none)'}")
            ts = TS.load(e.s2p)
            if ts.nports != 2:
                raise ValueError(f"{e.name}: {os.path.basename(e.s2p)} is not a two-port file")
            if not ts.covers(setup.f_start, setup.f_stop):
                warn.append(f"{e.name}: {os.path.basename(e.s2p)} covers {fmt_si(ts.f[0], 'Hz')}–"
                            f"{fmt_si(ts.f[-1], 'Hz')}; held constant outside")
        p = e.get_part()
        if p is not None and p.tol_unit == "%" and p.tol >= 100:
            warn.append(f"{e.name}: tolerance ≥ 100 %; the low end is clamped")
    for b in setup.bands:
        if not (setup.f_start <= b.f1 < b.f2 <= setup.f_stop):
            warn.append(f"Band {b.name}: outside the sweep or empty")
    return warn


def nominal(setup, f=None):
    f = setup.freqs() if f is None else f
    gam, gt, zin = M.response(setup, f)
    rl, il, diss, vswr = M.metrics_db(gam, gt)
    band, ok = band_metrics(setup, band_masks(setup, f), rl, il)
    return dict(f=f, gam=gam[0], gt=gt[0], zin=zin[0], rl=rl[0], il=il[0], diss=diss[0], vswr=vswr[0],
                band=[{k: v[0] for k, v in d.items()} for d in band], ok=bool(ok[0]))


def run(setup, do_corners=True, do_mc=True, do_worst=True, do_sens=True, progress=None, cancel=None):
    f = setup.freqs()
    res = Result(f, variables(setup), nominal(setup, f))
    res.warnings = check(setup)
    vars_, masks = res.vars, band_masks(setup, f)
    k = len(vars_)
    if k == 0:
        res.warnings.append("No tolerances: only the nominal response was computed")
        return res
    if do_corners:
        if k <= MAX_CORNER_VARS:
            res.corners = evaluate(setup, f, vars_, corners(vars_), masks, progress, cancel, "Corners")
        else:
            res.warnings.append(f"{k} variables: {2 ** k} corners is too many; corners skipped")
    if do_mc and setup.mc_n > 0:
        rng = np.random.default_rng(setup.seed)
        X = sample(vars_, int(setup.mc_n), setup.dist, rng)
        res.mc = evaluate(setup, f, vars_, X, masks, progress, cancel, "Monte Carlo")
    if do_sens:
        _sensitivity(setup, res, masks)
    if do_worst and setup.bands:
        _worst_case(setup, res, masks, progress, cancel)
    return res


def _sensitivity(setup, res, masks):
    vars_, k = res.vars, len(res.vars)
    X = np.repeat(nominal_x(vars_), 2 * k, axis=0)
    for j, v in enumerate(vars_):
        X[2 * j, j], X[2 * j + 1, j] = v.lo, v.hi
    st = evaluate(setup, res.f, vars_, X, masks)
    nb = len(setup.bands)
    oat = np.full((k, nb, 2, 2), np.nan)
    for b in range(nb):
        oat[:, b, 0, :] = st.band[b]["rl_worst"].reshape(k, 2)
        oat[:, b, 1, :] = st.band[b]["il_worst"].reshape(k, 2)
    res.oat = oat
    mc = res.mc
    if mc is None or mc.n < 3 * k + 3:
        return
    Z = (mc.X - mc.X.mean(0)) / np.maximum(mc.X.std(0), 1e-300)
    share = np.full((k, nb, 2), np.nan)
    for b in range(nb):
        for m, key in enumerate(("rl_worst", "il_worst")):
            y = mc.band[b][key].astype(float)
            if not np.all(np.isfinite(y)) or y.var() == 0:
                continue
            beta = np.linalg.lstsq(Z, y - y.mean(), rcond=None)[0]
            share[:, b, m] = beta ** 2 / y.var()
    res.share = share


def _margin(setup, b, metric, vals):
    """Spec margin (positive = passes) of band b for worst RL / IL values."""
    band = setup.bands[b]
    return vals - band.rl_req() if metric == "rl" else band.il_max - vals


def _worst_case(setup, res, masks, progress=None, cancel=None):
    """Bounded search for the worst point of each spec. Resonant responses are not monotonic in the
    part values, so the worst case can sit inside the tolerance box: start from the nominal point and
    the worst corners / Monte Carlo samples, then refine with bounded Powell."""
    from scipy.optimize import minimize
    vars_ = res.vars
    lo = np.array([v.lo for v in vars_])
    hi = np.array([v.hi for v in vars_])
    known = [s for s in (res.corners, res.mc) if s is not None]
    for b, (band, m) in enumerate(zip(setup.bands, masks)):
        if not m.any():
            continue
        fb = res.f[m]
        for metric, active in (("rl", band.rl_req() > 0), ("il", band.il_max > 0)):
            if not active:
                continue
            if cancel is not None and cancel():
                raise InterruptedError
            if progress:
                progress(f"Worst case: {band.name} {metric.upper()}")

            def g(u):
                x = lo + (np.clip(u, -1, 1) + 1) / 2 * (hi - lo)
                gam, gt, _ = M.response(setup, fb, vars_, x[None, :])
                rl, il, _, _ = M.metrics_db(gam, gt)
                return float(_margin(setup, b, metric, rl.min() if metric == "rl" else il.max()))

            starts = [nominal_x(vars_)[0]]
            for s in known:
                vals = s.band[b]["rl_worst" if metric == "rl" else "il_worst"]
                order = np.argsort(_margin(setup, b, metric, vals))[:3]
                starts += [s.X[i] for i in order]
            best = None
            for x0 in starts:
                u0 = np.where(hi > lo, 2 * (x0 - lo) / np.where(hi > lo, hi - lo, 1) - 1, 0)
                r = minimize(g, u0, method="Powell", bounds=[(-1, 1)] * len(vars_),
                             options=dict(maxfev=80 * len(vars_) + 200, xtol=1e-3, ftol=1e-4))
                cand = (min(r.fun, g(u0)), r.x if r.fun <= g(u0) else u0)
                if best is None or cand[0] < best[0]:
                    best = cand
            x = lo + (np.clip(best[1], -1, 1) + 1) / 2 * (hi - lo)
            margin = best[0]
            value = band.rl_req() + margin if metric == "rl" else band.il_max - margin
            res.worst.append(dict(band=b, metric=metric, value=value, margin=margin, x=x))


# ---------------------------------------------------------------- BOM
def bom(setup):
    rows = {}
    for e in setup.elements:
        p = e.get_part()
        if p is not None:
            key = (p.kind, p.name, p.value, p.tol, p.tol_unit, p.package)
            desc = dict(kind=M.KINDS[e.kind], value=fmt_si(p.value, p.unit), tol="±" + p.tol_text(),
                        part=e.lib or "(custom)", package=p.package, mfr=p.mfr, notes=p.notes)
        elif e.kind == "S2P":
            key = ("S2P", e.s2p)
            desc = dict(kind=M.KINDS[e.kind], value=os.path.basename(e.s2p), tol="fixed", part="", package="",
                        mfr="", notes="")
        else:
            continue                    # lines and stubs are copper, not BOM items
        r = rows.setdefault(key, dict(refs=[], **desc))
        r["refs"].append(e.name)
    out = []
    for r in rows.values():
        refs = r.pop("refs")
        out.append(dict(refs=", ".join(refs), qty=len(refs), **r))
    return out


def summary_lines(setup, res):
    """Text summary: nominal per band, yields, worst cases."""
    L = []
    for b, band in enumerate(setup.bands):
        nb = res.nominal["band"][b]
        s = (f"{band.name} ({fmt_si(band.f1, 'Hz')}–{fmt_si(band.f2, 'Hz')}): nominal worst RL "
             f"{nb['rl_worst']:.1f} dB, IL {nb['il_worst']:.2f} dB")
        for st, nm in ((res.corners, "corners"), (res.mc, "MC")):
            if st is not None:
                s += f"; {nm} pass {100 * st.yield_(b):.1f} %"
        L.append(s)
    for st, nm in ((res.corners, f"Corners ({res.corners.n if res.corners else 0})"),
                   (res.mc, f"Monte Carlo ({setup.mc_n}, {setup.dist})")):
        if st is not None and setup.bands:
            L.append(f"{nm}: overall yield {100 * st.yield_():.2f} %")
    for w in res.worst:
        band = setup.bands[w["band"]]
        L.append(f"Worst case {band.name} {'RL' if w['metric'] == 'rl' else 'IL'}: "
                 f"{w['value']:.2f} dB (margin {w['margin']:+.2f} dB)")
    return L
