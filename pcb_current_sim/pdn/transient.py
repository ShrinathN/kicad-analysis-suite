"""Load steps over time: the rail's voltage at each load while the loads' currents step.

Linear, from the same model as Z(f): each impedance Z_kj(f) (load k's voltage per amp drawn at load j)
is fitted with a rational function (vector fitting: poles and residues), whose response to a current
ramp has a closed form at any time; so nanosecond ringing and a millisecond regulator recovery come
out of one model without time stepping the mesh.  Each load's current is a SPICE-like PULSE (step,
delay, rise, width, fall, period); every edge adds its ramp response, with the regulator's
rising-step loop for rising load edges and its falling-step loop for falling ones (they differ on
real regulators; with one model this is exact superposition).  Voltages are deviations from the rail
before the first step (the DC drop at the base current is DC & Thermal's).
"""
import math
from dataclasses import dataclass, field

import numpy as np

# ---- vector fitting ------------------------------------------------------------------------


@dataclass
class Rational:
    """H(s) = d + s e + Σ r_i / (s − p_i) (complex poles come with their conjugates)."""
    poles: np.ndarray
    res: np.ndarray
    d: float = 0.0
    e: float = 0.0               # an inductance at the top of the band
    err: float = 0.0             # relative rms error of the fit

    def __call__(self, f):
        s = 2j * math.pi * np.asarray(f, float)[..., None]
        return self.d + s[..., 0] * self.e + np.sum(self.res / (s - self.poles), axis=-1)

    def ramp(self, t, tr):
        """Response to a unit input rising linearly from 0 at t = 0 to 1 at t = tr (0 = a step)."""
        t = np.asarray(t, float)
        p, r = self.poles, self.res

        def G(tt):                                   # ∫ step response
            tt = np.maximum(tt, 0.0)[..., None]
            with np.errstate(over="ignore", invalid="ignore"):
                g = r / p * (np.expm1(p * tt) / p - tt)
            return np.real(np.sum(g, axis=-1)) + self.d * tt[..., 0]

        def g(tt):                                   # step response
            tt = np.maximum(tt, 0.0)[..., None]
            with np.errstate(over="ignore", invalid="ignore"):
                y = r / p * np.expm1(p * tt)
            return np.real(np.sum(y, axis=-1)) + self.d * (tt[..., 0] > 0)

        if tr <= 0:
            return np.where(t >= 0, g(t), 0.0)
        ind = self.e * ((t >= 0) & (t < tr)) / tr          # L di/dt while the current ramps
        return (G(t) - G(t - tr)) / tr + ind


def _start_poles(f0, f1, n_pairs):
    w = 2 * math.pi * np.logspace(math.log10(f0), math.log10(f1), n_pairs)
    a = -w / 100
    return np.concatenate([a + 1j * w, a - 1j * w])


def _basis(s, poles):
    """Real-coefficient basis for poles (conjugate pairs -> two real functions each)."""
    cols, kinds = [], []
    used = np.zeros(len(poles), bool)
    for i, p in enumerate(poles):
        if used[i]:
            continue
        if abs(p.imag) < 1e-12 * max(abs(p), 1):
            cols.append(1 / (s - p.real))
            kinds.append(("r", p.real))
            used[i] = True
        else:
            j = next((k for k in range(len(poles)) if not used[k] and k != i and abs(poles[k] - p.conjugate())
                      < 1e-9 * abs(p)), None)
            if j is not None:
                used[j] = True
            pp = complex(p.real, abs(p.imag))
            cols.append(1 / (s - pp) + 1 / (s - pp.conjugate()))
            cols.append(1j / (s - pp) - 1j / (s - pp.conjugate()))
            kinds.append(("c", pp))
            used[i] = True
    return np.column_stack(cols), kinds


def vectfit(f, H, n_pairs=14, iters=8, weight=None):
    """Rational fit of samples H(f) (Gustavsen & Semlyen, relaxed-free, real coefficients, stable poles)."""
    f = np.asarray(f, float)
    H = np.asarray(H, complex)
    s = 2j * math.pi * f
    wgt = 1 / np.maximum(np.abs(H), 1e-12) if weight is None else weight
    poles = _start_poles(f[0], f[-1], n_pairs)
    for _ in range(iters):
        Phi, kinds = _basis(s, poles)
        n = Phi.shape[1]
        A = np.hstack([Phi, np.ones((len(s), 1)), s[:, None], -H[:, None] * Phi]) * wgt[:, None]
        b = H * wgt
        Ar = np.vstack([A.real, A.imag])
        br = np.concatenate([b.real, b.imag])
        sc = np.linalg.norm(Ar, axis=0)
        sc[sc == 0] = 1
        x = np.linalg.lstsq(Ar / sc, br, rcond=None)[0] / sc
        c = x[n + 2:]
        # zeros of sigma(s) = 1 + Σ c φ: eigenvalues of A − b cᵀ in the real block form
        blocks, bvec = [], []
        for kd in kinds:
            if kd[0] == "r":
                blocks.append(np.array([[kd[1]]]))
                bvec.append([1.0])
            else:
                p = kd[1]
                blocks.append(np.array([[p.real, p.imag], [-p.imag, p.real]]))
                bvec.append([2.0, 0.0])
        m = sum(bl.shape[0] for bl in blocks)
        Am = np.zeros((m, m))
        k = 0
        for bl in blocks:
            q = bl.shape[0]
            Am[k:k + q, k:k + q] = bl
            k += q
        bv = np.concatenate(bvec)
        z = np.linalg.eigvals(Am - np.outer(bv, c))
        z = np.where(z.real > 0, -z.real + 1j * z.imag, z)          # unstable -> mirrored
        z = np.where(np.abs(z.real) < 1e-6 * np.abs(z), -1e-6 * np.abs(z) + 1j * z.imag, z)
        poles = z
    # residues with the final poles
    Phi, kinds = _basis(s, poles)
    A = np.hstack([Phi, np.ones((len(s), 1)), s[:, None]]) * wgt[:, None]
    Ar = np.vstack([A.real, A.imag])
    br = np.concatenate([(H * wgt).real, (H * wgt).imag])
    sc = np.linalg.norm(Ar, axis=0)
    sc[sc == 0] = 1
    x = np.linalg.lstsq(Ar / sc, br, rcond=None)[0] / sc
    P, R, k = [], [], 0
    for kd in kinds:
        if kd[0] == "r":
            P.append(kd[1])
            R.append(x[k])
            k += 1
        else:
            p = kd[1]
            r = complex(x[k], x[k + 1])            # φ1 = 1/(s-p) + 1/(s-p*), φ2 = j/(s-p) - j/(s-p*)
            P += [p, p.conjugate()]
            R += [x[k] + 1j * x[k + 1], x[k] - 1j * x[k + 1]]
            k += 2
    rat = Rational(np.array(P, complex), np.array(R, complex), float(x[-2]), float(max(x[-1], 0.0)))
    rat.err = fit_error(rat, f, H)
    return rat


def fit_error(rat, f, H):
    """Relative rms error, each point against its own |H| but not below 5 % of the largest (deep notches
    between resonances would otherwise dominate)."""
    H = np.asarray(H, complex)
    ref = np.maximum(np.abs(H), 0.05 * np.abs(H).max())
    return float(np.sqrt(np.mean(np.abs((rat(f) - H) / ref) ** 2)))


def vectfit_auto(f, H, target=0.02):
    """vectfit with more poles until the error is below `target` (or the most tried)."""
    best = None
    for n in (12, 18, 26, 34):
        r = vectfit(f, H, n)
        if best is None or r.err < best.err:
            best = r
        if r.err < target:
            break
    return best


# ---- load currents --------------------------------------------------------------------------
def pulse_edges(ld, t_stop):
    """[(t0, ΔI, rise)] of a load's PULSE within [0, t_stop]: +step at delay, −step after rise + width."""
    out = []
    per = ld.period if ld.period > 0 else math.inf
    k, t0 = 0, ld.delay
    while t0 < t_stop * (1 - 1e-9) and k < 10000:
        out.append((t0, ld.i_step, ld.t_rise))
        t1 = t0 + ld.t_rise + ld.width
        if ld.width >= 0 and t1 < t_stop and (ld.period > 0 or ld.width > 0):
            out.append((t1, -ld.i_step, ld.t_fall))
        if not math.isfinite(per):
            break
        k += 1
        t0 = ld.delay + k * per
    return out


def current(ld, t, t_stop):
    i = np.zeros_like(np.asarray(t, float))
    for t0, di, tr in pulse_edges(ld, t_stop):
        i += di * (np.clip((t - t0) / tr, 0, 1) if tr > 0 else (t >= t0))
    return i


def time_grid(edges, t_stop, n=3000):
    """Uniform samples plus dense ones after every edge (log spaced), so fast ringing is resolved."""
    pts = [np.linspace(0, t_stop, n)]
    for t0, _, tr in edges:
        tr = max(tr, t_stop * 1e-9)
        pts.append(t0 + np.linspace(0, tr, 12))
        pts.append(t0 + tr + np.logspace(math.log10(tr / 50 + 1e-15), math.log10(max(t_stop - t0, tr)), 200))
    t = np.unique(np.concatenate(pts))
    return t[(t >= 0) & (t <= t_stop)]


# ---- simulation -----------------------------------------------------------------------------
@dataclass
class Waves:
    t: np.ndarray
    loads: list
    i: dict                      # load -> current change [A]
    v: dict                      # load -> ΔV at that load [V]
    parts: dict                  # (at load, from load) -> ΔV contribution [V]
    fit_err: float = 0.0         # worst relative error of the rational fits
    summary: list = field(default_factory=list)


def fit_matrix(f, Z):
    P = Z.shape[1]
    return [[vectfit_auto(f, Z[:, k, j]) for j in range(P)] for k in range(P)]


def f_top(loads, f_max):
    """Highest frequency the steps carry: about 50 / (the fastest edge), within the sweep."""
    tr = min((min(ld.t_rise, ld.t_fall) for ld in loads if min(ld.t_rise, ld.t_fall) > 0), default=1e-9)
    return float(min(f_max, max(50.0 / tr, 1e5)))


def simulate(loads, fits_up, fits_dn, t_stop, names):
    """loads: the setup's enabled Load objects in the order of `names` (the fitted matrix's order)."""
    edges = {ld.name: pulse_edges(ld, t_stop) for ld in loads}
    t = time_grid([e for v in edges.values() for e in v], t_stop)
    P = len(names)
    parts, v = {}, {n: np.zeros_like(t) for n in names}
    for k, at in enumerate(names):
        for j, frm in enumerate(names):
            acc = np.zeros_like(t)
            for t0, di, tr in edges.get(frm, []):
                rat = (fits_up if di > 0 else fits_dn)[k][j]
                acc -= di * rat.ramp(t - t0, tr)            # the load draws current: the rail drops
            parts[(at, frm)] = acc
            v[at] += acc
    cur = {ld.name: current(ld, t, t_stop) for ld in loads}
    err = max(f.err for row in fits_up + fits_dn for f in row) if P else 0.0
    return Waves(t, list(names), cur, v, parts, err)


def auto_t_stop(loads, settle):
    end = max((ld.delay + ld.t_rise + max(ld.width, 0) + ld.t_fall for ld in loads), default=0.0)
    return max(end + settle, 2 * settle)
