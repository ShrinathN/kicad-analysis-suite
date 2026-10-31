"""Switching transient of a cell: one hard-switched turn-on and one turn-off edge, in time.

The extracted copper (coupled inductors + resistors at the ringing frequency, board capacitance
between nets) and the parts form ONE circuit, so the power loop and the gate loops interact: the
switch node's dv/dt reaches the off FET's gate through C_gd and the shared copper, and di/dt
through the common-source inductance.

Solver: modified nodal analysis, trapezoidal integration (backward Euler on the first step),
Newton iterations with junction-voltage damping, fixed step with sub-stepping on failure.

Device models (datasheet level):
  MOSFET    square-law channel from V_th and g_fs (at a current), triode slope 1/R_DS(on) at the
            gate voltage it is given at; C_gs constant, C_gd(V) and C_ds(V) ~ 1/√(V + 1 V) through
            the datasheet values at V_ref (charge-conserving); body diode with stored charge
            τ·i (τ = Q_rr / I_test) for reverse recovery; package L per pin, internal R_G.
  diode     exponential with R_d, C_j(V), stored charge (Q_rr; 0 for a Schottky), package L.
  internal  (a switch inside a converter IC) a conductance ramped over its rise / fall time, C_oss,
            body diode.
  driver    a voltage source stepping 0 <-> V_drive in 1 ns, behind R out (source / sink).
  inductor  a constant current at the switch-node pad (the edge is far shorter than its time constant).
  supply    V_rail behind 50 mΩ + 200 nH at the biggest hot-loop cap (holds the rail; the caps
            supply the edge).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import model as M
from .analysis import pin_terms, term

VT = 0.02585            # V, thermal voltage
GMIN = 1e-9             # S, every node to ground
T0 = 2e-9               # s, when the driver starts to move
DRV_EDGE = 1e-9         # s, driver source ramp (its R and the gate C set the real edge)
R_TIE = 1e-5            # Ω, terminals on one mesh node
EXP_MAX = 40.0


# ---------------------------------------------------------------- device functions
FC = 0.5               # forward-bias coefficient (as SPICE's junction capacitance)


def qc_fun(c_ref, v_ref, phi=1.0):
    """Charge q(v) and capacitance c(v) of a junction-like capacitor: c_ref at v_ref, c0 / √(1 + v/φ)
    down to v = -FC·φ, then rising linearly as SPICE's junction capacitance does in forward bias (a
    MOSFET's C_gd does rise once V_DS drops below V_GS).  v_ref <= 0: a linear capacitor c_ref."""
    if v_ref <= 0 or c_ref <= 0:
        f = lambda v: (c_ref * v, c_ref)        # noqa: E731
        f.spec = ("lin", c_ref)
        return f
    A = 2 * c_ref * math.sqrt(v_ref + phi)
    c0 = c_ref * math.sqrt((v_ref + phi) / phi)
    sp = math.sqrt(phi)

    v1 = -FC * phi
    q1 = A * (math.sqrt(v1 + phi) - sp)
    F = (1 - FC) ** 1.5

    def f(v):
        if v >= v1:
            r = math.sqrt(v + phi)
            return A * (r - sp), A / (2 * r)
        c = c0 / F * (1 - 1.5 * FC - 0.5 * v / phi)
        q = q1 + c0 / F * ((1 - 1.5 * FC) * (v - v1) - 0.25 * (v * v - v1 * v1) / phi)
        return q, c
    f.spec = ("qc", A, phi, c0)
    return f


def junction(i_s, n):
    """Diode current and conductance (exponential, continued linearly above EXP_MAX)."""
    nvt = n * VT

    def f(v):
        x = v / nvt
        if x > EXP_MAX:
            e = math.exp(EXP_MAX)
            return i_s * (e * (1 + x - EXP_MAX) - 1), i_s * e / nvt
        e = math.exp(x)
        return i_s * (e - 1), i_s * e / nvt
    f.spec = (i_s, n)
    return f


def sat_current(vf, i_vf, n):
    return i_vf / (math.exp(min(vf / (n * VT), EXP_MAX)) - 1)


def channel(m):
    """MOSFET channel current i(vgs, vds) from drain to source (N-channel sense)."""
    K = m.gfs ** 2 / (4 * max(m.i_gfs, 1e-3))
    s = 0.05
    k_lin = 1.0 / (max(m.rds_on, 1e-5) * max(m.v_gs_on - m.vth, 0.5))

    def f(vgs, vds):
        vov = s * np.logaddexp(0.0, (vgs - m.vth) / s)
        isat = K * vov * vov
        g = vov * k_lin
        return isat * math.tanh(g * vds / (isat + 1e-12))
    f.spec = (K, m.vth, s, k_lin)
    return f


# ---------------------------------------------------------------- circuit
@dataclass
class _Q:          # nonlinear charge between a and b (polarity p): q = p·f(p·v)
    a: int
    b: int
    f: object
    p: float = 1.0
    q0: float = 0.0
    i0: float = 0.0


@dataclass
class _D:          # junction current a -> b, with stored charge tau·i
    a: int
    b: int
    f: object
    tau: float = 0.0
    q0: float = 0.0
    i0: float = 0.0


@dataclass
class _Ch:         # MOSFET channel between d and s, controlled by g; polarity p (+1 N, -1 P)
    d: int
    s: int
    g: int
    f: object
    p: float = 1.0


@dataclass
class _G:          # time-varying conductance a - b
    a: int
    b: int
    g: object


class Circuit:
    def __init__(self):
        self.names = {"0": -1}
        self.res = []          # (a, b, R)
        self.caps = []         # (a, b, C) linear
        self.qs, self.ds, self.chs, self.gs = [], [], [], []
        self.lbr = []          # inductive branches (a, b, R, L, label)
        self.blocks = []       # (first branch index, L (k, k), R (k, k)) coupled copper
        self.vsrc = []         # (a, b, E(t), R, label)
        self.isrc = []         # (a, b, I(t)): current leaves a, enters b
        self.subs = []         # vendor SPICE subcircuits (ngspice only): (name, file, [nodes])
        self.probes = {}

    def n(self, name):
        if name not in self.names:
            self.names[name] = len(self.names) - 1
        return self.names[name]

    def R(self, a, b, r):
        self.res.append((self.n(a), self.n(b), max(r, 1e-6)))

    def C(self, a, b, c):
        if c > 0:
            self.caps.append((self.n(a), self.n(b), c))

    def Qnl(self, a, b, f, p=1.0):
        self.qs.append(_Q(self.n(a), self.n(b), f, p))

    def L(self, a, b, l, r=0.0, label=""):
        self.lbr.append((self.n(a), self.n(b), r, max(l, 1e-13), label))
        return len(self.lbr) - 1

    def block(self, pairs, L, R):
        k0 = len(self.lbr)
        for a, b in pairs:
            self.lbr.append((self.n(a), self.n(b), 0.0, 0.0, ""))
        self.blocks.append((k0, np.asarray(L, float), np.asarray(R, float)))

    def V(self, a, b, e, r, label=""):
        self.vsrc.append((self.n(a), self.n(b), e, max(r, 1e-6), label))
        return len(self.vsrc) - 1

    def I(self, a, b, i):
        self.isrc.append((self.n(a), self.n(b), i))

    def D(self, a, b, f, tau=0.0):
        self.ds.append(_D(self.n(a), self.n(b), f, tau))

    def Ch(self, d, s, g, f, p=1.0):
        self.chs.append(_Ch(self.n(d), self.n(s), self.n(g), f, p))

    def Gt(self, a, b, g):
        self.gs.append(_G(self.n(a), self.n(b), g))

    def sub(self, name, path, nodes):
        self.subs.append((name, path, [self.n(x) for x in nodes]))

    # ------------------------------------------------------------- matrices
    def _setup(self):
        n = len(self.names) - 1
        mL, mV = len(self.lbr), len(self.vsrc)
        self.nn, self.mL, self.mV = n, mL, mV
        G = np.zeros((n, n))
        Cn = np.zeros((n, n))

        def st(Mx, a, b, v):
            if a >= 0:
                Mx[a, a] += v
            if b >= 0:
                Mx[b, b] += v
            if a >= 0 and b >= 0:
                Mx[a, b] -= v
                Mx[b, a] -= v
        for a, b, r in self.res:
            st(G, a, b, 1 / r)
        for a, b, c in self.caps:
            st(Cn, a, b, c)
        G[np.diag_indices(n)] += GMIN
        Bl = np.zeros((n, mL))
        Lm = np.zeros((mL, mL))
        Rm = np.zeros((mL, mL))
        for k, (a, b, r, l, _) in enumerate(self.lbr):
            if a >= 0:
                Bl[a, k] = 1
            if b >= 0:
                Bl[b, k] = -1
            Lm[k, k], Rm[k, k] = l, r
        for k0, L, R in self.blocks:
            k = len(L)
            Lm[k0:k0 + k, k0:k0 + k] = L
            Rm[k0:k0 + k, k0:k0 + k] = R
        Bv = np.zeros((n, mV))
        Rv = np.zeros(mV)
        for k, (a, b, _, r, _) in enumerate(self.vsrc):
            if a >= 0:
                Bv[a, k] = 1
            if b >= 0:
                Bv[b, k] = -1
            Rv[k] = r
        self.G, self.Cn, self.Bl, self.Lm, self.Rm, self.Bv, self.Rv = G, Cn, Bl, Lm, Rm, Bv, Rv

    def _A(self, mode, h):
        """The linear part of the Jacobian: mode 'dc', 'be' (backward Euler) or 'tr' (trapezoidal)."""
        n, mL, mV = self.nn, self.mL, self.mV
        N = n + mL + mV
        A = np.zeros((N, N))
        kc = 0.0 if mode == "dc" else (1 / h if mode == "be" else 2 / h)
        A[:n, :n] = self.G + kc * self.Cn
        A[:n, n:n + mL] = self.Bl
        A[:n, n + mL:] = self.Bv
        A[n:n + mL, :n] = self.Bl.T
        A[n:n + mL, n:n + mL] = -(self.Rm + kc * self.Lm)
        A[n + mL:, :n] = self.Bv.T
        A[n + mL:, n + mL:] = -np.diag(self.Rv)
        return A, kc

    # ------------------------------------------------------------- one Newton solve
    def _newton(self, A, kc, mode, t, x_prev, x0, hist, lam=1.0, maxit=60):
        n, mL = self.nn, self.mL
        N = len(x0)
        b = np.zeros(N)
        # sources
        for a, bb, i in self.isrc:
            I = lam * i(t)
            if a >= 0:
                b[a] -= I
            if bb >= 0:
                b[bb] += I
        for k, (_, _, e, _, _) in enumerate(self.vsrc):
            b[n + mL + k] = lam * e(t)
        # history of linear caps and inductive branches
        if mode != "dc":
            vp, jp = x_prev[:n], x_prev[n:n + mL]
            if mode == "tr":
                b[:n] += kc * (self.Cn @ vp) + hist["ic"]
                b[n:n + mL] += -self.Bl.T @ vp + self.Rm @ jp - kc * (self.Lm @ jp)
            else:
                b[:n] += kc * (self.Cn @ vp)
                b[n:n + mL] += -kc * (self.Lm @ jp)
        x = x0.copy()
        V = lambda v, k: v[k] if k >= 0 else 0.0          # noqa: E731
        for it in range(maxit):
            J = A.copy()
            F = A @ x - b
            v = x[:n]

            def add(a, bb, i, gab):          # current i from a to b, di/dv_ab = gab
                if a >= 0:
                    F[a] += i
                    J[a, a] += gab
                    if bb >= 0:
                        J[a, bb] -= gab
                if bb >= 0:
                    F[bb] -= i
                    J[bb, bb] += gab
                    if a >= 0:
                        J[bb, a] -= gab
            for q in self.qs:
                if mode == "dc":
                    continue
                vab = V(v, q.a) - V(v, q.b)
                qq, cc = q.f(q.p * vab)
                qq *= q.p
                if mode == "tr":
                    i = kc * (qq - q.q0) - q.i0
                else:
                    i = kc * (qq - q.q0)
                add(q.a, q.b, i, kc * cc)
            for d in self.ds:
                vab = V(v, d.a) - V(v, d.b)
                i, g = d.f(vab)
                if mode != "dc" and d.tau > 0:
                    qd = d.tau * i
                    i += (kc * (qd - d.q0) - (d.i0 if mode == "tr" else 0.0))
                    g += kc * d.tau * g
                add(d.a, d.b, i, g)
            for c in self.chs:
                vg, vd, vs = V(v, c.g), V(v, c.d), V(v, c.s)
                vgs, vds = c.p * (vg - vs), c.p * (vd - vs)
                i0 = c.f(vgs, vds)
                e = 1e-6
                dg = (c.f(vgs + e, vds) - i0) / e
                dd = (c.f(vgs, vds + e) - i0) / e
                i = c.p * i0
                # i = p f(p(vg-vs), p(vd-vs)): di/dvg = dg, di/dvd = dd, di/dvs = -(dg+dd)
                for row, sg in ((c.d, 1.0), (c.s, -1.0)):
                    if row < 0:
                        continue
                    F[row] += sg * i
                    for col, dv in ((c.g, dg), (c.d, dd), (c.s, -(dg + dd))):
                        if col >= 0:
                            J[row, col] += sg * dv
            for g in self.gs:
                gv = g.g(t)
                vab = V(v, g.a) - V(v, g.b)
                add(g.a, g.b, gv * vab, gv)
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return None, it
            # damping: junctions may not jump more than a few tenths of a volt forward
            alpha = 1.0
            for d in self.ds:
                v_old = V(v, d.a) - V(v, d.b)
                dv = (dx[d.a] if d.a >= 0 else 0.0) - (dx[d.b] if d.b >= 0 else 0.0)
                if dv > 0.2 and v_old + dv > 0.3:
                    alpha = min(alpha, 0.2 / dv)
            big = np.max(np.abs(dx[:n])) if n else 0.0
            if big > 50:
                alpha = min(alpha, 50 / big)
            x = x + alpha * dx
            if alpha == 1.0 and np.max(np.abs(dx[:n])) < 1e-6 + 1e-6 * np.max(np.abs(x[:n])) \
                    and (mL == 0 or np.max(np.abs(dx[n:])) < 1e-6 + 1e-6 * np.max(np.abs(x[n:]))):
                return x, it
        return None, maxit

    def _commit(self, x, mode, kc):
        """Update the charge elements' history after a converged step."""
        n = self.nn
        V = lambda k: x[k] if k >= 0 else 0.0          # noqa: E731
        for q in self.qs:
            qq, _ = q.f(q.p * (V(q.a) - V(q.b)))
            qq *= q.p
            q.i0 = 0.0 if mode == "dc" else (kc * (qq - q.q0) - q.i0 if mode == "tr" else kc * (qq - q.q0))
            q.q0 = qq
        for d in self.ds:
            i, _ = d.f(V(d.a) - V(d.b))
            qd = d.tau * i
            d.i0 = 0.0 if mode == "dc" else (kc * (qd - d.q0) - d.i0 if mode == "tr" else kc * (qd - d.q0))
            d.q0 = qd
        return n

    def dc(self, t=0.0):
        """Operating point at time t (inductors shorted, capacitors open); source stepping if
        Newton fails outright."""
        self._setup()
        A, kc = self._A("dc", 1.0)
        N = A.shape[0]
        x = np.zeros(N)
        sol, _ = self._newton(A, kc, "dc", t, x, x, None)
        if sol is None:
            for lam in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
                sol, _ = self._newton(A, kc, "dc", t, x, x, None, lam=lam, maxit=150)
                if sol is None:
                    raise RuntimeError("The operating point before the edge did not converge.")
                x = sol
        self._commit(sol, "dc", 0.0)
        return sol

    def run(self, t_end, h, progress=None):
        """Transient from the DC operating point at t = 0 to t_end with step h."""
        x = self.dc(0.0)
        n, mL = self.nn, self.mL
        ic = np.zeros(n)
        hist = {"ic": ic}
        ts, xs = [0.0], [x.copy()]
        cache = {}
        t = 0.0
        first = True
        steps = int(math.ceil(t_end / h))
        for k in range(steps):
            done, sub = 0.0, 1
            while done < h - 1e-18:
                hh = h / sub
                mode = "be" if first else "tr"
                key = (mode, sub)
                if key not in cache:
                    cache[key] = self._A(mode, hh)
                A, kc = cache[key]
                sol, _ = self._newton(A, kc, mode, t + hh, x, x, hist, maxit=40)
                if sol is None:
                    if sub >= 256:
                        raise RuntimeError(f"The transient did not converge at {(t + hh) * 1e9:.2f} ns.")
                    sub *= 4
                    continue
                # cap currents for the trapezoidal history (linear caps, as node vectors)
                vp = x[:n]
                if mode == "tr":
                    hist["ic"] = kc * (self.Cn @ (sol[:n] - vp)) - hist["ic"]
                else:
                    hist["ic"] = kc * (self.Cn @ (sol[:n] - vp))
                self._commit(sol, mode, kc)
                x, t, done, first = sol, t + hh, done + hh, False
            ts.append(t)
            xs.append(x.copy())
            if progress and k % 500 == 0:
                progress(k / steps)
        return np.array(ts), np.array(xs)

    def v(self, xs, name):
        k = self.names.get(name, -1)
        return xs[:, k] if k >= 0 else np.zeros(len(xs))

    def j(self, xs, branch):
        return xs[:, self.nn + branch]


# ---------------------------------------------------------------- the cell as a circuit
@dataclass
class Edge:
    kind: str                    # "on" | "off": the control switch turns on / off
    switch: str                  # its key
    other: str                   # the complementary position's main part (key)
    current: float               # A at the edge
    t: np.ndarray = None
    v_sw: np.ndarray = None      # V across the blocking device (die) at the end of the edge
    waves: dict = field(default_factory=dict)       # name -> array
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _psd(Mx, floor=1e-15):
    Mx = (Mx + Mx.T) / 2
    w, U = np.linalg.eigh(Mx)
    return (U * np.maximum(w, floor * max(w.max(), 1e-30))) @ U.T


def vendor_model(d):
    """(subckt name, file, pin order list, includes package) of a FET's vendor SPICE model, or None."""
    m = d.model
    if d.kind == "fet" and not d.internal and m.get("spice_file") and m.get("spice_subckt"):
        pins = (m.get("spice_pins") or "D G S").split()
        return m["spice_subckt"], m["spice_file"], pins, bool(m.get("spice_pkg", True))
    return None


def build(board, s, copper, edge, i_l, warnings, vendor=False):
    """Circuit for one edge: kind "on" / "off" of the control switch.  vendor: FETs with a vendor
    SPICE model are subcircuits (ngspice only)."""
    c = Circuit()
    gnd = copper["ret_ref"]
    ren = lambda name: "0" if name == gnd else name          # noqa: E731
    pairs = [(ren(a), ren(b)) for a, b in copper["ports"]]
    if pairs:
        c.block(pairs, _psd(copper["L"]), _psd(copper["R"], 0.0))
    for a, b in copper["ties"]:
        c.R(ren(a), ren(b), R_TIE)
    for a, b, cc in copper["caps"]:
        c.C(ren(a), ren(b), cc)
    direction, _ = s.current_dir(board)
    ctrl_pos = "upper" if direction == "out" else "lower"
    other_pos = "lower" if ctrl_pos == "upper" else "upper"
    ctrl = next((d for d in s.by_role(ctrl_pos) if d.kind == "fet"), None)
    if ctrl is None:
        raise ValueError(f"No switch in the {ctrl_pos} position to switch (inductor current "
                         f"{'out of' if direction == 'out' else 'into'} the switch node).")
    drives = {s.device(g.fet).key: g for g in s.drives if s.device(g.fet) is not None}
    used_r = {k for g in s.drives for k in g.chain}
    t_on = T0
    probes = {}
    for d in s.devices:
        if not d.enabled or not (d.role in ("upper", "lower", "cap", "link")
                                 or (d.role == "gate_r" and d.key in used_r)):
            continue
        T = {k: ren(v) for k, v in pin_terms(d).items()}
        k_ = d.key
        if d.kind == "fet" and vendor and vendor_model(d):
            name, path, order, pkg = vendor_model(d)
            m = d.fet()
            p = -1.0 if m.channel == "P" else 1.0
            dn, sn, gn = k_ + "#d", k_ + "#s", k_ + "#g"
            tiny = 1e-12
            bd = c.L(T["D"], dn, tiny if pkg else m.ld, label=f"{k_} drain")
            c.L(T["S"], sn, tiny if pkg else m.ls)
            if "KS" in T:
                c.L(T["KS"], sn, tiny if pkg else m.lks)
            gated = "G" in T and k_ in drives
            if gated:
                c.L(T["G"], gn, tiny if pkg else m.lg, r=0.0 if pkg else m.rg)
            else:
                c.R(gn, sn, 1e3)
            nodes = []
            for k2, pin in enumerate(order):
                pu = pin.upper()
                if pu in ("D", "G", "S"):
                    nodes.append({"D": dn, "G": gn, "S": sn}[pu])
                elif pu.startswith("T"):              # temperature pins of the model: 25 °C
                    if "tamb#" not in c.names:
                        c.V("tamb#", "0", lambda t: 25.0, 1e-3)
                    nodes.append("tamb#")
                else:
                    nc = f"{k_}#nc{k2}"
                    c.R(nc, "0", 1e9)
                    nodes.append(nc)
            c.sub(name, path, nodes)
            if gated:
                g = drives[k_]
                vd = g.v_drive * p
                if d is ctrl:
                    r = g.rout if edge == "on" else (g.rout_off or g.rout)
                    lo, hi = (0.0, vd) if edge == "on" else (vd, 0.0)
                    e = (lambda t, lo=lo, hi=hi: lo + (hi - lo) * min(max((t - t_on) / DRV_EDGE, 0.0), 1.0))
                else:
                    r, e = (g.rout_off or g.rout), (lambda t: 0.0)
                c.V(ren(A_term(g.out)), ren(A_term(g.ret)), e, r, f"{g.out[0]} driver")
            probes[k_] = dict(d=dn, s=sn, g=gn if gated else None, i=bd, pv=p, pi=p)
        elif d.kind == "fet":
            m = d.fet()
            p = -1.0 if m.channel == "P" else 1.0
            dn, sn, gn = k_ + "#d", k_ + "#s", k_ + "#g"
            if vendor_model(d):
                warnings.append(f"{k_}: its vendor SPICE model needs ngspice; the built-in simulator uses its "
                                "datasheet values.")
            bd = c.L(T["D"], dn, m.ld, label=f"{k_} drain")
            c.L(T["S"], sn, m.ls)
            if "KS" in T:
                c.L(T["KS"], sn, m.lks)
            v_ref = m.v_ref
            cgs, cgd_ref, cds_ref = m.ciss - m.crss, m.crss, max(m.coss - m.crss, 1e-15)
            c.Qnl(dn, sn, qc_fun(cds_ref, v_ref), p)
            # body diode: anode at the source (N)
            n_bd = 1.5
            i_s = sat_current(m.vsd, m.i_vsd, n_bd)
            tau = m.qrr / max(m.i_qrr, 1e-3)
            a_, k2 = (sn, dn) if p > 0 else (dn, sn)
            c.D(a_, k2, junction(i_s, n_bd), tau)
            switched = d is ctrl
            gated = "G" in T and k_ in drives and not d.internal
            if gated:
                c.L(T["G"], gn, m.lg, r=m.rg)
                c.C(gn, sn, max(cgs, 1e-15))
                c.Qnl(dn, gn, qc_fun(cgd_ref, v_ref), p)
                c.Ch(dn, sn, gn, channel(m), p)
                g = drives[k_]
                vd = g.v_drive * p
                if switched:
                    r = g.rout if edge == "on" else (g.rout_off or g.rout)
                    lo, hi = (0.0, vd) if edge == "on" else (vd, 0.0)
                    e = (lambda t, lo=lo, hi=hi: lo + (hi - lo) * min(max((t - t_on) / DRV_EDGE, 0.0), 1.0))
                else:           # held off by its driver
                    r, e = (g.rout_off or g.rout), (lambda t: 0.0)
                c.V(ren(A_term(g.out)), ren(A_term(g.ret)), e, r, f"{g.out[0]} driver")
            else:
                # an IC's internal switch, or a FET without a gate drive: a conductance ramped over
                # its rise / fall time
                if not d.internal and "G" in T:
                    warnings.append(f"{k_} has no gate drive: switched as an ideal ramp over its rise / fall time.")
                t_r = m.t_rise or M.INTERNAL_EDGE
                t_f = m.t_fall or M.INTERNAL_EDGE
                g_on = 1 / max(m.rds_on, 1e-5)
                # the conductance moves through RAMP_DECADES decades evenly in log over the edge, so the
                # resistance passes V / I (where the current changes over) in the middle of it, not in
                # its last sliver as a linear ramp would
                if switched and edge == "on":
                    gf = (lambda t, tr=t_r, g0=g_on: _log_ramp(g0, (t - t_on) / tr))
                elif switched:
                    gf = (lambda t, tf=t_f, g0=g_on: _log_ramp(g0, 1 - (t - t_on) / tf))
                else:
                    gf = (lambda t: 0.0)
                c.Gt(dn, sn, gf)
                c.Qnl(dn, sn, qc_fun(cgd_ref, v_ref), p)        # no gate: C_rss is across d-s too
            probes[k_] = dict(d=dn, s=sn, g=gn if gated else None, i=bd, pv=p, pi=p)
        elif d.kind == "diode":
            m = d.diode()
            an, aj = k_ + "#a", k_ + "#j"
            bd = c.L(T["A"], an, m.l_pkg, label=f"{k_} anode")
            c.R(an, aj, m.rd)
            n_d = 1.05 if m.qrr == 0 else 1.6
            c.D(aj, T["K"], junction(sat_current(m.vf, m.i_vf, n_d), n_d), m.qrr / max(m.i_qrr, 1e-3))
            c.Qnl(T["K"], an, qc_fun(m.cj, m.v_ref, 0.5))
            probes[k_] = dict(d=T["K"], s=an, g=None, i=bd, pv=1.0, pi=-1.0)
        elif d.kind == "cap":
            p_ = d.part()
            from ..suite.library import cap_factor
            mid = d.ref + "#c"
            c.L(T["1"], mid, p_.esl_eff(), r=p_.esr_eff())
            c.C(mid, T["2"], p_.value * float(cap_factor(p_, 25.0, s.v_rail)))
        elif d.kind == "res":
            c.L(T["1"], T["2"], d.model.get("ls", 0.5e-9), r=d.model.get("r", 0.0))
    # inductor current (constant during the edge) and the supply holding the rail
    if copper.get("inductor"):
        lt = ren(copper["inductor"][0])
    else:
        lt = ren(pin_terms(ctrl)["S" if ctrl_pos == "upper" else "D"])
        warnings.append("No inductor on the switch node: the current is injected at the switch's own pin.")
    if direction == "out":
        c.I(lt, "0", lambda t: i_l)
    else:
        c.I("0", lt, lambda t: i_l)
    caps = sorted(s.by_role("cap"), key=lambda d: -d.model.get("value", 0.0))
    if caps:
        T = pin_terms(caps[0])
        c.L(ren(T["1"]), "supply#", 200e-9, r=0.05)
        c.V("supply#", ren(T["2"]), lambda t: s.v_rail, 1e-3, "supply")
    c.probes = probes
    return c, ctrl, direction


def A_term(pin):
    return term(pin[0], pin[1])


RAMP_DECADES = 5.0


def _log_ramp(g_on, x):
    """Conductance of a ramped switch at x = 0 (off) .. 1 (on)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return g_on
    return g_on * 10 ** (-RAMP_DECADES * (1 - _smooth(x)))


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- the edges
def _metrics(t, vb, vrail, i_ctrl, i_l, v_ctrl, p_ctrl, vgs_other, vth_other, i_other, t0):
    """Numbers of one edge.  vb: voltage across the device that blocks at the end."""
    out = {}
    pk = int(np.argmax(vb))
    out["V_peak"] = float(vb[pk])
    out["overshoot"] = float(vb[pk] - vrail)
    # dv/dt: 10-90 % of the rail on the blocking device (it rises on this edge)
    def cross(y, lev, start=0):
        k = np.nonzero(y[start:] >= lev)[0]
        return start + int(k[0]) if k.size else None
    k10, k90 = cross(vb, 0.1 * vrail), cross(vb, 0.9 * vrail)
    out["dvdt_V_ns"] = (0.8 * vrail / max(t[k90] - t[k10], 1e-13) * 1e-9) if (k10 is not None and k90 is not None
                                                                         and k90 > k10) else float("nan")
    # di/dt of the control switch's current: max slope (smoothed over 5 samples)
    di = np.gradient(i_ctrl, t)
    if len(di) > 5:
        di = np.convolve(di, np.ones(5) / 5, mode="same")
    out["didt_A_ns"] = float(np.max(np.abs(di)) * 1e-9)
    # ringing after the peak: frequency from the spacing of its maxima, Q from their decay
    y = vb[pk:] - (vb[-1] if len(vb) else vrail)
    mx = np.nonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > 0))[0] + 1
    mx = mx[y[mx] > 0.05 * max(y[0], 1e-9)][:10]
    if len(mx) >= 3:
        tt = t[pk:][mx]
        out["f_ring"] = 1 / float(np.mean(np.diff(tt)))
        lr = np.log(y[mx][:-1] / y[mx][1:])
        dec = float(np.mean(lr))
        out["Q_ring"] = math.pi / dec if dec > 1e-6 else float("inf")
    else:
        out["f_ring"], out["Q_ring"] = float("nan"), float("nan")
    out["t_rise_ns"] = float((t[k90] - t[k10]) * 1e9) if (k10 is not None and k90 is not None and k90 > k10) \
        else float("nan")
    # switching energy in the control switch: ∫ v_ds i_d over the edge (until the power stays below
    # 2 % of its peak)
    pw = v_ctrl * i_ctrl * p_ctrl * p_ctrl
    k0 = int(np.searchsorted(t, t0))
    p_end = float(np.mean(pw[-max(len(pw) // 20, 1):]))          # conduction loss once settled
    pk_p = float(np.max(np.abs(pw[k0:] - p_end))) if len(pw) > k0 else 0.0
    above = np.nonzero(np.abs(pw[k0:] - p_end) > 0.02 * pk_p)[0]
    k1 = k0 + (int(above[-1]) + 1 if above.size else 0)
    out["E_uJ"] = float(np.trapezoid(pw[k0:k1 + 1], t[k0:k1 + 1]) * 1e6) if k1 > k0 else 0.0
    if vgs_other is not None:
        out["vgs_other_peak"] = float(np.max(vgs_other[k0:]))
        out["vgs_other_min"] = float(np.min(vgs_other[k0:]))
        out["vth_other"] = vth_other
        out["gate_margin"] = vth_other - out["vgs_other_peak"]
    if i_other is not None:
        out["i_rr_peak"] = float(-np.min(i_other[k0:]))   # reverse current through the other device
    return out


def _simulate(c, t_end, h, sim):
    """(t, V(name), J(branch)) on a uniform time grid of step h."""
    tu = np.arange(0.0, t_end + h / 2, h)
    if sim == "ngspice":
        from . import spice_tran as ST
        text, vec = ST.deck(c, t_end, h)
        want = set()
        for pr in c.probes.values():
            for key in ("d", "s", "g"):
                if pr.get(key) and pr[key] != "0":
                    want.add(vec[("v", pr[key])])
            want.add(vec[("i", pr["i"])])
        res = ST.run(text, sorted(want))
        t = res["time"]
        V = lambda name: (np.interp(tu, t, res[vec[("v", name)]]) if name != "0" else np.zeros(len(tu)))   # noqa: E731
        J = lambda k: np.interp(tu, t, res[vec[("i", k)]])          # noqa: E731
        return tu, V, J
    t, xs = c.run(t_end, h)
    return t, (lambda name: c.v(xs, name)), (lambda k: c.j(xs, k))


def simulator(board, s):
    """'builtin' or 'ngspice' for setup s (auto: ngspice when a FET has a vendor model and ngspice is
    there)."""
    from . import spice_tran as ST
    want = s.simulator
    if want == "auto":
        want = "ngspice" if any(vendor_model(d) for d in s.devices if d.enabled) and ST.available() else "builtin"
    if want == "ngspice" and not ST.available():
        if s.simulator == "ngspice":
            raise ValueError("ngspice is not available (KiCad's libngspice was not found): use the built-in "
                             "simulator.")
        want = "builtin"
    return want


def run(board, s, result, progress=lambda m: None):
    """Both hard-switched edges of cell s.  result: the frequency analysis (its copper)."""
    if not result.copper:
        raise ValueError("Run the analysis first (the transient uses its copper model).")
    if s.i_switch <= 0:
        raise ValueError("Give the inductor current at the edges (Operating point) to simulate the switching.")
    if s.v_rail <= 0:
        raise ValueError("Give the voltage across the switches (Operating point).")
    act = [c for c in result.commutations if c["active"]] or result.commutations
    f_ring = act[0]["f_ring"] if math.isfinite(act[0]["f_ring"]) else act[0]["f_eval"]
    q = act[0]["Q"] if math.isfinite(act[0]["Q"]) else 10.0
    edges = []
    sim = simulator(board, s)
    for kind in ("off", "on"):
        warnings = []
        i_l = max(s.i_switch + (s.i_ripple / 2 if kind == "off" else -s.i_ripple / 2), 1e-3)
        progress(f"Switching transient ({'ngspice' if sim == 'ngspice' else 'built-in'}): turn-{kind} edge at "
                 f"{i_l:.3g} A...")
        c, ctrl, direction = build(board, s, result.copper, kind, i_l, warnings, vendor=sim == "ngspice")
        m = ctrl.fet()
        other_pos = "lower" if ctrl.role == "upper" else "upper"
        others = s.by_role(other_pos)
        other = next((d for d in others if d.kind == "fet"), others[0] if others else None)
        # the edge's length: the gate charging through the drive, or the switch's own times
        g = next((g for g in s.drives if s.device(g.fet) is ctrl), None)
        if g is not None and not ctrl.internal:
            rg = (g.rout if kind == "on" else (g.rout_off or g.rout)) + m.rg + sum(
                s.device(k).model.get("r", 0.0) for k in g.chain if s.device(k))
            t_sw = 6 * rg * m.ciss + 4 * (m.crss * 4) * s.v_rail * rg / max(g.v_drive - m.vth, 0.5)
        else:
            t_sw = 1.5 * ((m.t_rise if kind == "on" else m.t_fall) or M.INTERNAL_EDGE)
        t_sw = min(max(t_sw, 2e-9), 300e-9)
        t_ring = min(6 * q / (math.pi * f_ring), 400e-9)
        t_end = T0 + 2 * t_sw + t_ring + 8 / f_ring
        h = min(1 / (f_ring * 40), t_sw / 60)
        h = min(max(h, 5e-12), 0.2e-9)
        if t_end / h > 12000:
            h = t_end / 12000
        t, Vf, Jf = _simulate(c, t_end, h, sim)
        pr = c.probes
        pc = pr[ctrl.key]
        v_ctrl = (Vf(pc["d"]) - Vf(pc["s"])) * pc["pv"]
        i_ctrl = Jf(pc["i"]) * pc["pi"]
        waves = {f"V_DS {ctrl.key}": v_ctrl, f"I_D {ctrl.key}": i_ctrl}
        if pc["g"]:
            waves[f"V_GS {ctrl.key}"] = (Vf(pc["g"]) - Vf(pc["s"])) * pc["pv"]
        vgs_o, i_o, vth_o = None, None, float("nan")
        if other is not None and other.key in pr:
            po = pr[other.key]
            v_o = (Vf(po["d"]) - Vf(po["s"])) * po["pv"]
            i_o = Jf(po["i"]) * po["pi"]
            lab_v = f"V_KA {other.key}" if other.kind == "diode" else f"V_DS {other.key}"
            waves[lab_v] = v_o
            # current through the other device, positive in its conducting direction (body diode /
            # rectifier forward = source -> drain for a FET)
            i_fwd = -i_o
            waves[f"I {other.key} (forward)"] = i_fwd
            if po["g"]:
                vgs_o = (Vf(po["g"]) - Vf(po["s"])) * po["pv"]
                waves[f"V_GS {other.key}"] = vgs_o
                vth_o = other.fet().vth
        else:
            v_o, i_fwd = None, None
        vb = v_ctrl if kind == "off" else (v_o if v_o is not None else v_ctrl)
        met = _metrics(t, vb, s.v_rail, i_ctrl, i_l, v_ctrl, 1.0, vgs_o, vth_o,
                       i_fwd if i_fwd is not None else None, T0)
        met["I"] = i_l
        met["blocking"] = ctrl.key if kind == "off" else (other.key if other else ctrl.key)
        e = Edge(kind, ctrl.key, other.key if other else "", i_l, t=t, v_sw=vb, waves=waves, metrics=met,
                 warnings=warnings)
        edges.append(e)
    e_on = next(e for e in edges if e.kind == "on").metrics["E_uJ"]
    e_off = next(e for e in edges if e.kind == "off").metrics["E_uJ"]
    summary = dict(direction=s.current_dir(board), simulator=sim, E_on_uJ=e_on, E_off_uJ=e_off,
                   P_sw_W=(e_on + e_off) * 1e-6 * s.f_sw, f_sw=s.f_sw)
    return edges, summary
