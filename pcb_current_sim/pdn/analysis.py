"""PDN analysis: Z(f) at the loads (bare board, + decaps, + VRM), the target check, resonance peaks,
the decap table, what-if (each decap removed) and voltage maps at a frequency."""
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..suite.library import cap_factor, part_from_dict
from . import network as PN


@dataclass
class Result:
    f: np.ndarray
    loads: list
    Z: np.ndarray                # (F, P, P) complete: board + decaps + VRM
    Z_caps: np.ndarray           # board + decaps (no VRM)
    Z_bare: np.ndarray           # board alone
    I_dec: dict                  # ref -> (F, P): decap current per 1 A at each load
    I_vrm: np.ndarray
    z_target: float
    f_target: float
    decaps: list                 # rows (dicts), see decap_rows
    checks: list                 # per load: dict(load, ok, worst_f, worst_z, ratio)
    peaks: dict                  # load -> [(f, |Z|)] local maxima of the complete Z, worst first
    warnings: list = field(default_factory=list)
    nodes: int = 0
    pitch: float = 0.0
    seconds: float = 0.0
    net: object = None           # the network (for maps and what-if; not saved)
    waves: object = None         # transient.Waves of the loads' steps (None: not simulated)
    wave_checks: list = field(default_factory=list)   # per load: dip, peak, allowed, ok, worst contributor
    layout: list = field(default_factory=list)        # layout.Issue: what in the layout raises Z, and where


def peaks_of(f, z, n=6):
    """Local maxima of |Z| (anti-resonances), largest first."""
    k = np.flatnonzero((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])) + 1
    k = k[np.argsort(-z[k])][:n]
    return [(float(f[i]), float(z[i])) for i in k]


def target_check(f, z, zt, ft, name):
    band = f <= ft
    if not band.any():
        return dict(load=name, ok=True, worst_f=float(f[0]), worst_z=float(z[0]), ratio=0.0)
    r = z[band] / zt
    k = int(np.argmax(r))
    return dict(load=name, ok=bool(r[k] <= 1.0), worst_f=float(f[band][k]), worst_z=float(z[band][k]),
                ratio=float(r[k]))


def _dist_to_loads(board, s, d):
    pads = {p.id: p for p in board.pads}
    a = [pads[x].pos for x in d.pads if x in pads]
    b = [pads[x].pos for ld in s.loads if ld.enabled for x in ld.pads if x in pads]
    if not a or not b:
        return math.nan
    A, B_ = np.array(a), np.array(b)
    return float(np.min(np.hypot(A[:, None, 0] - B_[None, :, 0], A[:, None, 1] - B_[None, :, 1])))


def decap_rows(board, s, sweep, l_ret):
    """Per decap: C at the rail voltage, ESR, ESL, mounting L, series resonance, distance to the
    nearest load, the largest share of the first load's current it carries and the band where it
    carries at least a quarter of it."""
    rows = []
    for d in s.decaps:
        p = part_from_dict(d.part) if d.part else None
        row = dict(ref=d.ref, enabled=d.enabled, name=(p.name if p else ""), value=(p.value if p else 0.0))
        if p is not None:
            c = p.value * float(cap_factor(p, s.T, s.target.v))
            lr, how = l_ret.get(d.ref, (0.0, ""))
            esl = p.esl_eff()
            row.update(c_eff=c, esr=p.esr_eff(), esl=esl, l_mount=lr, mount=how,
                       f_srf=1 / (2 * math.pi * math.sqrt(max(esl + lr, 1e-13) * c)) if c > 0 else math.nan)
        row["dist_mm"] = _dist_to_loads(board, s, d)
        cur = sweep.I_dec.get(d.ref)
        if cur is not None and cur.shape[1]:
            a = np.abs(cur[:, 0])
            k = int(np.argmax(a))
            row.update(i_peak=float(a[k]), f_peak=float(sweep.f[k]))
            band = sweep.f[a >= 0.25]
            row["band"] = (float(band.min()), float(band.max())) if band.size else None
        rows.append(row)
    return rows


def run(board, stackup, s, progress=lambda m: None):
    t0 = time.time()
    net = PN.build(board, stackup, s, progress)
    caps = [d for d in s.decaps if d.enabled]
    l_ret = {d.ref: PN.decap_l_ret(board, stackup, s, d) for d in s.decaps}
    progress("Board + decaps + regulator…")
    full = PN.sweep(net, board, stackup, s, progress=progress, l_ret=l_ret)
    progress("Board + decaps…")
    capz = PN.sweep(net, board, stackup, s, vrm=False, progress=lambda m: None, l_ret=l_ret)
    progress("Board alone…")
    bare = PN.sweep(net, board, stackup, s, decaps=[], vrm=False, progress=lambda m: None, l_ret=l_ret)
    zt, ft = s.target.z, s.target.f_max
    checks, peaks = [], {}
    for j, name in enumerate(full.loads):
        z = np.abs(full.Z[:, j, j])
        checks.append(target_check(full.f, z, zt, ft, name))
        peaks[name] = peaks_of(full.f, z)
    warnings = list(net.warnings)
    if not s.vrm.pads or "V" not in net.terms:
        warnings.append("No regulator output (VRM) on the rail: Z at low frequency is the decaps' alone (it rises "
                        "as 1/f). Pick the regulator's output pad.")
    if not caps:
        warnings.append("No decap is switched on.")
    lam = 3e8 / math.sqrt(4.5) / s.f_max * 1e3                      # mm, in FR4
    if net.pitch > lam / 20:
        warnings.append(f"The mesh ({net.pitch:.2f} mm) is coarser than λ/20 at f max ({lam / 20:.2f} mm): plane "
                        "resonances near f max read low; raise the cell count or lower f max.")
    if s.f_max < ft:
        warnings.append(f"f max ({s.f_max:.3g} Hz) is below the target's 0.35 / t_rise ({ft:.3g} Hz): the target "
                        "is only checked up to f max.")
    for d in s.decaps:
        if d.enabled and f"C:{d.ref}" not in net.terms:
            warnings.append(f"{d.ref}: no rail copper under its pad; left out.")
    res = Result(full.f, full.loads, full.Z, capz.Z, bare.Z, full.I_dec, full.I_vrm, zt, ft,
                 decap_rows(board, s, full, l_ret), checks, peaks, warnings, net.n_nodes, net.pitch,
                 time.time() - t0, net)
    from . import layout as LA
    res.layout = LA.check(board, s, res, l_ret)
    if s.transient and any(ld.enabled for ld in s.loads):
        steps(board, stackup, s, res, l_ret, progress)
    res.seconds = time.time() - t0
    return res


def _settle(rat):
    """Time for the slowest significant part of a fitted response to die out (5 time constants)."""
    w = np.abs(rat.res / rat.poles)
    keep = w > 0.01 * w.max() if w.size else np.zeros(0, bool)
    if not keep.any():
        return 1e-4
    return float(np.clip(5.0 / np.min(np.abs(rat.poles.real[keep])), 20e-6, 50e-3))


def steps(board, stackup, s, res, l_ret, progress=lambda m: None):
    """The loads' PULSE steps over time (fills res.waves and res.wave_checks)."""
    from . import transient as TT
    net = res.net
    on = [ld for ld in s.loads if ld.enabled]
    f = np.logspace(1, math.log10(TT.f_top(on, max(s.f_max, 1e5))), 160)
    progress("Load steps: impedances for the time response…")
    up = PN.sweep(net, board, stackup, s, f=f, l_ret=l_ret, direction=1)
    two = s.vrm.kind == "loop" and s.vrm.model and s.vrm.model.get("down") and s.vrm.enabled
    dn = PN.sweep(net, board, stackup, s, f=f, l_ret=l_ret, direction=-1) if two else up
    progress("Load steps: fitting and simulating…")
    fu = TT.fit_matrix(f, up.Z)
    fd = TT.fit_matrix(f, dn.Z) if two else fu
    loads = [next(ld for ld in s.loads if ld.name == n) for n in up.loads]
    settle = max(_settle(fu[k][k]) for k in range(len(loads)))
    t_stop = s.t_stop or TT.auto_t_stop(loads, settle)
    w = TT.simulate(loads, fu, fd, t_stop, up.loads)
    res.waves = w
    allowed = s.target.v * s.target.ripple / 100
    res.wave_checks = []
    for at in w.loads:
        v = w.v[at]
        k = int(np.argmax(np.abs(v)))
        worst = max(w.loads, key=lambda frm: abs(w.parts[(at, frm)][k]))
        res.wave_checks.append(dict(load=at, dip=float(v.min()), peak=float(v.max()), allowed=allowed,
                                    ok=bool(np.abs(v).max() <= allowed), t_worst=float(w.t[k]), worst_from=worst,
                                    share=float(abs(w.parts[(at, worst)][k]) / max(abs(v[k]), 1e-15))))
    if w.fit_err > 0.05:
        res.warnings.append(f"The load-step waveforms come from a rational fit of Z(f) that is {100 * w.fit_err:.0f} % "
                            "off somewhere in the band: treat them as approximate (more frequency points help).")


def what_if(board, stackup, s, res, n_f=61, progress=lambda m: None):
    """Each enabled decap removed in turn: the worst rise of |Z| at the first load within the target band
    (ratio to the complete Z) and where.  Coarser frequency grid (n_f points)."""
    net = res.net
    f = np.logspace(math.log10(s.f_min), math.log10(min(s.f_max, max(res.f_target, s.f_min * 10))), n_f)
    base = PN.sweep(net, board, stackup, s, f=f)
    zb = np.abs(base.Z[:, 0, 0])
    on = [d.ref for d in s.decaps if d.enabled]
    out = {}
    for k, ref in enumerate(on):
        progress(f"Without {ref} ({k + 1} of {len(on)})…")
        r = PN.sweep(net, board, stackup, s, f=f, decaps=[x for x in on if x != ref])
        z = np.abs(r.Z[:, 0, 0])
        q = z / zb
        i = int(np.argmax(q))
        out[ref] = dict(rise=float(q[i]), f=float(f[i]), z=float(z[i]),
                        over=bool((z / res.z_target).max() > 1.0))
    return out


def node_voltages(board, stackup, s, res, f, load=0):
    """|V| (per 1 A at load `load`) at every mesh node at frequency f, with decaps and the VRM."""
    net = res.net
    caps = [d for d in s.decaps if d.enabled and f"C:{d.ref}" in net.terms]
    Y = net.y_mesh(f)
    add = np.zeros(net.n_nodes, complex)
    for d in caps:
        lr = PN.decap_l_ret(board, stackup, s, d)[0]
        add[net.terms[f"C:{d.ref}"]] += 1.0 / PN.decap_z(d, np.array([f]), s.target.v, s.T, lr)[0]
    if s.vrm.enabled and "V" in net.terms:
        add[net.terms["V"]] += PN.vrm_y(s.vrm, np.array([f]))[0]
    rhs = np.zeros(net.n_nodes, complex)
    rhs[net.terms[f"L:{res.loads[load]}"]] = 1.0
    V = splu((Y + sp.diags(add)).tocsc(), permc_spec="MMD_AT_PLUS_A").solve(rhs)
    return np.abs(V)


EXPORTS = {
    "loads": "Loads, complete rail (board + decaps + regulator)",
    "board": "Board only, a port at every load, decap and the regulator",
}


def touchstone(board, stackup, s, res, path, what="loads", z0=50.0, progress=lambda m: None):
    """Write the rail's impedance as a Touchstone S-parameter file (.s<n>p, RI, reference z0).
    'loads': Z between the loads with every enabled part (res.Z, the analysed frequencies);
    'board': the bare copper with a port at each load, decap (enabled or not) and the regulator
    output, for adding the parts in another tool. Returns the port names."""
    from ..matching import touchstone as TS
    import datetime
    net = res.net
    head = [f"PDN impedance of rail {', '.join(s.rail)} on {os.path.basename(getattr(board, 'path', '') or '') or 'the board'}",
            f"Written {datetime.datetime.now():%Y-%m-%d %H:%M} by the PCB Analysis Suite (PDN)",
            f"Returns: {', '.join(s.returns)}; all ports are against the return (ideal reference)"]
    if what == "loads":
        f, Z, names = res.f, res.Z, list(res.loads)
        head.append("Ports at the loads; decaps and the regulator are inside the network")
    elif what == "board":
        if net is None:
            raise ValueError("Analyse again: this result has no network.")
        ports = [f"L:{n}" for n in res.loads]
        ports += [f"C:{d.ref}" for d in s.decaps if f"C:{d.ref}" in net.terms]
        if "V" in net.terms:
            ports.append("V")
        progress(f"Board alone, {len(ports)} ports…")
        sw = PN.sweep(net, board, stackup, s, f=res.f, decaps=[], vrm=False, ports=ports, progress=progress)
        f, Z = sw.f, sw.Z
        names = [p[2:] if p[:2] in ("L:", "C:") else f"regulator output {s.vrm.ref}".strip() for p in ports]
        kinds = ["load" if p.startswith("L:") else "decap" if p.startswith("C:") else "regulator" for p in ports]
        names = [f"{n} ({k})" for n, k in zip(names, kinds)]
        head.append("Bare copper: decaps and the regulator are NOT included, connect them at their ports")
    else:
        raise ValueError(f"unknown export {what!r}")
    n = Z.shape[1]
    root, ext = os.path.splitext(path)
    if ext.lower() != f".s{n}p":
        path = root + f".s{n}p"
    TS.write(path, f, Z, z0, names, head)
    return path, names
