"""Frequency-domain analysis of a switching cell: hot-loop and gate-loop parasitics.

For each commutation (one position turns off, the current moves to the other):
  - loop L and R seen across the switch that turns off: board copper alone, and with the device
    packages and the capacitors' ESL / ESR (the capacitors' C shorted: it is the loop's
    inductance), with the energy share of every part of the copper and of each package;
  - the ringing: the impedance across that switch with its own capacitance (Coss, or Cj) and
    the board's stray capacitance; its peak gives the ringing frequency and Q.
For each FET with a known gate drive: the gate-loop inductance (driver pin -> gate resistor ->
gate, and back through the source to the driver's return), and the common-source inductance,
the mutual between that loop and the power loop through this FET (shared copper, package and
magnetic coupling): V_gs,induced = -L_cs di/dt.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import raster as rz
from ..extract.spice import fmt
from . import circuit as CI
from . import peec as P
from . import checks
from . import model as M
from .model import SMALL_CAP, Setup


@dataclass
class Result:
    setup: dict
    window: tuple
    pitch: float
    cells: int
    bars: int
    meshes: int
    seconds: float
    terminals: list                       # names
    commutations: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    cap_nets: list = field(default_factory=list)
    cap_pF: list = field(default_factory=list)      # mutual C between nets, pF (matrix as nested lists)
    freqs: np.ndarray = None
    maps: dict = field(default_factory=dict)        # commutation index -> {"J", "E", "e"}: per mesh node
    display: dict = field(default_factory=dict)     # {"grid", "data": {layer: (ny, nx, 2 x commutations)}}
    mesh: object = None                             # the mesh (not saved)
    pitch_min: float = 0.0                          # finest cell [mm] (adaptive mesh)
    grid: object = None
    layers: list = field(default_factory=list)
    spice: str = ""
    warnings: list = field(default_factory=list)
    # the copper for the switching transient: {"ports": [(a, b)], "ties": [(a, b)], "R", "L": (p, p) at
    # f_ring, "caps": [(a, b, C)], "inductor": (terminal, ref) or None, "ret_ref": terminal}
    copper: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)       # switching transient: transient.Edge per edge
    tran: dict = field(default_factory=dict)        # its summary (energies, loss)
    tran_error: str = ""


def term(ref, net):
    return f"{ref}:{net}"


def pin_terms(d):
    """{pin: terminal name} of a part: one terminal per (footprint, net); a Kelvin pin its own."""
    T = {pin: (term(d.pads[pin], net) if d.pads.get(pin) else term(d.ref, net))
         for pin, net in d.nets.items() if net}
    if d.kind == "fet" and d.kelvin and d.nets.get("S"):
        T["KS"] = term(d.ref, d.nets["S"]) + ":KS"
    return T


def _tidy(h):
    e = 10 ** math.floor(math.log10(h))
    for m in (1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10):
        if m * e >= h - 1e-12:
            return round(m * e, 6)
    return h


class _Copper:
    """Copper port impedances at sample frequencies, interpolated (R and L per entry, in log f)."""

    def __init__(self, net, f_samples, progress):
        self.net = net
        self.fs = np.array(f_samples)
        Rs, Ls = [], []
        for f in self.fs:
            progress(f"Solving the copper at {fmt(f)}Hz...")
            Z, _ = net.solve(f)
            Rs.append(Z.real)
            Ls.append(Z.imag / (2 * math.pi * f))
        self.R, self.L = np.array(Rs), np.array(Ls)
        self.L_hf = net.solve(0, hf=True)[0]

    def __call__(self, f):
        lf = np.log(np.clip(f, self.fs[0], self.fs[-1]))
        x = np.log(self.fs)
        k = min(max(np.searchsorted(x, lf) - 1, 0), len(x) - 2)
        t = (lf - x[k]) / (x[k + 1] - x[k])
        R = self.R[k] * (1 - t) + self.R[k + 1] * t
        L = self.L[k] * (1 - t) + self.L[k + 1] * t
        return R + 2j * math.pi * f * L


def _pads_of(board, ref, net):
    return [k for k, p in enumerate(board.pads) if p.ref == ref and p.net == net]


def run(board, stackup, setup, progress=lambda m: None, _coarse=False):
    """The analysis.  With setup.adaptive a coarse first pass finds where the loop's magnetic
    energy sits; the second pass meshes there (and at every part's pads) finely, 2x and 4x coarser
    further away, at the same cell budget.  _coarse: internal first pass (no display, no SPICE)."""
    t0 = time.time()
    s = setup if isinstance(setup, Setup) else Setup.from_dict(setup)
    warnings = []
    drives = [g for g in s.drives if s.device(g.fet) and s.device(g.fet).enabled]
    used_r = {k for g in drives for k in g.chain}
    # a gate resistor only belongs to the circuit while its FET's drive does (else it floats)
    devs = [d for d in s.devices if d.enabled and (d.role in ("upper", "lower", "cap", "link")
                                                   or d.role == "gate_r" and d.key in used_r)]
    errs = checks.errors(s)
    if errs:
        raise ValueError(errs[0].text)

    # terminals: one per (part, net)
    terms = {}
    for d in devs:
        T = pin_terms(d)
        for pin, pads in d.pins(board).items():
            if pin in T:
                terms[T[pin]] = pads
    for g in drives:
        for ref, net in (g.out, g.ret):
            terms[term(ref, net)] = _pads_of(board, ref, net)
    lref, _far = M.switch_inductor(board, s.sw_net)
    if lref:                    # where the inductor current leaves (enters) the switch node
        terms[term(lref, s.sw_net)] = _pads_of(board, lref, s.sw_net)
    missing = [t for t, pads in terms.items() if not pads]
    if missing:
        raise ValueError("No pads for " + ", ".join(missing) + " (check the nets of these parts).")
    nets = sorted({t.split(":")[1] for t in terms})
    xs = [board.pads[k].pos for pads in terms.values() for k in pads]
    ext = [board.pads[k].geom.bbox for pads in terms.values() for k in pads if board.pads[k].geom is not None]
    X0, Y0, X1, Y1 = board.bbox()
    window = (max(min(b[0] for b in ext) - s.margin, X0), max(min(b[1] for b in ext) - s.margin, Y0),
              min(max(b[2] for b in ext) + s.margin, X1), min(max(b[3] for b in ext) + s.margin, Y1))
    # pitch from the copper area in the window
    g0 = rz.make_grid(board, 0.25, margin=0.0, bbox=window)
    R0 = rz.rasterise(board, nets, g0)
    area = sum(float(lr.f.sum()) for lr in R0.layers.values()) * 0.25 ** 2
    # cells: the user's target, capped so the dense matrices fit in memory (about 1.8 bars and
    # 0.9 meshes per cell)
    cells = max(s.cells, 100)
    while cells > 500 and P.dense_bytes(1.8 * cells, 0.9 * cells) > P.MEM_SHARE * P.available_ram():
        cells = int(cells * 0.85)
    if cells < s.cells:
        warnings.append(f"Mesh cells lowered from {s.cells} to {cells} to fit in the free memory.")
    pitch = _tidy(min(max(math.sqrt(area / cells), 0.05), 1.0))
    level = None
    if s.adaptive and not _coarse:
        progress("Pass 1 of 2: a coarse mesh to find where the loop current crowds...")
        s0 = Setup.from_dict(s.to_dict())
        s0.adaptive, s0.cells = False, max(600, min(1500, cells // 3))
        r0 = run(board, stackup, s0, progress=lambda m: progress("  " + m), _coarse=True)
        level, pitch = _refinement(r0, board, terms, cells)
        progress("Pass 2 of 2: fine where the current crowds, coarser elsewhere...")
    progress(f"Meshing {len(nets)} nets in {window[2] - window[0]:.0f} x {window[3] - window[1]:.0f} mm at {pitch:g} mm"
             + (" (finest)..." if level else "..."))
    mesh = P.build(board, stackup, nets, window, pitch, terms, level=level)
    for _ in range(4):        # adaptive: meshing is cheap next to the solve, so hit the cell budget
        if level is None or 0.8 * cells <= mesh.n_nodes <= 1.05 * cells or pitch <= 0.05:
            break
        pitch = max(0.05, round(pitch * math.sqrt(mesh.n_nodes / (0.92 * cells)), 3))
        progress(f"  re-meshing at {pitch:g} mm ({mesh.n_nodes} nodes for {cells} cells)...")
        mesh = P.build(board, stackup, nets, window, pitch, terms, level=level)
    warnings += mesh.warnings
    # ports: per net and per connected piece of its copper (a Kelvin pin's trace, or any copper of
    # a net joined only through parts, is its own piece with its own reference terminal)
    comp = P.components(mesh)
    by_piece, by_net = {}, {}
    for t in terms:
        by_piece.setdefault((t.split(":")[1], int(comp[mesh.terminals[t]])), []).append(t)
    for (n_, _), ts in by_piece.items():
        by_net.setdefault(n_, []).extend(ts)
    # two terminals on one mesh node (small pads close together on a coarse mesh) have no copper
    # between them: they are tied by a short in the circuits, not given a port of zero length
    # (or two names for one pad: a part's own pin and a pad-to-pad element on it)
    ports, ties = [], []
    for ts in by_piece.values():
        first = {}                      # mesh node -> the first terminal on it
        for t in ts:
            n_ = mesh.terminals[t]
            if n_ in first:
                ties.append((t, first[n_]))
            else:
                first[n_] = t
        reps = list(first.values())
        ports += [(t, reps[0]) for t in reps[1:]]
    net = P.Network(mesh, ports, progress)
    fs = np.logspace(math.log10(s.f_min) - 2, math.log10(s.f_max), 6)
    cop = _Copper(net, fs, progress)
    progress("Capacitance between nets...")
    Cmax = P.capacitance(mesh, stackup)
    ref_of = {n: ts[0] for n, ts in by_net.items()}

    # ---- circuits ------------------------------------------------------------------
    def build(off, mode, caps_c=True, fet_caps=True):
        """off: position turning off ("upper"/"lower"); mode "board" (copper only, parts as
        shorts) or "full".  caps_c=False shorts the capacitors' C (loop inductance);
        fet_caps=False drops the switching parts' capacitances (pure inductances)."""
        c = CI.Circuit()
        if ports:
            c.block(ports, cop)
        for t, r_ in ties:
            c.short(t, r_)
        if mode == "full" and caps_c:
            for i, a in enumerate(mesh.nets):
                for j, b in enumerate(mesh.nets):
                    if i < j and a in ref_of and b in ref_of and -Cmax[i, j] > 0:
                        c.C(ref_of[a], ref_of[b], -Cmax[i, j], "board C")
        on = "lower" if off == "upper" else "upper"
        for d in devs:
            T = pin_terms(d)
            state = "off" if d.role == off else "on" if d.role == on else None
            if d.kind == "fet":
                if mode == "board":
                    if state == "on":
                        c.short(T["D"], T["S"])
                    continue
                m = d.fet()
                k_ = d.key
                dn, sn, gn = k_ + "#d", k_ + "#s", k_ + "#g"
                c.L(T["D"], dn, m.ld, f"{k_} package (drain)")
                c.L(T["S"], sn, m.ls, f"{k_} package (source)")
                if "KS" in T:          # Kelvin source: the die's source on its own pin
                    c.L(T["KS"], sn, m.lks, f"{k_} package (Kelvin)")
                if "G" in T:
                    c.L(T["G"], gn, m.lg, f"{k_} package (gate)", r=m.rg)
                cgs, cgd, cds = m.caps(s.v_rail)
                if state == "on":
                    c.R(dn, sn, m.rds_on, f"{k_} R_on")
                if "G" not in T:        # no gate pin (an IC's internal switch): Coss across it
                    if fet_caps and state == "off":
                        c.C(dn, sn, cgd + cds, f"{k_} Coss")
                elif fet_caps or state == "on":
                    c.C(gn, sn, cgs, f"{k_} Cgs")
                    c.C(gn, dn, cgd, f"{k_} Cgd")
                    if state == "off":
                        c.C(dn, sn, cds, f"{k_} Cds")
            elif d.kind == "diode":
                m = d.diode()
                fet_on = any(x.kind == "fet" for x in s.by_role(d.role)) and state == "on"
                if mode == "board":
                    if state == "on" and not fet_on:
                        c.short(T["A"], T["K"])
                    continue
                an = d.key + "#a"
                c.L(T["A"], an, m.l_pkg, f"{d.key} package")
                if state == "on" and not fet_on:
                    c.R(an, T["K"], m.rd, f"{d.key} R_d")
                elif fet_caps or state == "on":
                    c.C(an, T["K"], m.cj_at(s.v_rail), f"{d.key} Cj")
            elif d.kind == "cap":
                a, b = T["1"], T["2"]
                if mode == "board":
                    c.short(a, b)
                    continue
                p = d.part()
                mid = d.ref + "#c"
                c.L(a, mid, p.esl_eff(), f"{d.ref} ESL", r=p.esr_eff())
                # loop L: a capacitor's C is a short at the ringing frequency, but not a small one
                # (a 12 pF filter cap shorted would be a near-zero-inductance path round the loop)
                if caps_c or p.value < SMALL_CAP:
                    from ..suite.library import cap_factor
                    c.C(mid, b, p.value * float(cap_factor(p, 25.0, s.v_rail)), f"{d.ref} C")
                else:
                    c.short(mid, b)
            elif d.kind == "res":
                a, b = T["1"], T["2"]
                if mode == "board":
                    c.short(a, b)
                else:
                    c.L(a, b, d.model.get("ls", 0.5e-9), f"{d.key} package", r=d.model.get("r", 0.0))
        for g in drives:
            o, r = term(*g.out), term(*g.ret)
            if mode == "board":
                c.short(o, r)
            else:
                c.R(o, r, g.rout, f"{g.out[0]} driver")
        return c

    def port_of(off, mode):
        """Measurement port across the main part turning off (a FET if there is one)."""
        parts = s.by_role(off)
        d = next((x for x in parts if x.kind == "fet"), parts[0])
        T = pin_terms(d)
        if d.kind == "fet":
            if mode == "board":
                return d, (T["D"], T["S"])
            return d, (d.key + "#d", d.key + "#s")
        if mode == "board":
            return d, (T["A"], T["K"])
        return d, (d.key + "#a", T["K"])

    freqs = np.logspace(math.log10(s.f_min), math.log10(s.f_max), 241)
    res = Result(s.to_dict(), window, pitch, int(sum((mesh.cell_node[L] >= 0).sum() for L in mesh.layers)),
                 mesh.n_branches, net.n_meshes, 0.0, list(terms), freqs=freqs, grid=mesh.grid,
                 layers=list(mesh.layers), warnings=warnings)
    active = s.active()
    for off in ("lower", "upper"):
        on = "lower" if off == "upper" else "upper"
        progress(f"Commutation: {off} position turns off...")
        d_off, pb = port_of(off, "board")
        _, pf = port_of(off, "full")
        cb = build(off, "board")
        cl = build(off, "full", caps_c=False, fet_caps=False)
        cr = build(off, "full")
        Zb = cb.zmatrix(freqs, [pb])[:, 0, 0]
        Zl = cl.zmatrix(freqs, [pf])[:, 0, 0]
        Zr = cr.zmatrix(freqs, [pf])[:, 0, 0]
        w = 2 * np.pi * freqs
        pk = CI.peak(freqs, Zr)
        f_eval = pk[0] if pk else min(100e6, s.f_max)
        k = int(np.argmin(np.abs(freqs - f_eval)))
        Lt, Lb_ = Zl.imag[k] / w[k], Zb.imag[k] / w[k]
        # energy breakdown at f_eval: copper (from its port currents) + named inductors
        V = cl.solve(f_eval, {pf: 1.0})
        Zp = cop(f_eval)
        vp = np.array([V(a) - V(b) for a, b in ports])
        Ip = np.linalg.solve(Zp, vp)
        _, X = net.solve(f_eval)
        Ib = net.currents(X, Ip)
        e = net.energy(Ib)
        parts = {}
        for kk in range(mesh.n_branches):
            nm = mesh.nets[mesh.br_net[kk]]
            lab = f"{nm} vias" if mesh.br_kind[kk] == 2 else f"{nm} {mesh.layers[mesh.br_layer[kk]]}"
            parts[lab] = parts.get(lab, 0.0) + e[kk]
        for lab, a, b, Lx in cl.inductors:
            el = next(x for x in cl.elements if x[3] == lab)
            I = (V(a) - V(b)) / complex(el[2](np.array(f_eval)))
            parts[lab] = parts.get(lab, 0.0) + Lx * abs(I) ** 2
        tot = sum(parts.values())
        brk = sorted(((lab, v) for lab, v in parts.items() if abs(v) > 1e-3 * abs(tot)), key=lambda t: -t[1])
        c_eff = 1 / ((2 * math.pi * pk[0]) ** 2 * Lt) if pk else float("nan")
        # di/dt of this edge: the off switch's current fall time, or (a diode turning off) the rise
        # time of the switch taking the current; overshoot ≈ L di/dt
        didt, why = float("nan"), ""
        if s.i_switch > 0:
            if d_off.kind == "fet" and d_off.fet().t_fall > 0:
                didt, why = s.i_switch / d_off.fet().t_fall, f"{d_off.key} t_f"
            else:
                d_on = next((x for x in s.by_role(on) if x.kind == "fet" and x.fet().t_rise > 0), None)
                if d_on is not None:
                    didt, why = s.i_switch / d_on.fet().t_rise, f"{d_on.key} t_r"
        res.commutations.append(dict(
            off=off, on=on, active=off in active,
            title=f"{', '.join(x.key for x in s.by_role(off))} turn{'s' if len(s.by_role(off)) == 1 else ''} off, "
                  f"{', '.join(x.key for x in s.by_role(on))} take{'s' if len(s.by_role(on)) == 1 else ''} the current",
            port=d_off.ref, f_eval=f_eval,
            L_total_nH=Lt * 1e9, L_board_nH=Lb_ * 1e9, L_parts_nH=(Lt - Lb_) * 1e9,
            R_mOhm=Zl.real[k] * 1e3, f_ring=pk[0] if pk else float("nan"), Q=pk[1] if pk else float("nan"),
            Z_peak=pk[2] if pk else float("nan"), C_eff_pF=c_eff * 1e12,
            didt_A_ns=didt * 1e-9, V_over=Lt * didt, didt_from=why,
            breakdown=[(lab, v * 1e9) for lab, v in brk],
            Z_board=Zb, Z_loop=Zl, Z_ring=Zr))
        res.maps[len(res.commutations) - 1] = _node_maps(mesh, Ib, e)

    # ---- gate loops ------------------------------------------------------------------
    for g in drives:
        d = s.device(g.fet)
        off = d.role
        c = build(off, "full", caps_c=False, fet_caps=False)
        m = d.fet()
        pw = (d.key + "#d", d.key + "#s")
        gt = (d.key + "#g", d.key + "#s")
        Z = c.zmatrix(freqs, [pw, gt])
        w = 2 * np.pi * freqs
        k = int(np.argmin(np.abs(freqs - min(100e6, s.f_max))))
        Lg, Lcs, Rg = Z[k, 1, 1].imag / w[k], Z[k, 0, 1].imag / w[k], Z[k, 1, 1].real
        cgs, cgd, _ = m.caps(s.v_rail)
        ciss = cgs + cgd
        fg = 1 / (2 * math.pi * math.sqrt(Lg * ciss)) if Lg > 0 else float("nan")
        if Rg > 1e3:
            warnings.append(f"The gate loop of {d.key} is open (R {Rg:.3g} Ω): the series parts do not join "
                            f"{g.out[0]}'s pin on {g.out[1]} to the gate; check the drive's series parts.")
        res.gates.append(dict(fet=d.key, driver=g.out[0], r_ext=", ".join(g.chain), L_gate_nH=Lg * 1e9, L_cs_nH=Lcs * 1e9,
                              R_gate=Rg, f_gate=fg, Q_gate=math.sqrt(Lg / ciss) / Rg if Rg > 0 and Lg > 0 else float("nan"),
                              Z=Z))
    for d in s.devices:
        if d.enabled and d.kind in ("fet", "diode") and d.model.get("assumed", False):
            warnings.append(f"{d.key}: generic {d.kind.upper() if d.kind == 'fet' else 'diode'} values; enter its "
                            "datasheet capacitances and package inductances.")
    no_drive = [d.key for d in s.devices if d.enabled and d.kind == "fet" and not d.internal
                and not any(s.device(g.fet) is d for g in drives)]
    if no_drive:
        warnings.append("No gate drive found for " + ", ".join(no_drive) + ": its gate loop is not analysed.")
    if pitch > 0.6:
        warnings.append(f"Coarse mesh ({pitch:g} mm): raise the cell count for narrow copper.")
    gmin = min((mesh.gap[i, j] for i in range(len(mesh.layers)) for j in range(len(mesh.layers)) if i != j),
               default=1.0) * 1e3
    if pitch > 3 * gmin:
        warnings.append(f"The mesh ({pitch:g} mm) is coarse against the thinnest dielectric ({gmin:.2f} mm): "
                        "loop L over a close plane may read a few % high.")
    res.cap_nets = list(mesh.nets)
    res.cap_pF = (-Cmax * 1e12).tolist()
    res.mesh, res.pitch_min = mesh, pitch
    if not _coarse:
        progress("Map...")
        res.display = _display(board, mesh, nets, window, res.maps)
        res.spice = _spice(res, ports, cop, Cmax, mesh, ref_of, res.commutations[0]["f_eval"], board)
    act = [c for c in res.commutations if c["active"]] or res.commutations
    f_t = act[0]["f_ring"] if math.isfinite(act[0]["f_ring"]) else act[0]["f_eval"]
    Zt = cop(f_t)
    caps_t = [(ref_of[a], ref_of[b], -Cmax[i, j]) for i, a in enumerate(mesh.nets) for j, b in enumerate(mesh.nets)
              if i < j and a in ref_of and b in ref_of and -Cmax[i, j] > 0]
    res.copper = dict(ports=list(ports), ties=list(ties), R=Zt.real, L=Zt.imag / (2 * math.pi * f_t), f=f_t,
                      caps=caps_t, inductor=(term(lref, s.sw_net), lref) if lref else None,
                      ret_ref=ref_of.get(s.ret_net), rail_ref={n: ref_of[n] for n in M.rail_nets(s) if n in ref_of})
    res.seconds = time.time() - t0
    return res


def _node_maps(mesh, I, e):
    """Per mesh node: |J| (A/mm per A of loop current), magnetic energy share e (nH) and its density
    E (nH/mm²), from the bars that meet there (half of each bar to each end)."""
    N = mesh.n_nodes
    jx, jy, en, area = (np.zeros(N) for _ in range(4))
    ip = mesh.br_kind != 2
    a, b = mesh.br_a[ip], mesh.br_b[ip]
    Jb = np.abs(I[ip]) / (mesh.br_w[ip] * 1e3) / 2
    sgn = np.sign(np.real(I[ip]) + 1e-30)
    for nd in (a, b):
        np.add.at(jx, nd, Jb * sgn * mesh.br_ux[ip])
        np.add.at(jy, nd, Jb * sgn * mesh.br_uy[ip])
    for nd in (mesh.br_a, mesh.br_b):
        np.add.at(en, nd, e * 1e9 / 2)
    h2 = mesh.grid.h ** 2
    for L in mesh.layers:
        m = mesh.cell_node[L] >= 0
        np.add.at(area, mesh.cell_node[L][m], mesh.cell_f[L][m] * h2)
    wsel = np.flatnonzero(mesh.br_kind == 3)
    wa = mesh.br_len[wsel] * mesh.br_w[wsel] * 1e6 / 2
    np.add.at(area, mesh.br_a[wsel], wa)
    np.add.at(area, mesh.br_b[wsel], wa)
    return dict(J=np.hypot(jx, jy), e=en, E=en / np.maximum(area, 1e-9))


DISPLAY_PX = 700           # display grid: about this many cells along the longer side


def _display(board, mesh, nets, window, maps):
    """The node values interpolated (linearly, per net and layer) onto a fine raster of the real
    copper, for a map that shows the layout's shapes.  NaN where there is no copper of the nets."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError
    W, H = window[2] - window[0], window[3] - window[1]
    pd = max(0.02, max(W, H) / DISPLAY_PX)
    g = rz.make_grid(board, pd, margin=0.0, bbox=window)
    R = rz.rasterise(board, nets, g)
    keys = sorted(maps)
    vals = np.column_stack([maps[k][q] for k in keys for q in ("J", "E")])
    X, Y = g.centres()
    # a part's pads are one node at the pad centre carrying the sum of the current entering at its
    # edges: left out, so the pads take the values of the copper around them
    is_term = np.zeros(mesh.n_nodes, bool)
    is_term[list(mesh.terminals.values())] = True
    out = {}
    for li, L in enumerate(mesh.layers):
        lr = R.layers[L]
        img = np.full(g.shape + (vals.shape[1],), np.nan, np.float32)
        for ni, net in enumerate(R.nets):
            tgt = (lr.f >= 0.3) & (lr.net == ni)
            if not tgt.any() or net not in mesh.nets:
                continue
            sel = np.flatnonzero((mesh.node_layer == li) & (mesh.node_net == mesh.nets.index(net)) & ~is_term)
            if not sel.size:
                continue
            pts, v = mesh.node_xy[sel], vals[sel]
            q = np.column_stack([X[tgt], Y[tgt]])
            near = NearestNDInterpolator(pts, v)(q)
            try:
                lin = LinearNDInterpolator(pts, v)(q) if len(sel) >= 4 else near
                lin = np.where(np.isnan(lin), near, lin)
            except (QhullError, ValueError):
                lin = near
            img[tgt] = lin
        out[L] = img
    return dict(grid=g, data=out, keys=[(k, q) for k in keys for q in ("J", "E")])


def _refinement(r0, board, terms, cells):
    """(level function, finest pitch) from a coarse first pass: cells holding 80 % of each
    commutation's magnetic energy, and every part's pads, are the hot region; on it the mesh is
    fine (1), 2x within 1.5 mm, 4x beyond.  The finest pitch fits the cell budget."""
    from scipy.spatial import cKDTree
    m = r0.mesh
    pts = []
    for k, mp in r0.maps.items():
        e, E = mp["e"], mp["E"]
        order = np.argsort(-E)
        cum = np.cumsum(np.clip(e[order], 0, None))
        if cum[-1] <= 0:
            continue
        hot = order[:int(np.searchsorted(cum, 0.8 * cum[-1])) + 1]
        pts.append(m.node_xy[hot])
    for pads in terms.values():
        for k in pads:
            x0, y0, x1, y1 = board.pads[k].geom.bbox
            xs, ys = np.meshgrid(np.linspace(x0, x1, 4), np.linspace(y0, y1, 4))
            pts.append(np.column_stack([xs.ravel(), ys.ravel()]))
    tree = cKDTree(np.vstack(pts))
    r1 = 0.75 * m.grid.h                   # the hot coarse cells themselves (and the pads)
    r2 = r1 + 1.5

    def level(X, Y):
        d, _ = tree.query(np.column_stack([np.ravel(X), np.ravel(Y)]))
        return np.where(d < r1, 1, np.where(d < r2, 2, 4)).reshape(np.shape(X))
    # copper area at each level, from the coarse cells
    A = {1: 0.0, 2: 0.0, 4: 0.0}
    g = m.grid
    for L in m.layers:
        jj, ii = np.nonzero(m.cell_node[L] >= 0)
        lv = level(g.x0 + (ii + 0.5) * g.h, g.y0 + (jj + 0.5) * g.h)
        a = m.cell_f[L][jj, ii] * g.h ** 2
        for v in A:
            A[v] += float(a[lv == v].sum())
    h = math.sqrt((A[1] + A[2] / 4 + A[4] / 16) / (0.85 * cells))
    return level, _tidy(min(max(h, 0.05), 1.0))


def _ident(t):
    return "".join(c if c.isalnum() else "_" for c in t)


def _spice(res, ports, cop, Cmax, mesh, ref_of, f_eval, board):
    """Coupled-inductor model of the copper at f_eval: one R + L branch per port (terminal to its
    net's reference terminal), K couplings, and the board capacitance between nets."""
    Zp = cop(f_eval)
    w = 2 * math.pi * f_eval
    Lm = Zp.imag / w
    Rm = Zp.real
    nm = lambda t: _ident(t)              # noqa: E731
    out = ["* Switching-loop copper extracted by kicad-analysis-suite",
           f"* board {board.path.rsplit('/', 1)[-1] if board.path else ''}; R and L at {fmt(f_eval)}Hz "
           "(the first commutation's ringing frequency)",
           "* Nodes are the parts' pins: <ref>_<net>.  Each branch runs from a pin to the reference pin of its net;",
           "* K lines couple them (shared copper and magnetic coupling).  Shared resistance between branches is",
           "* left out (small against wL at these frequencies).  Board C between nets at their reference pins.",
           f".subckt LOOP_COPPER {' '.join(sorted({nm(t) for t in res.terminals}))}"]
    for k, (t, r) in enumerate(ports):
        out.append(f"R_p{k + 1} {nm(t)} x{k + 1} {fmt(max(Rm[k, k], 1e-6))}")
        out.append(f"L_p{k + 1} x{k + 1} {nm(r)} {fmt(max(Lm[k, k], 1e-15))}")
    for i in range(len(ports)):
        for j in range(i + 1, len(ports)):
            kc = Lm[i, j] / math.sqrt(max(Lm[i, i] * Lm[j, j], 1e-40))
            if abs(kc) > 1e-4:
                out.append(f"K_{i + 1}_{j + 1} L_p{i + 1} L_p{j + 1} {min(max(kc, -0.9999), 0.9999):.5f}")
    for i, a in enumerate(mesh.nets):
        for j, b in enumerate(mesh.nets):
            if i < j and a in ref_of and b in ref_of and -Cmax[i, j] > 1e-16:
                out.append(f"C_{_ident(a)}_{_ident(b)} {nm(ref_of[a])} {nm(ref_of[b])} {fmt(-Cmax[i, j])}")
    out.append(".ends")
    return "\n".join(out) + "\n"
