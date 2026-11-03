"""Switching Loop in the batch runner (one test per switching cell).

[test.setup] = one cell as the Switching Loop window saves it: switch node, supply and return nets,
parts and their models, gate drives, operating point, snubbers, mesh, transient and EMI options.
Values marked 'hard-switched' are of the commutation(s) the converter actually performs.
"""
import csv
import math

import numpy as np

from . import Metric, Outcome

TITLE = "Switching Loop"

METRICS = {
    "loop_L_nH": Metric("nH", "hot-loop inductance (copper + parts) of the hard-switched commutation"),
    "loop_L_board_nH": Metric("nH", "the copper's share of the hot-loop inductance"),
    "loop_R_mOhm": Metric("mΩ", "hot-loop resistance at the ringing frequency"),
    "ring_MHz": Metric("MHz", "switch-node ringing frequency (frequency analysis)"),
    "ring_Q": Metric("", "ringing Q (frequency analysis; with the snubbers)"),
    "V_over_est": Metric("V", "estimated overshoot L × di/dt at the given inductor current"),
    "loop_area_mm2": Metric("mm²", "hot loop's effective magnetic area (radiated EMI)"),
    "commutation_L_nH": Metric("nH", "hot-loop L of each commutation ('<parts> off')", "commutation"),
    "gate_L_nH": Metric("nH", "gate-loop inductance of a FET", "fet"),
    "L_cs_nH": Metric("nH", "common-source inductance of a FET (shared by gate and power loops)", "fet"),
    "gate_Q": Metric("", "gate-loop ringing Q of a FET", "fet"),
    "overshoot_V": Metric("V", "highest switch overshoot above the rail in the simulated edges"),
    "V_peak_V": Metric("V", "peak voltage across the blocking switch", "edge"),
    "dvdt_V_ns": Metric("V/ns", "switch-node dv/dt (10-90 %)", "edge"),
    "didt_A_ns": Metric("A/ns", "highest di/dt in the loop", "edge"),
    "E_uJ": Metric("µJ", "switching energy of an edge", "edge"),
    "P_sw_W": Metric("W", "switching loss (E_on + E_off) × f_sw"),
    "gate_margin_V": Metric("V", "smallest V_th − peak gate voltage of the off FET during the edges (< 0: it "
                                 "turns on: shoot-through risk)"),
    "rad_margin_dB": Metric("dB", "radiated EMI estimate: worst margin below the limit (negative = over)"),
    "cond_margin_dB": Metric("dB", "conducted common-mode EMI estimate: worst margin below the limit"),
}

OUTPUTS = {
    "csv": "commutations.csv (L, breakdown, ringing), gates.csv, edges.csv (switching waveforms, when simulated)",
    "png": "impedance.png (across the switch) and edges.png (switching waveforms)",
    "spice": "loop.lib: coupled-inductor model of the cell's copper",
}


def from_scenario(scen, board, comps):
    d = scen.get("loop") or {}
    cells = d.get("cells") or ([d["setup"]] if d.get("setup") else [])
    out = []
    for k, c in enumerate(cells):
        c = {key: v for key, v in c.items() if key != "baseline"}
        name = c.get("name") or c.get("sw_net") or f"cell {k + 1}"
        out.append((f"Loop {name}", c))
    return out


def _fin(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def run(ctx):
    from ...loop import analysis as A
    from ...loop import emi as EM
    from ...loop import model as M
    from ...loop import saved as SV
    from ...loop import transient as TR
    setup = M.Setup.from_dict({k: v for k, v in ctx.test.setup.items() if k != "baseline"})
    SV.apply_components(setup, ctx.components)
    r = A.run(ctx.board, ctx.stackup, setup, progress=ctx.log)
    if setup.transient and setup.i_switch > 0:
        try:
            r.edges, r.tran = TR.run(ctx.board, setup, r, progress=ctx.log)
            r.emi = EM.analyse(r, setup)
        except (ValueError, RuntimeError) as e:
            r.tran_error = str(e)
    out = Outcome(warnings=list(r.warnings) + ([f"Switching edges: {r.tran_error}"] if r.tran_error else []))
    v = out.values
    act = [c for c in r.commutations if c["active"]] or r.commutations
    if act:
        w = max(act, key=lambda c: c["L_total_nH"])
        v["loop_L_nH"] = w["L_total_nH"]
        v["loop_L_board_nH"] = w["L_board_nH"]
        v["loop_R_mOhm"] = _fin(w["R_mOhm"])
        v["ring_MHz"] = _fin(w["f_ring"] / 1e6)
        v["ring_Q"] = _fin(w["Q"])
        v["V_over_est"] = _fin(w["V_over"])
        v["loop_area_mm2"] = _fin(w.get("A_loop_mm2"))
    v["commutation_L_nH"] = {f"{_keys(setup, c['off'])} off": c["L_total_nH"] for c in r.commutations}
    v["gate_L_nH"] = {g["fet"]: g["L_gate_nH"] for g in r.gates}
    v["L_cs_nH"] = {g["fet"]: g["L_cs_nH"] for g in r.gates}
    v["gate_Q"] = {g["fet"]: q for g in r.gates if (q := _fin(g["Q_gate"])) is not None}
    if r.edges:
        v["V_peak_V"] = {e.kind: e.metrics.get("V_peak") for e in r.edges}
        v["dvdt_V_ns"] = {e.kind: _fin(e.metrics.get("dvdt_V_ns")) for e in r.edges}
        v["didt_A_ns"] = {e.kind: _fin(e.metrics.get("didt_A_ns")) for e in r.edges}
        v["E_uJ"] = {e.kind: _fin(e.metrics.get("E_uJ")) for e in r.edges}
        for key in ("V_peak_V", "dvdt_V_ns", "didt_A_ns", "E_uJ"):
            v[key] = {k: x for k, x in v[key].items() if x is not None}
        ov = [e.metrics.get("overshoot") for e in r.edges if _fin(e.metrics.get("overshoot")) is not None]
        if ov:
            v["overshoot_V"] = max(ov)
        gm = [e.metrics["gate_margin"] for e in r.edges if _fin(e.metrics.get("gate_margin")) is not None]
        if gm:
            v["gate_margin_V"] = min(gm)
        if r.tran:
            v["P_sw_W"] = _fin(r.tran.get("P_sw_W"))
    if r.emi:
        v["rad_margin_dB"] = _fin(r.emi.get("rad_margin"))
        v["cond_margin_dB"] = _fin(r.emi.get("cond_margin"))
    for c in r.commutations:
        out.lines.append(f"{'●' if c['active'] else '○'} {c['title']}: L {c['L_total_nH']:.2f} nH (copper "
                         f"{c['L_board_nH']:.2f}), ringing {c['f_ring'] / 1e6:.4g} MHz, Q {c['Q']:.3g}")
    for g in r.gates:
        out.lines.append(f"Gate loop {g['fet']}: {g['L_gate_nH']:.2f} nH, common-source {g['L_cs_nH']:.2f} nH")
    for e in r.edges or []:
        m = e.metrics
        out.lines.append(f"Edge {e.kind}: peak {m.get('V_peak', float('nan')):.4g} V, overshoot "
                         f"{m.get('overshoot', float('nan')):.3g} V, dv/dt {m.get('dvdt_V_ns', float('nan')):.3g} V/ns, "
                         f"E {m.get('E_uJ', float('nan')):.3g} µJ")
    out.lines.append(f"{r.cells} cells at {r.pitch:g} mm, {r.seconds:.1f} s")
    _outputs(ctx, setup, r, out)
    return out


def _keys(setup, role):
    return ", ".join(x.key for x in setup.by_role(role))


def _outputs(ctx, setup, r, out):
    import os
    if ctx.want("csv"):
        with open(ctx.path("commutations.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["commutation", "hard_switched", "L_total_nH", "L_board_nH", "L_parts_nH", "R_mOhm", "f_ring_Hz",
                        "Q", "C_eff_pF", "didt_A_ns", "V_over", "A_loop_mm2", "breakdown (label: nH)"])
            for c in r.commutations:
                w.writerow([c["title"], c["active"], c["L_total_nH"], c["L_board_nH"], c["L_parts_nH"], c["R_mOhm"],
                            c["f_ring"], c["Q"], c["C_eff_pF"], c["didt_A_ns"], c["V_over"], c.get("A_loop_mm2"),
                            "; ".join(f"{lab}: {x:.4g}" for lab, x in c["breakdown"])])
        ctx.wrote(out, "commutations.csv")
        with open(ctx.path("gates.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fet", "driver", "series", "L_gate_nH", "L_cs_nH", "R_gate", "f_gate_Hz", "Q_gate"])
            for g in r.gates:
                w.writerow([g["fet"], g["driver"], g["r_ext"], g["L_gate_nH"], g["L_cs_nH"], g["R_gate"], g["f_gate"],
                            g["Q_gate"]])
        ctx.wrote(out, "gates.csv")
        if r.edges:
            with open(ctx.path("edges.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                for e in r.edges:
                    names = list(e.waves)
                    w.writerow([f"edge {e.kind}", "t_s"] + names)
                    for k in range(len(e.t)):
                        w.writerow(["", f"{e.t[k]:.6g}"] + [f"{e.waves[n][k]:.6g}" for n in names])
            ctx.wrote(out, "edges.csv")
    if ctx.want("spice") and r.spice:
        with open(ctx.path("loop.lib"), "w") as fh:
            fh.write(r.spice)
        ctx.wrote(out, "loop.lib")
    if ctx.want("png"):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 4.5), constrained_layout=True)
        ax = fig.add_subplot()
        for c in r.commutations:
            ax.loglog(r.freqs, np.abs(c["Z_ring"]), label=f"{_keys(setup, c['off'])} off")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|Z| across the switch [Ω]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(ctx.path("impedance.png"), dpi=110)
        ctx.wrote(out, "impedance.png")
        out.images.append(os.path.join(ctx.rel, "impedance.png"))
        if r.edges:
            fig = Figure(figsize=(8, 2.6 * len(r.edges)), constrained_layout=True)
            for k, e in enumerate(r.edges):
                ax = fig.add_subplot(len(r.edges), 1, k + 1)
                ax.plot(e.t * 1e9, e.v_sw, lw=1, label="V across the blocking switch")
                ax2 = ax.twinx()
                if e.i_loop is not None:
                    ax2.plot(e.t * 1e9, e.i_loop, lw=1, color="tab:orange", label="switch current")
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
                ax.set_title(f"turn-{e.kind} of {e.switch}", fontsize=9)
                ax.set_xlabel("t [ns]")
                ax.set_ylabel("V")
                ax2.set_ylabel("A")
                ax.grid(True, alpha=0.3)
            fig.savefig(ctx.path("edges.png"), dpi=110)
            ctx.wrote(out, "edges.png")
            out.images.append(os.path.join(ctx.rel, "edges.png"))
