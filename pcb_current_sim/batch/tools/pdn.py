"""PDN in the batch runner.

[test.setup] = the PDN window's setup: rail, returns, decaps (with their models), the regulator
(R + L or its fitted loop), loads with their load steps, target, frequency range and mesh.
"""
import csv
import math

import numpy as np

from . import Metric, Outcome

TITLE = "PDN (AC)"

METRICS = {
    "z_ratio": Metric("", "worst |Z| / the target impedance, up to the target's frequency (<= 1 passes)", "load"),
    "z_worst_mOhm": Metric("mΩ", "worst |Z| within the target's band", "load"),
    "z_target_mOhm": Metric("mΩ", "the target impedance (V × deviation / ΔI)"),
    "droop_mV": Metric("mV", "deepest dip of the rail at a load during the load steps", "load"),
    "rise_mV": Metric("mV", "highest rise of the rail at a load during the load steps", "load"),
    "step_ratio": Metric("", "largest |ΔV| during the load steps / the allowed deviation (<= 1 passes)", "load"),
    "decap_srf_MHz": Metric("MHz", "a decap's series resonance with its mounting", "decap"),
    "decap_mount_nH": Metric("nH", "a decap's return-side mounting inductance", "decap"),
    "decap_dist_mm": Metric("mm", "a decap's distance to the nearest load", "decap"),
    "layout_warnings": Metric("", "number of layout findings marked as warnings (Layout tab)"),
    "peak_mOhm": Metric("mΩ", "highest |Z| at a load over the whole sweep (incl. plane resonances)", "load"),
}

OUTPUTS = {
    "csv": "impedance.csv (|Z| per load: complete, + decaps, board alone), decaps.csv, layout.csv, steps.csv",
    "png": "impedance.png per load and steps.png",
    "touchstone": "the complete rail between the loads (.sNp, 50 Ω)",
    "touchstone_board": "the bare board with a port at every load, decap and the regulator (.sNp, 50 Ω)",
}


def from_scenario(scen, board, comps):
    d = (scen.get("pdn") or {}).get("setup")
    if not d or not d.get("rail"):
        return []
    return [(f"PDN {'+'.join(d['rail'])}", d)]


def run(ctx):
    from ...pdn import analysis as PA
    from ...pdn import model as PM
    s = PM.Setup.from_dict(ctx.test.setup)
    if not s.rail:
        raise ValueError("[test.setup] needs the rail (power net)")
    if not s.returns:
        raise ValueError("[test.setup] needs returns (the return / ground nets)")
    r = PA.run(ctx.board, ctx.stackup, s, progress=ctx.log)
    out = Outcome(warnings=list(r.warnings))
    v = out.values
    v["z_target_mOhm"] = r.z_target * 1e3
    v["z_ratio"] = {c["load"]: c["ratio"] for c in r.checks}
    v["z_worst_mOhm"] = {c["load"]: c["worst_z"] * 1e3 for c in r.checks}
    v["peak_mOhm"] = {n: float(np.abs(r.Z[:, j, j]).max()) * 1e3 for j, n in enumerate(r.loads)}
    if r.wave_checks:
        v["droop_mV"] = {c["load"]: -c["dip"] * 1e3 for c in r.wave_checks}
        v["rise_mV"] = {c["load"]: c["peak"] * 1e3 for c in r.wave_checks}
        v["step_ratio"] = {c["load"]: max(-c["dip"], c["peak"]) / c["allowed"] if c["allowed"] > 0 else math.inf
                           for c in r.wave_checks}
    v["decap_srf_MHz"] = {d["ref"]: d["f_srf"] / 1e6 for d in r.decaps if d.get("enabled") and "f_srf" in d
                          and math.isfinite(d["f_srf"])}
    v["decap_mount_nH"] = {d["ref"]: d.get("l_mount", 0.0) * 1e9 for d in r.decaps if d.get("enabled")}
    v["decap_dist_mm"] = {d["ref"]: d["dist_mm"] for d in r.decaps if d.get("enabled") and math.isfinite(d["dist_mm"])}
    v["layout_warnings"] = float(sum(1 for i in r.layout if i.level == "warn"))
    from ...suite.library import fmt_si
    out.lines.append(f"Rail {', '.join(s.rail)} over {', '.join(s.returns)}: target {fmt_si(r.z_target, 'Ω')} up to "
                     f"{fmt_si(r.f_target, 'Hz')}; {r.nodes} nodes at {r.pitch:.2f} mm, {r.seconds:.1f} s.")
    for c in r.checks:
        out.lines.append(f"{'✔' if c['ok'] else '✖'} {c['load']}: worst {fmt_si(c['worst_z'], 'Ω')} at "
                         f"{fmt_si(c['worst_f'], 'Hz')} ({100 * c['ratio']:.0f} % of the target)")
    for c in r.wave_checks:
        out.lines.append(f"{'✔' if c['ok'] else '✖'} {c['load']} load steps: dips {-c['dip'] * 1e3:.4g} mV, rises "
                         f"{c['peak'] * 1e3:.4g} mV (allowed ± {c['allowed'] * 1e3:.4g} mV)")
    for i in r.layout:
        if i.level == "warn":
            out.lines.append(f"⚠ {i.text}")
    _outputs(ctx, s, r, out)
    return out


def _outputs(ctx, s, r, out):
    from ...pdn import analysis as PA
    if ctx.want("csv"):
        with open(ctx.path("impedance.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["f_Hz"] + [f"{n} {k} [ohm]" for n in r.loads for k in ("complete", "decaps", "board")])
            for k, f in enumerate(r.f):
                w.writerow([f"{f:.6g}"] + [f"{abs(Z[k, j, j]):.6g}" for j in range(len(r.loads))
                                           for Z in (r.Z, r.Z_caps, r.Z_bare)])
        ctx.wrote(out, "impedance.csv")
        cols = ["ref", "enabled", "name", "value", "c_eff", "esr", "esl", "l_mount", "mount", "f_srf", "dist_mm",
                "i_peak", "f_peak"]
        with open(ctx.path("decaps.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for d in r.decaps:
                w.writerow([d.get(c, "") for c in cols])
        ctx.wrote(out, "decaps.csv")
        with open(ctx.path("layout.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["level", "kind", "refs", "x_mm", "y_mm", "text"])
            for i in r.layout:
                w.writerow([i.level, i.kind, " ".join(i.refs), *(i.at or ("", "")), i.text])
        ctx.wrote(out, "layout.csv")
        wv = r.waves
        if wv is not None:
            with open(ctx.path("steps.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_s"] + [f"I {n} [A]" for n in wv.loads] + [f"V at {n} [V]" for n in wv.loads])
                for k in range(len(wv.t)):
                    w.writerow([f"{wv.t[k]:.6g}"] + [f"{wv.i[n][k]:.6g}" for n in wv.loads]
                               + [f"{wv.v[n][k]:.6g}" for n in wv.loads])
            ctx.wrote(out, "steps.csv")
    if ctx.want("png"):
        _plots(ctx, s, r, out)
    if ctx.want("touchstone"):
        p, _ = PA.touchstone(ctx.board, ctx.stackup, s, r, ctx.path("rail.snp"), "loads")
        ctx.wrote(out, p.rsplit("/", 1)[-1])
    if ctx.want("touchstone_board"):
        p, _ = PA.touchstone(ctx.board, ctx.stackup, s, r, ctx.path("board.snp"), "board", progress=ctx.log)
        ctx.wrote(out, p.rsplit("/", 1)[-1])


def _plots(ctx, s, r, out):
    import os
    from matplotlib.figure import Figure
    for j, n in enumerate(r.loads):
        fig = Figure(figsize=(8, 4.5), constrained_layout=True)
        ax = fig.add_subplot()
        ax.loglog(r.f, np.abs(r.Z_bare[:, j, j]), color="0.6", lw=1, label="board alone")
        ax.loglog(r.f, np.abs(r.Z_caps[:, j, j]), "--", color="tab:blue", lw=1, label="+ decaps")
        ax.loglog(r.f, np.abs(r.Z[:, j, j]), color="tab:blue", lw=2, label="complete")
        band = r.f <= r.f_target
        ax.loglog(r.f[band], np.full(band.sum(), r.z_target), color="tab:red", label="target")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"|Z| at {n} [Ω]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        name = f"impedance_{_safe(n)}.png"
        fig.savefig(ctx.path(name), dpi=110)
        ctx.wrote(out, name)
        out.images.append(os.path.join(ctx.rel, name))
    wv = r.waves
    if wv is not None:
        fig = Figure(figsize=(8, 5), constrained_layout=True)
        a1, a2 = fig.add_subplot(211), fig.add_subplot(212, sharex=None)
        for n in wv.loads:
            a1.plot(wv.t * 1e6, wv.i[n], lw=1, label=n)
            a2.plot(wv.t * 1e6, wv.v[n] * 1e3, lw=1, label=n)
        allowed = s.target.v * s.target.ripple / 100 * 1e3
        for y in (allowed, -allowed):
            a2.axhline(y, color="tab:red", ls="--", lw=0.8)
        a1.set_ylabel("Load current change [A]")
        a2.set_ylabel("ΔV at the load [mV]")
        a2.set_xlabel("Time [µs]")
        for a in (a1, a2):
            a.grid(True, alpha=0.3)
            a.legend(fontsize=7)
        fig.savefig(ctx.path("steps.png"), dpi=110)
        ctx.wrote(out, "steps.png")
        out.images.append(os.path.join(ctx.rel, "steps.png"))


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
