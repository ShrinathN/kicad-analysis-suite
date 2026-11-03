"""Matching & Tolerance in the batch runner.

[test.setup] = the network as the Matching window saves it (source, elements, load, bands, sweep,
Monte Carlo size, distribution, temperature), plus an optional table
    [test.setup.analyses]  corners = true, mc = true, worst = true, sensitivity = false
"""
import csv
import math

import numpy as np

from . import Metric, Outcome

TITLE = "Matching & Tolerance"
ANALYSES = {"corners": True, "mc": True, "worst": True, "sensitivity": False}

METRICS = {
    "yield_pct": Metric("%", "Monte Carlo yield: samples meeting every band's specs"),
    "yield_corners_pct": Metric("%", "share of the tolerance corners meeting every band's specs"),
    "band_yield_pct": Metric("%", "Monte Carlo yield of one band", "band"),
    "rl_nominal_dB": Metric("dB", "nominal worst return loss in a band (higher is better)", "band"),
    "il_nominal_dB": Metric("dB", "nominal worst insertion loss in a band (lower is better)", "band"),
    "rl_worst_dB": Metric("dB", "worst-case return loss in a band over the tolerances (worst-case search)", "band"),
    "il_worst_dB": Metric("dB", "worst-case insertion loss in a band over the tolerances", "band"),
}

OUTPUTS = {
    "csv": "response.csv: nominal RL, IL, VSWR per frequency (+ Monte Carlo 5 / 50 / 95 % when run)",
    "png": "response.png: RL and IL against frequency with the bands",
}


def from_scenario(scen, board, comps):
    d = (scen.get("matching") or {}).get("setup")
    if not d or not d.get("elements"):
        return []
    return [("Matching network", dict(d, analyses=dict(ANALYSES)))]


def run(ctx):
    from ...matching import analysis as A
    from ...matching import model as M
    d = dict(ctx.test.setup)
    an = dict(ANALYSES, **(d.pop("analyses", None) or {}))
    s = M.Setup.from_dict(d)
    res = A.run(s, an["corners"], an["mc"], an["worst"], an["sensitivity"], progress=ctx.log)
    out = Outcome(warnings=list(res.warnings))
    v = out.values
    names = [b.name for b in s.bands]
    v["rl_nominal_dB"] = {n: float(res.nominal["band"][k]["rl_worst"]) for k, n in enumerate(names)}
    v["il_nominal_dB"] = {n: float(res.nominal["band"][k]["il_worst"]) for k, n in enumerate(names)}
    if res.mc is not None and s.bands:
        v["yield_pct"] = 100 * res.mc.yield_()
        v["band_yield_pct"] = {n: 100 * res.mc.yield_(k) for k, n in enumerate(names)}
    if res.corners is not None and s.bands:
        v["yield_corners_pct"] = 100 * res.corners.yield_()
    for w in res.worst:
        key = "rl_worst_dB" if w["metric"] == "rl" else "il_worst_dB"
        v.setdefault(key, {})[names[w["band"]]] = float(w["value"])
    for k in [k for k, x in v.items() if isinstance(x, dict)]:
        v[k] = {n: x for n, x in v[k].items() if math.isfinite(x)}
    out.lines = A.summary_lines(s, res)
    if ctx.want("csv"):
        nom = res.nominal
        with open(ctx.path("response.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            head = ["f_Hz", "RL_dB", "IL_dB", "VSWR"]
            pct = None
            if res.mc is not None and res.mc.n:
                pct = (np.nanpercentile(res.mc.rl, [5, 50, 95], axis=0), np.nanpercentile(res.mc.il, [5, 50, 95], axis=0))
                head += ["RL_p5", "RL_p50", "RL_p95", "IL_p5", "IL_p50", "IL_p95"]
            w.writerow(head)
            for k, f in enumerate(res.f):
                row = [f"{f:.6g}", f"{nom['rl'][k]:.5g}", f"{nom['il'][k]:.5g}", f"{nom['vswr'][k]:.5g}"]
                if pct is not None:
                    row += [f"{pct[0][j][k]:.5g}" for j in range(3)] + [f"{pct[1][j][k]:.5g}" for j in range(3)]
                w.writerow(row)
        ctx.wrote(out, "response.csv")
    if ctx.want("png"):
        import os
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 4.5), constrained_layout=True)
        a1 = fig.add_subplot()
        a1.plot(res.f / 1e6, res.nominal["rl"], label="return loss")
        a1.plot(res.f / 1e6, res.nominal["il"], label="insertion loss")
        for b in s.bands:
            a1.axvspan(b.f1 / 1e6, b.f2 / 1e6, color="tab:green", alpha=0.12)
        a1.set_xlabel("Frequency [MHz]")
        a1.set_ylabel("dB")
        a1.grid(True, alpha=0.3)
        a1.legend(fontsize=8)
        fig.savefig(ctx.path("response.png"), dpi=110)
        ctx.wrote(out, "response.png")
        out.images.append(os.path.join(ctx.rel, "response.png"))
    return out
