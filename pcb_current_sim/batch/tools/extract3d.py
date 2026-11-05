"""3D extraction in the batch runner.

[test.setup] = the 3D extraction window's setup: nets (conductors), ports ([{name, plus, minus}]
with pad ids), links, frequency range, cell pitch, filament limits (see em3d.pcb.Setup).
"""
import csv
import math

import numpy as np

from . import Metric, Outcome

TITLE = "3D extraction"

METRICS = {
    "R_dc_mOhm": Metric("mΩ", "DC resistance of a port", "port"),
    "L_dc_nH": Metric("nH", "inductance of a port with the DC current pattern", "port"),
    "R_fmax_mOhm": Metric("mΩ", "resistance of a port at f max (skin and proximity effect)", "port"),
    "L_fmax_nH": Metric("nH", "inductance of a port at f max", "port"),
    "R_ratio": Metric("", "R(f max) / R_dc of a port", "port"),
    "Q_max": Metric("", "highest Q = ωL / R of a port over the band", "port"),
    "k_max": Metric("", "largest |coupling factor| between two ports over the band (key: A-B)", "pair"),
    "filaments": Metric("", "number of filaments in the solve"),
}

OUTPUTS = {
    "spice": "em3d.lib: the wideband SPICE model (Foster R-L; controlled sources for several ports)",
    "csv": "impedance.csv: R and L of every port (and couplings) at every frequency",
}


def from_scenario(scen, board, comps):
    d = (scen.get("extract3d") or {}).get("setup") or {}
    if not d.get("ports"):
        return []
    return [("3D " + ", ".join(p.get("name", "P") for p in d["ports"]), d)]


def run(ctx):
    from ...em3d import analysis as A
    from ...em3d import pcb
    from ...em3d import spice
    s = pcb.Setup.from_dict(ctx.test.setup)
    res = A.run(ctx.board, ctx.stackup, s, progress=ctx.log)
    out = Outcome(warnings=list(res.warnings))
    v = out.values
    Rdc, Ldc = res.dc()
    names = res.names
    v["R_dc_mOhm"] = {n: float(Rdc[q] * 1e3) for q, n in enumerate(names)}
    v["L_dc_nH"] = {n: float(Ldc[q, q] * 1e9) for q, n in enumerate(names)}
    v["R_fmax_mOhm"] = {n: float(res.R[-1, q] * 1e3) for q, n in enumerate(names)}
    v["L_fmax_nH"] = {n: float(res.L[-1, q, q] * 1e9) for q, n in enumerate(names)}
    v["R_ratio"] = {n: float(res.R[-1, q] / Rdc[q]) for q, n in enumerate(names)}
    v["Q_max"] = {n: float(np.nanmax(res.Q[:, q])) for q, n in enumerate(names)}
    if len(names) > 1:
        k = res.k
        v["k_max"] = {f"{names[i]}-{names[j]}": float(np.nanmax(np.abs(k[:, i, j])))
                      for i in range(len(names)) for j in range(i + 1, len(names))}
    v["filaments"] = float(res.n_fils)
    for q, n in enumerate(names):
        out.lines.append(f"{n}: R_dc {Rdc[q] * 1e3:.4g} mΩ, L_dc {Ldc[q, q] * 1e9:.4g} nH; at {res.f[-1]:.3g} Hz "
                         f"R {res.R[-1, q] * 1e3:.4g} mΩ, L {res.L[-1, q, q] * 1e9:.4g} nH")
    out.lines.append(f"{res.n_fils} filaments, {res.seconds:.1f} s")
    if ctx.want("spice"):
        with open(ctx.path("em3d.lib"), "w") as f:
            f.write(spice.foster(res, "EM3D", header=[f"3D extraction of {ctx.board_path}"]))
        ctx.wrote(out, "em3d.lib")
    if ctx.want("csv"):
        with open(ctx.path("impedance.csv"), "w", newline="") as f:
            w = csv.writer(f)
            head = ["f_Hz"] + [f"R_{n}_ohm" for n in names] + [f"L_{n}_H" for n in names]
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            head += [f"k_{names[i]}_{names[j]}" for i, j in pairs]
            w.writerow(head)
            L, k = res.L, res.k
            for m, f_ in enumerate(res.f):
                w.writerow([f"{f_:.6g}"] + [f"{res.R[m, q]:.6g}" for q in range(len(names))]
                           + [f"{L[m, q, q]:.6g}" for q in range(len(names))]
                           + [f"{k[m, i, j]:.6g}" for i, j in pairs])
        ctx.wrote(out, "impedance.csv")
    if not all(math.isfinite(x) for x in v["R_dc_mOhm"].values()):
        raise ValueError("the solve gave no finite resistance")
    return out
