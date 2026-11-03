"""Trace R/L/C → SPICE in the batch runner.

[test.setup] = the Trace window's section: ports (pad ids, the first is P1), return_nets (empty or
missing: guessed as in the window) and options. Parts between pads (R, diode, C, L) are the board's
shared components switched on for the Trace tool ([[test.components]]).
"""
import csv
import math

from . import Metric, Outcome

TITLE = "Trace R/L/C → SPICE"
TRACE_MODELS = ("R", "V", "C", "L")

METRICS = {
    "R_mOhm": Metric("mΩ", "resistance from P1 to a port (key: the port's pad, e.g. J2.1)", "port"),
    "L_nH": Metric("nH", "loop inductance from P1 to a port", "port"),
    "C_pF": Metric("pF", "copper capacitance to the return, P1 to a port", "port"),
    "delay_ps": Metric("ps", "propagation delay from P1 to a port", "port"),
    "Z0_ohm": Metric("Ω", "average characteristic impedance from P1 to a port", "port"),
    "Z0_min_ohm": Metric("Ω", "lowest section Z0 on the path to a port", "port"),
    "Z0_max_ohm": Metric("Ω", "highest section Z0 on the path to a port", "port"),
    "C_total_pF": Metric("pF", "all copper capacitance to the return"),
    "impedance_steps": Metric("", "number of Z0 steps above 10 % along the path"),
}

OUTPUTS = {
    "spice": "<NET>.lib: the SPICE subcircuit (plain R, L, C)",
    "csv": "pairs.csv (totals per port), segments.csv, vias.csv, pads.csv",
}


def from_scenario(scen, board, comps):
    d = scen.get("extract") or {}
    if len(d.get("ports") or []) < 2:
        return []
    return [(f"Trace {d['ports'][0]} → {d['ports'][-1]}", {k: d[k] for k in ("ports", "return_nets", "options")
                                                           if k in d})]


def _fin(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def run(ctx):
    from dataclasses import fields
    from ...extract import rlc as X
    from ...extract import spice as XS
    d = ctx.test.setup
    ports = list(d.get("ports") or [])
    ids = {p.id for p in ctx.board.pads}
    missing = [p for p in ports if p not in ids]
    if missing:
        raise ValueError(f"pad(s) not on the board: {', '.join(missing)}")
    if len(ports) < 2:
        raise ValueError("[test.setup] ports needs at least two pads")
    known = {f.name for f in fields(X.Options)}
    opts = X.Options(**{k: v for k, v in (d.get("options") or {}).items() if k in known})
    nets = [n for n in (d.get("return_nets") or []) if n in ctx.board.nets]
    if not d.get("return_nets"):
        sig = next(p.net for p in ctx.board.pads if p.id == ports[0])
        nets = X.guess_return_nets(ctx.board, sig)
        ctx.log(f"return nets guessed: {', '.join(nets) or 'none'}")
    els = [e for e in ctx.components.view("extract") if e.model in TRACE_MODELS]
    ext = X.Extractor(ctx.board, ctx.stackup, nets, opts, progress=ctx.log, elements=els).run(ports)
    out = Outcome(warnings=list(ext.warnings))
    v = out.values
    for key, src in (("R_mOhm", "R_mOhm"), ("L_nH", "L_nH"), ("C_pF", "C_pF"), ("delay_ps", "delay_ps"),
                     ("Z0_ohm", "Z0_avg"), ("Z0_min_ohm", "Z0_min"), ("Z0_max_ohm", "Z0_max")):
        v[key] = {p["b"]: x for p in ext.pairs if (x := _fin(p.get(src))) is not None}
    v["C_total_pF"] = _fin(ext.summary.get("C_total_pF"))
    v["impedance_steps"] = float(len(ext.markers))
    s = ext.summary
    out.lines.append(f"Net {s.get('net')}, ports {s.get('ports')}, return {s.get('return_nets')}; "
                     f"{s.get('sections')} section(s), {s.get('seconds', 0):.1f} s")
    for p in ext.pairs:
        out.lines.append(f"{p['pair']} ({p['a']} → {p['b']}): R {p['R_mOhm']:.4g} mΩ, L {p['L_nH']:.4g} nH, "
                         f"C {p['C_pF']:.4g} pF, Z0 {p['Z0_avg']:.4g} Ω, delay {p['delay_ps']:.4g} ps")
    if ctx.want("spice"):
        name = XS.subckt_name(ext)
        fn = f"{name}.lib"
        with open(ctx.path(fn), "w") as fh:
            fh.write(XS.write(ext, name, getattr(ctx.board, "path", "")))
        ctx.wrote(out, fn)
    if ctx.want("csv"):
        for fn, rows in (("pairs.csv", ext.pairs), ("segments.csv", ext.segments), ("vias.csv", ext.vias),
                         ("pads.csv", ext.pads)):
            rows = [{k: x for k, x in r.items() if not k.startswith("_")} for r in rows]
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(ctx.path(fn), "w", newline="") as fh:
                w = csv.DictWriter(fh, cols)
                w.writeheader()
                w.writerows(rows)
            ctx.wrote(out, fn)
    return out
