"""DC & Thermal in the batch runner.

[test.setup] = the DC window's section: rows (excitations) and settings. The parts between pads
(resistors, elements, regulators) are the board's shared components: those the window has switched
on for DC, saved with the test in [[test.components]] (detected resistors are found on the board
again, as in the window). Copper thickness overrides are in [test.stackup] cu_thickness.
"""
import math

import numpy as np

from . import Metric, Outcome

TITLE = "DC & Thermal"

METRICS = {
    "T_max_C": Metric("°C", "hottest copper temperature (thermal on)"),
    "runaway": Metric("", "1 if the electro-thermal loop ran away (copper heating without limit), else 0"),
    "P_copper_W": Metric("W", "power dissipated in the copper"),
    "P_elements_W": Metric("W", "power dissipated in the parts between pads"),
    "J_max": Metric("A/mm²", "highest current density, over all nets"),
    "net_J_max": Metric("A/mm²", "highest current density of a net", "net"),
    "net_P_W": Metric("W", "power dissipated in a net's copper", "net"),
    "row_I_A": Metric("A", "current of an excitation", "row"),
    "row_dV_mV": Metric("mV", "voltage between an excitation's source and sink pads", "row"),
    "row_R_mOhm": Metric("mΩ", "resistance of a current-path excitation", "row"),
    "row_err_pct": Metric("%", "grid-error estimate of an excitation's result", "row"),
    "loop_L_nH": Metric("nH", "loop inductance of an excitation (magnetic on)", "row"),
    "pad_V": Metric("V", "voltage at a pad (as the window's Pads table: against each net's sink)", "pad"),
    "pad_T_C": Metric("°C", "temperature at a pad (thermal on)", "pad"),
    "el_I_A": Metric("A", "current through a part between pads", "part"),
    "el_P_W": Metric("W", "power in a part between pads", "part"),
    "el_util": Metric("", "a part's power / its rating (1 = at the rating)", "part"),
    "el_T_C": Metric("°C", "temperature at a part's pads", "part"),
    "reg_V_in": Metric("V", "a regulator's input voltage at its pins", "regulator"),
    "reg_V_out": Metric("V", "a regulator's output voltage at its pins", "regulator"),
    "reg_headroom_V": Metric("V", "an LDO's input - output voltage", "regulator"),
    "reg_dropout": Metric("", "1 if an LDO is in dropout, else 0", "regulator"),
    "reg_P_W": Metric("W", "a regulator's loss", "regulator"),
    "reg_T_j_C": Metric("°C", "a regulator's estimated junction temperature", "regulator"),
    "ipc_util_10_max": Metric("", "worst track current / its IPC-2221 10 °C rise current"),
    "ipc_util_20_max": Metric("", "worst track current / its IPC-2221 20 °C rise current"),
    "t_fuse_min_s": Metric("s", "shortest time for a track to fuse at these currents (Onderdonk)"),
    "via_util_max": Metric("", "worst via current / its 10 °C rise capacity"),
    "pulse_T_max_C": Metric("°C", "hottest temperature during the pulse (pulse on)"),
}

OUTPUTS = {
    "png": "every map as PNG (each quantity and layer)",
    "svg": "every map as SVG",
    "pdf": "multi-page PDF report: maps and tables",
    "csv": "the result tables as CSV",
    "html": "interactive HTML map page",
    "html3d": "interactive 3D HTML page (plotly inline)",
    "vtk": "ParaView files (paraview/board.vtm)",
    "npz": "the result itself (result.npz), to open in the DC viewer",
}


def from_scenario(scen, board, comps):
    dc = scen.get("dc") or {}
    if not dc.get("rows"):
        return []
    return [("DC & Thermal", {"rows": dc.get("rows", []), "settings": dc.get("settings", {})})]


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _items(rows, key, val, scale=1.0):
    out = {}
    for r in rows:
        x = _num(r.get(val))
        if x is not None and r.get(key) is not None:
            out[str(r[key])] = x * scale
    return out


def run(ctx):
    from ... import solver as S
    from ...scenario import dc_from_dict
    t = ctx.test
    if not (t.setup.get("rows")):
        raise ValueError("no excitation rows in [test.setup] rows")
    rows, s = dc_from_dict(t.setup)
    if ctx.stackup is not None:
        s.thickness = dict(ctx.stackup.cu_override)
    els = [e for e in ctx.components.view("dc") if s.auto_resistors or not e.auto]
    res = S.simulate(ctx.board, rows, s, progress=lambda m: ctx.log(m.strip()), elements=els)
    out = Outcome(warnings=list(res.warnings) + list(getattr(ctx.components, "warnings", []) or []))
    sm, v = res.summary, out.values
    if sm.get("thermal"):
        v["T_max_C"] = _num(sm.get("T_max"))
        v["runaway"] = 1.0 if sm.get("runaway") else 0.0
    v["P_copper_W"] = _num(sm.get("P_total_W"))
    v["P_elements_W"] = _num(sm.get("P_elements_W")) or 0.0
    v["net_J_max"] = _items(res.net_table, "net", "J_max")
    v["net_P_W"] = _items(res.net_table, "net", "P_W")
    if v["net_J_max"]:
        v["J_max"] = max(v["net_J_max"].values())
    v["row_I_A"] = _items(res.rows, "name", "I_A")
    v["row_dV_mV"] = _items(res.rows, "name", "dV_mV")
    v["row_R_mOhm"] = _items(res.rows, "name", "R_mOhm")
    v["row_err_pct"] = _items(res.rows, "name", "err_pct")
    if res.inductance:
        v["loop_L_nH"] = {r["row_a"]: float(r["L_nH"]) for r in res.inductance if r["row_a"] == r["row_b"]}
    v["pad_V"] = _items(res.pads, "pad", "V_mV", 1e-3)
    if sm.get("thermal"):
        v["pad_T_C"] = _items(res.pads, "pad", "T_C")
    v["el_I_A"] = _items(res.elements, "name", "I_A")
    v["el_P_W"] = _items(res.elements, "name", "P_W")
    v["el_util"] = _items(res.elements, "name", "util")
    if sm.get("thermal"):
        v["el_T_C"] = _items(res.elements, "name", "T_C")
    if res.regulators:
        v["reg_V_in"] = _items(res.regulators, "name", "V_in_V")
        v["reg_V_out"] = _items(res.regulators, "name", "V_out_V")
        v["reg_headroom_V"] = _items(res.regulators, "name", "headroom_V")
        v["reg_dropout"] = {r["name"]: 1.0 if r.get("dropout") else 0.0 for r in res.regulators}
        v["reg_P_W"] = _items(res.regulators, "name", "P_W")
        v["reg_T_j_C"] = _items(res.regulators, "name", "T_j_C")
    seg = [r for r in res.segments if r.get("kind") == "track"]
    for key, name in (("ipc_util_10", "ipc_util_10_max"), ("ipc_util_20", "ipc_util_20_max")):
        xs = [_num(r.get(key)) for r in seg]
        xs = [x for x in xs if x is not None]
        if xs:
            v[name] = max(xs)
    xs = [x for x in (_num(r.get("t_fuse_s")) for r in seg) if x is not None]
    if xs:
        v["t_fuse_min_s"] = min(xs)
    xs = [x for x in (_num(r.get("util")) for r in res.vias) if x is not None]
    if xs:
        v["via_util_max"] = max(xs)
    if res.pulse:
        T = np.asarray(res.pulse.get("T_max", []), float)
        if T.size:
            v["pulse_T_max_C"] = float(np.nanmax(T))
    out.lines = [ln for ln in res.summary_text().splitlines() if ln.strip()][:40]
    _outputs(ctx, res, out)
    return out


def _outputs(ctx, res, out):
    import os
    from ... import export
    folder = None
    for f in ("png", "svg"):
        if ctx.want(f):
            folder = ctx.path("")
            before = set(os.listdir(folder))
            export.export_images(res, folder, f)
            new = sorted(set(os.listdir(folder)) - before)
            for n in new:
                ctx.wrote(out, n)
            if f == "png":                       # the report shows current density and temperature, top layer first
                top = res.layers[0].replace(".", "_") if res.layers else ""
                pick = [n for q in ("J", "T", "Tp") for n in new if n == f"{top}_{q}.png"]
                pick += [n for n in new if n.endswith("_J.png") and n not in pick][:1]
                out.images += [os.path.join(ctx.rel, n) for n in pick[:3]]
    if ctx.want("pdf"):
        export.export_report_pdf(res, ctx.path("report.pdf"))
        ctx.wrote(out, "report.pdf")
    if ctx.want("html"):
        export.export_html(res, ctx.path("current_density.html"))
        ctx.wrote(out, "current_density.html")
    if ctx.want("html3d"):
        export.export_html_3d(res, ctx.path("current_density_3d.html"), js="inline")
        ctx.wrote(out, "current_density_3d.html")
    if ctx.want("vtk"):
        from ...vtk_export import export_vtk
        export_vtk(res, ctx.path("paraview"))
        ctx.wrote(out, "paraview/board.vtm")
    if ctx.want("csv"):
        folder = ctx.path("")
        before = set(os.listdir(folder))
        res.write_csv(folder)
        for n in sorted(set(os.listdir(folder)) - before):
            ctx.wrote(out, n)
    if ctx.want("npz"):
        res.save(ctx.path("result.npz"))
        ctx.wrote(out, "result.npz")
