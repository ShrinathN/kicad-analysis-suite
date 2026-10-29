"""Plotting shared by the viewer and the exporters (PNG/SVG/PDF/HTML)."""
import os
import textwrap

import numpy as np
import matplotlib
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

CMAPS = {"J": "inferno", "V": "viridis", "T": "magma", "p": "inferno", "B": "cividis", "Bn": "RdBu_r",
         "Tp": "magma"}


def plot_field(ax, res, layer, q="J", log=True, stream=False, outline=True, label_pads=True):
    """Draw one quantity of one layer into `ax`; returns the image artist (or None)."""
    ax.set_facecolor("#1e1e1e")
    A = res.field(layer, q)
    im = None
    if A is not None and np.isfinite(A).any():
        A = np.ma.masked_invalid(A)
        vmax = float(A.max())
        if q == "J":
            vmax = max(float(np.percentile(A.compressed(), 99.9)), 1e-6)
        if q == "Bn":
            m = max(float(np.percentile(np.abs(A.compressed()), 99.9)), 1e-9)
            norm = Normalize(-m, m)
        elif log and q in ("J", "p"):
            pos = A.compressed()
            pos = pos[pos > 0]
            lo = max(float(np.percentile(pos, 1)) if pos.size else 1e-6, vmax * 1e-3)
            norm = LogNorm(lo, max(vmax, lo * 10))
            A = np.ma.masked_less_equal(A, 0)
        if q != "Bn" and not (log and q in ("J", "p")):
            norm = Normalize(float(A.min()), vmax if vmax > A.min() else float(A.min()) + 1e-9)
        im = ax.imshow(A, extent=res.extent(q), cmap=CMAPS[q], norm=norm, interpolation="nearest",
                       origin="upper")
    if stream and q != "T":
        _streamlines(ax, res, layer)
    if outline:
        for kind, data in res.outlines.get(layer, []):
            if kind == "line":
                xy = np.asarray(data)
                ax.plot(xy[:, 0], xy[:, 1], color="#6fa8dc", lw=0.4, alpha=0.6)
            elif kind == "elem":
                (xa, ya), (xb, yb), name = data
                ax.plot([xa, xb], [ya, yb], color="#ffd966", lw=1.2, ls="--", alpha=0.9)
                ax.annotate(name, ((xa + xb) / 2, (ya + yb) / 2), xytext=(0, 6), textcoords="offset points",
                            ha="center", color="#ffd966", fontsize=7)
            elif kind == "pad" and label_pads:
                (x, y), pid = data
                ax.annotate(pid, (x, y), xytext=(0, -12), textcoords="offset points", ha="center",
                            color="cyan", fontsize=7)
    l, r, b, t = res.extent(q) if q in ("B", "Bn") else res.grid.extent
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    return im


def _streamlines(ax, res, layer, max_n=400):
    g, F = res.grid, res.fields[layer]
    s = max(1, int(np.ceil(max(g.nx, g.ny) / max_n)))
    Jx, Jy = F["Jx"][::s, ::s], F["Jy"][::s, ::s]
    if not np.isfinite(Jx).any():
        return
    xs = g.x0 + (np.arange(Jx.shape[1]) * s + 0.5) * g.h
    ys = g.y0 + (np.arange(Jx.shape[0]) * s + 0.5) * g.h
    U, V = np.nan_to_num(Jx), np.nan_to_num(Jy)
    speed = np.hypot(U, V)
    if speed.max() <= 0:
        return
    ax.streamplot(xs, ys, U, V, density=2.0, color="white", linewidth=0.5, arrowsize=0.6,
                  broken_streamlines=True)


def figure_for(res, layer, q, log=True, stream=False, title=True, size=(11, 6)):
    fig = Figure(figsize=size, constrained_layout=True)
    ax = fig.add_subplot()
    im = plot_field(ax, res, layer, q, log, stream)
    if im is not None:
        fig.colorbar(im, ax=ax, label=res.QUANTITIES[q], shrink=0.85)
    if title:
        ax.set_title(f"{res.QUANTITIES[q].split('[')[0].strip()} - {layer}")
    return fig


def export_images(res, outdir, fmt="png", quantities=None, stream=False, dpi=150):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for L in res.layers:
        if not _layer_has_current(res, L):
            continue
        for q in quantities or res.quantity_keys():
            if q not in res.quantity_keys() or res.field(L, q) is None:
                continue
            fig = figure_for(res, L, q, stream=stream)
            p = os.path.join(outdir, f"{L.replace('.', '_')}_{q}.{fmt}")
            fig.savefig(p, dpi=dpi)
            paths.append(p)
    return paths


def _layer_has_current(res, L):
    J = res.fields[L]["J"]
    return np.isfinite(J).any() and np.nanmax(J) > 1e-3


def _table_page(pdf, title, rows, cols, fmt=None, max_rows=40):
    if not rows:
        return
    fmt = fmt or {}
    for start in range(0, len(rows), max_rows):
        fig = Figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot()
        ax.axis("off")
        ax.set_title(title + (f" ({start + 1}-{min(start + max_rows, len(rows))} of {len(rows)})"
                              if len(rows) > max_rows else ""), loc="left")
        cells = []
        for r in rows[start:start + max_rows]:
            line = []
            for c in cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    line.append("-")
                elif isinstance(v, list):
                    line.append("+".join(v))
                elif c in fmt:
                    line.append(fmt[c].format(v))
                elif isinstance(v, (float, np.floating)):
                    line.append(f"{v:.4g}")
                else:
                    line.append(str(v)[:40])
            cells.append(line)
        t = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="left")
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1, 1.2)
        pdf.savefig(fig)


ROW_COLS = ["name", "kind", "src", "snk", "I_A", "dV_mV", "R_mOhm", "R_cold_mOhm", "err_pct", "R_extrap_mOhm"]
PAD_COLS = ["pad", "net", "role", "I_A", "V_mV", "T_C"]
VIA_COLS = ["id", "kind", "net", "drill_mm", "I_A", "I_cap_10C", "util", "group", "share_pct", "dV_mV", "P_W"]
ELEM_COLS = ["name", "a", "b", "net_a", "net_b", "R", "I_A", "V_mV", "P_W", "P_rated_W", "util", "T_C",
             "source", "board_heat"]
SEG_COLS = ["item", "net", "layer", "P_W", "share_pct", "dV_mV", "I_A", "J_med", "J_max", "crowding",
            "ipc_util_10", "ipc_util_20", "t_fuse_s", "I_fuse_1s", "T_pulse_C"]
IND_COLS = ["row_a", "row_b", "L_nH", "k", "closure_mm"]
PULSE_COLS = ["t_ms", "T_max_C"]


def export_report_pdf(res, path, stream=False):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        fig = Figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot()
        ax.axis("off")
        text = "\n".join(textwrap.fill(l, 150) if len(l) > 150 else l for l in res.summary_text().split("\n"))
        ax.text(0, 1, "PCB current-density simulation\n\n" + text, va="top", ha="left",
                family="monospace", fontsize=7.5, transform=ax.transAxes)
        pdf.savefig(fig)
        _table_page(pdf, "Excitations", res.rows, ROW_COLS)
        _table_page(pdf, "Pads (voltage / current)", res.pads, PAD_COLS)
        _table_page(pdf, "Vias and plated holes", res.vias, VIA_COLS)
        if res.elements:
            _table_page(pdf, "Resistance elements", res.elements, ELEM_COLS)
        _table_page(pdf, "Loss ranking by copper item", res.segments, SEG_COLS)
        if res.inductance:
            _table_page(pdf, "Loop inductance (self L and mutual M between rows)", res.inductance, IND_COLS)
        if res.pulse:
            pdf.savefig(pulse_figure(res, size=(11.69, 8.27)))
        for L in res.layers:
            if not _layer_has_current(res, L):
                continue
            for q in res.quantity_keys():
                if res.field(L, q) is None:
                    continue
                pdf.savefig(figure_for(res, L, q, stream=stream and q == "J", size=(11.69, 8.27)))
    return path


def pulse_figure(res, size=(9, 5)):
    """Hottest temperature versus time during the pulse."""
    p = res.pulse
    fig = Figure(figsize=size, constrained_layout=True)
    ax = fig.add_subplot()
    ax.plot(p["t_ms"], p["T_max"], color="#d95f02", lw=2)
    ax.axhline(1083, color="#888", ls="--", lw=1)
    ax.text(p["t_ms"][0], 1083, " copper melts (1083 °C)", va="bottom", color="#666", fontsize=8)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("hottest point [°C]")
    ax.set_title(f"Pulse: {p['scale']:g} x table values for {p['ms']:g} ms, from "
                 + ("steady state" if p["from_steady"] else "ambient"))
    if p["fused"]:
        ax.axvline(p["fused"]["t_ms"], color="#c00", lw=1)
    ax.grid(alpha=0.3)
    return fig


def export_html(res, path, max_cells=400_000):
    """Interactive Plotly page: layer x quantity dropdown with hover values."""
    import plotly.graph_objects as go
    traces, labels = [], []
    for L in res.layers:
        if not _layer_has_current(res, L):
            continue
        for q in res.quantity_keys():
            A = res.field(L, q)
            if A is None:
                continue
            g = {"T": res.tgrid, "Tp": res.pgrid, "B": res.bgrid, "Bn": res.bgrid}.get(q, res.grid)
            s = max(1, int(np.ceil(np.sqrt(A.size / max_cells))))
            Z = A[::s, ::s]
            x = g.x0 + (np.arange(Z.shape[1]) * s + 0.5) * g.h
            y = g.y0 + (np.arange(Z.shape[0]) * s + 0.5) * g.h
            zmax = float(np.nanpercentile(Z, 99.5)) if q == "J" else None
            traces.append(go.Heatmap(z=Z, x=x, y=y, colorscale={"cividis": "Cividis", "RdBu_r": "RdBu"}.get(CMAPS[q], CMAPS[q].capitalize()),
                                     reversescale=q == "Bn", zmax=zmax,
                                     colorbar=dict(title=res.QUANTITIES[q]), visible=False,
                                     hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>%{z:.4g}<extra></extra>"))
            labels.append(f"{L} {res.QUANTITIES[q]}")
    if not traces:
        return None
    traces[0].visible = True
    fig = go.Figure(traces)
    buttons = [dict(label=lab, method="update",
                    args=[{"visible": [k == i for k in range(len(traces))]}, {"title": lab}])
               for i, lab in enumerate(labels)]
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, y=1.12, xanchor="left")],
                      title=labels[0], plot_bgcolor="#1e1e1e", height=700)
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ---------------------------------------------------------------------------
# 3D scene: copper layers as sheets at their stackup heights (z exaggerated), vias as vertical bars
Q3D = ("J", "V", "T", "p", "Tp", "Bn")
COLORSCALES = {"J": "Inferno", "V": "Viridis", "T": "Magma", "p": "Inferno", "Tp": "Magma", "Bn": "RdBu"}


def _block(A, s, how):
    """Downsample by s x s blocks (NaN-aware max or mean)."""
    if s == 1:
        return A
    ny, nx = A.shape
    P = np.full((-(-ny // s) * s, -(-nx // s) * s), np.nan)
    P[:ny, :nx] = A
    B = P.reshape(P.shape[0] // s, s, P.shape[1] // s, s)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(B, axis=(1, 3)) if how == "max" else np.nanmean(B, axis=(1, 3))


def _on_grid(res, L, q):
    """Quantity q of layer L on the electrical grid, NaN off the simulated copper."""
    g = res.grid
    copper = res.fields[L]["f"] > 0
    if q in ("J", "V", "p"):
        A = np.array(res.fields[L][q], float)
    else:
        src, tg = (res.T, res.tgrid) if q == "T" else (res.Tp, res.pgrid)
        X, Y = g.centres()
        jj = np.clip(np.floor((Y - tg.y0) / tg.h).astype(int), 0, tg.ny - 1)
        ii = np.clip(np.floor((X - tg.x0) / tg.h).astype(int), 0, tg.nx - 1)
        A = np.array(src[L][jj, ii], float)
    A[~copper] = np.nan
    return A


def figure_3d(res, max_cells=20_000, z_aspect=0.4):
    """Plotly 3D figure with a quantity menu. Needs layer heights in res.summary (new results)."""
    import plotly.graph_objects as go
    s = res.summary
    zL = s.get("layer_z_mm")
    if not zL:
        return None
    g = res.grid
    step = max(1, int(np.ceil(np.sqrt(g.nx * g.ny / max_cells))))
    xs = g.x0 + (np.arange(-(-g.nx // step)) * step + step / 2) * g.h
    ys = g.y0 + (np.arange(-(-g.ny // step)) * step + step / 2) * g.h
    X, Y = np.meshgrid(xs, ys)
    z_top, z_bot = 0.0, s.get("z_bottom_mm", min(zL.values()))
    bh = s.get("b_height_mm", 1.0)
    avail = [q for q in Q3D if q in res.quantity_keys()]
    traces, group = [], []           # group: quantity key, or None for always-visible traces

    for q in avail:
        log = q in ("J", "p")
        if q == "Bn":
            # B planes above / below, copper sheets in neutral grey underneath
            for L in res.layers:
                A = _block(_on_grid(res, L, "V"), step, "mean")
                Z = np.where(np.isnan(A), np.nan, zL[L]).astype(np.float32)
                traces.append(go.Surface(x=xs, y=ys, z=Z, showscale=False,
                                         colorscale=[[0, "#b87333"], [1, "#b87333"]], opacity=0.6, hoverinfo="skip",
                                         visible=False, name=L))
                group.append(q)
            bg = res.bgrid
            bs = max(1, int(np.ceil(np.sqrt(bg.nx * bg.ny / max_cells))))
            bxs = bg.x0 + (np.arange(-(-bg.nx // bs)) * bs + bs / 2) * bg.h
            bys = bg.y0 + (np.arange(-(-bg.ny // bs)) * bs + bs / 2) * bg.h
            vals = {side: _block(np.array(b["Bz"], float), bs, "mean").astype(np.float32) for side, b in res.B.items()}
            m = max(float(np.nanpercentile(np.abs(v), 99.5)) for v in vals.values()) or 1e-9
            for side, v in vals.items():
                zp = z_top + bh if side == "above" else z_bot - bh
                traces.append(go.Surface(x=bxs, y=bys, z=np.full(v.shape, zp, np.float32), surfacecolor=v, cmin=-m, cmax=m,
                                         colorscale="RdBu", reversescale=True, opacity=0.85, visible=False,
                                         colorbar=dict(title="Bz [µT]", len=0.6), showscale=side == "above",
                                         name=f"B {side}",
                                         hovertemplate=f"B {bh:g} mm {side}<br>x=%{{x:.2f}} y=%{{y:.2f}} mm"
                                                       "<br>Bz=%{surfacecolor:.1f} µT<extra></extra>"))
                group.append(q)
            continue
        per = {L: _block(_on_grid(res, L, q), step, "max" if q in ("J", "p", "T", "Tp") else "mean")
               for L in res.layers}
        allv = np.concatenate([a[np.isfinite(a)] for a in per.values()]) if per else np.zeros(0)
        if log:
            allv = allv[allv > 0]
        if allv.size == 0:
            continue
        lo, hi = float(np.percentile(allv, 1)), float(np.percentile(allv, 99.9))
        if log:
            lo = max(lo, hi * 1e-3)
            cmin, cmax = np.log10(lo), np.log10(hi)
            ticks = [v for v in range(int(np.floor(cmin)), int(np.ceil(cmax)) + 1)]
            cbar = dict(title=res.QUANTITIES[q], tickvals=ticks, ticktext=[f"{10 ** t:g}" for t in ticks], len=0.75)
        else:
            cmin, cmax = lo, hi if hi > lo else lo + 1e-9
            cbar = dict(title=res.QUANTITIES[q], len=0.75)
        first = True
        for L in res.layers:
            A = per[L]
            if not np.isfinite(A).any():
                continue
            C = np.log10(np.where(A > 0, A, np.nan)) if log else A
            Z = np.where(np.isnan(A), np.nan, zL[L]).astype(np.float32)
            name, unit = res.QUANTITIES[q].split("[")[0].strip(), res.QUANTITIES[q].split("[")[-1].rstrip("]")
            val = "%{customdata:.4g}" if log else "%{surfacecolor:.4g}"
            traces.append(go.Surface(
                x=xs.astype(np.float32), y=ys.astype(np.float32), z=Z, surfacecolor=C.astype(np.float32),
                customdata=A.astype(np.float32) if log else None, cmin=cmin, cmax=cmax, colorscale=COLORSCALES[q],
                showscale=first, colorbar=cbar, visible=False, name=L,
                hovertemplate=f"{L}<br>x=%{{x:.2f}} y=%{{y:.2f}} mm<br>{name} = {val} {unit}<extra></extra>"))
            group.append(q)
            first = False

    # vias / barrels: vertical bars coloured by current
    vx, vy, vz, vc, vt = [], [], [], [], []
    for v in res.vias:
        if v.get("L_top") not in zL or v.get("L_bot") not in zL:
            continue
        for z in (zL[v["L_top"]], zL[v["L_bot"]], None):
            vx.append(v["x"] if z is not None else None)
            vy.append(v["y"] if z is not None else None)
            vz.append(z)
            vc.append(v["I_A"])
            vt.append(f"{v['id']} ({v['net']})<br>I = {v['I_A']:.3g} A")
    if vx:
        traces.append(go.Scatter3d(x=vx, y=vy, z=vz, mode="lines", text=vt, hoverinfo="text", name="vias",
                                   line=dict(width=7, color=vc, colorscale="Turbo", cmin=0,
                                             cmax=max(max(vc), 1e-9))))
        group.append(None)
    # resistance elements as links just above their side of the board
    for e in res.elements:
        items = [d for d in res.outlines.get(res.layers[0], []) if d[0] == "elem" and d[1][2] == e["name"]]
        if not items:
            continue
        (xa, ya), (xb, yb), _ = items[0][1]
        z = z_top + 0.15
        traces.append(go.Scatter3d(x=[xa, xb], y=[ya, yb], z=[z, z], mode="lines+text", text=["", e["name"]],
                                   line=dict(width=5, color="#ffd966", dash="dash"), name=e["name"],
                                   hovertext=f"{e['name']} {e['R']}<br>I = {e['I_A']:.3g} A, P = {e['P_W']:.3g} W",
                                   hoverinfo="text", textfont=dict(color="#ffd966")))
        group.append(None)
    # board outline at both surfaces
    for loop in s.get("board_outline", []):
        a = np.asarray(loop)
        for z in (z_top, z_bot):
            traces.append(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=np.full(len(a), z), mode="lines", hoverinfo="skip",
                                       line=dict(width=2, color="#8fbc8f"), showlegend=False))
            group.append(None)
    if not any(gk for gk in group):
        return None
    shown = avail[0] if avail[0] in group else next(gk for gk in group if gk)
    for t, gk in zip(traces, group):
        t.visible = gk is None or gk == shown
    labels = {q: res.QUANTITIES[q] for q in avail if q in group}
    buttons = [dict(label=lab, method="update", args=[{"visible": [gk is None or gk == q for gk in group]}])
               for q, lab in labels.items()]
    W, H = g.nx * g.h, g.ny * g.h
    zr = (z_bot - (bh if res.B else 0) - 0.2, z_top + (bh if res.B else 0) + 0.3)
    exag = (z_aspect * W) / (zr[1] - zr[0])
    fig = go.Figure(traces)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="#333",
                          font=dict(color="#ddd"))],
        paper_bgcolor="#1e1e1e", font=dict(color="#ddd"), margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"{os.path.basename(res.board_path)} - 3D (z exaggerated x{exag:.0f})", x=0.5, font=dict(size=13)),
        legend=dict(x=0.01, y=0.85),
        scene=dict(xaxis=dict(title="x [mm]", range=[g.x0, g.x0 + W], backgroundcolor="#1e1e1e", gridcolor="#444"),
                   yaxis=dict(title="y [mm]", range=[g.y0 + H, g.y0], backgroundcolor="#1e1e1e", gridcolor="#444"),
                   zaxis=dict(title="z [mm]", range=list(zr), backgroundcolor="#1e1e1e", gridcolor="#444"),
                   aspectmode="manual", aspectratio=dict(x=1, y=H / W, z=z_aspect),
                   # from the board's front edge (large y), x increasing to the right as in the 2D map
                   camera=dict(eye=dict(x=0.25, y=-1.5, z=1.1), up=dict(x=0, y=0, z=1))))
    return fig


def plotly_js_path():
    """plotly.min.js shipped with the plotly Python package (for offline pages)."""
    import plotly
    p = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    return p if os.path.exists(p) else None


def export_html_3d(res, path, js="inline"):
    """3D scene as a page.  js: "inline" (self-contained, works offline, +4.6 MB), "cdn", or
    "local" (a <script> pointing at the installed plotly.min.js - for the embedded viewer)."""
    fig = figure_3d(res)
    if fig is None:
        return None
    local = plotly_js_path() if js == "local" else None
    inc = False if local else {"inline": True, "cdn": "cdn"}.get(js, True)
    html = fig.to_html(include_plotlyjs=inc, full_html=True, config=dict(displaylogo=False, responsive=True),
                       default_height="100vh", default_width="100%")
    if local:
        html = html.replace("<head>", f'<head><script src="file://{local}"></script>', 1)
    html = html.replace("<body>", '<body style="margin:0;background:#1e1e1e">', 1)
    with open(path, "w") as f:
        f.write(html)
    return path
