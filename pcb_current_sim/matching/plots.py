"""Matplotlib figures for the matching tool (toolkit-neutral: each function fills a Figure).

Conventions: S11 = −RL and S21 = −IL in dB. Monte Carlo is drawn as a min–max band (light)
and a 5–95 % band (darker); corners as a dashed worst-case envelope; specs as red segments.
"""
import numpy as np
from matplotlib.patches import Circle

NOM = "#1f6fd1"
MC = "#e8590c"
CORNER = "#7b2cbf"
SPEC = "#d62828"
GRID = "0.6"


def fscale(setup):
    return (1e9, "GHz") if setup.f_stop >= 1e9 else (1e6, "MHz")


def _bands(ax, setup, div):
    for b in setup.bands:
        ax.axvspan(b.f1 / div, b.f2 / div, color="0.5", alpha=0.12, lw=0)


def _spec(ax, setup, div, val_of):
    for b in setup.bands:
        v = val_of(b)
        if v is not None:
            ax.plot([b.f1 / div, b.f2 / div], [v, v], color=SPEC, lw=2.2, solid_capstyle="butt")


def _env(ax, f, st, key, sign, label):
    a = sign * getattr(st, key)
    lo, hi = a.min(0), a.max(0)
    p5, p95 = np.percentile(a, [5, 95], axis=0)
    ax.fill_between(f, lo, hi, color=MC, alpha=0.18, lw=0, label=f"{label} min–max")
    ax.fill_between(f, p5, p95, color=MC, alpha=0.38, lw=0, label=f"{label} 5–95 %")


def _corners(ax, f, st, key, sign, worst):
    a = sign * getattr(st, key)
    ax.plot(f, a.max(0) if worst == "max" else a.min(0), "--", color=CORNER, lw=1.2,
            label=f"corners worst ({st.n})")


def _finish(ax, setup, unit, ylabel):
    ax.set_xlabel(f"f [{unit}]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(setup.f_start / fscale(setup)[0], setup.f_stop / fscale(setup)[0])
    if setup.log_f:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def plot_s11(fig, setup, nom, res=None):
    div, unit = fscale(setup)
    ax = fig.subplots()
    f = nom["f"] / div
    _bands(ax, setup, div)
    if res is not None and res.mc is not None:
        _env(ax, f, res.mc, "rl", -1, "MC")
    if res is not None and res.corners is not None:
        _corners(ax, f, res.corners, "rl", -1, "max")
    ax.plot(f, -nom["rl"], color=NOM, lw=1.8, label="nominal")
    _spec(ax, setup, div, lambda b: -b.rl_req() if b.rl_req() > 0 else None)
    lo = min(-40, float(np.nanmin(-nom["rl"])) - 3)
    ax.set_ylim(max(lo, -60), 1)
    ax.set_title("Input match S11 (vs source)")
    _finish(ax, setup, unit, "S11 [dB]")


def plot_s21(fig, setup, nom, res=None):
    div, unit = fscale(setup)
    ax = fig.subplots()
    f = nom["f"] / div
    _bands(ax, setup, div)
    if res is not None and res.mc is not None:
        _env(ax, f, res.mc, "il", -1, "MC")
    if res is not None and res.corners is not None:
        _corners(ax, f, res.corners, "il", -1, "min")
    ax.plot(f, -nom["il"], color=NOM, lw=1.8, label="nominal transducer gain")
    ax.plot(f, -nom["diss"], ":", color=NOM, lw=1.4, label="loss in the parts only")
    _spec(ax, setup, div, lambda b: -b.il_max if b.il_max > 0 else None)
    ax.set_ylim(max(float(np.nanmin(-nom["il"])) - 3, -40), 0.5)
    ax.set_title("Transmission S21 (insertion loss incl. mismatch)")
    _finish(ax, setup, unit, "S21 [dB]")


def plot_vswr(fig, setup, nom, res=None):
    div, unit = fscale(setup)
    ax = fig.subplots()
    f = nom["f"] / div
    _bands(ax, setup, div)

    def vs(rl):
        g = 10 ** (-np.asarray(rl, float) / 20)
        return (1 + g) / (1 - np.minimum(g, 1 - 1e-9))
    if res is not None and res.mc is not None:
        v = vs(res.mc.rl)
        ax.fill_between(f, v.min(0), v.max(0), color=MC, alpha=0.18, lw=0, label="MC min–max")
        p5, p95 = np.percentile(v, [5, 95], axis=0)
        ax.fill_between(f, p5, p95, color=MC, alpha=0.38, lw=0, label="MC 5–95 %")
    if res is not None and res.corners is not None:
        ax.plot(f, vs(res.corners.rl).max(0), "--", color=CORNER, lw=1.2, label=f"corners worst ({res.corners.n})")
    ax.plot(f, nom["vswr"], color=NOM, lw=1.8, label="nominal")
    _spec(ax, setup, div, lambda b: (1 + 10 ** (-b.rl_req() / 20)) / (1 - 10 ** (-b.rl_req() / 20))
          if b.rl_req() > 0 else None)
    ax.set_ylim(1, 6)
    ax.set_title("VSWR")
    _finish(ax, setup, unit, "VSWR")


def smith_grid(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    outer = Circle((0, 0), 1, fill=False, lw=1.2, ec=GRID)
    ax.add_patch(outer)
    for r in (0.2, 0.5, 1, 2, 5):
        ax.add_patch(Circle((r / (1 + r), 0), 1 / (1 + r), fill=False, lw=0.5, ec=GRID))
        ax.text(r / (1 + r) - 1 / (1 + r) + 0.01, 0.015, f"{r:g}", fontsize=6, color="0.4")
    for x in (0.2, 0.5, 1, 2, 5):
        for sg in (1, -1):
            c = Circle((1, sg / x), 1 / x, fill=False, lw=0.5, ec=GRID)
            ax.add_patch(c)
            c.set_clip_path(outer)
            g = complex(0, sg * x)
            p = (g - 1) / (g + 1)
            ax.text(p.real * 1.07, p.imag * 1.07, f"{'+' if sg > 0 else '−'}j{x:g}", fontsize=6, color="0.4",
                    ha="center", va="center")
    ax.plot([-1, 1], [0, 0], lw=0.5, color=GRID)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)


def plot_smith(fig, setup, nom, res=None):
    ax = fig.subplots()
    smith_grid(ax)
    z0 = setup.z_ref

    def g(z):
        return (z - z0) / (z + z0)
    f = nom["f"]
    masks = [(f >= b.f1) & (f <= b.f2) for b in setup.bands]
    if res is not None and res.mc is not None and len(res.mc.zin):
        label = "MC samples (in band)" if masks else "MC samples"
        for z in res.mc.zin:
            gm = g(z.astype(complex))
            for m in masks or [np.ones(len(f), bool)]:
                ax.plot(gm[m].real, gm[m].imag, color=MC, lw=0.5, alpha=0.18, label=label)
                label = None
    zl = g(setup.load.Z(f))
    ax.plot(zl.real, zl.imag, "--", color="0.45", lw=1, label="load alone")
    gm = g(nom["zin"])
    ax.plot(gm.real, gm.imag, color=NOM, lw=1, alpha=0.6, label="nominal Zin")
    for b, m in zip(setup.bands, masks):
        if m.any():
            ax.plot(gm[m].real, gm[m].imag, color=NOM, lw=3)
            for i in (np.argmax(m), len(m) - 1 - np.argmax(m[::-1])):
                ax.plot(gm[i].real, gm[i].imag, "o", color=NOM, ms=4)
            ax.annotate(b.name, (gm[m][len(gm[m]) // 2].real, gm[m][len(gm[m]) // 2].imag), fontsize=7,
                        xytext=(6, 6), textcoords="offset points", color=NOM)
    src = setup.source
    if src.kind == "Z" and src.x == 0 and abs(src.r - z0) < 1e-9:
        for b in setup.bands:
            if b.rl_req() > 0:
                ax.add_patch(Circle((0, 0), 10 ** (-b.rl_req() / 20), fill=False, ec=SPEC, lw=1, ls="--"))
    ax.plot([g(nom["zin"][0]).real], [g(nom["zin"][0]).imag], "s", color=NOM, ms=4)
    ax.set_title(f"Smith chart, Zin normalised to {z0:g} Ω (■ = sweep start)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-0.02, 1.0))


def plot_hist(fig, setup, nom, res):
    st = res.mc if res is not None and res.mc is not None else (res.corners if res is not None else None)
    nb = len(setup.bands)
    if st is None or not nb:
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, "Run a Monte Carlo analysis with at least one band", ha="center", va="center")
        return
    cols = 2 if any(b.il_max > 0 for b in setup.bands) else 1
    axes = np.atleast_2d(fig.subplots(nb, cols, squeeze=False))
    for k, b in enumerate(setup.bands):
        for c, (key, lim, lab, sgn) in enumerate((("rl_worst", b.rl_req(), "worst RL in band [dB]", 1),
                                                  ("il_worst", b.il_max, "worst IL in band [dB]", -1))[:cols]):
            ax = axes[k, c]
            v = st.band[k][key]
            v = v[np.isfinite(v)]
            if not len(v):
                ax.axis("off")
                continue
            ax.hist(v, bins=60, color=MC, alpha=0.8)
            ax.axvline(nom["band"][k][key], color=NOM, lw=1.5, label="nominal")
            if lim > 0:
                ax.axvline(lim, color=SPEC, lw=2, label="spec")
                ok = (v >= lim) if sgn > 0 else (v <= lim)
                ax.set_title(f"{b.name}: {100 * ok.mean():.1f} % pass", fontsize=9)
            else:
                ax.set_title(b.name, fontsize=9)
            ax.set_xlabel(lab)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)


def plot_sens(fig, setup, nom, res):
    if res is None or res.oat is None or not setup.bands:
        ax = fig.subplots()
        ax.axis("off")
        ax.text(0.5, 0.5, "Run an analysis with tolerances and at least one band", ha="center", va="center")
        return
    nb = min(len(setup.bands), 3)
    axes = fig.subplots(1, nb, squeeze=False)[0]
    names = [v.label for v in res.vars]
    for b in range(nb):
        ax = axes[b]
        base = nom["band"][b]["rl_worst"]
        lo, hi = res.oat[:, b, 0, 0] - base, res.oat[:, b, 0, 1] - base
        order = np.argsort(np.maximum(abs(lo), abs(hi)))
        y = np.arange(len(order))
        ax.barh(y, lo[order], color="#4c9be8", label="variable at its low limit")
        ax.barh(y, hi[order], color="#f4a259", alpha=0.85, label="at its high limit")
        lab = []
        for i in order:
            s = names[i]
            if res.share is not None and np.isfinite(res.share[i, b, 0]):
                s += f"  ({100 * res.share[i, b, 0]:.0f} %)"
            lab.append(s)
        ax.set_yticks(y, lab, fontsize=7)
        ax.axvline(0, color="0.3", lw=0.8)
        t = f"{setup.bands[b].name}: Δ worst RL [dB]"
        if res.share is not None:
            t += f"\nlinear share of MC variance: {100 * np.nansum(res.share[:, b, 0]):.0f} %"
        ax.set_title(t, fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        if b == 0:
            ax.legend(fontsize=7, loc="lower left")


PLOTS = (("Smith", plot_smith), ("S11", plot_s11), ("S21 / loss", plot_s21), ("VSWR", plot_vswr),
         ("Histograms", plot_hist), ("Sensitivity", plot_sens))
