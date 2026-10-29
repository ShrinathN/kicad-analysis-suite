"""2D DC current-density simulation of F.Cu copper in test.kicad_pcb.

Copper is rasterised (tracks + pads, drills removed) and the Laplace equation
div(sigma grad V) = 0 is solved with a 5-point finite-volume scheme.  Each
plated barrel is an equipotential electrode: the source pad barrel at J1 and the
sink barrel at J2.  The solution is scaled so exactly I_TOTAL flows.
"""
import re, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PCB = sys.argv[1] if len(sys.argv) > 1 else "test.kicad_pcb"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
I_TOTAL = 10.0          # A
T_CU = 35e-6            # m (1 oz)
RHO = 1.72e-8           # ohm*m, copper @ 20 C
H = 0.02                # grid pitch, mm
R_SHEET = RHO / T_CU    # ohm/square

txt = open(PCB).read()

# ---------- parse ----------
def blocks(src, head):
    """Yield the text of each top-level '(head ...)' s-expression."""
    for m in re.finditer(r"\(" + head + r"[\s\n]", src):
        i, depth = m.start(), 0
        for j in range(i, len(src)):
            depth += src[j] == "("
            depth -= src[j] == ")"
            if depth == 0:
                yield src[i:j + 1]
                break

num = r"(-?[\d.]+)"
segments = []
for b in blocks(txt, "segment"):
    if '"F.Cu"' not in b:
        continue
    s = re.search(r"\(start " + num + " " + num, b).groups()
    e = re.search(r"\(end " + num + " " + num, b).groups()
    w = float(re.search(r"\(width " + num, b).group(1))
    net = re.search(r'\(net "([^"]*)"', b).group(1)
    segments.append((*map(float, s), *map(float, e), w, net))

pads = []   # (ref, num, x, y, sx, sy, shape, drill, net)
for fp in blocks(txt, "footprint"):
    ref = re.search(r'\(property "Reference" "([^"]+)"', fp).group(1)
    fx, fy = map(float, re.search(r"\(at " + num + " " + num, fp).groups())
    for p in blocks(fp, "pad"):
        pn, shape = re.match(r'\(pad "([^"]+)" \S+ (\S+)', p).groups()
        px, py = map(float, re.search(r"\(at " + num + " " + num, p).groups())
        sx, sy = map(float, re.search(r"\(size " + num + " " + num, p).groups())
        dr = float(re.search(r"\(drill " + num, p).group(1))
        net = re.search(r'\(net "([^"]*)"', p).group(1)
        pads.append((ref, pn, fx + px, fy + py, sx, sy, shape, dr, net))

# ---------- rasterise ----------
xs = [v for s in segments for v in (s[0], s[2])] + [p[2] for p in pads]
ys = [v for s in segments for v in (s[1], s[3])] + [p[3] for p in pads]
x0, x1 = min(xs) - 2, max(xs) + 2
y0, y1 = min(ys) - 2, max(ys) + 2
nx, ny = int((x1 - x0) / H) + 1, int((y1 - y0) / H) + 1
X, Y = np.meshgrid(x0 + H * np.arange(nx), y0 + H * np.arange(ny))

copper = np.zeros((ny, nx), bool)
for (ax, ay, bx, by, w, _) in segments:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
    copper |= np.hypot(X - ax - t * dx, Y - ay - t * dy) <= w / 2
holes = {}
for (ref, pn, px, py, sx, sy, shape, dr, net) in pads:
    if shape == "rect":
        copper |= (abs(X - px) <= sx / 2) & (abs(Y - py) <= sy / 2)
    else:
        copper |= np.hypot(X - px, Y - py) <= sx / 2
    holes[(ref, pn)] = np.hypot(X - px, Y - py) <= dr / 2
for hmask in holes.values():
    copper &= ~hmask

def barrel(key):
    """Copper cells touching the drill wall (the plated barrel)."""
    h = holes[key]
    ring = np.zeros_like(h)
    ring[1:] |= h[:-1]; ring[:-1] |= h[1:]
    ring[:, 1:] |= h[:, :-1]; ring[:, :-1] |= h[:, 1:]
    return ring & copper

# ---------- solve ----------
def solve(src_key, snk_key):
    src, snk = barrel(src_key), barrel(snk_key)
    idx = -np.ones(copper.shape, int)
    cells = np.flatnonzero(copper)
    idx.flat[cells] = np.arange(cells.size)
    n = cells.size
    rows, cols, vals = [], [], []
    for sl_a, sl_b in [((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))]:
        a, b = idx[sl_a], idx[sl_b]
        m = (a >= 0) & (b >= 0)
        a, b = a[m], b[m]
        # each square-cell link has conductance 1/R_sheet
        rows += [a, b, a, b]; cols += [a, b, b, a]
        g = np.full(a.size, 1 / R_SHEET)
        vals += [g, g, -g, -g]
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (n, n))
    fixed = np.zeros(n, bool); vfix = np.zeros(n)
    fixed[idx[src]] = True; vfix[idx[src]] = 1.0
    fixed[idx[snk]] = True
    free = ~fixed
    # only solve the connected component (other net is floating)
    ncomp, lab = sp.csgraph.connected_components(G, directed=False)
    active = np.isin(lab, np.unique(lab[fixed]))
    free &= active
    A = G[free][:, free]
    rhs = -G[free][:, fixed] @ vfix[fixed]
    V = np.full(n, np.nan); V[fixed] = vfix[fixed]
    V[free] = spla.spsolve(A.tocsc(), rhs)
    I_unit = (G[fixed & (vfix > 0)] @ np.nan_to_num(V)).sum()   # A for 1 V
    scale = I_TOTAL / I_unit
    Vg = np.full(copper.shape, np.nan); Vg.flat[cells] = V * scale
    return Vg, 1 / I_unit

V = np.full(copper.shape, np.nan)
results = {}
# Supply leaves J1 pin 2, goes to load at J2 pin 2, returns J2 pin 1 -> J1 pin 1
for name, a, b in [("Net-(J1-Pin_2)  [J1.2 -> J2.2]", ("J1", "2"), ("J2", "2")),
                   ("GND  [J2.1 -> J1.1 return]", ("J2", "1"), ("J1", "1"))]:
    Vn, R = solve(a, b)
    m = ~np.isnan(Vn)
    V[m] = Vn[m]
    results[name] = R

gy, gx = np.gradient(V, H * 1e-3)
J = np.hypot(gx, gy) / RHO                  # A/m^2
J[~copper] = np.nan
J_mm2 = J * 1e-6                            # A/mm^2
J_mm2[np.isnan(V)] = np.nan

print(f"Grid {nx}x{ny} @ {H*1000:.0f} um, copper cells {copper.sum()}")
print(f"Sheet resistance {R_SHEET*1e3:.3f} mOhm/sq, I = {I_TOTAL} A")
tot_P = 0
for k, R in results.items():
    P = I_TOTAL ** 2 * R; tot_P += P
    print(f"{k:34s} R = {R*1e3:7.2f} mOhm   dV = {I_TOTAL*R*1e3:7.1f} mV   P = {P:5.2f} W")
print(f"Total loop: R = {sum(results.values())*1e3:.2f} mOhm, P = {tot_P:.2f} W")
print(f"Peak J = {np.nanmax(J_mm2):.0f} A/mm^2 ({np.nanpercentile(J_mm2, 99.9):.0f} at 99.9th pct)")
for w in sorted({s[4] for s in segments}):
    print(f"  {w:.1f} mm trace: nominal J = {I_TOTAL/(w*T_CU*1e3):.0f} A/mm^2")

np.savez_compressed(f"{OUT}/result.npz", J=J_mm2, V=V, x0=x0, y0=y0, h=H)

# ---------- plot ----------
ext = [x0, x0 + nx * H, y0 + ny * H, y0]
fig, ax = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
im = ax[0].imshow(J_mm2, extent=ext, cmap="inferno", norm=LogNorm(5, np.nanmax(J_mm2)),
                  interpolation="nearest")
fig.colorbar(im, ax=ax[0], label="Current density |J|  [A/mm²]")
ax[0].set_title(f"Current density, F.Cu, 35 µm Cu, {I_TOTAL:.0f} A loop  (J1 pin 2 → J2 pin 2, return via GND)")
im2 = ax[1].imshow(V * 1e3, extent=ext, cmap="viridis", interpolation="nearest")
fig.colorbar(im2, ax=ax[1], label="Potential relative to each net's sink [mV]")
ax[1].set_title("Voltage drop along each net")
for a in ax:
    for (ref, pn, px, py, *_r) in pads:
        a.annotate(f"{ref}.{pn}", (px, py), xytext=(0, -14), textcoords="offset points",
                   ha="center", color="cyan", fontsize=9)
    a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]"); a.set_facecolor("#222")
fig.savefig(f"{OUT}/current_density.png", dpi=130)

# interactive (decimated 2x to keep the HTML small)
import plotly.graph_objects as go
d = 2
Jd = J_mm2[::d, ::d]
fig = go.Figure(go.Heatmap(z=Jd, x=x0 + H * d * np.arange(Jd.shape[1]), y=y0 + H * d * np.arange(Jd.shape[0]),
                           colorscale="Inferno", zmin=0, zmax=float(np.nanpercentile(J_mm2, 99.5)),
                           colorbar=dict(title="|J| A/mm²"),
                           hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>J=%{z:.0f} A/mm²<extra></extra>"))
fig.update_yaxes(autorange="reversed", scaleanchor="x")
fig.update_layout(title=f"Current density, {I_TOTAL:.0f} A, 1 oz Cu (hover for values)",
                  plot_bgcolor="#222", height=650)
fig.write_html(f"{OUT}/current_density.html", include_plotlyjs="cdn")
