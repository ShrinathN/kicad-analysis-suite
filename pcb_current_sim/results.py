"""Result container: field maps plus the summary tables."""
import csv
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import elements as EL
from .raster import Grid

MIL2 = 0.0254 ** 2           # mm^2 per mil^2
T_MELT = 1083.0


def ipc2221(area_mm2, dT, outer):
    """IPC-2221 curve fit: allowed current [A] for a cross-section and temperature rise."""
    k = 0.048 if outer else 0.024
    return k * dT ** 0.44 * (area_mm2 / MIL2) ** 0.725


def onderdonk_time(area_mm2, I, Ta):
    """Seconds until a copper conductor fuses at current I (Onderdonk)."""
    cmil = area_mm2 / MIL2 * 4 / math.pi
    return math.log10(1 + (T_MELT - Ta) / (234 + Ta)) * cmil ** 2 / (33 * I * I) if I > 0 else math.inf


def onderdonk_current(area_mm2, t, Ta):
    cmil = area_mm2 / MIL2 * 4 / math.pi
    return cmil * math.sqrt(math.log10(1 + (T_MELT - Ta) / (234 + Ta)) / (33 * t))


@dataclass
class Result:
    board_path: str
    layers: list
    nets: list
    grid: Grid
    fields: dict                 # layer -> {f, net, V [mV], Jx, Jy, J [A/mm^2], p [W/mm^2]}
    tgrid: Grid = None
    T: dict = None               # layer -> coarse T [C]
    rows: list = field(default_factory=list)
    pads: list = field(default_factory=list)
    vias: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    elements: list = field(default_factory=list)
    net_table: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    outlines: dict = field(default_factory=dict)   # layer -> list of (kind, data) for drawing
    bgrid: Grid = None
    B: dict = None               # "above"/"below" -> {Bx, By, Bz} [µT] on bgrid
    inductance: list = field(default_factory=list)
    Tp: dict = None              # layer -> T at the end of the pulse [C] on pgrid
    pgrid: Grid = None
    pulse_frames: list = field(default_factory=list)   # [(t_ms, {layer: T on pgrid})]
    pulse: dict = None           # transient: t, T_max, ...

    QUANTITIES = {"J": "|J|  [A/mm²]", "V": "V  [mV]", "T": "T  [°C]", "p": "Power density  [W/mm²]",
                  "B": "|B|  [µT]", "Bn": "B normal (up)  [µT]", "Tp": "T end of pulse  [°C]"}

    def quantity_keys(self):
        have = {"T": self.T is not None, "B": self.B is not None, "Bn": self.B is not None,
                "Tp": self.Tp is not None}
        return [k for k in self.QUANTITIES if have.get(k, True)]

    @property
    def pulse_table(self):
        if not self.pulse:
            return []
        return [dict(t_ms=t, T_max_C=T) for t, T in zip(self.pulse["t_ms"], self.pulse["T_max"])]

    def b_side(self, layer):
        """B maps belong to the outer layers: the plane above the top / below the bottom surface."""
        if self.B is None:
            return None
        if layer == self.layers[0]:
            return "above"
        if layer == self.layers[-1]:
            return "below"
        return None

    # ---- point queries -------------------------------------------------
    def sample(self, layer, x, y):
        g = self.grid
        j, i = g.cell(x, y)
        out = dict(x=x, y=y, layer=layer)
        if 0 <= j < g.ny and 0 <= i < g.nx:
            F = self.fields[layer]
            nid = int(F["net"][j, i])
            out.update(net=self.nets[nid] if nid >= 0 else None,
                       J=float(F["J"][j, i]), V=float(F["V"][j, i]), p=float(F["p"][j, i]))
        for key, arr, tg in (("T", self.T, self.tgrid), ("Tp", self.Tp, self.pgrid)):
            if arr is not None:
                tj, ti = tg.cell(x, y)
                if 0 <= tj < tg.ny and 0 <= ti < tg.nx:
                    out[key] = float(arr[layer][tj, ti])
        side = self.b_side(layer)
        if side:
            bj, bi = self.bgrid.cell(x, y)
            if 0 <= bj < self.bgrid.ny and 0 <= bi < self.bgrid.nx:
                b = self.B[side]
                out.update(B_side=side, Bx=float(b["Bx"][bj, bi]), By=float(b["By"][bj, bi]),
                           Bz=float(b["Bz"][bj, bi]))
        return out

    def field(self, layer, q):
        if q in ("T", "Tp"):
            a = self.T if q == "T" else self.Tp
            return a[layer] if a is not None else None
        if q in ("B", "Bn"):
            side = self.b_side(layer)
            if side is None:
                return None
            b = self.B[side]
            return np.sqrt(b["Bx"] ** 2 + b["By"] ** 2 + b["Bz"] ** 2) if q == "B" else b["Bz"]
        return self.fields[layer][q]

    def extent(self, q):
        if q == "T":
            return self.tgrid.extent
        if q == "Tp":
            return self.pgrid.extent
        if q in ("B", "Bn"):
            return self.bgrid.extent
        return self.grid.extent

    # ---- persistence ---------------------------------------------------
    def save(self, path):
        arrs = {}
        for L, F in self.fields.items():
            for k, a in F.items():
                arrs[f"{L}|{k}"] = a
            if self.T is not None:
                arrs[f"{L}|T"] = self.T[L]
            if self.Tp is not None:
                arrs[f"{L}|Tp"] = self.Tp[L]
        for side, b in (self.B or {}).items():
            for k, a in b.items():
                arrs[f"B:{side}|{k}"] = a
        for n, (t, maps) in enumerate(self.pulse_frames):
            for L, a in maps.items():
                arrs[f"P:{n}|{L}"] = a
        np.savez_compressed(path, **arrs)
        meta = dict(board_path=self.board_path, layers=self.layers, nets=self.nets,
                    grid=vars(self.grid), tgrid=vars(self.tgrid) if self.tgrid else None,
                    rows=self.rows, pads=self.pads, vias=self.vias, segments=self.segments,
                    elements=self.elements, net_table=self.net_table, summary=self.summary, warnings=self.warnings,
                    outlines=self.outlines, bgrid=vars(self.bgrid) if self.bgrid else None,
                    pgrid=vars(self.pgrid) if self.pgrid else None,
                    inductance=self.inductance, pulse=self.pulse,
                    pulse_frame_t=[t for t, _ in self.pulse_frames])
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=1, default=float)

    @classmethod
    def load(cls, path):
        meta = json.load(open(os.path.splitext(path)[0] + ".json"))
        z = np.load(path)
        fields, T, Tp, Bf, frames = {}, {}, {}, {}, {}
        for key in z.files:
            L, k = key.split("|")
            if L.startswith("B:"):
                Bf.setdefault(L[2:], {})[k] = z[key]
            elif L.startswith("P:"):
                frames.setdefault(int(L[2:]), {})[k] = z[key]
            elif k == "T":
                T[L] = z[key]
            elif k == "Tp":
                Tp[L] = z[key]
            else:
                fields.setdefault(L, {})[k] = z[key]
        r = cls(meta["board_path"], meta["layers"], meta["nets"], Grid(**meta["grid"]), fields,
                Grid(**meta["tgrid"]) if meta["tgrid"] else None, T or None)
        for k in ("rows", "pads", "vias", "segments", "elements", "net_table", "summary", "warnings", "outlines",
                  "inductance"):
            setattr(r, k, meta.get(k, []))
        r.Tp, r.B, r.pulse = Tp or None, Bf or None, meta.get("pulse")
        r.pulse_frames = [(t, frames[n]) for n, t in enumerate(meta.get("pulse_frame_t", [])) if n in frames]
        r.bgrid = Grid(**meta["bgrid"]) if meta.get("bgrid") else None
        r.pgrid = Grid(**meta["pgrid"]) if meta.get("pgrid") else None
        return r

    def write_csv(self, outdir, stem="current_sim"):
        paths = []
        for name, rows in (("rows", self.rows), ("pads", self.pads), ("vias", self.vias),
                           ("segments", self.segments), ("elements", self.elements),
                           ("inductance", self.inductance), ("pulse", self.pulse_table),
                           ("nets", self.net_table)):
            if not rows:
                continue
            p = os.path.join(outdir, f"{stem}_{name}.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow({k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items()})
            paths.append(p)
        return paths

    # ---- text ------------------------------------------------------------
    def summary_text(self):
        s = self.summary
        out = [f"Board: {os.path.basename(self.board_path)}",
               f"Grid: {s['pitch_um']:g} µm, {s['nodes']:,} nodes; solve {s['seconds']:.1f} s",
               "Copper: " + ", ".join(f"{L} {t:.1f} µm" for L, t in s["thickness_um"].items())]
        if s["thermal"]:
            hdesc = (f"h from temperature (convection + radiation, ε = {s['emissivity']:g}), mean {s['h_conv']:.1f}"
                     if s.get("h_auto") else f"h = {s['h_conv']:g}")
            out.append(f"Thermal: ambient {s['T_amb']:g} °C, {hdesc} W/m²K per face, "
                       f"{s['iterations']} iteration(s), "
                       + ("converged" if s["converged"] else "NOT converged")
                       + (" - THERMAL RUNAWAY (copper would melt)" if s["runaway"] else ""))
            out.append(f"Hottest point: {s['T_max']:.1f} °C at ({s['T_max_xy'][0]:.2f}, {s['T_max_xy'][1]:.2f}) "
                       f"on {s['T_max_layer']}")
        else:
            out.append(f"Thermal coupling off: copper at {s['T_cold']:g} °C")
        out.append("")
        out.append("Excitations (solved simultaneously):")
        for r in self.rows:
            kind = r.get("kind", "I")
            pads = f"{'+'.join(r['src'])} -> {'+'.join(r['snk'])}"
            if kind == "V":
                line = (f"  {r['name']}: voltage source {pads} (+ -> -) {r['dV_mV'] / 1e3:.4g} V, "
                        f"delivers {r['I_A']:.4g} A")
            elif kind == "load":
                line = (f"  {r['name']}: load {pads} (+ -> -) {r['I_A']:g} A, "
                        f"sees {r['dV_mV'] / 1e3:.5g} V")
            else:
                line = (f"  {r['name']}: {pads}  {r['I_A']:g} A   "
                        f"dV = {r['dV_mV']:.2f} mV   R_eff = {r['R_mOhm']:.3f} mΩ")
                if s["thermal"] and r.get("R_cold_mOhm") is not None:
                    line += f"   (cold {r['R_cold_mOhm']:.3f} mΩ)"
            if r.get("err_pct") is not None:
                line += f"   grid err {r['err_pct']:.2f} %"
            out.append(line)
        out.append("")
        out.append(f"Total dissipation in copper: {s['P_total_W']:.3f} W")
        for n in self.net_table:
            out.append(f"  {n['net']}: {n['P_W']:.3f} W, peak |J| {n['J_max']:.0f} A/mm²")
        if self.elements:
            out.append(f"Resistance elements: {s.get('P_elements_W', 0):.3f} W in {len(self.elements)} part(s)")
            for e in self.elements:
                line = (f"  {e['name']} ({e['a']}-{e['b']}) {e['R']}: I = {e['I_A']:.4g} A, "
                        f"V = {e['V_mV']:.4g} mV, P = {e['P_W']:.4g} W")
                if e.get("util") is not None:
                    line += f" ({e['util'] * 100:.0f} % of {e['P_rated_W']:g} W)"
                out.append(line)
        for L, d in s["J_peak"].items():
            if d:
                out.append(f"Peak |J| on {L}: {d['J']:.0f} A/mm² at ({d['x']:.2f}, {d['y']:.2f}); "
                           f"99.5th pct {d['J995']:.0f} A/mm²")
        over = [g for g in self.segments if g.get("ipc_util_10") and g["ipc_util_10"] > 1]
        if over:
            out.append(f"{len(over)} track(s) exceed the IPC-2221 10 °C rise current "
                       f"(worst x{max(g['ipc_util_10'] for g in over):.1f}).")
        fuse = [g for g in self.segments if g.get("t_fuse_s") is not None and g["t_fuse_s"] < 10]
        if fuse:
            out.append(f"{len(fuse)} track(s) would fuse in < 10 s (Onderdonk, adiabatic).")
        if self.pulse:
            p = self.pulse
            out.append("")
            out.append(f"Pulse: {p['scale']:g} x table values for {p['ms']:g} ms, from "
                       + ("steady state" if p["from_steady"] else "ambient") + ":")
            if p["fused"]:
                f = p["fused"]
                out.append(f"  COPPER MELTS after {f['t_ms']:.3g} ms at ({f['x']:.2f}, {f['y']:.2f}) on {f['layer']}")
            else:
                w = p["where"]
                out.append(f"  T max at the end of the pulse: {p['T_end_max']:.1f} °C at ({w['x']:.2f}, {w['y']:.2f}) "
                           f"on {w['layer']}")
        if self.inductance:
            out.append("")
            out.append("Loop inductance (DC current distribution; each loop closed straight from sink to source):")
            for d in self.inductance:
                if d["row_a"] == d["row_b"]:
                    out.append(f"  {d['row_a']}: L = {d['L_nH']:.2f} nH (closure {d['closure_mm']:.1f} mm)")
                else:
                    out.append(f"  {d['row_a']} / {d['row_b']}: M = {d['L_nH']:.2f} nH, k = {d['k']:.3f}"
                               if d["k"] is not None else f"  {d['row_a']} / {d['row_b']}: M = {d['L_nH']:.2f} nH")
            if s.get("W_mag_nJ") is not None:
                out.append(f"  Magnetic energy at the operating point: {s['W_mag_nJ']:.3g} nJ")
        if self.B is not None:
            for side, b in self.B.items():
                m = np.sqrt(b["Bx"] ** 2 + b["By"] ** 2 + b["Bz"] ** 2)
                j, i = np.unravel_index(np.argmax(m), m.shape)
                g = self.bgrid
                out.append(f"Peak |B| {s.get('b_height_mm', 1):g} mm {side} the board: {m[j, i]:.1f} µT at "
                           f"({g.x0 + (i + 0.5) * g.h:.2f}, {g.y0 + (j + 0.5) * g.h:.2f})")
        if self.warnings:
            out.append("")
            out += ["Warning: " + w for w in self.warnings]
        out.append("")
        out.append("Notes: R_eff = dV/I per row; with shared copper (or elements) it includes the other rows' "
                   "drops. Thermal model is a 2D sheet (resistance elements heat their pads; no other "
                   "component heat, no enclosure, still-air h).")
        return "\n".join(out)


# ---------------------------------------------------------------------------
def build_result(board, R, net, sol, cold, coarse, tm, rows, s, thick, info, mag=None):
    g = R.grid
    h2 = g.h * g.h
    fields = {}
    for L in board.copper_layers:
        lr = R.layers[L]
        J = sol.J(L)
        V = sol.V_layers[L]
        fields[L] = dict(f=lr.f, net=lr.net, V=(V * 1e3).astype(np.float32),
                         Jx=sol.Jx[L].astype(np.float32), Jy=sol.Jy[L].astype(np.float32),
                         J=J.astype(np.float32),
                         p=np.where(np.isnan(V), np.nan, sol.q[L] / h2).astype(np.float32))
    res = Result(board.path, board.copper_layers, R.nets, g, fields,
                 tm.tg if tm else None, tm.T if tm else None)
    res.warnings = list(board.warnings) + list(R.warnings)
    Tfine = sol.T

    def T_at(L, x, y):
        if Tfine is None:
            return s.T_cold
        j, i = g.cell(x, y)
        return float(Tfine[L][min(max(j, 0), g.ny - 1), min(max(i, 0), g.nx - 1)])

    # rows: the error estimate tracks each kind's key quantity
    #   current path: R_eff = dV/I;  load: the voltage across the load;  voltage source: its current
    for k, row in enumerate(rows):
        dV, I = sol.row_dV[k], sol.row_I[k]
        d = dict(name=row.name, kind=row.kind, src=list(row.src), snk=list(row.snk), I_A=I, dV_mV=dV * 1e3,
                 R_mOhm=None, R_cold_mOhm=None, R_err_mOhm=None, R_err_pct=None, R_extrap_mOhm=None,
                 err_pct=None)
        if row.kind == "I" and I:
            d["R_mOhm"] = dV / I * 1e3
            d["R_cold_mOhm"] = cold.row_dV[k] / cold.row_I[k] * 1e3
        if coarse is not None:
            key = {"I": lambda S: S.row_dV[k] / S.row_I[k] if S.row_I[k] else 0.0,
                   "load": lambda S: S.row_dV[k], "V": lambda S: S.row_I[k]}[row.kind]
            r1, r2 = key(cold), key(coarse)
            d["err_pct"] = abs(r1 - r2) / abs(r1) * 100 if r1 else 0.0
            if row.kind == "I":
                d["R_err_mOhm"] = abs(r1 - r2) * 1e3
                d["R_err_pct"] = d["err_pct"]
                d["R_extrap_mOhm"] = (r1 + (r1 - r2) / 3) * 1e3
            if d["err_pct"] > 2:
                res.warnings.append(f"Row '{row.name}': grid error {d['err_pct']:.1f} % - use a finer pitch.")
        res.rows.append(d)

    # per-net power and peak J
    P_net = {n: 0.0 for n in R.nets}
    J_net = {n: 0.0 for n in R.nets}
    for L in board.copper_layers:
        lr, q, J = R.layers[L], sol.q[L], fields[L]["J"]
        for k, n in enumerate(R.nets):
            m = lr.net == k
            P_net[n] += float(q[m].sum())
            if m.any():
                J_net[n] = max(J_net[n], float(np.nanmax(np.where(m, J, np.nan), initial=0)))
    for bi, L1, L2, I, P, dV in sol.barrel_links:
        P_net[net.barrels[bi]["hole"]["net"]] += P
    res.net_table = [dict(net=n, P_W=P_net[n], J_max=J_net[n]) for n in R.nets]

    # pads
    roles = {}
    for r, row in enumerate(rows):
        for side, ids in (("src", row.src), ("snk", row.snk)):
            for pid in ids:
                roles.setdefault(pid, []).append(f"{side}:{row.name}")
    for e in net.elements:
        for pid in (e.a, e.b):
            roles.setdefault(pid, []).append(f"element {e.name}")
    for k, p in enumerate(board.pads):
        if p.net not in R.nets:
            continue
        b = net.hole_barrel.get(("pad", k))
        if b is not None:
            Vs = [sol.V[nd] for nd in b["nodes"].values()]
        else:
            Vs = []
            for L in p.layers:
                cells = R.pad_cells.get((k, L), np.zeros(0, np.int64))
                Vs += list(sol.V_layers[L].ravel()[cells])
        Vs = np.array(Vs, float)
        I = sum(v for (kk, r, side), v in sol.pad_current.items() if kk == k)
        res.pads.append(dict(pad=p.id, net=p.net, role=", ".join(roles.get(p.id, [])) or "-",
                             I_A=float(I), V_mV=float(np.nanmean(Vs)) * 1e3 if np.isfinite(Vs).any() else float("nan"),
                             T_C=T_at(p.layers[0], *p.pos) if p.layers else float("nan"),
                             x=p.pos[0], y=p.pos[1]))

    # resistance elements
    P_el = 0.0
    pad_of = {p.id: p for p in board.pads}
    for k, e in enumerate(net.elements):
        I, dV = sol.element_I[k], sol.element_dV[k]
        P = abs(I * dV)
        P_el += P
        pa, pb = pad_of[e.a], pad_of[e.b]
        d = dict(name=e.name, a=e.a, b=e.b, net_a=pa.net, net_b=pb.net, R=EL.fmt_ohms(e.R), R_ohm=e.R,
                 I_A=I, V_mV=dV * 1e3, P_W=P, P_rated_W=e.P_rated,
                 util=P / e.P_rated if e.P_rated else None,
                 T_C=max(T_at(pa.layers[0], *pa.pos), T_at(pb.layers[0], *pb.pos)),
                 source="auto" if e.auto else "user", board_heat=e.heats,
                 x=(pa.pos[0] + pb.pos[0]) / 2, y=(pa.pos[1] + pb.pos[1]) / 2)
        if d["util"] is not None and d["util"] > 1:
            res.warnings.append(f"{e.name} dissipates {P:.3g} W, above its {e.P_rated:g} W rating.")
        res.elements.append(d)

    # vias and THT barrels
    per_barrel = {}
    for bi, L1, L2, I, P, dV in sol.barrel_links:
        e = per_barrel.setdefault(bi, dict(I=0.0, P=0.0))
        e["I"] = max(e["I"], abs(I))
        e["P"] += P
    vias = []
    for bi, b in enumerate(net.barrels):
        hole = b["hole"]
        Ls = list(b["nodes"])
        e = per_barrel.get(bi, dict(I=0.0, P=0.0))
        A = math.pi * s.t_plate * 1e3 * (hole["drill"] + s.t_plate * 1e3)      # mm^2
        cap = ipc2221(A, 10, outer=False)
        Vt, Vb = sol.V[b["nodes"][Ls[0]]], sol.V[b["nodes"][Ls[-1]]]
        vias.append(dict(id=hole["label"], kind="via" if hole["kind"] == "via" else "THT", L_top=Ls[0], L_bot=Ls[-1],
                         net=hole["net"], x=hole["pos"][0], y=hole["pos"][1], drill_mm=hole["drill"],
                         I_A=e["I"], dV_mV=float(np.nan_to_num(Vt - Vb)) * 1e3, P_W=e["P"],
                         I_cap_10C=cap, util=e["I"] / cap if cap else 0.0, group="-", share_pct=None))
    # group parallel vias (same net, close together)
    vi = [k for k, v in enumerate(vias) if v["kind"] == "via"]
    parent = {k: k for k in vi}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sizes = {k: board.vias[net.barrels[k]["hole"]["index"]].size for k in vi}
    for a in vi:
        for b in vi:
            if a < b and vias[a]["net"] == vias[b]["net"]:
                d = math.dist((vias[a]["x"], vias[a]["y"]), (vias[b]["x"], vias[b]["y"]))
                if d <= max(1.5, 4 * max(sizes[a], sizes[b])):
                    parent[find(a)] = find(b)
    groups = {}
    for k in vi:
        groups.setdefault(find(k), []).append(k)
    for gn, (root, mem) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        tot = sum(vias[k]["I_A"] for k in mem)
        for k in mem:
            vias[k]["group"] = f"G{gn + 1} ({len(mem)})"
            vias[k]["share_pct"] = vias[k]["I_A"] / tot * 100 if tot else 0.0
    res.vias = sorted(vias, key=lambda v: -v["I_A"])

    # per-primitive loss ranking
    segs, seg_cells = [], []
    for L in board.copper_layers:
        lr = R.layers[L]
        F = fields[L]
        ok = (lr.owner >= 0) & ~np.isnan(F["V"])
        own = lr.owner[ok]
        if own.size == 0:
            continue
        flat = np.flatnonzero(ok)
        order = np.argsort(own, kind="stable")
        own_s, flat_s = own[order], flat[order]
        cuts = np.flatnonzero(np.diff(own_s)) + 1
        for grp in np.split(np.arange(own_s.size), cuts):
            pid = int(own_s[grp[0]])
            pr = R.prims[pid]
            cells = flat_s[grp]
            Jc = F["J"].ravel()[cells]
            Vc = F["V"].ravel()[cells]
            P = float(sol.q[L].ravel()[cells].sum())
            Jmed = float(np.nanmedian(Jc)) if Jc.size else 0.0
            Jmax = float(np.nanmax(Jc)) if Jc.size else 0.0
            d = dict(item=pr.label, kind=pr.kind, net=pr.net, layer=L, P_W=P, share_pct=0.0,
                     dV_mV=float(np.nanmax(Vc) - np.nanmin(Vc)), J_med=Jmed, J_max=Jmax,
                     crowding=Jmax / Jmed if Jmed > 0 else float("nan"),
                     width_mm=pr.width or None, I_A=None, ipc_util_10=None, ipc_util_20=None,
                     t_fuse_s=None, I_fuse_10ms=None, I_fuse_1s=None,
                     x=pr.centre[0] if pr.centre else None, y=pr.centre[1] if pr.centre else None)
            if pr.kind in ("track", "arc") and pr.width:
                A = pr.width * thick[L] * 1e3          # mm^2
                I = Jmed * A
                outer = board.is_outer(L)
                d.update(I_A=I, ipc_util_10=I / ipc2221(A, 10, outer), ipc_util_20=I / ipc2221(A, 20, outer),
                         t_fuse_s=onderdonk_time(A, I, s.T_amb), I_fuse_10ms=onderdonk_current(A, 0.01, s.T_amb),
                         I_fuse_1s=onderdonk_current(A, 1.0, s.T_amb))
                # end-to-end drop along the track centreline
                ys, xs = np.divmod(cells, g.nx)
                cx, cy = g.x0 + (xs + 0.5) * g.h, g.y0 + (ys + 0.5) * g.h
                va = Vc[np.argmin(np.hypot(cx - pr.start[0], cy - pr.start[1]))]
                vb = Vc[np.argmin(np.hypot(cx - pr.end[0], cy - pr.end[1]))]
                d["dV_mV"] = float(abs(va - vb))
            segs.append(d)
            seg_cells.append((L, cells))
    for d in segs:
        tot = P_net.get(d["net"], 0)
        d["share_pct"] = d["P_W"] / tot * 100 if tot else 0.0
    pulse = info.get("pulse")
    if pulse is not None:
        Tpf = pulse["fine"]
        for d, (L, cells) in zip(segs, seg_cells):
            d["T_pulse_C"] = float(np.max(Tpf[L].ravel()[cells])) if len(cells) else None
        res.Tp = {L: a.astype(np.float32) for L, a in pulse["coarse"].items()}
        res.pgrid = pulse["tgrid"]
        res.pulse = {k: v for k, v in pulse.items() if k not in ("coarse", "fine", "tgrid", "frames")}
        res.pulse_frames = pulse["frames"]
        if pulse["fused"]:
            f = pulse["fused"]
            res.warnings.append(f"Pulse: copper reaches its melting point after {f['t_ms']:.3g} ms at "
                                f"({f['x']:.2f}, {f['y']:.2f}) on {f['layer']}.")
    res.segments = sorted(segs, key=lambda d: -d["P_W"])

    # summary
    peaks = {}
    for L in board.copper_layers:
        J = fields[L]["J"]
        if np.isfinite(J).any() and np.nanmax(J) > 1e-3:
            j, i = np.unravel_index(np.nanargmax(J), J.shape)
            peaks[L] = dict(J=float(J[j, i]), x=g.x0 + (i + 0.5) * g.h, y=g.y0 + (j + 0.5) * g.h,
                            J995=float(np.nanpercentile(J[J > 0], 99.5)))
        else:
            peaks[L] = None
    from .magnetic import layer_z
    zL, zb = layer_z(board, thick)
    summ = dict(pitch_um=g.h * 1e3, nodes=net.n, seconds=info["seconds"],
                layer_z_mm={L: z * 1e3 for L, z in zL.items()}, z_bottom_mm=zb * 1e3,
                board_outline=[np.asarray(lp)[::max(1, len(lp) // 400)].tolist() for lp in board.outline],
                thickness_um={L: thick[L] * 1e6 for L in board.copper_layers},
                thermal=bool(s.thermal), T_amb=s.T_amb, T_cold=s.T_cold,
                h_conv=tm.h_mean() if tm is not None else s.h_conv, h_auto=bool(getattr(s, "h_auto", False)),
                emissivity=getattr(s, "emissivity", None),
                iterations=info["iterations"], converged=info["converged"], runaway=info["runaway"],
                P_total_W=sum(P_net.values()), P_elements_W=P_el, J_peak=peaks, stackup_found=board.stackup_found)
    if tm is not None and tm.T is not None:
        best = (-1e9, None, None)
        for L, Tc in tm.T.items():
            if np.isfinite(Tc).any():
                j, i = np.unravel_index(np.nanargmax(Tc), Tc.shape)
                if Tc[j, i] > best[0]:
                    best = (float(Tc[j, i]), L, (tm.tg.x0 + (i + 0.5) * tm.tg.h, tm.tg.y0 + (j + 0.5) * tm.tg.h))
        summ.update(T_max=best[0], T_max_layer=best[1], T_max_xy=best[2])
        if not info["converged"]:
            res.warnings.append("Electro-thermal loop did not converge"
                                + (" (thermal runaway: copper exceeds its melting point)." if info["runaway"] else "."))
    if not board.stackup_found:
        res.warnings.append("No stackup in the board file: copper thickness defaults to 35 µm "
                            "unless overridden (outer layers are often 45-60 µm after plating).")
    if mag is not None:
        res.bgrid = mag["grid"]
        res.B = {side: dict(Bx=(b[0] * 1e6).astype(np.float32), By=(b[1] * 1e6).astype(np.float32),
                            Bz=(b[2] * 1e6).astype(np.float32)) for side, b in mag["B"].items()}
        Lm, clos = mag["L"], mag["closure"]
        for a, ra in enumerate(rows):
            for b, rb in enumerate(rows):
                if b < a or not np.isfinite(Lm[a, b]):
                    continue
                k = Lm[a, b] / math.sqrt(Lm[a, a] * Lm[b, b]) if a != b and Lm[a, a] > 0 and Lm[b, b] > 0 else None
                res.inductance.append(dict(row_a=ra.name, row_b=rb.name, L_nH=Lm[a, b] * 1e9, k=k,
                                           closure_mm=clos[a] if a == b else None))
        I = np.array([d["I_A"] for d in res.rows])
        ok = [k for k in range(len(rows)) if np.isfinite(Lm[k, k])]
        if ok:
            sub = Lm[np.ix_(ok, ok)]
            summ["W_mag_nJ"] = float(0.5 * I[ok] @ sub @ I[ok]) * 1e9
        for k, row in enumerate(rows):
            if not np.isfinite(Lm[k, k]):
                res.warnings.append(f"Row '{row.name}': no closed loop on its own (e.g. a source whose loads are "
                                    "current sinks); no inductance.")
            elif clos[k] and clos[k] > 5:
                res.warnings.append(f"Row '{row.name}': its loop is closed by a {clos[k]:.1f} mm straight line from "
                                    "sink back to source; the inductance includes that assumed external path.")
        summ["b_height_mm"] = s.b_height
    res.summary = summ

    # outlines for drawing (track centrelines + pad shapes of simulated nets)
    for L in board.copper_layers:
        items = []
        for t in board.tracks:
            if t.layer == L and t.net in R.nets:
                if t.kind == "segment":
                    items.append(["line", [t.start, t.end]])
                else:
                    items.append(["line", [p.a for p in t.geom.parts] + [t.geom.parts[-1].b]])
        for p in board.pads:
            if L in p.layers and p.net in R.nets:
                items.append(["pad", [p.pos, p.id]])
        for e in net.elements:
            pa, pb = pad_of[e.a], pad_of[e.b]
            items.append(["elem", [pa.pos, pb.pos, e.name]])
        res.outlines[L] = items
    return res
