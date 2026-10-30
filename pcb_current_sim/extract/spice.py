"""SPICE subcircuit of an extraction: plain R, L, C elements only, so it runs in any SPICE."""
import datetime
import math
import re

SUFFIX = ((1e12, "T"), (1e9, "G"), (1e6, "Meg"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"),
          (1e-9, "n"), (1e-12, "p"), (1e-15, "f"))


def fmt(v):
    """SPICE number with a scale suffix ('Meg' for 1e6: plain 'M' is milli in SPICE)."""
    if v == 0 or not math.isfinite(v):
        return "0"
    for m, s in SUFFIX:
        if abs(v) >= m * 0.9999995:
            return f"{v / m:.6g}{s}"
    return f"{v:.6g}"


def subckt_name(ext):
    return re.sub(r"[^A-Za-z0-9_]", "_", f"TRACE_{ext.net}_{'_'.join(ext.ports[:2])}")


def write(ext, name=None, board_name=""):
    name = name or subckt_name(ext)
    o, s = ext.opts, ext.summary
    ports = [f"P{i + 1}" for i in range(len(ext.ports))]
    out = [f"* {name}: RLC model of net {ext.net} extracted by kicad-analysis-suite",
           f"* board {board_name}, {datetime.date.today().isoformat()}"]
    for p, pid in zip(ports, ext.ports):
        out.append(f"*   {p} = pad {pid}")
    out.append(f"*   REF = return ({s.get('return_nets', '')}); connect it to the circuit's ground at the ports")
    out.append(f"* Sections <= lambda/10 at {fmt(o.f_max)}Hz; R is DC"
               + (" (scaled to the DC field solution)" if "R_scale" in s else " (rho l / w t)")
               + "; L, C quasi-static (2D field solver per cross-section).")
    for pr in ext.pairs:
        out.append(f"* {pr['pair']}: length {pr['length_mm']:.2f} mm, R {pr['R_mOhm']:.3g} mOhm, "
                   f"L {pr['L_nH']:.3g} nH, C {pr['C_pF']:.3g} pF, Z0 {pr['Z0_avg']:.1f} Ohm, "
                   f"delay {pr['delay_ps']:.1f} ps")
    out.append(f".subckt {name} {' '.join(ports)} REF")
    last = None
    diodes = False
    for nm, kind, a, b, v, tag in ext.elements:          # no inline comments: SPICE3 has none
        if tag != last:
            out.append(f"* {tag or 'shunt capacitance to the return'}")
            last = tag
        if kind == "D":
            # constant drop: Vf source + a near-ideal diode (about 15 mV at 1 A) + series R
            vf, r = v
            base = nm[2:]
            x, y = f"{base}_x", f"{base}_y"
            out.append(f"V_{base} {a} {x} {fmt(vf)}")
            out.append(f"{nm} {x} {y if r > 0 else b} DVF")
            if r > 0:
                out.append(f"R_{base}_s {y} {b} {fmt(r)}")
            diodes = True
        else:
            out.append(f"{nm} {a} {b} {fmt(v)}")
    if diodes:
        out.append(".model DVF D(IS=1e-12 N=0.02)")
    out.append(f".ends {name}")
    text = "\n".join(out).replace("→", "->") + "\n"
    return text.encode("ascii", "replace").decode()          # older SPICE readers want plain ASCII
