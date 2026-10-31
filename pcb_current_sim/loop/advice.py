"""Plain-language reading of a Switching Loop result: how good the loop is, what makes it so, and
what to change.  Rough rules of thumb for hard-switched SMD power stages, not limits."""
import math

from .model import Setup

# loop inductance bands [nH]: (upper limit, word)
L_BANDS = ((1.0, "excellent"), (2.5, "good"), (5.0, "fair"), (math.inf, "high"))


def rating(l_nh):
    return next(w for lim, w in L_BANDS if l_nh < lim)


def _kind(label, s):
    """What a breakdown entry is: 'via', 'copper', 'esl', 'link', 'switch', 'part'."""
    if label.endswith(" vias"):
        return "via"
    if "ESL" in label:
        return "esl"
    if "package" in label:
        ref = label.split()[0]
        d = s.device(ref) or next((x for x in s.devices if label.startswith(x.key)), None)
        if d is not None and d.role == "link":
            return "link"
        if d is not None and d.role in ("upper", "lower"):
            return "switch"
        return "part"
    return "copper"


def _tip(kind, label, s):
    name = label.split(" (")[0]
    if kind == "copper":
        net = label.rsplit(" ", 1)[0]
        what = ("the return path" if net == s.ret_net else "the switch node" if net == s.sw_net
                else "the supply side")
        fix = ("keep it directly under the supply-side copper, unbroken (no slots or splits in the way)"
               if net == s.ret_net else "shorten and widen it, with the return directly underneath")
        return f"copper of {net} ({what}): {fix}; or move a capacitor closer so the loop closes sooner"
    if kind == "via":
        return (f"vias of {label[:-5]}: add more vias in parallel at the capacitor and switch pads, or keep the "
                "loop on one layer over its return plane")
    if kind == "esl":
        return (f"{name}: a smaller package (0402 / 0603) or a reverse-geometry / several caps in parallel "
                "cut it")
    if kind == "link":
        return (f"{name}: the rail passes through this 0 Ω part before reaching its caps; put a cap on the "
                "switch side of it, right at the switch pins")
    if kind == "switch":
        return f"{name}: inside the part, fixed by its package"
    return f"{name}"


def verdict(r):
    """Lines of plain text for result r."""
    s = Setup.from_dict(r.setup)
    lines = []
    comms = [c for c in r.commutations if c["active"]] or r.commutations
    if len(comms) == 2 and abs(comms[0]["L_total_nH"] - comms[1]["L_total_nH"]) < 0.03 * comms[0]["L_total_nH"]:
        comms = [dict(comms[0], title="The hot loop (both edges)")]
    for c in comms:
        L, Lb = c["L_total_nH"], c["L_board_nH"]
        lines.append(f"{c['title']}: the hot loop is {L:.1f} nH ({rating(L)}); {Lb:.1f} nH of it is board copper, "
                     f"{L - Lb:.1f} nH inside the parts.")
        if math.isfinite(c["f_ring"]):
            q = c["Q"]
            lines.append(f"  It rings at {c['f_ring'] / 1e6:.0f} MHz"
                         + (f", lightly damped (Q {q:.0f}): expect a long ring on every edge, a source of EMI."
                            if q > 5 else f" (Q {q:.1f})."))
        if math.isfinite(c.get("V_over", math.nan)) and s.v_rail > 0:
            pct = 100 * c["V_over"] / s.v_rail
            lines.append(f"  Overshoot ≈ {c['V_over']:.1f} V on {s.v_rail:g} V ({pct:.0f} %)"
                         + (": check the switches' voltage rating." if pct > 30 else "."))
        top = sorted(c["breakdown"], key=lambda t: -t[1])
        fixable = [(lab, v) for lab, v in top if _kind(lab, s) != "switch" and v > 0.08 * L][:3]
        if fixable:
            lines.append("  Biggest contributors you can change:")
            for lab, v in fixable:
                lines.append(f"   • {v:.2f} nH: {_tip(_kind(lab, s), lab, s)}.")
        lines.append("")
    if len(r.commutations) > 1 and len(comms) == 1:
        lines.append("(The other edge only matters for a synchronous rectifier's reverse recovery; it is "
                     "shown in the table.)")
        lines.append("")
    lines += _transient(r, s)
    for g in r.gates:
        lg = g["L_gate_nH"] * 1e-9
        f = g.get("f_gate", math.nan)
        q = g.get("Q_gate", math.nan)
        line = f"Gate of {g['fet']}: {g['L_gate_nH']:.1f} nH loop, {g['R_gate']:.1f} Ω in it"
        if math.isfinite(f) and f > 0 and lg > 0:
            c_iss = 1 / ((2 * math.pi * f) ** 2 * lg)
            r_crit = 2 * math.sqrt(lg / c_iss)
            if math.isfinite(q) and q > 1:
                line += (f"; it rings (Q {q:.1f}): about {r_crit:.1f} Ω in total damps it (raise the gate "
                         "resistor, or shorten the gate loop).")
            else:
                line += "; well damped."
        else:
            line += "."
        if g["L_cs_nH"] > 0.5:
            line += (f" Common-source inductance {g['L_cs_nH']:.2f} nH: the power current's di/dt fights the gate "
                     "drive; return the driver to the FET's source pin (Kelvin) rather than to shared ground copper.")
        lines.append(line)
    return lines


def _transient(r, s):
    """Lines about the simulated edges."""
    if r.tran_error:
        return [f"The switching simulation failed: {r.tran_error}", ""]
    if not r.edges:
        return []
    out = ["Switching edges (simulated):"]
    for e in r.edges:
        m = e.metrics
        pct = 100 * m["overshoot"] / s.v_rail if s.v_rail > 0 else 0.0
        line = (f"  {e.switch} turning {e.kind} at {e.current:.3g} A: {m['blocking']} sees {m['V_peak']:.1f} V peak "
                f"({m['overshoot']:+.1f} V, {pct:.0f} % over the supply)")
        if pct > 50:
            line += "; check its voltage rating, and shrink the hot loop or add a snubber"
        out.append(line + ".")
        if "gate_margin" in m and math.isfinite(m["gate_margin"]):
            vg, vth = m["vgs_other_peak"], m["vth_other"]
            if m["gate_margin"] < 0:
                out.append(f"   ✖ {e.other}'s gate is pushed to {vg:.2f} V, above its {vth:g} V threshold: it "
                           "turns on briefly (shoot-through). Lower its gate-loop R / L when held off, add a gate-"
                           "source cap, a negative off-voltage, or slow this edge.")
            elif m["gate_margin"] < 0.5 * vth:
                out.append(f"   ⚠ {e.other}'s gate is pushed to {vg:.2f} V, close to its {vth:g} V threshold "
                           "(and V_th falls when hot).")
            else:
                out.append(f"   {e.other}'s gate stays at {vg:.2f} V, well below its {vth:g} V threshold.")
        if e.kind == "on" and m.get("i_rr_peak", 0) > 0.3 * e.current:
            out.append(f"   Reverse recovery: {m['i_rr_peak']:.2f} A back through {e.other} at turn-on; a "
                       "Schottky or a lower-Q_rr part reduces it.")
    t = r.tran
    out.append(f"  Switching loss ≈ {t['P_sw_W']:.3g} W at {t['f_sw'] / 1e3:g} kHz (E_on {t['E_on_uJ']:.3g} µJ, "
               f"E_off {t['E_off_uJ']:.3g} µJ).")
    out.append("")
    return out
