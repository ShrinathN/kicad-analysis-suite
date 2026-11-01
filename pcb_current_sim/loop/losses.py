"""Losses of a switching cell's parts at the operating point, for the thermal map (DC & Thermal).

  hard-switched switch   (E_on + E_off) f_sw from the simulated edges + I_rms² R_on over its on-time (duty)
  the other position     a FET: I_rms² R_on over the rest of the period; a diode: V_f I + R_d I_rms²
I_rms² of the inductor current (mean I, peak-to-peak ripple ΔI) is I² + ΔI²/12.  Dead time (the body
diode carrying the current) is not included: it is short and needs the driver's timing.  An IC's
internal switches add up on the IC.  Snubbers are placed ideally (no footprint), so their C V² f_sw
is reported but not placed.
"""


def part_losses(board, s, tran):
    """([(ref, W, what)], notes) from the setup and the transient summary (None: conduction only)."""
    d_ctrl = min(max(s.duty, 0.0), 1.0)
    i2 = s.i_switch ** 2 + s.i_ripple ** 2 / 12
    direction, _ = s.current_dir(board)
    ctrl_pos = "upper" if direction == "out" else "lower"
    out, notes = {}, []

    def add(ref, w, what):
        r, ww = out.get(ref, (0.0, []))
        out[ref] = (r + w, ww + [what])
    for pos in ("upper", "lower"):
        share = d_ctrl if pos == ctrl_pos else 1 - d_ctrl
        parts = s.by_role(pos)
        fets = [d for d in parts if d.kind == "fet"]
        for d in parts:
            if d.kind == "fet":
                m = d.fet()
                p = i2 * share * m.rds_on / max(len(fets), 1)
                add(d.ref, p, f"{d.key} conduction {p:.3g} W")
            elif d.kind == "diode" and not fets:
                m = d.diode()
                p = share * (m.vf * s.i_switch + m.rd * i2)
                add(d.ref, p, f"{d.key} conduction {p:.3g} W")
        if pos == ctrl_pos and tran:
            sw = next((d for d in fets), None)
            if sw is not None:
                add(sw.ref, tran["P_sw_W"], f"{sw.key} switching {tran['P_sw_W']:.3g} W")
    if not tran:
        notes.append("The switching edges were not simulated: conduction losses only.")
    for sn in s.snubbers_on():
        notes.append(f"Snubber across the {sn.pos} switch: {sn.c * s.v_rail ** 2 * s.f_sw:.3g} W (not placed: "
                     "it has no footprint).")
    rows = [(ref, w, ", ".join(what)) for ref, (w, what) in out.items() if w > 0]
    missing = [ref for ref, _, _ in rows if not any(p.ref == ref for p in board.pads)]
    if missing:
        notes.append("Not on the board: " + ", ".join(missing))
    return rows, notes

