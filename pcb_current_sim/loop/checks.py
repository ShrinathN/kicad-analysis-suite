"""Readiness checks of a switching cell, before (and independent of) the analysis.

check(board, s) returns Issues, worst first.  "error" blocks the analysis (analysis.run raises the
first one), "warn" makes the results doubtful, "info" says what more input would give.  Each issue
may carry a fix: an action key the GUI turns into a button:
    "detect"            detect the cell's parts again
    "edit:<key>"        open a part's model
    "off:<k1>,<k2>"     switch those parts off
    "drive:<key>"       add a gate drive for that FET
    "opt:<field>"       go to an option field
"""
import math
import os
from dataclasses import dataclass

from . import model as M
from ..suite.library import fmt_si

SMALL_CAP, SMALL_CAP_MM = M.SMALL_CAP, M.SMALL_CAP_MM


@dataclass
class Issue:
    level: str          # "error" | "warn" | "info"
    text: str
    fix: str = ""
    fix_label: str = ""


def _cap_mm(board, s, ref):
    """Distance of a part from the middle of the cell's switches [mm]."""
    sw = [d for d in s.devices if d.role in ("upper", "lower")]
    cx = [M._centre(board, d.ref) for d in sw]
    if not cx:
        return 0.0
    x0 = sum(x for x, _ in cx) / len(cx)
    y0 = sum(y for _, y in cx) / len(cx)
    x, y = M._centre(board, ref)
    return math.hypot(x - x0, y - y0)


def errors(s):
    """Problems that make the analysis meaningless (board-free: the setup alone)."""
    out = []
    for pos in ("upper", "lower"):
        if not s.by_role(pos):
            out.append(Issue("error", f"Nothing in the {pos} position: the cell needs a part (or an IC's internal "
                                      f"switch) between {'the rail and the switch node' if pos == 'upper' else 'the switch node and the return'}.",
                             "detect", "Detect parts"))
    if not s.by_role("cap"):
        out.append(Issue("error", "No hot-loop capacitor is on: the loop has no path back.", "detect", "Detect parts"))
    group = M.rail_nets(s)
    stray = [d for d in s.by_role("cap") if not (set(d.nets.values()) & group and s.ret_net in d.nets.values())]
    if stray:
        out.append(Issue(
            "error",
            "Hot-loop capacitors not across the cell's rail and return ("
            + ", ".join(f"{d.ref} on {' / '.join(sorted(set(d.nets.values())))}" for d in stray)
            + f"; rail {' / '.join(sorted(group))}, return {s.ret_net}): the loop cannot close through them. "
              "Detect again, switch them off, or add the 0 Ω part joining their net to the rail as a rail link.",
            "detect", "Detect parts"))
    for pos in ("upper", "lower"):
        want = (group, {s.sw_net}) if pos == "upper" else ({s.sw_net}, {s.ret_net})
        for d in s.by_role(pos):
            hi, lo = (d.nets.get("D"), d.nets.get("S")) if d.kind == "fet" else (d.nets.get("K"), d.nets.get("A"))
            if d.kind == "fet" and d.fet().channel == "P":
                hi, lo = lo, hi
            if hi not in want[0] or lo not in want[1]:
                out.append(Issue("error", f"{d.key} ({pos} switch) sits between {hi} and {lo}, but the {pos} position "
                                          f"is between {' / '.join(sorted(want[0]))} and {' / '.join(sorted(want[1]))}.",
                                 f"edit:{d.key}", "Edit pins"))
    return out


def check(board, s):
    """All issues of cell s on this board, worst first."""
    if not s.sw_net:
        return [Issue("error", "No switch node: pick it and press Detect parts.", "detect", "Detect parts")]
    out = []
    # the board changed since the cell was set up: parts or pins that are gone
    refs = {p.ref for p in board.pads}
    gone = [d.key for d in s.devices if d.enabled and not d.pads and d.ref not in refs]
    moved = [d.key for d in s.devices if d.enabled and d.ref in refs and not d.pads
             and any(net and not any(p.ref == d.ref and p.net == net for p in board.pads)
                     for pin, net in d.nets.items() if not (d.internal and pin == "G"))]
    if gone or moved:
        out.append(Issue("error", "The board changed since this cell was set up: "
                         + "; ".join(x for x in (f"not on the board: {', '.join(gone)}" if gone else "",
                                                 f"pins on other nets now: {', '.join(moved)}" if moved else "") if x)
                         + ".", "detect", "Detect parts"))
    out += errors(s)
    for g in s.drives:
        for what, (ref, net) in (("output", g.out), ("return", g.ret)):
            if not any(p.ref == ref and p.net == net for p in board.pads):
                out.append(Issue("error", f"{g.fet}'s gate drive: {ref} has no pin on {net} (its driver {what} pin).",
                                 "", ""))
    # doubtful results
    generic = [d for d in s.devices if d.enabled and d.kind in ("fet", "diode") and d.model.get("assumed")]
    for d in generic:
        what = "switch" if d.kind == "fet" else "diode"
        out.append(Issue("warn", f"{d.key}: generic {what} values. Enter its datasheet capacitances"
                                 f"{', R_on' if d.kind == 'fet' else ''} and package inductances.",
                         f"edit:{d.key}", "Edit model"))
    small = [d for d in s.by_role("cap") if d.model.get("value", 1.0) < SMALL_CAP
             and _cap_mm(board, s, d.ref) > SMALL_CAP_MM]
    if small:
        out.append(Issue("warn", f"Small capacitors far from the switches counted as hot-loop caps: "
                                 f"{', '.join(d.ref + ' ' + fmt_si(d.model.get('value', 0), 'F') for d in small)}. "
                                 "These are usually filter / RF caps.",
                         "off:" + ",".join(d.key for d in small), "Switch off"))
    far = [d for d in s.by_role("cap") if _cap_mm(board, s, d.ref) > 15.0 and d not in small]
    if far:
        out.append(Issue("info", f"Hot-loop caps more than 15 mm from the switches: {', '.join(d.ref for d in far)}. "
                                 "Fine if they are the only ones; a cap right at the switch pins makes a much "
                                 "smaller loop."))
    for d in s.devices:
        if d.enabled and d.kind == "fet" and not d.internal and d.nets.get("G") \
                and not any(s.device(g.fet) is d for g in s.drives):
            out.append(Issue("warn", f"{d.key} has no gate drive: its gate loop is not analysed.",
                             f"drive:{d.key}", "Add drive"))
    vendor = [d for d in s.devices if d.enabled and d.kind == "fet" and d.model.get("spice_file")
              and d.model.get("spice_subckt")]
    for d in vendor:
        if not os.path.exists(d.model["spice_file"]):
            out.append(Issue("error", f"{d.key}: its SPICE model file is missing ({d.model['spice_file']}).",
                             f"edit:{d.key}", "Edit model"))
    if vendor and s.transient and s.simulator != "builtin":
        from . import spice_tran as SP
        if SP.available() is None:
            out.append(Issue("warn", "ngspice was not found, so the vendor SPICE models can't be used: the "
                                     "switching simulation uses the datasheet values.", "", ""))
    if s.v_rail <= 0:
        out.append(Issue("warn", "The voltage across the switches is 0: capacitances can't be scaled.",
                         "opt:v_rail", "Set voltage"))
    if not s.i_switch:
        out.append(Issue("info", "Give the inductor current (Operating point) for the switching simulation and "
                                 "the overshoot in volts.", "opt:i_switch", "Set current"))
    if s.transient and s.sw_net and s.by_role("upper") and s.by_role("lower"):
        d, why = s.current_dir(board)
        if why.startswith("guessed"):
            out.append(Issue("warn", f"The inductor current's direction is a guess ({'out of' if d == 'out' else 'into'} "
                                     "the switch node, as in a buck). It decides which switch is hard-switched: set "
                                     "it under Operating point.", "opt:i_dir", "Set direction"))
        pos = "upper" if d == "out" else "lower"
        ctrl = next((x for x in s.by_role(pos) if x.kind == "fet"), None)
        if ctrl is None:
            out.append(Issue("error" if s.i_switch else "warn",
                             f"No switch in the {pos} position to simulate (inductor current "
                             f"{'out of' if d == 'out' else 'into'} the switch node): check the direction.",
                             "opt:i_dir", "Set direction"))
        elif ctrl.internal and not (ctrl.model.get("t_rise") and ctrl.model.get("t_fall")):
            out.append(Issue("info", f"{ctrl.key}: no rise / fall time given; the switching simulation assumes "
                                     f"{M.INTERNAL_EDGE * 1e9:g} ns edges, which sets its di/dt and overshoot.",
                             f"edit:{ctrl.key}", "Edit model"))
    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda i: order[i.level])
