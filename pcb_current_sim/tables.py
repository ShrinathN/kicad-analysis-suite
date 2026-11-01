"""Toolkit-neutral helpers shared by the Tk and wx viewers."""
import math
import re

import numpy as np

from . import export

def parse_amount(text, unit):
    """A row's value with SI prefixes and an optional unit: '1m', '500mA', '2.5 A', '1u', '1k', '1M',
    '12 V', '3V3' (= 3.3 V), '-2m'.  unit: "A" or "V".  Raises ValueError."""
    from .suite.library import parse_si
    t = str(text).strip().replace(" ", "").replace(",", ".")
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    m = re.fullmatch(rf"(\d+)[{unit}{unit.lower()}](\d+)", t)          # 3V3, 1A5
    if m:
        return sign * float(f"{m.group(1)}.{m.group(2)}")
    if t[-1:] in (unit, unit.lower()):
        t = t[:-1]
    if not t:
        raise ValueError(text)
    return sign * parse_si(t)


def fmt_amount(v, unit):
    """'1 mA', '12 V', '1.23456 A': a row's value as the dialogs and tables show it, every digit kept
    (editing a row must not round it)."""
    from .suite.library import PREFIXES
    if not v or not math.isfinite(v):
        return f"{v:g} {unit}"
    for p, k in PREFIXES:
        if abs(v) >= k * 0.9999999:
            return f"{v / k:.12g} {p}{unit}"
    return f"{v:.12g} {unit}"


TABLES = [
    ("Excitations", "rows", export.ROW_COLS),
    ("Pads", "pads", export.PAD_COLS),
    ("Vias / barrels", "vias", export.VIA_COLS),
    ("Elements", "elements", export.ELEM_COLS),
    ("Regulators", "regulators", export.REG_COLS),
    ("Loss ranking", "segments", export.SEG_COLS),
    ("Inductance", "inductance", export.IND_COLS),
    ("Pulse", "pulse_table", export.PULSE_COLS),
]


ROW_TEXT = {  # kind -> (left list label, right list label, value label, hint)
    "I": ("Source pads (current enters)", "Sink pads (current leaves)", "Current [A]",
          "A current path: the current enters the copper at the source and leaves at the sink.\n"
          "Source and sink may be on different nets if resistance elements connect them."),
    "load": ("+ pads (load supply pin)", "− pads (load return pin)", "Current [A]",
             "A load (e.g. an IC) drawing this current from its + pad and returning it at its − pad.\n"
             "Pair it with a voltage source to see the voltage the load really gets."),
    "V": ("+ pads (source output)", "− pads (source return)", "Voltage [V]",
          "A voltage source (regulator, supply) holding V(+) − V(−) at this value.\n"
          "Its current is set by the loads and resistance elements connected to it."),
}

FLOAT_SETTINGS = ("T_cold", "T_amb", "h_conv", "emissivity", "k_fr4", "thermal_pitch", "rho20", "b_height",
                  "pulse_scale", "pulse_ms")
CHECKS = [("thermal", "Electro-thermal coupling (ρ(T), temperature map)"),
          ("h_auto", "h from temperature (natural convection + radiation)"),
          ("error_estimate", "Grid error estimate (extra coarse solve)"),
          ("auto_resistors", "Include resistor footprints automatically"),
          ("crop", "Grid only around the simulated copper"),
          ("magnetic", "Magnetic field maps + loop inductance"),
          ("pulse", "Pulse: transient heating / fusing"),
          ("pulse_from_steady", "Pulse starts from steady state (else ambient)")]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return " + ".join(v)
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v):
            return "-" if math.isnan(v) else "inf"
        return f"{v:.4g}"
    return str(v)


def sort_key(v):
    try:
        return (0, float(v))
    except ValueError:
        return (1, v)


def readout(res, layer, x, y, ref=None):
    """Multi-line cursor readout text; ref = (layer, x, y, V_mV) or None."""
    s = res.sample(layer, x, y)
    lines = [f"x = {s['x']:9.3f} mm", f"y = {s['y']:9.3f} mm", f"layer {s['layer']}"]
    if s.get("net"):
        lines.append(f"net {s['net'][:26]}")
    if s.get("J") is not None and math.isfinite(s["J"]):
        lines.append(f"|J| = {s['J']:10.2f} A/mm²")
        lines.append(f"V   = {s['V']:10.3f} mV")
        if ref:
            lines.append(f"ΔV  = {s['V'] - ref[3]:10.3f} mV (vs ref)")
        lines.append(f"p   = {s['p']:10.4f} W/mm²")
    else:
        lines.append("no simulated copper")
    if s.get("T") is not None and math.isfinite(s["T"]):
        lines.append(f"T   = {s['T']:10.1f} °C")
    if s.get("Tp") is not None and math.isfinite(s["Tp"]):
        lines.append(f"T_pulse = {s['Tp']:6.1f} °C (end)")
    if s.get("B_side"):
        b = math.sqrt(s["Bx"] ** 2 + s["By"] ** 2 + s["Bz"] ** 2)
        h = res.summary.get("b_height_mm", 1)
        lines.append(f"B {h:g} mm {s['B_side']}: |B| {b:.1f} µT")
        lines.append(f"  Bx {s['Bx']:.1f}  By {s['By']:.1f}  Bz {s['Bz']:.1f} µT")
    return "\n".join(lines)
