"""A switching cell's operating point from the converter's own terms (V_in, V_out, I_out, L, f_sw).

The cell needs the voltage across the off switch, the inductor current the switches hand over at the
edges (mean and peak-to-peak ripple) and the controlled switch's duty cycle.  Ideal converter in
steady state (no switch or inductor drops; a boost's input current divided by the efficiency):

  buck:   V_sw = V_in,  D = V_out / V_in,      I_L = I_out,                  ΔI = (V_in - V_out) D / (L f)
  boost:  V_sw = V_out, D = 1 - V_in / V_out,  I_L = I_out V_out / (V_in η), ΔI = V_in D / (L f)
  half-bridge: as a buck with the bus as V_in and the mean phase voltage as V_out; the sign of the
          phase current says whether the leg sources (out of the switch node) or sinks
  4-switch buck-boost: the input leg bucks when V_out < V_in, the output leg boosts when V_out > V_in;
          the other leg then sits still (input leg: upper on, D = 1; output leg: upper on, D = 0)

With a diode rectifier the converter goes discontinuous when ΔI/2 > I_L: the switch then turns on at
zero current and off at the peak I_pk of a triangle that ends before the period does.  A synchronous
converter in forced PWM stays continuous: the current goes negative in the valley, so the turn-on is
soft (the simulation runs it at about zero current).
"""
import math

from . import model as M


def from_converter(topo, v_in, v_out, i_out, l, f_sw, eff=1.0, sync=True):
    """dict(v_rail, i_switch, i_ripple, duty, i_dir, switching, mode, notes) for topology key topo.
    Raises ValueError for inputs the topology can't have."""
    if topo not in M.TOPO_NAME:
        raise ValueError("Pick the topology first (the cell's kind decides the formulas).")
    if v_in <= 0 or f_sw <= 0:
        raise ValueError("V_in and f_sw must be above 0.")
    eff = min(max(eff, 0.3), 1.0)
    notes = []
    sign = 1.0
    if topo == "half_bridge":
        if not 0 <= v_out <= v_in:
            raise ValueError("A half-bridge's mean phase voltage is between 0 and the bus voltage.")
        sign = -1.0 if i_out < 0 else 1.0
        i_out = abs(i_out)
        kind = "buck"
    elif topo in ("bb_in", "bb_out"):
        if v_out <= 0:
            raise ValueError("V_out must be above 0.")
        kind = "buck" if v_out < v_in else "boost"
        if abs(v_out - v_in) < 0.1 * v_in:
            notes.append("V_out is within 10 % of V_in: many controllers switch both legs here (buck-boost "
                         "mode). The values assume one leg at a time.")
        active = (topo == "bb_in") == (kind == "buck")
        if not active:
            # this leg sits still with its upper switch on: it carries the inductor current, no edges
            other = from_converter("buck" if kind == "buck" else "boost", v_in, v_out, i_out, l, f_sw, eff, sync)
            return dict(v_rail=v_in if topo == "bb_in" else v_out, i_switch=other["i_switch"],
                        i_ripple=other["i_ripple"], duty=1.0 if topo == "bb_in" else 0.0,
                        i_dir="out" if topo == "bb_in" else "in", switching=False, mode=other["mode"],
                        notes=notes + [f"This leg does not switch while the converter "
                                       f"{'bucks' if kind == 'buck' else 'boosts'} (V_out "
                                       f"{'<' if kind == 'buck' else '>'} V_in): its upper switch stays on and "
                                       "carries the inductor current. The switching simulation is turned off; "
                                       "the conduction loss still counts."])
    else:
        kind = topo
    if kind == "buck":
        if not 0 < v_out < v_in and topo != "half_bridge":
            raise ValueError("A buck needs 0 < V_out < V_in.")
        v_sw, duty, i_l = v_in, v_out / v_in, i_out
        di = (v_in - v_out) * duty / (l * f_sw) if l > 0 else 0.0
        on_slope, off_slope = v_in - v_out, v_out
    else:
        if not v_out > v_in:
            raise ValueError("A boost needs V_out > V_in.")
        v_sw, duty = v_out, 1 - v_in / v_out
        i_l = i_out * v_out / (v_in * eff)
        di = v_in * duty / (l * f_sw) if l > 0 else 0.0
        on_slope, off_slope = v_in, v_out - v_in
    mode = "CCM"
    if l <= 0:
        notes.append("No inductance: the ripple is left at 0 (both edges at the mean current).")
    elif di / 2 > i_l and i_l > 0:
        if sync or min(on_slope, off_slope) <= 0:
            notes.append(f"The ripple ({di:.3g} A p-p) is larger than twice the mean current: in forced PWM the "
                         "current goes negative in the valley, so the turn-on is soft (simulated at about 0 A).")
        else:
            # discontinuous: a triangle of height I_pk, rising for t_on and falling for t_off
            #   I_L = I_pk^2 L f / 2 (1 / on_slope + 1 / off_slope)
            i_pk = math.sqrt(2 * i_l / (l * f_sw * (1 / on_slope + 1 / off_slope)))
            duty = i_pk * l * f_sw / on_slope
            mode = "DCM"
            notes.append(f"Discontinuous (DCM): the switch turns on at 0 A and off at the peak {i_pk:.3g} A; "
                         f"the duty cycle falls to {duty:.3g}. Conduction losses read about 1/(D + D_off) high.")
            i_l, di = i_pk / 2, i_pk
    if topo == "half_bridge":
        i_dir = "out" if sign > 0 else "in"
    else:
        i_dir = "out" if kind == "buck" else "in"
    if kind == "boost" and eff < 1:
        notes.append(f"The input current includes the {eff:.0%} efficiency; the duty cycle is the ideal one.")
    return dict(v_rail=v_sw, i_switch=i_l, i_ripple=di, duty=duty, i_dir=i_dir, switching=True, mode=mode,
                notes=notes)
