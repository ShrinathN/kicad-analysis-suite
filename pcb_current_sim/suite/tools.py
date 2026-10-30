"""The suite's tools, grouped as on the Home window. GUI-neutral: the wx/Tk layers map keys to windows."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    key: str
    title: str
    group: str
    blurb: str
    ready: bool = False


GROUPS = ("POWER", "SIGNAL & FIELDS", "RF / CIRCUIT")

TOOLS = (
    Tool("dc", "DC & Thermal", "POWER",
         "Current density, IR drop, resistance elements, electro-thermal, pulse / fusing, DC B-field.", True),
    Tool("pdn", "PDN (AC)", "POWER",
         "Z(f) against target impedance, decaps, VRM, plane resonances, loop L(f)."),
    Tool("extract", "Trace R/L/C → SPICE", "SIGNAL & FIELDS",
         "R, L, C, Z₀ and delay of the copper between any pads (2D field solve); SPICE RLC subcircuit.", True),
    Tool("si", "SI & Fields", "SIGNAL & FIELDS",
         "Trace Z₀ and εeff, crosstalk, return path at frequency, HV E-field and clearance."),
    Tool("extract3d", "3D extraction", "SIGNAL & FIELDS",
         "FastHenry / FastCap: loop L, via arrays, R_ac(f), coils, parasitic C; SPICE subcircuits."),
    Tool("matching", "Matching & Tolerance", "RF / CIRCUIT",
         "Ladder networks with real parts: corners, Monte Carlo, worst case, yield, Smith chart.", True),
    Tool("openems", "Full-wave (openEMS)", "RF / CIRCUIT",
         "Antennas, interconnect S-parameters, TDR, near / far field. Export or run."),
)


def get(key):
    return next(t for t in TOOLS if t.key == key)
