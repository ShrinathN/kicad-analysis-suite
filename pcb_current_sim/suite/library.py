"""Component library: real-part models shared by the matching and PDN tools.

Models (f in Hz; every function broadcasts, so `value` may be an array of
Monte Carlo samples shaped (S, 1) against f shaped (F,)):
  L: (jωL + R(f)) ∥ C_p,   R(f) = DCR + k√f with k fitted to Q at f_Q,  C_p = 1/((2π·SRF)²·L)
  C: ESR + 1/(jωC) + jω·ESL,  ESR from Q at f_Q if not given, ESL from SRF if not given;
     C drifts with temperature (C0G / X7R / X5R curve) and DC bias
  R: (R + jωL_s) ∥ C_p

The user library is one JSON file shared by the KiCad (Flatpak) and host
installs: ~/.config/pcb_current_sim/library.json. Built-in parts are generic
typical values, not datasheet data.
"""
import json
import math
import os
import re
from dataclasses import asdict, dataclass, fields

import numpy as np

USER_LIBRARY = os.path.expanduser("~/.config/pcb_current_sim/library.json")
KINDS = {"L": ("Inductor", "H"), "C": ("Capacitor", "F"), "R": ("Resistor", "Ω")}
C_TOL_CODES = {"A": 0.05e-12, "B": 0.1e-12, "C": 0.25e-12, "D": 0.5e-12}   # ± absolute
DIELECTRICS = ("C0G", "X7R", "X5R", "Y5V")

_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0,
       "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "R": 1.0, "r": 1.0}
_SI_RE = re.compile(r"^\s*([0-9]*\.?[0-9]*)\s*([fpnuµμmkKMGRr]?)([0-9]*)\s*(?:e([+-]?\d+))?\s*"
                    r"(?:H|F|Ω|ohm|Ohm|Hz)?\s*$")


def parse_si(text):
    """'4.7n', '4n7', '100 nH', '2u2', '0R1', '1e-9' -> float. Raises ValueError."""
    t = str(text).strip()
    try:
        return float(t)
    except ValueError:
        pass
    m = _SI_RE.match(t)
    if not m or not (m.group(1) or m.group(3)):
        raise ValueError(f"not a value: {text!r}")
    a, p, frac, ex = m.groups()
    if frac and "." in a:
        raise ValueError(f"not a value: {text!r}")
    v = float((a or "0") + ("." + frac if frac else ""))
    return v * _SI[p] * (10.0 ** int(ex) if ex else 1.0)


PREFIXES = (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9),
            ("p", 1e-12), ("f", 1e-15))


def fmt_like(v, ref, unit=""):
    """v in the SI prefix that suits ref (a tolerance next to its value: '±0.1 pF', not '±100 fF')."""
    if not ref or not math.isfinite(ref):
        return fmt_si(v, unit)
    for p, s in PREFIXES:
        if abs(ref) >= s * 0.9999:
            return f"{v / s:.4g} {p}{unit}".strip()
    return fmt_si(v, unit)


def fmt_si(v, unit=""):
    if v is None or not math.isfinite(v):
        return "-"
    if v == 0:
        return f"0 {unit}".strip()
    for p, s in PREFIXES:
        if abs(v) >= s * 0.9999:
            return f"{v / s:.4g} {p}{unit}".strip()
    return f"{v:.3g} {unit}".strip()


@dataclass
class Part:
    kind: str                    # "L", "C" or "R"
    name: str                    # library key, e.g. a manufacturer part number
    value: float                 # H, F or Ω
    tol: float = 5.0             # ± tolerance, in % or absolute base units (tol_unit)
    tol_unit: str = "%"          # "%" or "abs"
    package: str = ""
    mfr: str = ""
    notes: str = ""
    # inductor
    dcr: float = 0.0             # Ω
    q: float = 0.0               # quality factor at f_q (0 = lossless apart from DCR); also for C
    f_q: float = 100e6           # Hz
    srf: float = 0.0             # Hz, self-resonance (0 = none); for C it sets ESL when esl = 0
    # capacitor
    esr: float = 0.0             # Ω (0 = from Q)
    esl: float = 0.0             # H (0 = from SRF)
    dielectric: str = "C0G"
    v_rated: float = 0.0         # V, for the DC bias curve
    bias_loss: float = 0.0       # fraction of C lost at the rated voltage (class 2 ceramics)
    # resistor
    ls: float = 0.0              # H
    cp: float = 0.0              # F
    builtin: bool = False

    @property
    def unit(self):
        return KINDS[self.kind][1]

    def label(self):
        return f"{self.name} ({fmt_si(self.value, self.unit)} ±{self.tol_text()})"

    def tol_text(self):
        return f"{self.tol:g} %" if self.tol_unit == "%" else fmt_like(self.tol, self.value, self.unit)

    def tol_abs(self, value=None):
        v = self.value if value is None else value
        return abs(v) * self.tol / 100 if self.tol_unit == "%" else self.tol

    # ---- impedance ------------------------------------------------------
    def Z(self, f, value=None, T=25.0, vdc=0.0):
        f = np.asarray(f, float)
        v = self.value if value is None else np.asarray(value, float)
        w = 2 * np.pi * f
        if self.kind == "L":
            Rf = self.dcr + self._k_skin() * np.sqrt(f)
            Z = Rf + 1j * w * v
            if self.srf > 0:
                Cp = 1.0 / ((2 * np.pi * self.srf) ** 2 * self.value)
                Z = Z / (1 + 1j * w * Cp * Z)
            return Z
        if self.kind == "C":
            C = v * cap_factor(self, T, vdc)
            return self.esr_eff() + 1 / (1j * w * C) + 1j * w * self.esl_eff()
        Z = v + 1j * w * self.ls
        return Z / (1 + 1j * w * self.cp * Z) if self.cp > 0 else Z

    def _k_skin(self):
        """k in R(f) = DCR + k√f so that Q(f_q) = ω L / R matches the stated Q."""
        if self.kind != "L" or self.q <= 0 or self.f_q <= 0:
            return 0.0
        r_q = 2 * math.pi * self.f_q * self.value / self.q
        return max(r_q - self.dcr, 0.0) / math.sqrt(self.f_q)

    def esr_eff(self):
        if self.esr > 0 or self.q <= 0:
            return self.esr
        return 1.0 / (2 * math.pi * self.f_q * self.value * self.q)

    def esl_eff(self):
        if self.esl > 0 or self.srf <= 0:
            return self.esl
        return 1.0 / ((2 * math.pi * self.srf) ** 2 * self.value)


def cap_factor(part, T=25.0, vdc=0.0):
    """C(T, Vdc) / C(25 °C, 0 V). Generic curves; real parts vary, so give bias_loss from the datasheet."""
    T = np.asarray(T, float)
    d = part.dielectric
    if d == "C0G":
        k = 1 + 0 * T
    elif d in ("X7R", "X5R"):                  # roughly -15 % at the class limits, flat near 25 °C
        span = 100.0 if d == "X7R" else 60.0
        k = 1 - 0.15 * ((T - 25) / span) ** 2
    else:                                      # Y5V: +22 / -82 %
        k = 1 - 0.8 * ((T - 25) / 60) ** 2
    if part.v_rated > 0 and part.bias_loss > 0:
        k = k * (1 - part.bias_loss * np.clip(abs(vdc) / part.v_rated, 0, 1.5) ** 2)
    return k


def _builtins():
    P = Part
    out = [
        # generic 0402 multilayer RF inductors: Q and SRF roughly typical for the class
        P("L", "generic 0402 RF 1n0", 1.0e-9, 0.3e-9, "abs", "0402", notes="typical", dcr=0.045, q=20, f_q=500e6,
          srf=12e9),
        P("L", "generic 0402 RF 4n7", 4.7e-9, 0.3e-9, "abs", "0402", notes="typical", dcr=0.13, q=20, f_q=500e6,
          srf=5e9),
        P("L", "generic 0402 RF 10n", 10e-9, 5, "%", "0402", notes="typical", dcr=0.2, q=20, f_q=500e6, srf=3.2e9),
        P("L", "generic 0402 RF 22n", 22e-9, 5, "%", "0402", notes="typical", dcr=0.35, q=20, f_q=250e6, srf=2e9),
        P("L", "generic 0603 wirewound 47n", 47e-9, 2, "%", "0603", notes="typical", dcr=0.4, q=40, f_q=200e6,
          srf=1.5e9),
        P("C", "generic 0402 C0G 1p0", 1.0e-12, 0.1e-12, "abs", "0402", notes="typical", q=400, f_q=1e9, srf=20e9),
        P("C", "generic 0402 C0G 10p", 10e-12, 5, "%", "0402", notes="typical", q=300, f_q=1e9, esl=0.4e-9),
        P("C", "generic 0402 C0G 100p", 100e-12, 5, "%", "0402", notes="typical", q=100, f_q=500e6, esl=0.4e-9),
        P("C", "generic 0402 X7R 100n 16V", 100e-9, 10, "%", "0402", notes="typical decap", esr=0.03,
          esl=0.4e-9, dielectric="X7R", v_rated=16, bias_loss=0.3),
        P("C", "generic 0603 X7R 1u 25V", 1e-6, 10, "%", "0603", notes="typical decap", esr=0.015,
          esl=0.6e-9, dielectric="X7R", v_rated=25, bias_loss=0.4),
        P("C", "generic 0805 X5R 10u 10V", 10e-6, 20, "%", "0805", notes="typical decap", esr=0.005,
          esl=0.8e-9, dielectric="X5R", v_rated=10, bias_loss=0.6),
        P("R", "generic 0402 thick film", 50.0, 1, "%", "0402", notes="typical", ls=0.4e-9, cp=0.05e-12),
    ]
    for p in out:
        p.builtin = True
    return out


class Library:
    def __init__(self, path=USER_LIBRARY):
        self.path = path
        self.parts = _builtins()
        self.error = None
        if os.path.exists(path):
            try:
                self.parts += [part_from_dict(d) for d in json.load(open(path)).get("parts", [])]
            except Exception as e:      # noqa: BLE001 - a broken file must not stop the suite
                self.error = f"Could not read {path}: {e}"

    def by_kind(self, kind):
        return [p for p in self.parts if p.kind == kind]

    def get(self, name):
        return next((p for p in self.parts if p.name == name), None)

    def put(self, part):
        """Add or replace a user part by name."""
        part.builtin = False
        self.parts = [p for p in self.parts if p.name != part.name or p.builtin] + [part]

    def remove(self, name):
        self.parts = [p for p in self.parts if p.name != name or p.builtin]

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        user = [asdict(p) for p in self.parts if not p.builtin]
        for d in user:
            d.pop("builtin")
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(version=1, parts=user), f, indent=1)
        os.replace(tmp, self.path)


def part_from_dict(d):
    k = {f.name for f in fields(Part)}
    return Part(**{a: b for a, b in d.items() if a in k})
