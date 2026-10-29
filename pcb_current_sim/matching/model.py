"""Ladder network between a source and a load, evaluated as an ABCD cascade.

Elements run from the source (port 1) to the load (port 2). Each is in a series
or a shunt slot:
  L, C, R          series or shunt; a real part (suite.library.Part)
  TL               series transmission line: ideal (Z0, εeff, loss) or microstrip (w, h, εr, tanδ)
  stub_open/short  shunt stub, ideal or microstrip like TL
  S2P              series two-port from a Touchstone file (fixed, no tolerance)
Sample-dependent parameters (tolerances, temperature) are arrays shaped (S, 1)
against f shaped (F,); everything broadcasts to (S, F).
"""
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from ..suite.library import Part, fmt_si, part_from_dict
from . import microstrip as MS
from . import touchstone as TS

C0 = 299792458.0
KINDS = {"L": "Inductor", "C": "Capacitor", "R": "Resistor", "TL": "Transmission line",
         "stub_open": "Open stub", "stub_short": "Shorted stub", "S2P": "Two-port (S2P)"}
SERIES_ONLY, SHUNT_ONLY = {"TL", "S2P"}, {"stub_open", "stub_short"}
T_NOM = 25.0


@dataclass
class Element:
    kind: str
    shunt: bool = False
    name: str = ""
    part: dict = None             # Part fields for L / C / R (a copy, so the scenario is self-contained)
    lib: str = ""                 # library part it was taken from ("" = custom)
    vdc: float = 0.0              # DC bias across a capacitor
    line: str = "ideal"           # TL / stubs: "ideal" or "ms"
    z0: float = 50.0
    eeff: float = 1.0
    length: float = 0.01          # m
    loss_db_m: float = 0.0        # ideal line loss
    z0_tol: float = 0.0           # % (ideal line)
    w: float = 3.0e-3             # microstrip, m
    h: float = 1.5e-3
    er: float = 4.3
    tand: float = 0.02
    w_tol: float = 0.0            # ± m (etching)
    er_tol: float = 0.0           # %
    s2p: str = ""

    def __post_init__(self):
        if self.kind in SERIES_ONLY:
            self.shunt = False
        elif self.kind in SHUNT_ONLY:
            self.shunt = True

    def get_part(self):
        return part_from_dict(self.part) if self.part else None

    def line_params(self, f):
        """(Z0, γ per m, εeff) at f for TL / stubs, nominal values."""
        if self.line == "ms":
            z0, g = MS.gamma(f, self.w, self.h, self.er, self.tand)
            return z0, g, MS.z0_eeff(self.w, self.h, self.er)[1]
        beta = 2 * np.pi * np.asarray(f, float) * math.sqrt(self.eeff) / C0
        return self.z0, self.loss_db_m / 8.686 + 1j * beta, self.eeff

    def value_text(self, f0=None):
        """Short description for the schematic."""
        p = self.get_part()
        if p is not None:
            return fmt_si(p.value, p.unit)
        if self.kind == "S2P":
            import os
            return os.path.basename(self.s2p) or "(no file)"
        if self.line == "ms":
            z0, eeff = MS.z0_eeff(self.w, self.h, self.er)
            s = f"w {self.w * 1e3:.3g} mm, {z0:.1f} Ω"
        else:
            z0, eeff, s = self.z0, self.eeff, f"{self.z0:.4g} Ω"
        s += f", {self.length * 1e3:.4g} mm"
        if f0:
            s += f" ({360 * f0 * self.length * math.sqrt(eeff) / C0:.0f}° @ {fmt_si(f0, 'Hz')})"
        return s

    def tol_text(self):
        p = self.get_part()
        if p is not None:
            return "±" + p.tol_text() if p.tol else "exact"
        if self.kind == "S2P":
            return "fixed"
        if self.line == "ms":
            t = [f"w ±{self.w_tol * 1e6:g} µm" if self.w_tol else "", f"εr ±{self.er_tol:g} %" if self.er_tol else ""]
        else:
            t = [f"Z0 ±{self.z0_tol:g} %" if self.z0_tol else ""]
        return ", ".join(x for x in t if x) or "exact"


@dataclass
class Port:
    """Source or load impedance."""
    kind: str = "Z"               # "Z" (R + jX), "series RLC", "parallel RLC", "s1p"
    r: float = 50.0
    x: float = 0.0
    l: float = 0.0
    c: float = 0.0
    path: str = ""

    def Z(self, f):
        f = np.asarray(f, float)
        w = 2 * np.pi * f
        if self.kind == "Z":
            return np.full(f.shape, complex(self.r, self.x))
        if self.kind == "series RLC":
            z = self.r + 1j * w * self.l + 0j
            return z + (1 / (1j * w * self.c) if self.c > 0 else 0)
        if self.kind == "parallel RLC":
            y = (1 / self.r if self.r > 0 else 0) + 1j * w * self.c + 0j
            if self.l > 0:
                y = y + 1 / (1j * w * self.l)
            return 1 / y
        ts = TS.load(self.path)
        g = ts.at(f)[:, 0, 0]
        return ts.z0 * (1 + g) / (1 - g)

    def text(self):
        if self.kind == "Z":
            return f"{self.r:g} {'+' if self.x >= 0 else '−'} j{abs(self.x):g} Ω"
        if self.kind == "s1p":
            import os
            return os.path.basename(self.path) or "(no file)"
        parts = [fmt_si(self.r, "Ω")] + ([fmt_si(self.l, "H")] if self.l else []) + \
                ([fmt_si(self.c, "F")] if self.c else [])
        return ("ser " if self.kind.startswith("series") else "par ") + ", ".join(parts)


@dataclass
class Band:
    name: str = "band"
    f1: float = 2.4e9
    f2: float = 2.5e9
    rl_min: float = 10.0          # dB, 0 = no limit
    il_max: float = 0.0           # dB, 0 = no limit
    vswr_max: float = 0.0         # 0 = no limit

    def rl_req(self):
        """The return-loss requirement implied by rl_min and vswr_max (0 if none)."""
        r = self.rl_min
        if self.vswr_max > 1:
            r = max(r, -20 * math.log10((self.vswr_max - 1) / (self.vswr_max + 1)))
        return r


@dataclass
class Setup:
    elements: list = field(default_factory=list)
    source: Port = field(default_factory=Port)
    load: Port = field(default_factory=Port)
    f_start: float = 1e9
    f_stop: float = 4e9
    n_f: int = 301
    log_f: bool = False
    z_ref: float = 50.0           # Smith chart normalisation
    bands: list = field(default_factory=list)
    mc_n: int = 10000
    dist: str = "uniform"         # "uniform" or "gauss" (3σ = tolerance, truncated at ±tolerance)
    seed: int = 1
    temp: bool = False            # sweep temperature T_min..T_max as an extra variable
    T_min: float = -40.0
    T_max: float = 85.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        def pick(klass, x):
            k = {f.name for f in fields(klass)}
            return klass(**{a: b for a, b in (x or {}).items() if a in k})
        s = pick(cls, {k: v for k, v in d.items() if k not in ("elements", "source", "load", "bands")})
        s.elements = [pick(Element, e) for e in d.get("elements", [])]
        s.source, s.load = pick(Port, d.get("source")), pick(Port, d.get("load"))
        s.bands = [pick(Band, b) for b in d.get("bands", [])]
        return s

    def freqs(self):
        """Sweep grid plus 41 points in each band, so narrow bands are always resolved."""
        g = (np.geomspace if self.log_f else np.linspace)(self.f_start, self.f_stop, max(int(self.n_f), 2))
        extra = [np.linspace(b.f1, b.f2, 41) for b in self.bands if b.f2 > b.f1]
        return np.unique(np.concatenate([g] + extra))

    def next_name(self, kind):
        pre = {"L": "L", "C": "C", "R": "R", "TL": "TL", "stub_open": "ST", "stub_short": "ST", "S2P": "X"}[kind]
        used = {e.name for e in self.elements}
        n = 1
        while f"{pre}{n}" in used:
            n += 1
        return f"{pre}{n}"


def new_element(setup, kind, shunt=False, part=None, stackup=None, f0=None):
    """An element with sensible defaults (a quarter wave at f0 for lines; microstrip on the stackup's top layer)."""
    e = Element(kind, shunt, setup.next_name(kind))
    if kind in ("L", "C", "R"):
        p = part or Part(kind, f"{e.name} custom", {"L": 10e-9, "C": 1e-12, "R": 50.0}[kind], 5.0)
        e.part, e.lib = asdict(p), (part.name if part is not None else "")
    elif kind in ("TL", "stub_open", "stub_short"):
        if stackup is not None and len(stackup.dielectrics):
            d = stackup.dielectrics[0]
            e.h, e.er, e.tand = d.t, d.er, d.tand
        e.w = MS.width_for(50.0, e.h, e.er)
        e.line = "ms" if stackup is not None else "ideal"
        f0 = f0 or 1e9
        eeff = MS.z0_eeff(e.w, e.h, e.er)[1] if e.line == "ms" else e.eeff
        e.length = C0 / (4 * f0 * math.sqrt(eeff))
    return e


# ---------------------------------------------------------------- evaluation
class Params:
    """Per-element parameter lookup: sampled arrays (S, 1) where a variable exists, else the nominal value."""

    def __init__(self, var_list=(), X=None):
        self.cols = {}
        if X is not None:
            for k, v in enumerate(var_list):
                self.cols[(v.elem, v.attr)] = X[:, k:k + 1]

    def get(self, i, attr, nominal):
        return self.cols.get((i, attr), nominal)


def element_abcd(e, i, f, P):
    """('series', Z) | ('shunt', Y) | ('abcd', (A, B, C, D)) for element e at f."""
    T = P.get(-1, "T", T_NOM)
    if e.kind in ("L", "C", "R"):
        p = e.get_part()
        Z = p.Z(f, value=P.get(i, "value", p.value), T=T, vdc=e.vdc)
        return ("shunt", 1 / Z) if e.shunt else ("series", Z)
    if e.kind == "S2P":
        ts = TS.load(e.s2p)
        return "abcd", TS.s2abcd(ts.at(f), ts.z0)
    if e.line == "ms":
        z0, g = MS.gamma(f, P.get(i, "w", e.w), e.h, P.get(i, "er", e.er), e.tand)
    else:
        z0 = P.get(i, "z0", e.z0)
        g = e.loss_db_m / 8.686 + 1j * 2 * np.pi * f * math.sqrt(e.eeff) / C0
    gl = g * e.length
    if e.kind == "TL":
        ch, sh = np.cosh(gl), np.sinh(gl)
        return "abcd", (ch, z0 * sh, sh / z0, ch)
    t = np.tanh(gl)
    Y = t / z0 if e.kind == "stub_open" else 1 / (z0 * t)
    return "shunt", Y


def cascade(elements, f, P, shape):
    A = np.ones(shape, complex)
    B = np.zeros(shape, complex)
    C = np.zeros(shape, complex)
    D = np.ones(shape, complex)
    for i, e in enumerate(elements):
        how, v = element_abcd(e, i, f, P)
        if how == "series":
            B, D = A * v + B, C * v + D
        elif how == "shunt":
            A, C = A + B * v, C + D * v
        else:
            a, b, c, d = v
            A, B, C, D = A * a + B * c, A * b + B * d, C * a + D * c, C * b + D * d
    return A, B, C, D


def response(setup, f, var_list=(), X=None):
    """Γin (power-wave reflection seen by the source), GT (transducer gain), Zin; each (S, F)."""
    f = np.asarray(f, float)
    S = 1 if X is None else X.shape[0]
    P = Params(var_list, X)
    A, B, C, D = cascade(setup.elements, f, P, (S, len(f)))
    Zs, Zl = setup.source.Z(f), setup.load.Z(f)
    zin = (A * Zl + B) / (C * Zl + D)
    gam = (zin - np.conj(Zs)) / (zin + Zs)
    den = A * Zl + B + C * Zs * Zl + D * Zs
    gt = 4 * Zs.real * Zl.real / np.abs(den) ** 2
    return gam, gt, zin


def metrics_db(gam, gt):
    """Return loss, insertion loss (transducer) and dissipative loss in dB, and VSWR."""
    mag = np.clip(np.abs(gam), 1e-12, 1 - 1e-12)
    rl = -20 * np.log10(mag)
    il = -10 * np.log10(np.maximum(gt, 1e-30))
    eff = np.clip(gt / np.maximum(1 - mag ** 2, 1e-30), 1e-30, 1)
    return rl, il, -10 * np.log10(eff), (1 + mag) / (1 - mag)
