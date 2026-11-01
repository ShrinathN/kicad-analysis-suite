"""Lumped two-terminal elements between pads: a resistance, or a constant voltage drop.

Resistor footprints are detected automatically (2 pads, a parseable Value).
Any other part (inductor DCR, FET R_on, fuse, connector, external load ...)
can be added by hand as a resistance between two of its pads.  Each element
pad becomes an equipotential electrode on its copper, and the two electrodes
are joined by 1/R.  The element's I^2 R is a heat source in the thermal model.

A constant-drop element (model "V", e.g. a diode) is Vf in series with R and
conducts from pad a (anode) to pad b only; it is open while V_a - V_b < Vf.
It dissipates I (Vf + I R).

A regulator (model "LDO" or "SMPS") has three pads: a = IN, b = OUT, g = GND.  It holds
V_OUT - V_GND at Vout (measured at its own pins, so ground drops show at the loads) and draws
its input current from IN, returned at GND:
  linear (LDO):     I_in = I_out + Iq; below the dropout (V_IN - V_OUT < Vf) it becomes a constant
                    drop Vf (+ R) from IN to OUT and no longer regulates
  switching (SMPS): I_in = V_out I_out / (eta V_in) + Iq (power balance, any topology)
Its loss V_in I_in - V_out I_out heats the part's pads (shared by pad area).  DC only: no
regulation loop, soft start, current limit or thermal shutdown.
"""
import re
from dataclasses import dataclass, field, replace

R_MIN = 1e-6           # ohm; 0 ohm parts are modelled as this (ideal link)
REGULATORS = ("LDO", "SMPS")

# imperial size code -> typical thick-film rating [W]
SMD_RATING = {"0201": 0.05, "0402": 0.0625, "0603": 0.1, "0805": 0.125, "1206": 0.25, "1210": 0.5,
              "1218": 1.0, "2010": 0.75, "2512": 1.0, "2817": 2.0}
METRIC_TO_IMPERIAL = {"0603": "0201", "1005": "0402", "1608": "0603", "2012": "0805", "3216": "1206",
                      "3225": "1210", "3246": "1218", "5025": "2010", "6332": "2512", "7142": "2817"}
DIN_RATING = {"0204": 0.25, "0207": 0.25, "0309": 0.5, "0411": 1.0, "0414": 2.0, "0516": 2.0,
              "0617": 3.0, "0918": 4.0, "0922": 5.0}


@dataclass
class Element:
    name: str                # usually the reference, e.g. "R3"
    a: str                   # pad id, e.g. "R3.1"
    b: str
    R: float                 # ohm
    P_rated: float = None    # W, None = no check
    enabled: bool = True
    auto: bool = False       # detected from a resistor footprint (not user-defined)
    heats: bool = True       # its I^2 R heats the board (off for an external load or wire)
    model: str = "R"         # "R" resistance, "V" constant drop Vf + R conducting a -> b only (diode),
    #                          "C" capacitor, "L" inductor (ideal; RLC extraction; DC: C open, L short)
    Vf: float = 0.0          # V, model "V" only
    C: float = 0.0           # F, model "C"
    L: float = 0.0           # H, model "L"
    # regulators ("LDO", "SMPS"): a = IN, b = OUT, g = GND; Vf = an LDO's dropout voltage and R its
    # resistance in dropout
    g: str = ""              # GND pad
    Vout: float = 0.0        # V, output voltage (OUT - GND)
    Iq: float = 0.0          # A, quiescent current (IN -> GND)
    eta: float = 0.9         # SMPS efficiency (P_out / P_in)
    psi_jb: float = 0.0      # °C/W, junction to board (datasheet Ψ_JB); 0 = no junction estimate
    Tj_max: float = 125.0    # °C, for the warning
    # which tools use it ({"dc": True, "extract": False, ...}): the element is defined once for the
    # board (components.Components) and each tool switches it on or off for itself
    use: dict = field(default_factory=dict)
    # real model of a C / L (suite.library.Part as a dict: ESR, ESL, tolerance, DC bias ...), set by the
    # tools that need it (Switching Loop, PDN); DC & Thermal and Trace use the ideal value C / L / R
    part: dict = field(default_factory=dict)

    @property
    def regulator(self):
        return self.model in REGULATORS

    def pads(self):
        return (self.a, self.b, self.g) if self.regulator else (self.a, self.b)

    def label(self):
        if self.regulator:
            return f"{self.name} (IN {self.a}, OUT {self.b}, GND {self.g})"
        return f"{self.name} ({self.a}-{self.b})"

    def describe(self):
        """Value column text: '4.7 kΩ', or '0.7 V + 10 mΩ' (→) for a constant drop."""
        if self.model == "LDO":
            return f"LDO {self.Vout:g} V (dropout {self.Vf:g} V)"
        if self.model == "SMPS":
            return f"switching {self.Vout:g} V, η {self.eta * 100:.0f} %"
        if self.model == "C":
            from .suite.library import fmt_si
            return fmt_si(self.C, "F")
        if self.model == "L":
            from .suite.library import fmt_si
            return fmt_si(self.L, "H")
        if self.model != "V":
            return fmt_ohms(self.R)
        return f"{self.Vf:g} V →" + (f" + {fmt_ohms(self.R)}" if self.R else "")


def parse_ohms(text):
    """'10k' '4k7' '0R1' 'R005' '5m' '1M' '0.1' '22 Ohm' '10k 1%' -> ohms, or None."""
    if not text:
        return None
    t = text.strip().replace("Ω", "").replace("ω", "").replace(",", ".")
    t = re.split(r"[\s/_]", t)[0]
    t = re.sub(r"(?i)ohms?$", "", t)
    t = re.sub(r"(?i)meg", "M", t)
    mult = {"": 1.0, "R": 1.0, "r": 1.0, "E": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "k": 1e3, "K": 1e3,
            "M": 1e6, "G": 1e9}
    m = re.fullmatch(r"(\d*)([RrEkKMmGuµ])(\d+)", t)          # RKM code: 4k7, 0R1, R005
    if m and (m.group(1) or m.group(2) in "Rr"):
        return float(f"{m.group(1) or 0}.{m.group(3)}") * mult[m.group(2)]
    m = re.fullmatch(r"(\d+\.?\d*|\.\d+)(?:e([-+]?\d+))?([RrEkKMmGuµ]?)(?:[Rr])?", t)
    if m:
        v = float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1)
        return v * mult[m.group(3)]
    return None


def natural_key(s):
    """Sort key for names in alphabetical order, numbers by value (R2 < R10, pad 2 < pad 10), case
    ignored.  Numbers sort before letters at the same place, so mixed names compare without error."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", str(s))]


def fmt_ohms(r):
    for s, m in (("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6)):
        if abs(r) >= m or m == 1e-6:
            return f"{r / m:.4g} {s}Ω"


def guess_rating(lib):
    """Power rating from the footprint library name, or None."""
    m = re.search(r"DIN(\d{4})", lib)
    if m and m.group(1) in DIN_RATING:
        return DIN_RATING[m.group(1)]
    m = re.search(r"_(\d{4})_(\d{4})Metric", lib)
    if m:
        return SMD_RATING.get(m.group(1))
    m = re.search(r"[RL]_(\d{4})\b", lib)
    if m:
        code = m.group(1)
        return SMD_RATING.get(METRIC_TO_IMPERIAL.get(code, code))
    return None


def is_resistor(fp):
    return (re.match(r"R[S]?\d+$", fp.ref) is not None or fp.lib.split(":")[0].startswith("Resistor")) \
        and not fp.lib.split(":")[0].startswith(("Potentiometer", "Resistor_Network"))


def detect(board):
    """Auto elements for resistor footprints; returns (elements, warnings)."""
    out, warn = [], []
    for fp in board.footprints:
        if fp.dnp or not is_resistor(fp):
            continue
        nums = []
        for k in fp.pads:
            p = board.pads[k]
            if p.layers and p.number and p.number not in nums:
                nums.append(p.number)
        if len(nums) != 2:
            warn.append(f"{fp.ref} has {len(nums)} pads: add it as a resistance element by hand "
                        "(pick the two pads).")
            continue
        R = parse_ohms(fp.value)
        if R is None:
            warn.append(f"{fp.ref}: cannot read a resistance from value '{fp.value}'; set it by hand.")
            continue
        out.append(Element(fp.ref, f"{fp.ref}.{nums[0]}", f"{fp.ref}.{nums[1]}", R,
                           guess_rating(fp.lib), True, True))
    return out, warn


def merge(auto, saved):
    """Combine detected resistors with saved elements.

    A saved user element (auto=False) replaces an auto one with the same name,
    e.g. a resistor whose value was edited.  A saved auto entry only carries
    its enabled flag, so value changes on the board are picked up."""
    saved = list({e.name: e for e in saved or []}.values())        # last entry per name wins
    user = {e.name: e for e in saved if not e.auto}
    flags = {e.name: e.enabled for e in saved if e.auto}
    out = []
    for e in auto:
        if e.name in user:
            continue
        out.append(replace(e, enabled=flags.get(e.name, e.enabled)))
    return out + [e for e in saved if not e.auto]


def to_save(elements):
    """The part of an element list worth saving (user elements and disabled auto ones)."""
    return [e for e in elements if not e.auto or not e.enabled]


def active(board, row_nets, elements):
    """Elements that can carry current for the given rows.

    Starting from the rows' nets, follow enabled elements to further nets, then
    drop dead-end nets (reached through a single element and not excited): no
    current can flow into them.  Returns (nets, elements, warnings)."""
    pad_net = {}
    for p in board.pads:
        if p.layers:
            pad_net.setdefault(p.id, p.net)
    warn, els = [], []
    for e in elements:
        if not e.enabled:
            continue
        ns = [pad_net.get(pid) for pid in e.pads()]
        bad = next((pid for pid, n in zip(e.pads(), ns) if n is None), None)
        empty = next((pid for pid, n in zip(e.pads(), ns) if n == ""), None)
        if bad is not None:
            warn.append(f"Element {e.name}: pad {bad or '(none)'} not found; ignored.")
        elif empty is not None:
            warn.append(f"Element {e.name}: pad {empty} has no net; ignored.")
        else:
            els.append((e, set(ns)))
    nets = set(row_nets)
    grown = True
    while grown:
        grown = False
        for e, ns in els:
            if ns & nets and not ns <= nets:
                nets |= ns
                grown = True
    use = [(e, ns) for e, ns in els if ns <= nets]
    while True:
        deg = {}
        for e, ns in use:
            if len(ns) > 1:
                for n in ns:
                    deg[n] = deg.get(n, 0) + 1
        # a regulator's nets are never dead ends: its output feeds its loads by itself, its GND is the
        # output's reference, and an input left without a source gets its own clear error
        fed = {n for e, ns in use if e.regulator for n in ns}
        dead = {n for n, d in deg.items() if d == 1 and n not in row_nets and n not in fed}
        if not dead:
            break
        use = [(e, ns) for e, ns in use if not ns & dead]
        nets -= dead
    order = list(row_nets) + sorted(nets - set(row_nets))
    return order, [e for e, _ in use], warn
