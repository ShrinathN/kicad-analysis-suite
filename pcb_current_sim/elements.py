"""Lumped two-terminal resistive elements between pads.

Resistor footprints are detected automatically (2 pads, a parseable Value).
Any other part (inductor DCR, FET R_on, fuse, connector, external load ...)
can be added by hand as a resistance between two of its pads.  Each element
pad becomes an equipotential electrode on its copper, and the two electrodes
are joined by 1/R.  The element's I^2 R is a heat source in the thermal model.
"""
import re
from dataclasses import dataclass, replace

R_MIN = 1e-6           # ohm; 0 ohm parts are modelled as this (ideal link)

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

    def label(self):
        return f"{self.name} ({self.a}-{self.b})"


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
    """Sort key so R2 < R10 and pad 2 < pad 10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


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
        na, nb = pad_net.get(e.a), pad_net.get(e.b)
        if na is None or nb is None:
            warn.append(f"Element {e.name}: pad {e.a if na is None else e.b} not found; ignored.")
        elif not na or not nb:
            warn.append(f"Element {e.name}: pad {e.a if not na else e.b} has no net; ignored.")
        else:
            els.append((e, na, nb))
    nets = set(row_nets)
    grown = True
    while grown:
        grown = False
        for e, na, nb in els:
            if (na in nets) != (nb in nets):
                nets |= {na, nb}
                grown = True
    use = [(e, na, nb) for e, na, nb in els if na in nets and nb in nets]
    while True:
        deg = {}
        for e, na, nb in use:
            if na != nb:
                deg[na] = deg.get(na, 0) + 1
                deg[nb] = deg.get(nb, 0) + 1
        dead = {n for n, d in deg.items() if d == 1 and n not in row_nets}
        if not dead:
            break
        use = [(e, na, nb) for e, na, nb in use if na not in dead and nb not in dead]
        nets -= dead
    order = list(row_nets) + sorted(nets - set(row_nets))
    return order, [e for e, _, _ in use], warn
