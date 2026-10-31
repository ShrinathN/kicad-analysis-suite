"""Switching Loop setup: the cell (devices in roles), device models and auto-detection.

A hard-switched power stage is one cell: an upper and a lower switching position between a rail
and its return, the switch node between them, and the hot-loop capacitors across rail-return.
Buck, boost and half-bridge legs differ only in which position is active:
  buck:  upper = control FET, lower = synchronous FET or diode (hot-loop caps = input caps)
  boost: lower = control FET, upper = synchronous FET or diode (hot-loop caps = output caps)
  half-bridge / motor phase: both FETs (caps = bus caps)
A position may hold several parts in parallel (a FET with a Schottky across it).  Each FET's
gate drive is: driver output pin -> optional gate resistor -> gate pin, returning to the driver's
return pin (the source net: return for a low-side FET, switch node for a high-side one).

Pins are found by net, not by pin number: all of a part's pads on one net are one pin (three
source pins = one node).  A FET may name Kelvin-source pads (by pad number): they become their own
pin, on the same net, joined to the die's source inside the package.  A converter IC with
internal switches gets "internal switch" parts on its footprint (D and S on its pins, no gate pin).
"""
import math
import re
from dataclasses import asdict, dataclass, field, fields

from .. import elements as EL
from ..extract.rlc import GND_WORDS
from ..suite.library import Part, part_from_dict, parse_si

TOPOLOGIES = {
    "auto": "Detect from the parts",
    "buck": "Buck (upper switch active)",
    "boost": "Boost (lower switch active)",
    "half_bridge": "Half-bridge / bridge leg (both active)",
}


@dataclass
class FetModel:
    """MOSFET small-signal model from datasheet values (C at v_ref, scaled as 1/√V), plus its
    switching times (used for di/dt now, and by the transient later)."""
    name: str = "generic N-MOSFET"
    channel: str = "N"           # "N" or "P" (a P-channel high-side switch has its source on the rail)
    rds_on: float = 10e-3        # Ω
    ciss: float = 1500e-12       # F at v_ref
    coss: float = 400e-12
    crss: float = 40e-12
    v_ref: float = 15.0          # V, V_DS of the datasheet capacitances
    rg: float = 1.0              # Ω, internal gate resistance
    ld: float = 0.2e-9           # H, package (drain, source, gate)
    ls: float = 0.3e-9
    lg: float = 0.5e-9
    lks: float = 0.1e-9          # H, Kelvin-source pin (when the part has one)
    t_rise: float = 0.0          # s, datasheet rise / fall time and turn-on / off delays (0 = unknown)
    t_fall: float = 0.0
    t_d_on: float = 0.0
    t_d_off: float = 0.0
    # switching transient (step 2): channel from the threshold and transconductance, body diode
    vth: float = 2.0             # V, gate threshold V_GS(th)
    gfs: float = 30.0            # S, forward transconductance...
    i_gfs: float = 10.0          # A, ...at this drain current (sets the channel's square law)
    v_gs_on: float = 10.0        # V, the gate voltage R_DS(on) is given at
    vsd: float = 0.8             # V, body-diode forward voltage...
    i_vsd: float = 1.0           # A, ...at this current
    qrr: float = 20e-9           # C, body-diode reverse-recovery charge...
    i_qrr: float = 10.0          # A, ...at this forward current (sets the stored charge per amp)
    assumed: bool = True         # generic values: enter the datasheet's

    def caps(self, v):
        """(Cgs, Cgd, Cds) at V_DS = v.  v_ref = 0: the capacitances as entered (no V scaling, e.g. a
        regulator datasheet's switch-node capacitance); C_oss = 0 leaves the switch's C out."""
        if self.v_ref <= 0:
            k = 1.0
        else:
            k = math.sqrt(self.v_ref / max(v, 0.5))
            k = min(max(k, 0.2), 6.0)
        cgd = self.crss * k
        return max(self.ciss - self.crss, 1e-15), cgd, max(self.coss - self.crss, 1e-15) * k

    def coss_at(self, v):
        cgs, cgd, cds = self.caps(v)
        return cgd + cds


@dataclass
class DiodeModel:
    name: str = "generic Schottky"
    cj: float = 200e-12          # F at v_ref
    v_ref: float = 4.0
    rd: float = 30e-3            # Ω, on-state slope resistance
    vf: float = 0.45
    l_pkg: float = 1.5e-9        # H, total lead / package
    i_vf: float = 1.0            # A, the current V_f is given at
    qrr: float = 0.0             # C, reverse-recovery charge (0: Schottky)...
    i_qrr: float = 1.0           # A, ...at this forward current
    assumed: bool = True

    def cj_at(self, v):
        return self.cj * min(max(math.sqrt((self.v_ref + 0.5) / (max(v, 0.0) + 0.5)), 0.2), 6.0)


# generic switches inside a converter IC (typical of small buck / boost regulators)
INTERNAL_FET = dict(name="internal switch", rds_on=0.1, ciss=300e-12, coss=100e-12, crss=10e-12, v_ref=0.0,
                    rg=0.0, ld=0.4e-9, ls=0.4e-9, lg=0.0, qrr=2e-9, i_qrr=1.0)
INTERNAL_EDGE = 5e-9            # s, an internal switch's edge when its datasheet gives no rise / fall time


@dataclass
class Device:
    ref: str                     # footprint reference
    role: str                    # upper | lower | cap | gate_r
    kind: str                    # fet | diode | cap | res
    nets: dict                   # pin -> net: fet D/S/G, diode A/K, cap and res "1"/"2"
    model: dict = field(default_factory=dict)
    enabled: bool = True
    name: str = ""               # unique name ("" = ref); an IC's switches are "U1 high-side", ...
    kelvin: list = field(default_factory=list)    # pad numbers of a Kelvin-source pin (FET)
    internal: bool = False       # a switch inside an IC: no gate pin, no gate loop
    pads: dict = field(default_factory=dict)       # pin -> pad id ("R7.2"): an element between any two pads

    @property
    def key(self):
        return self.name or self.ref

    def pins(self, board):
        """{pin: [board pad indices]}; a Kelvin pin takes its pads out of the source pin; a pin given
        as a pad id (an element between any two pads) is that pad alone."""
        out = {}
        for pin, net in self.nets.items():
            if not net:
                continue
            if self.pads.get(pin):
                out[pin] = [k for k, p in enumerate(board.pads) if p.id == self.pads[pin]]
                continue
            out[pin] = [k for k, p in enumerate(board.pads) if p.ref == self.ref and p.net == net
                        and not (pin == "S" and p.number in self.kelvin)]
        if self.kind == "fet" and self.kelvin and self.nets.get("S"):
            out["KS"] = [k for k, p in enumerate(board.pads) if p.ref == self.ref and p.net == self.nets["S"]
                         and p.number in self.kelvin]
        return out

    def fet(self):
        return FetModel(**{k: v for k, v in self.model.items() if k in {f.name for f in fields(FetModel)}})

    def diode(self):
        return DiodeModel(**{k: v for k, v in self.model.items() if k in {f.name for f in fields(DiodeModel)}})

    def part(self):
        return part_from_dict(self.model)


@dataclass
class GateDrive:
    """Driver output pin -> series parts (any number, any order: the copper links them) -> gate pin,
    returning from the source to the driver's return pin."""
    fet: str                     # FET (its key)
    out: list                    # [driver ref, net] of the output pin
    ret: list                    # [driver ref, net] of the return pin
    chain: list = field(default_factory=list)      # keys of the series parts (gate_r devices)
    rout: float = 2.0            # Ω, driver output resistance (turning the FET on: source)
    rout_off: float = 0.0        # Ω, turning it off (sink); 0 = same as rout
    v_drive: float = 10.0        # V, the driver's output swing (its supply)

    @property
    def r_ext(self):             # the first series part (older setups had one gate resistor)
        return self.chain[0] if self.chain else ""


@dataclass
class Setup:
    name: str = ""               # the cell's name (a board may have several cells: "" = its switch node)
    topology: str = "auto"
    sw_net: str = ""
    rail_net: str = ""
    ret_net: str = ""
    devices: list = field(default_factory=list)
    drives: list = field(default_factory=list)
    v_rail: float = 12.0         # rail voltage: across the off switch; sets C(V) of the switches and the caps' DC bias
    margin: float = 3.0          # mm around the parts
    cells: int = 4000            # target mesh cells (sets the pitch)
    f_min: float = 1e6
    f_max: float = 1e9
    adaptive: bool = True        # fine cells where the loop current crowds (two passes)
    i_switch: float = 0.0        # A, inductor current at the edges (mean; with the parts' rise / fall times: di/dt)
    i_ripple: float = 0.0        # A peak-to-peak: turn-on at I - ΔI/2, turn-off at I + ΔI/2
    f_sw: float = 500e3          # Hz, switching frequency (losses = energy per edge x f_sw)
    i_dir: str = "auto"          # inductor current: "out" of the switch node (buck-like), "in" (boost-like)
    transient: bool = True       # simulate the switching edges (needs the switched current)
    simulator: str = "auto"      # "auto" (ngspice when a FET has a vendor model), "builtin", "ngspice"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        s = Setup(**{k: v for k, v in d.items() if k in {f.name for f in fields(Setup)} - {"devices", "drives"}})
        s.devices = [Device(**{k: v for k, v in x.items() if k in {f.name for f in fields(Device)}})
                     for x in d.get("devices", [])]
        s.drives = []
        for x in d.get("drives", []):
            g = GateDrive(**{k: v for k, v in x.items() if k in {f.name for f in fields(GateDrive)}})
            if not g.chain and x.get("r_ext"):              # setups from before series chains
                g.chain = [x["r_ext"]]
            s.drives.append(g)
        return s

    def by_role(self, role):
        return [x for x in self.devices if x.role == role and x.enabled]

    def device(self, key):
        """A part by its name (key), else by its reference."""
        return next((x for x in self.devices if x.key == key), None) or \
            next((x for x in self.devices if x.ref == key), None)

    def label(self):
        up = {x.kind for x in self.by_role("upper")}
        lo = {x.kind for x in self.by_role("lower")}
        if self.topology != "auto":
            return TOPOLOGIES[self.topology]
        ic = any(x.internal for x in self.by_role("upper") + self.by_role("lower"))
        if ic:
            if "fet" in up and "fet" in lo:
                return "Converter IC with internal switches (synchronous)"
            return "Converter IC with an internal switch" + (" and a diode" if up | lo >= {"diode"} else "")
        if "fet" in up and "fet" in lo:
            return "Synchronous half-bridge (sync buck / sync boost / bridge leg)"
        if "fet" in up:
            return "Buck with diode rectifier"
        if "fet" in lo:
            return "Boost with diode rectifier"
        return "Switching cell"

    def current_dir(self, board):
        """("out" | "in", reason): the inductor current's direction at the switch node.  "out" (a
        buck, a bridge sourcing): the upper switch is hard-switched; "in" (a boost): the lower."""
        if self.i_dir in ("out", "in"):
            return self.i_dir, "set by hand"
        if self.topology in ("buck", "boost"):
            return ("out" if self.topology == "buck" else "in"), "from the topology"
        up = {x.kind for x in self.by_role("upper")}
        lo = {x.kind for x in self.by_role("lower")}
        if up == {"diode"} and "fet" in lo:
            return "in", "diode in the upper position (a boost)"
        if lo == {"diode"} and "fet" in up:
            return "out", "diode in the lower position (a buck)"
        _, far = switch_inductor(board, self.sw_net)
        # the converter regulates its output: the feedback divider sits on it
        fb = feedback_nets(board)
        if far and self.rail_net in fb and far not in fb:
            return "in", "the feedback divider is on the supply side (a boost)"
        if far and far in fb and self.rail_net not in fb:
            return "out", "the feedback divider is on the inductor's far side (a buck)"
        name = lambda n: n.upper().replace("/", "")       # noqa: E731
        if far and any(w in name(far) for w in INPUT_WORDS):
            return "in", "the inductor's far side looks like an input (a boost)"
        return "out", "guessed (a buck); set it under Operating point if wrong"

    def active(self):
        """Positions that switch actively ('upper', 'lower')."""
        if self.topology == "buck":
            return ["upper"]
        if self.topology == "boost":
            return ["lower"]
        return [p for p in ("upper", "lower") if any(x.kind == "fet" for x in self.by_role(p))]


# ---- defaults for parts read from the board -------------------------------------------
ESL_BY_SIZE = {"0201": 0.3e-9, "0402": 0.4e-9, "0603": 0.6e-9, "0805": 0.8e-9, "1206": 1.0e-9, "1210": 1.0e-9,
               "1812": 1.2e-9, "2220": 1.2e-9}


def _size(lib):
    for a, b in EL.METRIC_TO_IMPERIAL.items():
        if f"_{b}_" in lib or f"_{a}Metric" in lib or lib.endswith(f"_{b}"):
            return b
    m = re.search(r"_(0201|0402|0603|0805|1206|1210|1812|2220)", lib)
    return m.group(1) if m else ""


def cap_part(fp):
    """A capacitor model for a footprint: value from its Value field, ESL / ESR by package
    (generic: replace with a library part)."""
    try:
        c = parse_si(re.split(r"[\s/_]", fp.value.replace("µ", "u"))[0])
    except ValueError:
        c = 1e-6
    size = _size(fp.lib)
    ceramic = bool(size)
    esl = ESL_BY_SIZE.get(size, 5e-9 if not ceramic else 1e-9)
    p = Part("C", f"{fp.value} {size or 'cap'} (generic)", c, 10, "%", size, notes="assumed from footprint",
             esr=0.005 if ceramic else 0.05, esl=esl, dielectric="X7R" if c >= 1e-9 else "C0G")
    return asdict(p)


def _pads(board, ref):
    return [p for p in board.pads if p.ref == ref and p.net]


def _fp(board, ref):
    return next((f for f in board.footprints if f.ref == ref), None)


def _centre(board, ref):
    ps = _pads(board, ref)
    return (sum(p.pos[0] for p in ps) / len(ps), sum(p.pos[1] for p in ps) / len(ps)) if ps else (0.0, 0.0)


def _is_ret(board, net):
    if any(w in net.upper().replace("/", "") for w in GND_WORDS) or net.upper().endswith(("_N", "-")):
        return True
    return False


def switch_nets(board):
    """Candidate switch nodes, best first: nets joining an inductor and a transistor or diode."""
    out = []
    for net in board.nets:
        refs = {p.ref for p in board.pads if p.net == net}
        has_l = any(r.startswith("L") for r in refs)
        sw = [r for r in refs if r[:1] in "QD" or r.startswith(("U", "IC"))]
        if has_l and sw:
            score = 3 * sum(r.startswith("Q") for r in refs) + 2 * sum(r.startswith("D") for r in refs) + len(sw)
            out.append((score, net))
    return [n for _, n in sorted(out, key=lambda t: (-t[0], t[1]))]


def detect(board, sw_net=None, max_cap_mm=20.0):
    """Best-guess Setup for a switch node (auto-picked if None).  Raises ValueError if none."""
    if not sw_net:
        cands = switch_nets(board)
        if not cands:
            raise ValueError("No switch node found (a net joining an inductor and a transistor or diode). "
                             "Pick it by hand.")
        sw_net = cands[0]
    s = Setup(sw_net=sw_net)
    on_sw = sorted({p.ref for p in board.pads if p.net == sw_net}, key=EL.natural_key)
    power = []                       # (ref, kind, other power net, gate net)
    for ref in on_sw:
        nets = {}
        for p in _pads(board, ref):
            nets[p.net] = nets.get(p.net, 0) + 1
        others = [n for n in nets if n != sw_net]
        if ref.startswith("Q") and len(others) >= 2:
            # the gate: fewest pads on the part; ties go to the net that looks least like power
            # (no pour, fewest connections on the board, not a return)
            zone_area = {n: sum(z.geom.area() for z in board.zones if z.net == n) for n in others}
            n_pads = {n: sum(1 for q in board.pads if q.net == n) for n in others}
            gate = min(others, key=lambda n: (nets[n], zone_area[n] > 0, n_pads[n], _is_ret(board, n)))
            pw = max((n for n in others if n != gate), key=lambda n: nets[n])
            power.append((ref, "fet", pw, gate))
        elif ref.startswith("D") and len(others) == 1:
            power.append((ref, "diode", others[0], None))
    ics = [r for r in on_sw if r.startswith(("U", "IC"))]
    if not power and not ics:
        raise ValueError(f"No transistor, diode or converter IC on {sw_net}.")
    pnets = sorted({pw for _, _, pw, _ in power})
    ic_ret = {}
    for u in ics:            # an IC's return pin: its GND-like net with the most pads (PGND)
        cnt = {}
        for p in _pads(board, u):
            if p.net != sw_net and _is_ret(board, p.net):
                cnt[p.net] = cnt.get(p.net, 0) + 1
        if cnt:
            ic_ret[u] = max(cnt, key=cnt.get)
    if not pnets and ic_ret:
        pnets = [next(iter(ic_ret.values()))]
    ret = [n for n in pnets if _is_ret(board, n)]
    if not ret:
        area = {n: sum(z.geom.area() for z in board.zones if z.net == n) for n in pnets}
        ret = [max(pnets, key=lambda n: area[n])] if len(pnets) > 1 else []
    s.ret_net = ret[0] if ret else ""
    if not s.ret_net and ic_ret:
        s.ret_net = next(iter(ic_ret.values()))
    rails = [n for n in pnets if n != s.ret_net]
    s.rail_net = rails[0] if rails else ""
    ic_up = None
    if not s.rail_net or not any(pw != s.ret_net for _, _, pw, _ in power):
        for u in ics:            # the IC's hot-loop rail: decoupled to return, not the inductor's far end
            r_ = _ic_rail(board, u, sw_net, s.ret_net)
            if r_:
                s.rail_net, ic_up = s.rail_net or r_, u
                break
    if not s.ret_net or not s.rail_net:
        raise ValueError(f"Could not tell the rail from the return among {', '.join(pnets)}: only one side of "
                         f"the cell has a switching part on {sw_net}.")
    for ref, kind, pw, gate in power:
        role = "lower" if pw == s.ret_net else "upper"
        fp = _fp(board, ref)
        if kind == "fet":
            m = asdict(FetModel(name=fp.value if fp else ref))
            nets = dict(D=sw_net if role == "lower" else pw, S=pw if role == "lower" else sw_net, G=gate)
        else:
            m = asdict(DiodeModel(name=fp.value if fp else ref))
            nets = dict(A=sw_net, K=pw) if role == "upper" else dict(A=pw, K=sw_net)
        s.devices.append(Device(ref, role, kind, nets, m))
    # positions with no external part: the converter IC's internal switches
    for role in ("upper", "lower"):
        if any(d.role == role for d in s.devices):
            continue
        u = ic_up if role == "upper" else next((x for x in ics if ic_ret.get(x) == s.ret_net), None)
        if u is None and role == "upper":
            u = next((x for x in ics if any(p.net == s.rail_net for p in _pads(board, x))), None)
        if u is None:
            continue
        fp = _fp(board, u)
        nets = dict(D=s.rail_net, S=sw_net) if role == "upper" else dict(D=sw_net, S=s.ret_net)
        m = {**asdict(FetModel()), **INTERNAL_FET, "name": f"{fp.value if fp else u} internal"}
        s.devices.append(Device(u, role, "fet", nets, m, name=f"{u} {'high' if role == 'upper' else 'low'}-side",
                                internal=True))
    if not s.by_role("upper") or not s.by_role("lower"):
        missing = "upper" if not s.by_role("upper") else "lower"
        raise ValueError(f"Nothing found for the {missing} position on {sw_net}: add the part (or the IC's "
                         "internal switch) by hand.")
    s.devices += find_links(board, s)
    s.devices += find_caps(board, s, max_cap_mm)
    # gate drives: from the gate, through series two-pin parts (R, FB, L), to a driver IC's pin; the
    # return is the driver's pin on the FET's source net
    for d in [x for x in s.devices if x.kind == "fet" and not x.internal and x.nets.get("G")]:
        g = drive_from_gate(board, s, d)
        if g is not None:
            s.drives.append(g)
    s.topology = "auto"
    return s


SERIES_PREFIXES = ("R", "FB", "L")


def series_part(board, ref, net_a, net_b):
    """A series part of a gate drive (a gate_r device): a resistor / bead footprint between two nets."""
    fp = _fp(board, ref)
    r = (EL.parse_ohms(fp.value) if fp and ref.startswith("R") else 0.0) or 0.0
    return Device(ref, "gate_r", "res", {"1": net_a, "2": net_b}, dict(r=r, ls=0.5e-9, name=fp.value if fp else ref))


def drive_from_gate(board, s, d, max_parts=4):
    """GateDrive for FET d: walk from its gate net through series two-pin parts (resistors, beads)
    until a net with a driver IC pin; the parts found are added to s.devices.  None if no driver."""
    net, chain, seen = d.nets["G"], [], {d.ref}
    src = d.nets["S"]
    for _ in range(max_parts + 1):
        drv = [p.ref for p in board.pads if p.net == net and p.ref not in seen and p.ref.startswith(("U", "IC"))
               and any(q.net == src for q in _pads(board, p.ref))]
        if drv:
            for part in chain:
                if not s.device(part.key):
                    s.devices.append(part)
            return GateDrive(d.key, [drv[0], net], [drv[0], src], [c.key for c in chain])
        step = None
        for fp in board.footprints:
            if fp.ref in seen or not fp.ref.startswith(SERIES_PREFIXES):
                continue
            nets = [p.net for p in _pads(board, fp.ref)]
            if len(nets) == 2 and net in nets and len(set(nets)) == 2:
                step = fp.ref, next(n for n in nets if n != net)
                break
        if step is None:
            return None
        ref, nxt = step
        seen.add(ref)
        chain.append(series_part(board, ref, nxt, net))
        net = nxt
    return None


def element_part(board, name, pad_a, pad_b, r, ls=0.5e-9):
    """A series part between any two pads (a pad-to-pad element, as in DC & Thermal)."""
    nets = {p.id: p.net for p in board.pads}
    ref = pad_a.rsplit(".", 1)[0]
    return Device(ref, "gate_r", "res", {"1": nets.get(pad_a, ""), "2": nets.get(pad_b, "")},
                  dict(r=r, ls=ls, name=name), name=name, pads={"1": pad_a, "2": pad_b})


SMALL_CAP = 1e-9        # F: below this a capacitor is a filter / RF cap, not a hot-loop cap...
SMALL_CAP_MM = 5.0      # ...unless it sits this close to the switches (an HF bypass right at the pins)


def find_caps(board, s, max_cap_mm=20.0, old=()):
    """Hot-loop capacitors of cell s: across its rail and return, enabled within max_cap_mm of the
    switching parts.  Parts in `old` (devices from before) keep their model and on / off state."""
    keep = {d.ref: d for d in old if d.kind == "cap"}
    group = rail_nets(s)
    sw = [d for d in s.devices if d.role in ("upper", "lower")]
    cx = [_centre(board, d.ref) for d in sw] or [(0.0, 0.0)]
    c0 = (sum(x for x, _ in cx) / len(cx), sum(y for _, y in cx) / len(cx))
    out = []
    for fp in board.footprints:
        if not fp.ref.startswith("C") or fp.dnp:
            continue
        nets = {p.net for p in _pads(board, fp.ref)}
        rail = next((n for n in nets if n in group), None)
        if rail and nets == {rail, s.ret_net}:
            x, y = _centre(board, fp.ref)
            pins = {"1": rail, "2": s.ret_net}
            if fp.ref in keep:
                d = keep[fp.ref]
                d.nets = pins
                out.append(d)
            else:
                m, dist = cap_part(fp), math.hypot(x - c0[0], y - c0[1])
                # small caps (filter / RF) only count when they sit right at the switches
                small = m.get("value", 1.0) < SMALL_CAP and dist > SMALL_CAP_MM
                out.append(Device(fp.ref, "cap", "cap", pins, m, enabled=dist <= max_cap_mm and not small))
    return out


def set_cell_nets(board, s, rail=None, ret=None):
    """Change the cell's rail and / or return: the switching parts' pins and the drivers' return pins
    on the old nets move to the new ones, and the hot-loop capacitors are searched again."""
    old_rail, old_ret = s.rail_net, s.ret_net
    s.rail_net = rail or s.rail_net
    s.ret_net = ret or s.ret_net
    for d in s.devices:
        if d.role in ("upper", "lower"):
            for pin, net in d.nets.items():
                if net == old_rail:
                    d.nets[pin] = s.rail_net
                elif net == old_ret:
                    d.nets[pin] = s.ret_net
    for g in s.drives:
        if g.ret[1] == old_ret:
            g.ret = [g.ret[0], s.ret_net]
    caps = [d for d in s.devices if d.kind == "cap"]
    s.devices = [d for d in s.devices if d.kind != "cap" and d.role != "link"]
    s.devices += find_links(board, s)
    s.devices += find_caps(board, s, old=caps)


LINK_PREFIXES = ("R", "JP", "SJ", "NT", "SB")


def is_link(board, ref):
    """A fitted 0 Ω part joining two nets: a 0 Ω resistor, a solder jumper or a net tie (not a ferrite
    bead: at the ringing frequency a bead is a high impedance, so caps behind it are not in the loop)."""
    fp = _fp(board, ref)
    if fp is None or fp.dnp or not ref.startswith(LINK_PREFIXES):
        return False
    if len({p.net for p in _pads(board, ref)}) != 2:
        return False
    if ref.startswith("R"):
        return EL.parse_ohms(fp.value) == 0
    return True


def link_part(board, ref, net_a, net_b):
    fp = _fp(board, ref)
    return Device(ref, "link", "res", {"1": net_a, "2": net_b}, dict(r=0.0, ls=0.5e-9, name=fp.value if fp else ref))


def rail_nets(s):
    """The cell's rail and the nets joined to it by its enabled rail links (0 Ω parts)."""
    group, grow = {s.rail_net}, True
    links = [d for d in s.devices if d.role == "link" and d.enabled]
    while grow:
        grow = False
        for d in links:
            a, b = d.nets.get("1"), d.nets.get("2")
            if (a in group) != (b in group):
                group |= {a, b}
                grow = True
    return group


def find_links(board, s):
    """Rail links of cell s: 0 Ω parts that carry its rail on to other nets (an output cap reached
    through a 0 Ω option resistor).  Never into the switch node, the return or the inductor's far end
    (a boost's input: a 0 Ω from VOUT to VIN is a bypass option, not part of the hot loop)."""
    stop = {s.sw_net, s.ret_net, ""} | (_inductor_far(board, s.sw_net) - {s.rail_net})
    group, out, todo = {s.rail_net}, [], [s.rail_net]
    while todo:
        net = todo.pop()
        for ref in sorted({p.ref for p in board.pads if p.net == net}, key=EL.natural_key):
            if any(d.ref == ref for d in out) or not is_link(board, ref):
                continue
            other = next(p.net for p in _pads(board, ref) if p.net != net)
            if other in stop or other in group:
                continue
            out.append(link_part(board, ref, net, other))
            group.add(other)
            todo.append(other)
    return out


INPUT_WORDS = ("PVIN", "VIN", "IN", "VCC", "VDD", "VBAT", "VSUP")


def _inductor_far(board, sw_net):
    """Nets at the other end of the inductors on the switch node (buck: the output, boost: the input)."""
    out = set()
    for fp in board.footprints:
        if fp.ref.startswith("L"):
            nets = {p.net for p in _pads(board, fp.ref)}
            if sw_net in nets:
                out |= nets - {sw_net}
    return out


def feedback_nets(board):
    """Nets joined by a resistor to a feedback net (a name with FB / VFB / FEEDBACK in it)."""
    fbn = {n for n in board.nets if set(re.split(r"[^A-Z0-9]+", n.upper())) & {"FB", "VFB", "FEEDBACK"}}
    out = set()
    for fp in board.footprints:
        if fp.ref.startswith("R") and not fp.dnp:
            ns = {p.net for p in _pads(board, fp.ref)}
            if len(ns) == 2 and ns & fbn:
                out |= ns - fbn
    return out


def switch_inductor(board, sw_net):
    """(ref, far net) of the inductor on the switch node, or (None, None)."""
    for fp in board.footprints:
        if fp.ref.startswith("L") and not fp.dnp:
            nets = {p.net for p in _pads(board, fp.ref)}
            if sw_net in nets and len(nets) == 2:
                return fp.ref, next(n for n in nets if n != sw_net)
    return None, None


def _ic_rail(board, u, sw_net, ret_net):
    """The hot-loop rail of converter IC u: one of its nets (not the switch node or the return) with a
    capacitor to the return and not at the inductor's far end (a buck IC's input, a boost IC's
    output: SW -> internal high-side switch -> VOUT pin); named like an input if several remain, then
    the one with most pins."""
    cands = {}
    for p in _pads(board, u):
        if p.net in (sw_net, ret_net) or not p.net:
            continue
        cands[p.net] = cands.get(p.net, 0) + 1
    far = _inductor_far(board, sw_net)
    decoupled = set()
    for fp in board.footprints:
        if fp.ref.startswith("C") and not fp.dnp:
            ns = {p.net for p in _pads(board, fp.ref)}
            if ret_net in ns and len(ns) == 2:
                decoupled |= ns - {ret_net}
    for n in list(cands):        # decoupled through a 0 Ω link (the output cap behind an option resistor)
        tmp = Setup(sw_net=sw_net, rail_net=n, ret_net=ret_net)
        tmp.devices = find_links(board, tmp)
        if rail_nets(tmp) & decoupled:
            decoupled.add(n)
    good = [n for n in cands if n in decoupled]
    good = [n for n in good if n not in far] or good
    good = [n for n in good if not _control(n)] or good
    if not good:
        return None

    def score(n):
        name = n.upper().replace("/", "").replace("+", "")
        word = max((len(INPUT_WORDS) - i for i, w in enumerate(INPUT_WORDS) if w in name), default=0)
        return (word, cands[n])
    return max(good, key=score)


CONTROL_WORDS = {"FB", "EN", "SS", "COMP", "BST", "BOOT", "PG", "PGOOD", "ILIM", "RT", "MODE", "SYNC", "TRK",
                 "SENSE", "VSNS", "CS", "NC"}


def _control(net):
    """A converter IC's control-pin net (FB, EN, SS, …), never its power rail."""
    return bool(set(re.split(r"[^A-Z0-9]+", net.upper())) & CONTROL_WORDS)


def make_device(board, s, ref, role):
    """A Device for footprint `ref` in `role` of setup s, its pins assigned from the cell's nets."""
    fp = _fp(board, ref)
    pads = _pads(board, ref)
    nets = {}
    for p in pads:
        nets[p.net] = nets.get(p.net, 0) + 1
    if role == "cap" or (role in ("upper", "lower") and ref.startswith("C")):
        ns = sorted(nets)
        if len(ns) != 2:
            raise ValueError(f"{ref} needs pads on exactly two nets.")
        return Device(ref, "cap", "cap", {"1": ns[0], "2": ns[1]}, cap_part(fp) if fp else {})
    if role == "gate_r":
        ns = sorted(nets)
        return series_part(board, ref, ns[0], ns[-1])
    if role == "link":
        ns = sorted(nets, key=lambda n: n not in rail_nets(s))
        if len(ns) != 2:
            raise ValueError(f"{ref} needs pads on exactly two nets.")
        return link_part(board, ref, ns[0], ns[1])
    if role in ("upper_ic", "lower_ic"):           # a switch inside an IC
        pos = role[:-3]
        hi, lo = (s.rail_net, s.sw_net) if pos == "upper" else (s.sw_net, s.ret_net)
        fp_v = fp.value if fp else ref
        m = {**asdict(FetModel()), **INTERNAL_FET, "name": f"{fp_v} internal"}
        return Device(ref, pos, "fet", dict(D=hi if hi in nets else "", S=lo if lo in nets else ""), m,
                      name=f"{ref} {'high' if pos == 'upper' else 'low'}-side", internal=True)
    hi, lo = (s.rail_net, s.sw_net) if role == "upper" else (s.sw_net, s.ret_net)
    if hi not in nets or lo not in nets:
        raise ValueError(f"{ref} has no pads on {hi} and {lo} (the {role} position).")
    others = [n for n in nets if n not in (hi, lo)]
    if others or ref.startswith("Q"):
        m = asdict(FetModel(name=fp.value if fp else ref))
        return Device(ref, role, "fet", dict(D=hi, S=lo, G=others[0] if others else ""), m)
    m = asdict(DiodeModel(name=fp.value if fp else ref))
    return Device(ref, role, "diode", dict(A=lo, K=hi), m)
