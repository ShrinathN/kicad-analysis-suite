"""PDN setup: the rail, its decoupling capacitors, the regulator feeding it (VRM) and the loads.

A rail is one or more power nets (joined by the copper of the board only; a ferrite bead or a load
switch between two rails comes later as a component) over one or more return nets.  The return
copper is the ideal reference of v1: the power copper's inductance is that of copper over the
nearest return plane (no ground bounce between distant GND pins).

Pads are given by id ("C12.1"); a part's pins on the rail are one terminal node, its pins on a
return net its return.
"""
import re
from dataclasses import asdict, dataclass, field

from .. import elements as EL
from ..extract.rlc import guess_return_nets


@dataclass
class Decap:
    ref: str
    pads: list                   # pad ids on the rail
    ret: list                    # pad ids on a return net
    part: dict = field(default_factory=dict)      # suite.library.Part as a dict (kind "C")
    enabled: bool = True
    l_ret: float = -1.0          # H, return-side mounting L (pad + vias to the return plane); < 0 = estimate

    @property
    def key(self):
        return self.ref


@dataclass
class Vrm:
    """The source feeding the rail, seen at its output pin: R + L (step 3 adds a regulator loop model
    fitted to the datasheet's load-transient plot)."""
    ref: str = ""
    pads: list = field(default_factory=list)      # pad ids on the rail (the regulator's OUT, or the
    #                                               buck inductor's output pad)
    r_out: float = 5e-3          # Ω
    l_out: float = 20e-9         # H: about 1 / ((2π f_bw)² C_out) for a loop of bandwidth f_bw
    enabled: bool = True
    how: str = ""                # how it was found
    kind: str = "RL"             # "RL": r_out + L_out; "loop": the fitted control loop (`model`)
    model: dict = field(default_factory=dict)     # pdn.regulator.RegulatorModel as a dict


@dataclass
class Load:
    """A chip on the rail.  Its current over time is a PULSE (as in SPICE): it rises by i_step at
    `delay` over t_rise, stays for `width` (< 0: stays on), falls over t_fall, repeating every
    `period` (0: once).  Stagger the loads' delays to see each one's effect on its own."""
    name: str
    pads: list                   # pad ids on the rail (all one terminal)
    enabled: bool = True
    i_step: float = 1.0          # A
    delay: float = 20e-6         # s
    t_rise: float = 1e-6
    width: float = 200e-6
    t_fall: float = 1e-6
    period: float = 0.0


@dataclass
class Target:
    v: float = 3.3               # V, the rail voltage
    ripple: float = 5.0          # %, allowed deviation
    i_step: float = 1.0          # A, load step
    t_rise: float = 1e-6         # s, its rise time: the target holds up to 0.35 / t_rise

    @property
    def z(self):
        return self.v * self.ripple / 100 / max(self.i_step, 1e-12)

    @property
    def f_max(self):
        return 0.35 / max(self.t_rise, 1e-15)


@dataclass
class Setup:
    rail: list = field(default_factory=list)            # power net(s)
    returns: list = field(default_factory=list)         # return net(s)
    decaps: list = field(default_factory=list)
    vrm: Vrm = field(default_factory=Vrm)
    loads: list = field(default_factory=list)
    target: Target = field(default_factory=Target)
    f_min: float = 1e3
    f_max: float = 1e9
    n_f: int = 161
    cells: int = 12000           # mesh size limit (the pitch follows from the rail's copper area)
    pitch: float = 0.0           # mm, 0 = from cells
    margin: float = 2.0          # mm around the rail's copper
    T: float = 25.0              # °C, for the capacitors' temperature curve
    transient: bool = True       # also simulate the loads' steps over time
    t_stop: float = 0.0          # s, simulated time (0: until the regulator has settled after the last edge)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        s = Setup(**{k: v for k, v in d.items() if k not in ("decaps", "vrm", "loads", "target")
                     and k in Setup.__dataclass_fields__})
        s.decaps = [Decap(**x) for x in d.get("decaps", [])]
        s.vrm = Vrm(**d.get("vrm", {}))
        s.loads = [Load(**x) for x in d.get("loads", [])]
        s.target = Target(**d.get("target", {}))
        return s


# ---- detection ---------------------------------------------------------------------------
def _pads_by_net(board, ref):
    out = {}
    for p in board.pads:
        if p.ref == ref and p.net and p.layers:
            out.setdefault(p.net, []).append(p.id)
    return out


def _fp(board, ref):
    return next((f for f in board.footprints if f.ref == ref), None)


def rail_candidates(board):
    """Power nets, most likely first: nets with decoupling capacitors to a return net, by count."""
    rets = set(guess_return_nets(board))
    count = {}
    for fp in board.footprints:
        if not fp.ref.startswith("C") or fp.dnp:
            continue
        nets = set(_pads_by_net(board, fp.ref))
        if len(nets) == 2 and nets & rets:
            (n,) = nets - rets or {""}
            if n:
                count[n] = count.get(n, 0) + 1
    return sorted(count, key=lambda n: (-count[n], EL.natural_key(n)))


def cap_model(board, fp, components=None):
    """The capacitor's model: the shared component's real model, else a generic one from the footprint."""
    from ..loop.model import cap_part
    base = cap_part(fp)
    if components is not None:
        e = components.get(fp.ref)
        if e is not None and e.model == "C" and (e.part or e.C > 0):
            from ..components import cap_model as shared
            return shared(e, base)
    return base


def find_decaps(board, rail, returns, components=None, old=()):
    keep = {d.ref: d for d in old}
    out = []
    for fp in board.footprints:
        if not fp.ref.startswith("C") or fp.dnp:
            continue
        by = _pads_by_net(board, fp.ref)
        on = [pid for n in rail for pid in by.get(n, [])]
        ret = [pid for n in returns for pid in by.get(n, [])]
        if not on or not ret or len(by) != 2:
            continue
        if fp.ref in keep:
            d = keep[fp.ref]
            d.pads, d.ret = on, ret
            out.append(d)
        else:
            out.append(Decap(fp.ref, on, ret, cap_model(board, fp, components)))
    return sorted(out, key=lambda d: EL.natural_key(d.ref))


SW_WORDS = ("SW", "LX", "PH", "PHASE")


def find_vrm(board, rail, components=None):
    """The rail's source: a DC & Thermal regulator component with its OUT pad on the rail, else the
    output pad of an inductor whose other end is a switch
    node.  Vrm() with no pads if none (LDOs not in Components: the user picks the output pad)."""
    pad_net = {p.id: p.net for p in board.pads}
    if components is not None:
        for e in components.items:
            if e.regulator and pad_net.get(e.b) in rail:
                return Vrm(e.name, [e.b], how=f"regulator {e.name} (Components)")
    for fp in board.footprints:
        if not fp.ref.startswith("L") or fp.dnp:
            continue
        by = _pads_by_net(board, fp.ref)
        if len(by) != 2:
            continue
        on = [n for n in by if n in rail]
        other = [n for n in by if n not in rail]
        if on and other:
            o = other[0].upper()
            ic = any(p.net == other[0] and p.ref.startswith(("U", "IC")) for p in board.pads)
            if ic or any(w in re.split(r"[^A-Z0-9]+", o) for w in SW_WORDS):
                return Vrm(fp.ref, by[on[0]], how=f"output of inductor {fp.ref} (switch node {other[0]})")
    return Vrm(how="not found: pick the regulator's output pad")


def find_loads(board, rail, vrm_ref=""):
    out = []
    for fp in board.footprints:
        if fp.dnp or fp.ref == vrm_ref or not fp.ref.startswith(("U", "IC")):
            continue
        on = [pid for n in rail for pid in _pads_by_net(board, fp.ref).get(n, [])]
        if on:
            out.append(Load(fp.ref, on))
    return sorted(out, key=lambda x: EL.natural_key(x.name))


def detect(board, rail=None, components=None):
    """A setup for one rail (the best candidate if None)."""
    if rail is None:
        c = rail_candidates(board)
        if not c:
            raise ValueError("No power net with decoupling capacitors to a return net was found: pick the rail.")
        rail = c[0]
    rail = [rail] if isinstance(rail, str) else list(rail)
    returns = [n for n in guess_return_nets(board) if n not in rail]
    s = Setup(rail=rail, returns=returns)
    s.decaps = find_decaps(board, rail, returns, components)
    s.vrm = find_vrm(board, rail, components)
    s.loads = find_loads(board, rail, s.vrm.ref)
    m = re.search(r"(\d+)V(\d*)", rail[0].upper())
    if m:
        try:
            s.target.v = float(f"{m.group(1)}.{m.group(2) or 0}")
        except ValueError:
            pass
    return s
