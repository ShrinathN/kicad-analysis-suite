"""The board's shared component list: parts between pads, defined once and used by every tool.

A component is an `elements.Element` (pads + model: resistance, diode, C, L, regulator).  It is
stored once per board in the scenario's "components" section; each tool switches it on or off for
itself (`Element.use[tool]`), so an inductor's DCR entered in DC & Thermal is the same part the
Trace tool and PDN see, and switching it off in one tool leaves the others alone.

Resistor footprints are detected from the board on every load (`elements.detect`) and are on in
DC & Thermal and off elsewhere unless saved otherwise; only user components and auto ones with
non-default switches are saved.

A tool works on a *view* (copies whose `enabled` is that tool's switch) and hands its edited list
back with `sync(tool, elements)`.  Older scenario files kept separate lists in the "dc" and
"extract" sections: they are merged on the first read (same name and same values: one component
used by both; different values: the Trace one is renamed).
"""
from dataclasses import asdict, fields, replace

from . import elements as EL

TOOLS = {"dc": "DC & Thermal", "extract": "Trace R/L/C", "loop": "Switching Loop", "pdn": "PDN"}
SECTION = "components"
_KEYS = {f.name for f in fields(EL.Element)}


def default_use(e, tool):
    """A component's switch in a tool that never set it: detected resistors are on in DC only."""
    return bool(e.auto) and tool == "dc"


def used(e, tool):
    return bool(e.use.get(tool, default_use(e, tool)))


def from_dict(d):
    e = EL.Element(**{k: v for k, v in d.items() if k in _KEYS})
    e.use, e.part = dict(e.use or {}), dict(e.part or {})
    return e


def cap_model(e, base=None):
    """Loop / PDN capacitor model (library Part dict) of a shared C component: its saved real model,
    else `base` (e.g. the footprint's generic model), with the component's capacitance."""
    m = dict(e.part or base or {})
    if e.C > 0:
        m["value"] = e.C
    return m


def _same(a, b):
    """Same part (ignoring the on / off switches)."""
    da, db = asdict(a), asdict(b)
    for k in ("enabled", "use"):
        da.pop(k), db.pop(k)
    return da == db


def _unique(name, taken):
    if name not in taken:
        return name
    k = 2
    while f"{name}_{k}" in taken:
        k += 1
    return f"{name}_{k}"


def migrate(d):
    """Saved components from a scenario dict: its "components" section, or the old per-tool lists."""
    sec = d.get(SECTION)
    if sec is not None:
        return [from_dict(x) for x in sec.get("items", [])]
    items = {}
    for tool in ("dc", "extract"):
        for raw in (d.get(tool) or {}).get("elements", []):
            e = from_dict(raw)
            on = bool(raw.get("enabled", True))
            prev = items.get(e.name)
            if prev is not None and _same(prev, e):
                prev.use[tool] = on
                continue
            if prev is not None:
                e.name = _unique(f"{e.name}_trace", items)
            e.use, e.enabled = {tool: on}, True
            items[e.name] = e
    return list(items.values())


class Components:
    """The live list for one board (auto-detected resistors merged with the saved components)."""

    def __init__(self, items=(), warnings=()):
        self.items = list(items)
        self.warnings = list(warnings)

    # ---- loading / saving --------------------------------------------------
    @classmethod
    def build(cls, board, saved):
        """Detected resistors + saved components.  A saved user component replaces a detected one with
        the same name (e.g. a resistor whose value was edited); a saved auto entry only carries its
        switches, so value changes on the board are picked up."""
        auto, warn = EL.detect(board) if board is not None else ([], [])
        saved = list({e.name: e for e in saved}.values())          # last entry per name wins
        user = {e.name: e for e in saved if not e.auto}
        flags = {e.name: e.use for e in saved if e.auto}
        out = [replace(e, use=dict(flags.get(e.name, {}))) for e in auto if e.name not in user]
        return cls(out + [e for e in saved if not e.auto], warn)

    @classmethod
    def load(cls, board, path):
        from . import scenario as SC
        try:
            d = SC.read(path) if path else {}
        except (OSError, ValueError):
            d = {}
        return cls.build(board, migrate(d))

    def to_save(self):
        """User components, and detected ones whose switches differ from the defaults."""
        out = []
        for e in self.items:
            use = {t: v for t, v in e.use.items() if bool(v) != default_use(e, t)}
            if e.auto and not use:
                continue
            out.append(replace(e, use=use if e.auto else dict(e.use), enabled=True))
        return out

    def section(self):
        return {"items": [asdict(e) for e in self.to_save()]}

    def save(self, path):
        from . import scenario as SC
        SC.save_section(path, SECTION, self.section())

    # ---- tool views ----------------------------------------------------------
    def get(self, name):
        return next((e for e in self.items if e.name == name), None)

    def view(self, tool):
        """Copies for one tool, `enabled` = that tool's switch."""
        return [replace(e, use=dict(e.use), enabled=used(e, tool)) for e in self.items]

    def put(self, tool, el):
        """Add or update one component from a tool that keeps no view of the list (Switching Loop):
        its other tools' switches are kept and the tool's is set on."""
        old = self.get(el.name)
        use = dict(old.use) if old is not None else {}
        use[tool] = True
        new = replace(el, use=use, enabled=True)
        self.items = [new if e.name == el.name else e for e in self.items] if old is not None \
            else self.items + [new]
        return new

    def edit(self, old_name, new, use=None):
        """Replace (or add) a component edited outside any tool (the Components window): the old one's
        switches and real model stay with it unless `use` is given."""
        old = self.get(old_name) if old_name else None
        keep_use = dict(old.use) if old is not None else {}
        part = new.part or (old.part if old is not None and old.model == new.model else {})
        new = replace(new, use=dict(use if use is not None else keep_use), part=dict(part), enabled=True)
        self.items = [e for e in self.items if e.name not in (old_name, new.name)] + [new]
        return new

    def remove(self, names):
        """Delete user components; detected resistors cannot be deleted (returns their names)."""
        names = set(names)
        kept = [e.name for e in self.items if e.name in names and e.auto]
        self.items = [e for e in self.items if e.name not in names or e.auto]
        return kept

    def switch(self, names, tool, on=None):
        """Switch components on / off in a tool (on=None: all on if any was off, else all off)."""
        sel = [e for e in self.items if e.name in set(names)]
        if on is None:
            on = not all(used(e, tool) for e in sel)
        for e in sel:
            if on == default_use(e, tool):
                e.use.pop(tool, None)
            else:
                e.use[tool] = on
        return on

    def users(self, e, but=None):
        """Names of the tools that use component e (except `but`)."""
        return [TOOLS[t] for t in TOOLS if t != but and used(e, t)]

    def sync(self, tool, elements):
        """Take a tool's edited list back.  `elements` is its whole view, or only the part worth
        saving (elements.to_save: user components and switched-off detected ones): a user component
        missing from it was deleted, a detected one missing from it is on (the default)."""
        by_name = {e.name: e for e in elements}
        out = []
        for e in self.items:
            new = by_name.pop(e.name, None)
            if new is None:
                if e.auto:                                 # left out by to_save: the tool's default
                    e.use.pop(tool, None)
                    out.append(e)
                continue                                   # a user component the tool deleted
            use = dict(e.use)
            if tool in use or bool(new.enabled) != used(e, tool):     # only switches the tool changed
                use[tool] = bool(new.enabled)
            # a tool that only edits the ideal value (DC, Trace) keeps the real model other tools gave it
            part = new.part or (e.part if new.model == e.model else {})
            out.append(replace(new, use=use, enabled=True, part=dict(part)))
        for new in by_name.values():                       # added by the tool
            out.append(replace(new, use={**(new.use or {}), tool: bool(new.enabled)}, enabled=True))
        self.items = out
        return self
