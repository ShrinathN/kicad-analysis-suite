"""Save/load the suite scenario (<board>.current_sim.json).

One file per board with one section per tool:
  {"version": 2, "stackup": {...}, "dc": {"rows", "elements", "settings"}, "matching": {...}, ...}
Saving a section rewrites only that section, so the tools never clobber each other.
Version-1 files (rows / elements / settings at the top level) load as the "dc" section.

Copper-thickness overrides belong to the shared stackup ("stackup.cu_thickness");
the DC tool's Settings.thickness is filled from there on load and written back on save.

Only user-defined elements and edited auto elements need saving; the rest
are re-detected from the board on load (see elements.merge)."""
import json
import os
from dataclasses import asdict, fields

from .elements import Element
from .solver import Row, Settings

VERSION = 2


def scenario_path(board_path):
    return os.path.splitext(board_path)[0] + ".current_sim.json"


def read(path):
    """The whole scenario as a version-2 dict ({} sections when the file is missing)."""
    if not os.path.exists(path):
        return {"version": VERSION}
    with open(path) as f:
        d = json.load(f)
    if d.get("version", 1) < 2:
        dc = {k: d[k] for k in ("rows", "elements", "settings") if k in d}
        d = {"version": VERSION, "dc": dc}
        th = dc.get("settings", {}).get("thickness")
        if th:
            d["stackup"] = {"cu_thickness": th}
    return d


def section(path, name):
    return read(path).get(name) or {}


def save_section(path, name, data, **more):
    """Replace one section (and any `more` sections) and keep the rest of the file."""
    d = read(path)
    d[name] = data
    d.update(more)
    d["version"] = VERSION
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, path)


def save_scenario(path, rows, settings, elements=()):
    st = section(path, "stackup")
    st["cu_thickness"] = dict(settings.thickness)
    s = asdict(settings)
    s.pop("thickness")
    save_section(path, "dc", dict(rows=[asdict(r) for r in rows], elements=[asdict(e) for e in elements],
                                  settings=s), stackup=st)


def load_scenario(path):
    d = read(path)
    dc = d.get("dc") or {}
    rows = [Row(**r) for r in dc.get("rows", [])]
    known = {f.name for f in fields(Settings)}
    s = Settings(**{k: v for k, v in dc.get("settings", {}).items() if k in known})
    s.thickness = dict((d.get("stackup") or {}).get("cu_thickness") or s.thickness)
    ek = {f.name for f in fields(Element)}
    els = [Element(**{k: v for k, v in e.items() if k in ek}) for e in dc.get("elements", [])]
    return rows, s, els
