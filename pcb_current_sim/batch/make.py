"""Tests from the setups the tool windows have saved (the board's .current_sim.json), or from a window's
current setup: each test gets the tool's setup plus the board's stackup overrides and shared components,
so it is self-contained."""
from .spec import TOOLS, Test


def component_dict(e):
    """A shared component as written to the tests file: fields at their default left out."""
    from dataclasses import MISSING, fields
    out = {}
    for f in fields(e):
        v = getattr(e, f.name)
        if f.default is not MISSING:
            dflt = f.default
        elif f.default_factory is not MISSING:
            dflt = f.default_factory()
        else:
            out[f.name] = v
            continue
        if v != dflt:
            out[f.name] = v
    return out


def shared(scen):
    """(stackup overrides, component items) of a scenario dict."""
    from .. import components as CO
    stack = dict(scen.get("stackup") or {})
    items = [component_dict(e) for e in CO.migrate(scen)]
    return stack, items


def from_session(session, tool, name, setup):
    """A test from a window's current setup (session: suite.session.Session)."""
    stack = session.stackup.overrides() if session.stackup is not None else {}
    items = [component_dict(e) for e in session.components.to_save()] if session.components is not None else []
    return Test(name, tool, setup, stackup=_clean(stack), components=items)


def _clean(stack):
    """Leave out empty parts of the stackup overrides."""
    out = {}
    if stack.get("cu_thickness"):
        out["cu_thickness"] = stack["cu_thickness"]
    if stack.get("model"):
        out["model"] = stack["model"]
    if any(d is not None for d in stack.get("dielectrics") or []):
        out["dielectrics"] = stack["dielectrics"]
    return out


def from_scenario(board_path, tools=TOOLS, board=None):
    """Tests for every saved setup of the given tools on this board."""
    from .. import board as B
    from .. import components as CO
    from .. import scenario as SC
    from .tools import get
    scen = SC.read(SC.scenario_path(board_path))
    board = board or B.load(board_path)
    stack, items = shared(scen)
    comps = CO.Components.build(board, CO.migrate(scen))
    out = []
    for tool in tools:
        for name, setup in get(tool).from_scenario(scen, board, comps):
            out.append(Test(name, tool, setup, stackup=_clean(stack), components=items))
    return out
