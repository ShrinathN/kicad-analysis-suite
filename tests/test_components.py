import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pcb_current_sim import components as CO   # noqa: E402
from pcb_current_sim import elements as EL     # noqa: E402
from pcb_current_sim import scenario as SC     # noqa: E402
from pcb_current_sim import solver as S        # noqa: E402
from pcb_current_sim.suite.session import Session  # noqa: E402
from test_solver import part, series_board, smd, make_board  # noqa: E402


def two_resistors():
    return make_board(smd("A", (0, 0), "N1") + smd("C", (20, 0), "N2")
                      + part("R1", "0R1", [("1", (9, 0), "N1"), ("2", (11, 0), "N2")])
                      + part("R2", "10k", [("1", (9, 5), "N1"), ("2", (11, 5), "N2")])
                      + part("L1", "10u", [("1", (9, 9), "N1"), ("2", (11, 9), "N2")], lib="Inductor_SMD:L_1206_3216Metric"))


def test_old_per_tool_lists_merge_into_one_component_list():
    old = {"version": 2,
           "dc": {"elements": [asdict_el("L1", "L1.1", "L1.2", 0.02),
                               asdict_el("Q1", "A.1", "C.1", 0.005),
                               dict(asdict_el("R2", "R2.1", "R2.2", 1e4, auto=True), enabled=False)]},
           "extract": {"elements": [asdict_el("L1", "L1.1", "L1.2", 0.02),             # same part
                                    asdict_el("Q1", "A.1", "C.1", 0.0, model="C", C=1e-6)]}}  # different
    items = {e.name: e for e in CO.migrate(old)}
    assert items["L1"].use == {"dc": True, "extract": True}
    assert items["Q1"].use == {"dc": True} and items["Q1_trace"].use == {"extract": True}
    assert items["Q1_trace"].model == "C" and items["R2"].use == {"dc": False}


def asdict_el(name, a, b, R, **kw):
    from dataclasses import asdict
    d = asdict(EL.Element(name, a, b, R, **kw))
    d.pop("use")                     # files written before the shared list had no "use"
    return d


def test_views_and_switches_per_tool():
    b = two_resistors()
    c = CO.Components.build(b, [])
    assert {e.name for e in c.items} == {"R1", "R2"}            # resistors only are detected
    dc, tr = c.view("dc"), c.view("extract")
    assert all(e.enabled for e in dc) and not any(e.enabled for e in tr)
    # Trace adds the inductor and uses R1; DC switches R2 off
    tr[0 if tr[0].name == "R1" else 1].enabled = True
    tr.append(EL.Element("L1", "L1.1", "L1.2", 0.0, model="L", L=10e-6))
    c.sync("extract", tr)
    dc = c.view("dc")
    next(e for e in dc if e.name == "R2").enabled = False
    c.sync("dc", dc)
    on = {t: {e.name for e in c.view(t) if e.enabled} for t in ("dc", "extract", "pdn")}
    assert on == {"dc": {"R1"}, "extract": {"R1", "L1"}, "pdn": set()}
    assert c.users(c.get("L1")) == ["Trace R/L/C"]
    saved = {e.name: e.use for e in c.to_save()}
    assert saved == {"R1": {"extract": True}, "R2": {"dc": False}, "L1": {"extract": True}}
    # reload: same switches, detected values from the board
    c2 = CO.Components.build(b, c.to_save())
    assert {t: {e.name for e in c2.view(t) if e.enabled} for t in ("dc", "extract")} == \
        {t: on[t] for t in ("dc", "extract")}
    # a tool deleting a user component removes it for everyone; detected ones fall back to the default
    c2.sync("extract", [e for e in c2.view("extract") if e.name != "L1"])
    assert c2.get("L1") is None


def test_dc_scenario_and_session_share_the_list(tmp_path):
    bp = tmp_path / "s.kicad_pcb"
    src = series_board()
    # write a board file: reuse the text of the test board via the parser's round trip is not available,
    # so build the session on a parsed board directly
    s = Session(library_path=str(tmp_path / "lib.json"))
    s.board, s.board_path = src, str(bp)
    s.components = CO.Components.load(src, s.scenario_path)
    seen = []
    s.listen(lambda w: seen.append((w, s.sharing)))
    tr = s.components.view("extract")
    tr.append(EL.Element("D1", "A.1", "C.1", 0.01, model="V", Vf=0.4))
    s.share("extract", tr)
    assert seen == [("components", "extract")]
    d = json.loads((tmp_path / "s.current_sim.json").read_text())
    assert [x["name"] for x in d["components"]["items"]] == ["D1"]
    # the DC tool (file API, as the Tk GUI and CLI use it) sees D1 switched off and keeps it on save
    rows, st, els = SC.load_scenario(s.scenario_path)
    assert [(e.name, e.enabled) for e in els] == [("D1", False)]
    SC.save_scenario(s.scenario_path, rows, st, EL.to_save(EL.merge(EL.detect(src)[0], els)))
    c = CO.Components.load(src, s.scenario_path)
    assert c.get("D1").use == {"extract": True}            # DC never switched it
    assert c.get("D1").model == "V" and c.get("R1").use == {}
    # the DC solver ignores a component DC has off
    r = S.simulate(src, [S.Row("r", ["A.1"], ["C.1"], 1.0)],
                   S.Settings(pitch=0.05, thermal=False, error_estimate=False, magnetic=False),
                   progress=lambda m: None, elements=[e for e in c.view("dc")])
    assert "D1" not in {e["name"] for e in r.elements}


def test_loop_capacitor_model_is_shared():
    from pcb_current_sim.loop import model as M
    from pcb_current_sim.loop import saved as SV
    from test_loop import buck_cell
    b = buck_cell()
    s = M.detect(b)
    c1 = s.device("C1")
    c1.model = dict(c1.model, esl=0.25e-9, esr=0.002, name="GRM31 10u X7R")     # edited in Switching Loop
    comps = CO.Components.build(b, [])
    comps.put("loop", SV.cap_component(b, c1))
    e = comps.get("C1")
    assert (e.model, e.use) == ("C", {"loop": True}) and abs(e.C - 10e-6) < 1e-12 and e.part["esl"] == 0.25e-9
    # another cell (or a fresh detection) gets the same model
    s2 = M.detect(b)
    assert SV.apply_components(s2, comps) == ["C1"] and s2.device("C1").model["esl"] == 0.25e-9
    # Trace edits the ideal value: the real model is kept, with the new capacitance
    tr = comps.view("extract")
    next(x for x in tr if x.name == "C1").__dict__.update(C=22e-6, part={})
    comps.sync("extract", tr)
    assert comps.get("C1").part["esl"] == 0.25e-9
    SV.apply_components(s2, comps)
    assert s2.device("C1").model["value"] == 22e-6 and s2.device("C1").model["name"] == "GRM31 10u X7R"


def test_components_window_operations():
    c = CO.Components.build(two_resistors(), [])
    c.edit("", EL.Element("Q1", "A.1", "C.1", 0.005), use={"dc": True, "extract": True})
    assert c.switch(["Q1", "R1"], "extract") is True            # R1 was off: all on
    assert c.get("R1").use == {"extract": True}
    assert c.switch(["Q1", "R1"], "extract") is False           # all were on: all off
    assert not CO.used(c.get("Q1"), "extract") and c.get("R1").use == {}
    c.edit("Q1", EL.Element("Q1", "A.1", "C.1", 0.004))         # switches kept
    assert c.get("Q1").R == 0.004 and c.get("Q1").use == {"dc": True}         # off = the default, not stored
    assert c.remove(["Q1", "R2"]) == ["R2"] and c.get("Q1") is None and c.get("R2") is not None
