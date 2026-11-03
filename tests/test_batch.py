"""Batch command line: the tests file, checks, the runner, the report and the CLI."""
import json
import math
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pcb_current_sim.batch import tomlio                      # noqa: E402
from pcb_current_sim.batch import runner as R                  # noqa: E402
from pcb_current_sim.batch.spec import Check, Test, TestFile   # noqa: E402
from pcb_current_sim.batch.tools import Metric                 # noqa: E402

DEMO = os.path.join(ROOT, "sim", "demo")


def test_toml_round_trip_of_awkward_values():
    d = {"a": 1, "b": 2.5e-05, "c": [1, None, "x"], "d": {"e": None, "f": {"g": [{"h": 1}, {"h": float("inf")}]}},
         "weird key": "quote \" and \\ and\nnewline", "empty": {}, "lst": [], "t": True, "small": {"x": 1}}
    back = tomlio.loads(tomlio.dumps({"top": d}, ["comment"]))["top"]
    assert back == d


def test_tests_file_parsing_errors(tmp_path):
    p = tmp_path / "x.simtests.toml"
    p.write_text('board = "b.kicad_pcb"\n[[test]]\nname = "a"\ntool = "nope"\n')
    with pytest.raises(ValueError, match="tool must be"):
        TestFile.load(str(p))
    p.write_text('board = "b.kicad_pcb"\n[[test]]\nname = "a"\ntool = "dc"\n[[test.check]]\nwhat = "T_max_C"\nmax = 1\n'
                 'limit = 2\n')
    with pytest.raises(ValueError, match="unknown key"):
        TestFile.load(str(p))
    p.write_text('board = "b.kicad_pcb"\n[[test]]\nname = "a"\ntool = "dc"\n[[test]]\nname = "a"\ntool = "pdn"\n')
    with pytest.raises(ValueError, match="unique"):
        TestFile.load(str(p))


def test_check_evaluation():
    mets = {"T": Metric("°C", "t"), "V": Metric("V", "v", "pad")}
    vals = {"T": 80.0, "V": {"U1.1": 3.25, "U1.2": 3.1, "U2.1": 3.3}}
    ev = lambda c: R.evaluate(c, vals, mets)
    assert ev(Check("T", max=85)).status == R.PASS
    assert ev(Check("T", min=90)).status == R.FAIL
    r = ev(Check("V", min=3.2))                                  # every pad: U1.2 fails and is named
    assert r.status == R.FAIL and r.item == "U1.2" and r.value == pytest.approx(3.1)
    assert ev(Check("V", at="u1.1", min=3.2)).status == R.PASS    # case-insensitive item
    r = ev(Check("V", max=3.4))
    assert r.status == R.PASS and r.item == "U2.1"               # the closest to the limit is reported
    assert ev(Check("V", at="X1", min=1)).status == R.ERROR
    assert ev(Check("T", at="X1", min=1)).status == R.ERROR
    assert ev(Check("nope", max=1)).status == R.ERROR
    assert R.evaluate(Check("T", max=1), {"T": float("nan")}, mets).status == R.FAIL


def matching_setup():
    from test_matching import lmatch
    s = lmatch(tol=10)
    s.mc_n, s.seed = 2000, 1
    return dict(s.to_dict(), analyses={"corners": True, "mc": True, "worst": True, "sensitivity": False})


def test_run_report_exit_codes_and_comparison(tmp_path):
    """A matching test (fast, needs no copper) and a DC test on the demo board: exit codes, files, the
    report, results.json and the comparison with the previous run."""
    shutil.copy(os.path.join(DEMO, "demo.kicad_pcb"), tmp_path)
    from pcb_current_sim import scenario as SC
    from pcb_current_sim.batch.make import shared
    tf = TestFile.new(str(tmp_path / "demo.kicad_pcb"))
    _, tf.components = shared(SC.read(os.path.join(DEMO, "demo.current_sim.json")))    # Q1, LOAD, …
    tf.add(Test("match", "matching", matching_setup(), [Check("rl_nominal_dB", at="b", min=15),
                                                        Check("yield_pct", min=1)], ["csv", "png"]))
    dc = {"rows": [{"name": "supply", "src": ["J1.1"], "snk": ["J1.2"], "current": 12, "kind": "V"}],
          "settings": {"thermal": False, "magnetic": False, "error_estimate": False}}
    tf.add(Test("dc cold", "dc", dc, [Check("pad_V", at="J2.1", min=11.5), Check("el_util", max=1.0)]))
    tf.save()
    rr = R.run_tests(tf.path, all_tests=True)
    assert [t.status for t in rr.tests] == [R.PASS, R.FAIL] and rr.code == 1
    fail = rr.tests[1].checks[1]
    assert fail.item == "Q1" and fail.value > 1                  # the FET over its rating
    assert os.path.dirname(rr.folder) == str(tmp_path / R.RUNS_DIR)
    for f in ("report.html", "report.md", "results.json", "tests.toml", "log.txt"):
        assert os.path.exists(os.path.join(rr.folder, f))
    assert os.path.exists(os.path.join(rr.folder, "match", "response.csv"))
    assert not os.path.exists(os.path.join(rr.folder, "dc_cold"))      # no outputs asked: no folder
    js = json.load(open(os.path.join(rr.folder, "results.json")))
    assert js["code"] == 1 and js["tests"][0]["values"]["rl_nominal_dB"]["b"] > 15
    html = open(os.path.join(rr.folder, "report.html")).read()
    assert "Checks failed" in html and "match/response.png" in html
    # only the named test, report-only: exit 0, compared with the run before
    rr2 = R.run_tests(tf.path, names=["dc*"], report_only=True)
    assert [t.name for t in rr2.tests] == ["dc cold"] and rr2.code == 0 and rr2.previous == rr.folder
    assert rr2.tests[0].checks[0].previous == pytest.approx(rr.tests[1].checks[0].value)
    with pytest.raises(ValueError, match="no test named"):
        R.run_tests(tf.path, names=["missing"])


def test_a_test_that_cannot_run_is_an_error_not_a_crash(tmp_path):
    shutil.copy(os.path.join(DEMO, "demo.kicad_pcb"), tmp_path)
    tf = TestFile.new(str(tmp_path / "demo.kicad_pcb"))
    tf.add(Test("trace", "extract", {"ports": ["J1.1", "X9.1"]}))
    tf.add(Test("bad output", "matching", matching_setup(), outputs=["gif"]))
    tf.save()
    rr = R.run_tests(tf.path, all_tests=True)
    assert [t.status for t in rr.tests] == [R.ERROR, R.ERROR] and rr.code == 2
    assert "X9.1" in rr.tests[0].error and "gif" in rr.tests[1].error


def test_init_and_cli(tmp_path, capsys):
    """init builds tests from the board's saved setups (shared parts in [shared]); list, check, metrics."""
    from pcb_current_sim.batch import cli
    shutil.copy(os.path.join(DEMO, "demo.kicad_pcb"), tmp_path)
    shutil.copy(os.path.join(DEMO, "demo.current_sim.json"), tmp_path)
    board = str(tmp_path / "demo.kicad_pcb")
    assert cli.main(["init", board, "--tools", "dc"]) == 0
    tf = TestFile.load(str(tmp_path / "demo.simtests.toml"))
    assert [t.tool for t in tf.tests] == ["dc"] and tf.tests[0].components is None
    assert tf.components and all("name" in c for c in tf.components)
    assert tf.tests[0].setup["rows"][0]["name"] == "supply"
    assert cli.main(["init", board, "--tools", "dc"]) == 0                   # again: renamed, not replaced
    assert [t.name for t in TestFile.load(tf.path).tests] == ["DC & Thermal", "DC & Thermal (2)"]
    assert cli.main(["list", tf.path]) == 0
    assert cli.main(["check", tf.path]) == 0
    assert cli.main(["metrics", "pdn"]) == 0
    assert "z_ratio" in capsys.readouterr().out
    assert cli.main(["run", tf.path]) == 2                                   # nothing named
