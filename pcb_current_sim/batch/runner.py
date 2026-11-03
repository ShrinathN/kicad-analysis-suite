"""Run tests from a tests file and write simulations/<timestamp>/ with the report.

    run_tests(path, names=["3V3 rail impedance"])        # or all_tests=True
    -> RunResult(folder, tests=[TestResult], code)        # code: 0 pass, 1 a check failed, 2 a test failed to run

Per run folder: report.html, report.md, results.json (every value and check, for scripts), tests.toml (the
file as it ran), log.txt, and a sub-folder per test for the outputs it lists.
"""
import datetime
import fnmatch
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field

from . import report
from .spec import TestFile
from .tools import Context, get, scalar_or_items

RUNS_DIR = "simulations"
PASS, FAIL, ERROR, DONE = "pass", "fail", "error", "done"     # DONE: ran, no checks


@dataclass
class CheckResult:
    what: str
    at: str
    min: float
    max: float
    status: str                  # pass | fail | error
    value: float = None          # the deciding value (worst item when there are several)
    item: str = ""               # which item it was
    text: str = ""
    unit: str = ""
    previous: float = None

    def to_dict(self):
        return dict(what=self.what, at=self.at, min=self.min, max=self.max, status=self.status, value=self.value,
                    item=self.item, text=self.text, unit=self.unit, previous=self.previous)


@dataclass
class TestResult:
    name: str
    tool: str
    status: str
    seconds: float = 0.0
    checks: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    images: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    error: str = ""
    note: str = ""
    changes: list = field(default_factory=list)      # (value name, previous, now) for scalars that moved

    def to_dict(self):
        return dict(name=self.name, tool=self.tool, status=self.status, seconds=self.seconds, note=self.note,
                    checks=[c.to_dict() for c in self.checks], values=self.values, files=self.files,
                    images=self.images, warnings=self.warnings, lines=self.lines, error=self.error,
                    changes=[list(c) for c in self.changes])


@dataclass
class RunResult:
    folder: str
    board: str
    tests_file: str
    started: str
    tests: list = field(default_factory=list)
    report_only: bool = False
    previous: str = ""

    @property
    def code(self):
        if self.report_only:
            return 0
        if any(t.status == ERROR for t in self.tests):
            return 2
        if any(t.status == FAIL for t in self.tests):
            return 1
        return 0

    @property
    def counts(self):
        out = {}
        for t in self.tests:
            out[t.status] = out.get(t.status, 0) + 1
        return out

    def to_dict(self):
        return dict(board=self.board, tests_file=self.tests_file, started=self.started, folder=self.folder,
                    previous=self.previous, code=self.code, report_only=self.report_only,
                    tests=[t.to_dict() for t in self.tests])


# ---- checks ------------------------------------------------------------------------------
def _ok(x, c):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    if c.min is not None and x < c.min:
        return False
    if c.max is not None and x > c.max:
        return False
    return True


def _miss(x, c):
    """How far x is outside the limits (0 inside), for picking the worst item."""
    if x is None or math.isnan(x):
        return math.inf
    m = 0.0
    if c.min is not None and x < c.min:
        m = max(m, (c.min - x) / max(abs(c.min), 1e-30))
    if c.max is not None and x > c.max:
        m = max(m, (x - c.max) / max(abs(c.max), 1e-30))
    return m


def evaluate(c, values, metrics):
    met = metrics.get(c.what)
    unit = met.unit if met else ""
    r = CheckResult(c.what, c.at, c.min, c.max, ERROR, unit=unit)
    if met is None:
        r.text = f"unknown result value {c.what!r} (see: python3 -m pcb_current_sim metrics)"
        return r
    v = values.get(c.what)
    if v is None or (isinstance(v, dict) and not v):
        r.text = f"{c.what} was not computed in this run (switched off in the setup, or nothing to measure)"
        return r
    if isinstance(v, dict):
        if c.at:
            key = c.at if c.at in v else next((k for k in v if k.lower() == c.at.lower()), None)
            if key is None:
                some = ", ".join(list(v)[:12]) + (" …" if len(v) > 12 else "")
                r.text = f"{c.what} has no {met.per or 'item'} {c.at!r} (there are: {some})"
                return r
            items = {key: v[key]}
        else:
            items = v
    else:
        if c.at:
            r.text = f"{c.what} is a single number: remove 'at'"
            return r
        items = {"": v}
    bad = {k: x for k, x in items.items() if not _ok(x, c)}
    if bad:
        k = max(bad, key=lambda k: _miss(bad[k], c))
        r.status, r.item, r.value = FAIL, k, bad[k]
    else:
        if c.max is not None:
            k = max(items, key=lambda k: items[k])
        elif c.min is not None:
            k = min(items, key=lambda k: items[k])
        else:
            k = next(iter(items))
        r.status, r.item, r.value = PASS, k, items[k]
    if c.min is None and c.max is None:
        r.status = PASS
        r.text = "recorded (no limit)"
    return r


# ---- running -------------------------------------------------------------------------
def select(tf, names=None, all_tests=False):
    """The tests to run: those named (exact, or shell patterns like 'PDN*'), else every enabled one with
    all_tests. Raises ValueError for a name that matches nothing."""
    if all_tests:
        return [t for t in tf.tests if t.enabled]
    out = []
    for n in names or []:
        hit = [t for t in tf.tests if t.name == n] or [t for t in tf.tests if fnmatch.fnmatchcase(t.name, n)] \
            or [t for t in tf.tests if t.name.lower() == n.lower()]
        if not hit:
            raise ValueError(f"no test named {n!r} in {os.path.basename(tf.path)} "
                             f"(tests: {', '.join(t.name for t in tf.tests) or 'none'})")
        out += [t for t in hit if t not in out]
    return out


def _safe(name):
    import re
    s = re.sub(r"[^A-Za-z0-9.-]+", "_", name).strip("_.")
    return s or "test"


def run_folder(project, stamp=None):
    root = os.path.join(project, RUNS_DIR)
    stamp = stamp or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    p, k = os.path.join(root, stamp), 2
    while os.path.exists(p):
        p, k = os.path.join(root, f"{stamp}_{k}"), k + 1
    os.makedirs(p)
    return p


def previous_run(root, exclude=""):
    """The latest earlier run folder with a results.json."""
    if not os.path.isdir(root):
        return ""
    runs = sorted((d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "results.json"))
                   and os.path.join(root, d) != exclude), reverse=True)
    return os.path.join(root, runs[0]) if runs else ""


def run_tests(path, names=None, all_tests=False, report_only=False, out=None, log=None, board=None):
    """Run the selected tests of the tests file at path. out: the run folder (default
    <project>/simulations/<timestamp>). log(text) gets progress lines (also written to log.txt).
    board: an already parsed board (else the file's board is loaded)."""
    from .. import board as B
    from .. import components as CO
    from ..suite import stackup as ST
    tf = TestFile.load(path)
    chosen = select(tf, names, all_tests)
    if not chosen:
        raise ValueError("no test selected: name them with --test NAME, or use --all")
    bpath = tf.board_path
    if board is None and not os.path.exists(bpath):
        raise ValueError(f"board not found: {bpath}")
    project = os.path.dirname(bpath)
    folder = out or run_folder(project)
    os.makedirs(folder, exist_ok=True)
    lines = []

    def say(msg):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        lines.append(f"{stamp} {msg}")
        if log:
            log(msg)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    rr = RunResult(folder, bpath, os.path.abspath(path), started, report_only=report_only)
    shutil.copyfile(path, os.path.join(folder, "tests.toml"))
    try:
        if board is None:
            say(f"Loading {os.path.basename(bpath)}…")
            board = B.load(bpath)
        for t in chosen:
            rr.tests.append(_run_one(t, tf, board, bpath, folder, say, ST, CO))
    finally:
        prev = previous_run(os.path.join(project, RUNS_DIR), folder) if out is None else ""
        rr.previous = prev
        if prev:
            _compare(rr, prev)
        with open(os.path.join(folder, "results.json"), "w") as fh:
            json.dump(rr.to_dict(), fh, indent=1, default=str)
        report.write(rr, folder)
        with open(os.path.join(folder, "log.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return rr


def _run_one(t, tf, board, bpath, folder, say, ST, CO):
    tool = get(t.tool)
    tr = TestResult(t.name, t.tool, ERROR, note=t.note)
    t0 = time.time()
    say(f"── {t.name} ({tool.TITLE})")
    unknown = [o for o in t.outputs if o not in tool.OUTPUTS]
    try:
        if unknown:
            raise ValueError(f"unknown output(s) {', '.join(unknown)} for {t.tool} "
                             f"(available: {', '.join(tool.OUTPUTS)})")
        stackup = ST.from_board(board, tf.stackup_of(t) or None)
        comps = CO.Components.build(board, [CO.from_dict(x) for x in tf.components_of(t)])
        sub = _safe(t.name)
        ctx = Context(board, bpath, t, stackup, comps, os.path.join(folder, sub), sub, lambda m: say(f"   {m}"))
        res = tool.run(ctx)
        tr.values = {k: scalar_or_items(v) for k, v in res.values.items() if v is not None}
        tr.files, tr.images, tr.warnings, tr.lines = res.files, res.images, res.warnings, res.lines
        tr.checks = [evaluate(c, tr.values, tool.METRICS) for c in t.checks]
        st = [c.status for c in tr.checks]
        tr.status = ERROR if ERROR in st else FAIL if FAIL in st else PASS if st else DONE
    except Exception as e:      # noqa: BLE001
        known = isinstance(e, (ValueError, KeyError, RuntimeError)) or type(e).__name__ == "SimError"
        tr.error = str(e) if known and str(e) else traceback.format_exc()
        tr.status = ERROR
        say(f"   failed: {tr.error.splitlines()[-1] if tr.error else type(e).__name__}")
        if not known:
            say(traceback.format_exc())
    tr.seconds = time.time() - t0
    say(f"   {tr.status.upper()} in {tr.seconds:.1f} s")
    for c in tr.checks:
        say(f"   {c.status:5s} {c.what}{' at ' + c.at if c.at else ''}: {_fmt(c.value)} {c.unit}"
            f"{' (' + c.item + ')' if c.item and not c.at else ''}{' - ' + c.text if c.text else ''}")
    return tr


def _fmt(x):
    if x is None:
        return "-"
    return f"{x:.4g}"


def _compare(rr, prev):
    try:
        with open(os.path.join(prev, "results.json")) as fh:
            old = {t["name"]: t for t in json.load(fh).get("tests", [])}
    except (OSError, ValueError):
        return
    for t in rr.tests:
        o = old.get(t.name)
        if not o:
            continue
        ov = o.get("values") or {}
        for c in t.checks:
            p = ov.get(c.what)
            if isinstance(p, dict):
                p = p.get(c.item) if c.item else None
            c.previous = p if isinstance(p, (int, float)) else None
        for k, now in t.values.items():
            was = ov.get(k)
            if isinstance(now, (int, float)) and isinstance(was, (int, float)) and math.isfinite(now) \
                    and math.isfinite(was) and abs(now - was) > 0.01 * max(abs(was), abs(now), 1e-12):
                t.changes.append((k, was, now))
