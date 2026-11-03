"""The tests file (<board>.simtests.toml): which simulations to run on a board, with all their parameters,
the pass / fail checks and the files to keep.

    board = "Magnus_Gateway.kicad_pcb"          # relative to this file

    [[test]]
    name = "3V3 rail impedance"
    tool = "pdn"                                 # dc | pdn | loop | extract | matching
    outputs = ["csv", "touchstone"]              # nothing besides the report unless listed

      [[test.check]]
      what = "z_ratio"                           # a result value (python3 -m pcb_current_sim metrics pdn)
      at = "U1"                                  # one item of a per-item value (else: every item)
      max = 1.0                                  # and / or min

      [test.setup]                               # the tool's setup, as its window saves it
      ...
    [shared.stackup]                             # stackup overrides (copper, dielectrics) for every test
    [[shared.components]]                        # the board's parts between pads, for every test
    (a test may have its own [test.stackup] / [[test.components]] instead)

A test is self-contained: it does not read what the windows currently have saved, so a test keeps
meaning the same thing until its file is edited.
"""
import copy
import os
from dataclasses import dataclass, field

from . import tomlio

SUFFIX = ".simtests.toml"
TOOLS = ("dc", "pdn", "loop", "extract", "matching")
HEADER = (
    "PCB Analysis Suite: simulation tests for one board.",
    "Run:   python3 -m pcb_current_sim run THIS_FILE --test NAME   (or --all)",
    "List:  python3 -m pcb_current_sim list THIS_FILE;  result values: python3 -m pcb_current_sim metrics TOOL",
    "Each [[test]] holds every parameter of its simulation ([test.setup], as the tool's window saves it),",
    "its pass / fail checks ([[test.check]]: what, optional at, min and / or max) and the files to keep (outputs).",
    "[shared] at the end: the stackup and parts between pads every test uses (a test may have its own).",
    "Results go to simulations/<timestamp>/ in the project folder.",
)


def path_for(board_path):
    return os.path.splitext(board_path)[0] + SUFFIX


@dataclass
class Check:
    what: str
    at: str = ""                 # item of a per-item value ("" = every item must pass)
    min: float = None
    max: float = None
    note: str = ""

    def to_dict(self):
        d = {"what": self.what}
        if self.at:
            d["at"] = self.at
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or not d.get("what"):
            raise ValueError(f"a check needs 'what': {d!r}")
        unknown = set(d) - {"what", "at", "min", "max", "note"}
        if unknown:
            raise ValueError(f"check {d['what']}: unknown key(s) {', '.join(sorted(unknown))}")
        num = lambda v: None if v is None else float(v)
        return cls(str(d["what"]), str(d.get("at") or ""), num(d.get("min")), num(d.get("max")),
                   str(d.get("note") or ""))

    def label(self):
        lim = []
        if self.min is not None:
            lim.append(f">= {self.min:g}")
        if self.max is not None:
            lim.append(f"<= {self.max:g}")
        return f"{self.what}{f' at {self.at}' if self.at else ''} {' and '.join(lim) or '(recorded)'}"


@dataclass
class Test:
    __test__ = False             # not a pytest class
    name: str
    tool: str
    setup: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    stackup: dict = None         # None: the file's [shared] stackup
    components: list = None      # None: the file's [shared] components
    note: str = ""
    enabled: bool = True

    def to_dict(self):
        d = {"name": self.name, "tool": self.tool}
        if self.note:
            d["note"] = self.note
        if not self.enabled:
            d["enabled"] = False
        d["outputs"] = list(self.outputs)
        if self.checks:
            d["check"] = [c.to_dict() for c in self.checks]
        d["setup"] = self.setup
        if self.stackup is not None:
            d["stackup"] = self.stackup
        if self.components is not None:
            d["components"] = self.components
        return d

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise ValueError("each [[test]] must be a table")
        name = str(d.get("name") or "").strip()
        if not name:
            raise ValueError("a test needs a name")
        tool = str(d.get("tool") or "")
        if tool not in TOOLS:
            raise ValueError(f"test {name!r}: tool must be one of {', '.join(TOOLS)} (got {tool!r})")
        unknown = set(d) - {"name", "tool", "note", "enabled", "outputs", "check", "setup", "stackup", "components"}
        if unknown:
            raise ValueError(f"test {name!r}: unknown key(s) {', '.join(sorted(unknown))}")
        try:
            checks = [Check.from_dict(c) for c in d.get("check") or []]
        except ValueError as e:
            raise ValueError(f"test {name!r}: {e}") from None
        outs = d.get("outputs") or []
        if isinstance(outs, str):
            outs = [outs]
        return cls(name, tool, dict(d.get("setup") or {}), checks, [str(o) for o in outs],
                   dict(d["stackup"]) if d.get("stackup") is not None else None,
                   list(d["components"]) if d.get("components") is not None else None,
                   str(d.get("note") or ""), bool(d.get("enabled", True)))

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class TestFile:
    __test__ = False
    board: str                   # as written (relative to the file's folder, or absolute)
    tests: list = field(default_factory=list)
    path: str = ""
    stackup: dict = field(default_factory=dict)      # [shared] stackup overrides
    components: list = field(default_factory=list)   # [shared] parts between pads

    def stackup_of(self, t):
        return self.stackup if t.stackup is None else t.stackup

    def components_of(self, t):
        return self.components if t.components is None else t.components

    @property
    def folder(self):
        return os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()

    @property
    def board_path(self):
        b = os.path.expanduser(self.board)
        return b if os.path.isabs(b) else os.path.normpath(os.path.join(self.folder, b))

    def get(self, name):
        return next((t for t in self.tests if t.name == name), None)

    def unique_name(self, name):
        base, k, n = name, 2, name
        while self.get(n) is not None:
            n, k = f"{base} ({k})", k + 1
        return n

    def add(self, test, replace=False):
        """Add a test (renamed if the name is taken, or replacing it with replace=True). Its stackup and
        components move to [shared] when the file has none yet, or are dropped when equal to [shared]."""
        if not self.tests and not self.stackup and not self.components:
            self.stackup, self.components = test.stackup or {}, test.components or []
        if test.stackup is not None and test.stackup == self.stackup:
            test.stackup = None
        if test.components is not None and test.components == self.components:
            test.components = None
        old = self.get(test.name)
        if old is not None and replace:
            self.tests[self.tests.index(old)] = test
            return test
        test.name = self.unique_name(test.name)
        self.tests.append(test)
        return test

    def to_dict(self):
        d = {"board": self.board, "test": [t.to_dict() for t in self.tests]}
        if self.stackup or self.components:
            d["shared"] = {}
            if self.stackup:
                d["shared"]["stackup"] = self.stackup
            if self.components:
                d["shared"]["components"] = self.components
        return d

    def save(self, path=None):
        self.path = path or self.path
        tomlio.dump(self.path, self.to_dict(), HEADER)

    @classmethod
    def from_dict(cls, d, path=""):
        if not d.get("board"):
            raise ValueError("the file needs board = \"<file>.kicad_pcb\"")
        unknown = set(d) - {"board", "test", "shared"}
        if unknown:
            raise ValueError(f"unknown top-level key(s) {', '.join(sorted(unknown))}")
        tests = [Test.from_dict(t) for t in d.get("test") or []]
        names = [t.name for t in tests]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"test names must be unique: {', '.join(dup)}")
        sh = d.get("shared") or {}
        return cls(str(d["board"]), tests, path, dict(sh.get("stackup") or {}), list(sh.get("components") or []))

    @classmethod
    def load(cls, path):
        return cls.from_dict(tomlio.load(path), path)

    @classmethod
    def new(cls, board_path, path=None):
        path = path or path_for(board_path)
        rel = os.path.relpath(os.path.abspath(board_path), os.path.dirname(os.path.abspath(path)))
        return cls(rel, [], path)

    @classmethod
    def open_or_new(cls, board_path, path=None):
        path = path or path_for(board_path)
        return cls.load(path) if os.path.exists(path) else cls.new(board_path, path)
