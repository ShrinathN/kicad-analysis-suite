"""Per-tool adapters for the batch runner.

Each module defines
    TITLE                       the tool's name
    METRICS = {name: Metric}    result values a check can use
    OUTPUTS = {name: text}      files a test can ask for (none are written unless listed)
    from_scenario(scen, board, comps) -> [(test name, setup dict)]   tests from the windows' saved setups
    run(ctx) -> Outcome
"""
import importlib
from dataclasses import dataclass, field

from ..spec import TOOLS


@dataclass
class Metric:
    unit: str
    text: str
    per: str = ""                # "" = one number; else what the items are ("load", "pad", "row", ...)


@dataclass
class Outcome:
    values: dict = field(default_factory=dict)       # name -> float | {item: float}
    files: list = field(default_factory=list)        # written, relative to the run folder
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)        # key results in words, for the report
    images: list = field(default_factory=list)       # pictures for the report (relative paths)


@dataclass
class Context:
    board: object
    board_path: str
    test: object                 # spec.Test
    stackup: object              # suite.stackup.Stackup with the test's overrides
    components: object           # components.Components (the board's parts + the test's)
    folder: str                  # this test's output folder (created when a file is written)
    rel: str                     # that folder relative to the run folder
    log: object = print

    def path(self, name):
        import os
        os.makedirs(self.folder, exist_ok=True)
        return os.path.join(self.folder, name)

    def wrote(self, out, name):
        import os
        out.files.append(os.path.join(self.rel, name))

    def want(self, key):
        return key in self.test.outputs


def get(tool):
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r} (one of {', '.join(TOOLS)})")
    return importlib.import_module(f"{__name__}.{tool}")


def scalar_or_items(v):
    """A metric value for JSON: float / None, or {item: float}."""
    if isinstance(v, dict):
        return {str(k): _f(x) for k, x in v.items()}
    return _f(v)


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None
