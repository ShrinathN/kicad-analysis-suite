"""Command line for simulation tests.

  python3 -m pcb_current_sim init    BOARD.kicad_pcb [--tools dc,pdn] [-o FILE]   tests from the windows' saved setups
  python3 -m pcb_current_sim list    TESTS.toml                                    tests, checks and outputs in a file
  python3 -m pcb_current_sim check   TESTS.toml                                    validate without simulating
  python3 -m pcb_current_sim run     TESTS.toml --test NAME [--test NAME …] | --all  [--report-only] [--out DIR]
  python3 -m pcb_current_sim metrics [TOOL]                                        result values and outputs per tool

Exit codes of run: 0 every check passes (or --report-only), 1 a check failed, 2 a test could not run.
Results: simulations/<timestamp>/ in the board's (project) folder.
"""
import argparse
import os
import sys

COMMANDS = ("run", "init", "list", "check", "metrics")


def _err(msg, code=2):
    print(f"error: {msg}", file=sys.stderr)
    return code


def cmd_run(a):
    from .runner import run_tests
    if not a.test and not a.all:
        return _err("name the tests to run with --test NAME (repeatable, shell patterns allowed), or use --all")
    log = None if a.quiet else (lambda m: print(m, file=sys.stderr))
    try:
        rr = run_tests(a.file, a.test, a.all, a.report_only, a.out, log)
    except (ValueError, OSError) as e:
        return _err(e)
    from .report import MARK, WORD
    print(f"\n{os.path.basename(rr.board)}: {len(rr.tests)} test(s)")
    for t in rr.tests:
        ok = sum(1 for c in t.checks if c.status == "pass")
        print(f"  {MARK[t.status]} {t.name:40s} {WORD[t.status]:16s} {ok}/{len(t.checks)} checks  {t.seconds:6.1f} s")
        for c in t.checks:
            if c.status != "pass":
                where = c.at or c.item
                val = "-" if c.value is None else f"{c.value:.4g} {c.unit}".strip()
                print(f"      {MARK[c.status]} {c.what}{' at ' + where if where else ''}: {val}"
                      f"{' — ' + c.text if c.text else ''}")
        if t.error:
            print(f"      ⚠ {t.error.strip().splitlines()[-1]}")
    print(f"\nReport: {os.path.join(rr.folder, 'report.html')}")
    return rr.code


def cmd_init(a):
    from .make import from_scenario
    from .spec import TOOLS, TestFile
    tools = [t.strip() for t in a.tools.split(",")] if a.tools else list(TOOLS)
    bad = [t for t in tools if t not in TOOLS]
    if bad:
        return _err(f"unknown tool(s) {', '.join(bad)} (one of {', '.join(TOOLS)})")
    if not os.path.exists(a.board):
        return _err(f"board not found: {a.board}")
    try:
        tf = TestFile.open_or_new(a.board, a.output)
        tests = from_scenario(a.board, tools)
    except (ValueError, OSError) as e:
        return _err(e)
    if not tests:
        print("No saved setups found for these tools: set the simulations up in the windows first "
              "(each window saves its setup into the board's .current_sim.json).", file=sys.stderr)
        return 1
    for t in tests:
        got = tf.add(t, replace=a.replace)
        print(f"  + {got.name} ({got.tool})")
    tf.save()
    print(f"Wrote {tf.path}: {len(tf.tests)} test(s). Add [[test.check]] entries (what / at / min / max) and "
          f"outputs, then: python3 -m pcb_current_sim run {os.path.basename(tf.path)} --all")
    return 0


def _load(path):
    from .spec import TestFile
    return TestFile.load(path)


def cmd_list(a):
    try:
        tf = _load(a.file)
    except (ValueError, OSError) as e:
        return _err(e)
    print(f"{tf.path}: board {tf.board}")
    for t in tf.tests:
        print(f"  {'' if t.enabled else '(off) '}{t.name}  [{t.tool}]"
              + (f"  outputs: {', '.join(t.outputs)}" if t.outputs else ""))
        for c in t.checks:
            print(f"      check {c.label()}")
    return 0


def cmd_check(a):
    from .tools import get
    try:
        tf = _load(a.file)
    except (ValueError, OSError) as e:
        return _err(e)
    problems = []
    if not os.path.exists(tf.board_path):
        problems.append(f"board not found: {tf.board_path}")
    for t in tf.tests:
        tool = get(t.tool)
        for c in t.checks:
            if c.what not in tool.METRICS:
                problems.append(f"{t.name}: unknown result value {c.what!r}")
            elif c.at and not tool.METRICS[c.what].per:
                problems.append(f"{t.name}: {c.what} is a single number, 'at' does not apply")
            if c.min is not None and c.max is not None and c.min > c.max:
                problems.append(f"{t.name}: {c.what} min > max")
        for o in t.outputs:
            if o not in tool.OUTPUTS:
                problems.append(f"{t.name}: unknown output {o!r} (available: {', '.join(tool.OUTPUTS)})")
        if not t.setup:
            problems.append(f"{t.name}: empty [test.setup]")
    for p in problems:
        print(f"  ✖ {p}")
    print(f"{len(tf.tests)} test(s), {len(problems)} problem(s).")
    return 1 if problems else 0


def cmd_metrics(a):
    from .spec import TOOLS
    from .tools import get
    tools = [a.tool] if a.tool else list(TOOLS)
    for tool in tools:
        try:
            m = get(tool)
        except ValueError as e:
            return _err(e)
        print(f"{tool}: {m.TITLE}")
        print("  result values (checks: what = NAME, at = ITEM for values per item, min / max):")
        for name, met in m.METRICS.items():
            per = f" [per {met.per}]" if met.per else ""
            unit = f" [{met.unit}]" if met.unit else ""
            print(f"    {name:20s}{unit}{per}  {met.text}")
        print("  outputs (outputs = [...]):")
        for name, text in m.OUTPUTS.items():
            print(f"    {name:20s}  {text}")
        print()
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="python3 -m pcb_current_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run tests and write simulations/<timestamp>/")
    p.add_argument("file", help="the tests file (<board>.simtests.toml)")
    p.add_argument("--test", "-t", action="append", default=[], metavar="NAME",
                   help="a test to run (repeatable; shell patterns like 'PDN*' allowed)")
    p.add_argument("--all", action="store_true", help="run every enabled test in the file")
    p.add_argument("--report-only", action="store_true", help="always exit 0 (failed checks are still reported)")
    p.add_argument("--out", help="run folder (default <project>/simulations/<timestamp>)")
    p.add_argument("--quiet", "-q", action="store_true", help="no progress lines")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("init", help="tests from the setups the windows saved for a board")
    p.add_argument("board")
    p.add_argument("--tools", help=f"comma-separated subset (default all)")
    p.add_argument("-o", "--output", help="tests file (default <board>.simtests.toml; added to if it exists)")
    p.add_argument("--replace", action="store_true", help="replace tests with the same name (else renamed)")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("list", help="the tests in a file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("check", help="validate a tests file without simulating")
    p.add_argument("file")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("metrics", help="result values and outputs of each tool")
    p.add_argument("tool", nargs="?")
    p.set_defaults(fn=cmd_metrics)
    a = ap.parse_args(argv)
    return a.fn(a)
