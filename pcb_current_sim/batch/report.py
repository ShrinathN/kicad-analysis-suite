"""report.html and report.md of a batch run (self-contained; pictures are the run folder's PNGs)."""
import html
import os

MARK = {"pass": "✔", "fail": "✖", "error": "⚠", "done": "•"}
WORD = {"pass": "pass", "fail": "FAIL", "error": "ERROR", "done": "ran (no checks)"}


def _f(x, unit=""):
    if x is None:
        return "–"
    try:
        s = f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)
    return f"{s} {unit}".strip()


def _limits(c):
    out = []
    if c.min is not None:
        out.append(f"≥ {c.min:g}")
    if c.max is not None:
        out.append(f"≤ {c.max:g}")
    return " and ".join(out) or "–"


def _headline(rr):
    n = rr.counts
    parts = [f"{n[k]} {WORD[k] if k != 'done' else 'without checks'}" for k in ("pass", "fail", "error", "done")
             if n.get(k)]
    verdict = {0: "All checks pass", 1: "Checks failed", 2: "Some tests could not run"}[rr.code if not rr.report_only
                                                                                           else _code(rr)]
    return verdict, ", ".join(parts)


def _code(rr):
    st = [t.status for t in rr.tests]
    return 2 if "error" in st else 1 if "fail" in st else 0


def write(rr, folder):
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(rr))
    with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(page(rr))


# ---- markdown ---------------------------------------------------------------------------
def markdown(rr):
    verdict, counts = _headline(rr)
    L = [f"# Simulation report: {os.path.basename(rr.board)}", "",
         f"{rr.started} · tests file `{os.path.basename(rr.tests_file)}`", "",
         f"**{verdict}** ({counts})." + (" Reports only: exit code 0." if rr.report_only else ""), ""]
    if rr.previous:
        L += [f"Compared with the previous run `{os.path.basename(rr.previous)}`.", ""]
    L += ["| Test | Tool | Result | Checks | Time |", "|---|---|---|---|---|"]
    for t in rr.tests:
        ok = sum(1 for c in t.checks if c.status == "pass")
        L.append(f"| {t.name} | {t.tool} | {MARK[t.status]} {WORD[t.status]} | {ok}/{len(t.checks)} | "
                 f"{t.seconds:.1f} s |")
    for t in rr.tests:
        L += ["", f"## {MARK[t.status]} {t.name}", ""]
        if t.note:
            L += [t.note, ""]
        if t.error:
            L += ["Could not run:", "", "```", t.error.strip(), "```", ""]
        if t.checks:
            L += ["| | Check | Limit | Value | Previous | |", "|---|---|---|---|---|---|"]
            for c in t.checks:
                where = c.at or c.item
                L.append(f"| {MARK[c.status]} | {c.what}{' at ' + where if where else ''} | {_limits(c)} | "
                         f"{_f(c.value, c.unit)} | {_f(c.previous, c.unit)} | {c.text} |")
            L.append("")
        if t.lines:
            L += ["```"] + t.lines + ["```", ""]
        if t.changes:
            L += ["Changed since the previous run:", ""]
            L += [f"- {k}: {_f(a)} → {_f(b)}" for k, a, b in t.changes[:30]] + [""]
        if t.warnings:
            L += ["Warnings:", ""] + [f"- {w}" for w in t.warnings] + [""]
        if t.images:
            L += [f"![{os.path.basename(p)}]({p})" for p in t.images] + [""]
        if t.files:
            L += ["Files: " + ", ".join(f"[{os.path.basename(p)}]({p})" for p in t.files), ""]
    return "\n".join(L).rstrip() + "\n"


# ---- html -------------------------------------------------------------------------------
CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f21;--mute:#6b6f75;--line:#e1e2e4;--card:#ffffff;--pass:#1e7d3c;--fail:#b3261e;
--err:#9a5b00;--code:#f2f3f4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#17181a;--fg:#e6e7e8;--mute:#9a9ea4;
--line:#2c2e31;--card:#1f2124;--pass:#5cc47e;--fail:#ff7a6e;--err:#e3a74c;--code:#26282b}}
:root[data-theme=dark]{--bg:#17181a;--fg:#e6e7e8;--mute:#9a9ea4;--line:#2c2e31;--card:#1f2124;--pass:#5cc47e;
--fail:#ff7a6e;--err:#e3a74c;--code:#26282b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.5rem;margin:0 0 4px}
h2{font-size:1.15rem;margin:32px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
.mute{color:var(--mute)}.verdict{font-size:1.1rem;font-weight:600;margin:16px 0}
.pass{color:var(--pass)}.fail{color:var(--fail)}.error{color:var(--err)}.done{color:var(--mute)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:8px 0 12px;font-size:.92rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:500}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
pre{background:var(--code);padding:10px 12px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem;margin:8px 0}
img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff;margin:6px 0}
ul{margin:4px 0 12px;padding-left:20px}a{color:inherit}
"""


def _cls(s):
    return {"pass": "pass", "fail": "fail", "error": "error"}.get(s, "done")


def page(rr):
    e = html.escape
    verdict, counts = _headline(rr)
    code = _code(rr)
    H = ["<!doctype html><html lang=en><head><meta charset=utf-8>",
         "<meta name=viewport content='width=device-width,initial-scale=1'>",
         f"<title>Simulation report</title><style>{CSS}</style></head><body><main>",
         f"<h1>Simulation report: {e(os.path.basename(rr.board))}</h1>",
         f"<div class=mute>{e(rr.started)} · {e(os.path.basename(rr.tests_file))}"
         + (f" · compared with {e(os.path.basename(rr.previous))}" if rr.previous else "") + "</div>",
         f"<div class='verdict {_cls(['pass', 'fail', 'error'][code])}'>{e(verdict)} "
         f"<span class=mute>({e(counts)}){' · reports only' if rr.report_only else ''}</span></div>",
         "<div class=scroll><table><tr><th>Test</th><th>Tool</th><th>Result</th><th>Checks</th><th>Time</th></tr>"]
    for k, t in enumerate(rr.tests):
        ok = sum(1 for c in t.checks if c.status == "pass")
        H.append(f"<tr><td><a href='#t{k}'>{e(t.name)}</a></td><td>{e(t.tool)}</td>"
                 f"<td class={_cls(t.status)}>{MARK[t.status]} {WORD[t.status]}</td>"
                 f"<td class=num>{ok} / {len(t.checks)}</td><td class=num>{t.seconds:.1f} s</td></tr>")
    H.append("</table></div>")
    for k, t in enumerate(rr.tests):
        H.append(f"<h2 id='t{k}' class={_cls(t.status)}>{MARK[t.status]} <span style='color:var(--fg)'>"
                 f"{e(t.name)}</span></h2>")
        if t.note:
            H.append(f"<p>{e(t.note)}</p>")
        if t.error:
            H.append(f"<p class=error>Could not run:</p><pre>{e(t.error.strip())}</pre>")
        if t.checks:
            H.append("<div class=scroll><table><tr><th></th><th>Check</th><th>Limit</th><th>Value</th>"
                     "<th>Previous</th><th></th></tr>")
            for c in t.checks:
                where = c.at or c.item
                H.append(f"<tr><td class={_cls(c.status)}>{MARK[c.status]}</td>"
                         f"<td>{e(c.what)}{(' at <b>' + e(where) + '</b>') if where else ''}</td>"
                         f"<td class=num>{e(_limits(c))}</td><td class=num>{e(_f(c.value, c.unit))}</td>"
                         f"<td class='num mute'>{e(_f(c.previous, c.unit))}</td><td class=mute>{e(c.text)}</td></tr>")
            H.append("</table></div>")
        if t.lines:
            H.append("<pre>" + e("\n".join(t.lines)) + "</pre>")
        if t.changes:
            H.append("<p class=mute>Changed since the previous run:</p><ul>"
                     + "".join(f"<li>{e(k)}: {e(_f(a))} → {e(_f(b))}</li>" for k, a, b in t.changes[:30]) + "</ul>")
        if t.warnings:
            H.append("<p class=mute>Warnings:</p><ul>" + "".join(f"<li>{e(w)}</li>" for w in t.warnings) + "</ul>")
        for p in t.images:
            H.append(f"<img src='{e(p)}' alt='{e(os.path.basename(p))}' loading=lazy>")
        if t.files:
            H.append("<p class=mute>Files: " + ", ".join(f"<a href='{e(p)}'>{e(os.path.basename(p))}</a>"
                                                       for p in t.files) + "</p>")
    H.append("</main></body></html>")
    return "\n".join(H)
