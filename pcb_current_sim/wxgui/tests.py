"""Simulation tests in the GUI: "Save as test…" from a tool window, and the Tests window that edits the tests
file (<board>.simtests.toml: checks, outputs, on / off), runs tests and opens their reports.

The command line runs the same file:  python3 -m pcb_current_sim run <board>.simtests.toml --test NAME
"""
import os
import threading
import traceback

import wx

from ..batch import make as MK
from ..batch import runner as RN
from ..batch.spec import Check, TestFile, path_for
from ..batch.tools import get as tool_mod
from .layout import AutoList, refit, scrolled

TOOL_TITLES = {"dc": "DC & Thermal", "pdn": "PDN", "loop": "Switching Loop", "extract": "Trace R/L/C",
               "matching": "Matching"}
STATUS = {"pass": "✔ pass", "fail": "✖ FAIL", "error": "⚠ error", "done": "• ran"}


def _num_or_none(text):
    t = text.strip()
    if not t:
        return None
    from ..suite.library import parse_si
    return float(parse_si(t))


# ---------------------------------------------------------------- save as test
class SaveTestDialog(wx.Dialog):
    def __init__(self, parent, tf, name):
        super().__init__(parent, title="Save as a simulation test")
        self.tf = tf
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(wx.StaticText(self, label="The window's current setup, with every parameter, is stored as a test in\n"
                                        f"{tf.path}\nRun it from the command line or the Tests window; add its "
                                        "pass / fail checks and outputs there."), 0, wx.ALL, 10)
        g = wx.FlexGridSizer(0, 2, 6, 8)
        g.AddGrowableCol(1)
        self.name = wx.ComboBox(self, value=name, choices=[t.name for t in tf.tests], size=(360, -1))
        self.name.SetToolTip("The test's name (used with --test NAME on the command line). Pick an existing test's "
                             "name to replace its setup (its checks and outputs are kept)")
        g.AddMany([(wx.StaticText(self, label="Test name"), 0, wx.ALIGN_CENTER_VERTICAL), (self.name, 1, wx.EXPAND)])
        s.Add(g, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        self.open = wx.CheckBox(self, label="Open the Tests window afterwards")
        self.open.SetToolTip("Edit the test's checks and outputs, or run it")
        self.open.SetValue(True)
        s.Add(self.open, 0, wx.ALL, 10)
        s.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(s)


def save_as_test(frame, session, tool, setup, name):
    """Store a window's setup as a test in the board's tests file (asks for the name)."""
    if not session.board_path:
        return wx.MessageBox("Load a board first.", "Save as test", parent=frame)
    try:
        tf = TestFile.open_or_new(session.board_path)
    except (ValueError, OSError) as e:
        return wx.MessageBox(f"Could not read the tests file:\n{e}", "Save as test", parent=frame)
    d = SaveTestDialog(frame, tf, name)
    if d.ShowModal() != wx.ID_OK:
        return d.Destroy()
    name, open_after = d.name.GetValue().strip(), d.open.GetValue()
    d.Destroy()
    if not name:
        return
    t = MK.from_session(session, tool, name, setup)
    old = tf.get(name)
    if old is not None:
        if old.tool != tool:
            return wx.MessageBox(f"'{name}' is a {TOOL_TITLES.get(old.tool, old.tool)} test: pick another name.",
                                 "Save as test", parent=frame)
        if wx.MessageBox(f"Replace the setup of the test '{name}'? Its checks and outputs are kept.", "Save as test",
                         wx.YES_NO, parent=frame) != wx.YES:
            return
        t.checks, t.outputs, t.note, t.enabled = old.checks, old.outputs, old.note, old.enabled
    tf.add(t, replace=True)
    try:
        tf.save()
    except OSError as e:
        return wx.MessageBox(f"Could not save {tf.path}:\n{e}", "Save as test", parent=frame)
    msg = f"Saved the test '{t.name}' in {os.path.basename(tf.path)}."
    st = getattr(frame, "_status", None)
    if st:
        st(msg)
    w = session.windows.get("tests")
    if w:
        w.reload(select=t.name)
    if open_after:
        from .suite import open_window
        w = open_window(session, "tests", frame)
        if w:
            w.reload(select=t.name)


def add_button(parent, frame, getter, label="Save as test…"):
    """A "Save as test…" button: getter() -> (tool, setup dict, default name) or None (problem shown)."""
    b = wx.Button(parent, label=label, style=wx.BU_EXACTFIT)
    b.SetToolTip("Store this window's setup, with every parameter, as a simulation test in the board's tests file "
                 "(<board>.simtests.toml): run it later from the command line or the Tests window, with pass / fail "
                 "checks and a report")

    def go(e):
        got = getter()
        if got:
            save_as_test(frame, frame.session, *got)
    b.Bind(wx.EVT_BUTTON, go)
    return b


# ---------------------------------------------------------------- check dialog
class CheckDialog(wx.Dialog):
    def __init__(self, parent, tool, check=None, items=None):
        super().__init__(parent, title="Check", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.mod, self.items = tool_mod(tool), items or {}
        c = check or Check("")
        s = wx.BoxSizer(wx.VERTICAL)
        g = wx.FlexGridSizer(0, 2, 6, 8)
        g.AddGrowableCol(1)
        self.names = list(self.mod.METRICS)
        self.what = wx.Choice(self, choices=[self._label(n) for n in self.names])
        self.what.SetToolTip("The result value to check")
        self.what.Bind(wx.EVT_CHOICE, lambda e: self._sync())
        self.help = wx.StaticText(self, label="", size=(420, -1))
        self.at = wx.ComboBox(self, value=c.at, size=(240, -1))
        self.at.SetToolTip("For values per item (per load, pad, part, …): the one to check. Empty: every item "
                           "must pass. The list holds the items of the last run")
        self.min = wx.TextCtrl(self, value="" if c.min is None else f"{c.min:g}", size=(120, -1))
        self.min.SetToolTip("Lowest value that passes (empty: no lower limit); SI prefixes allowed (3m3, 1k)")
        self.max = wx.TextCtrl(self, value="" if c.max is None else f"{c.max:g}", size=(120, -1))
        self.max.SetToolTip("Highest value that passes (empty: no upper limit). Neither: the value is only recorded")
        self.note = wx.TextCtrl(self, value=c.note, size=(360, -1))
        self.note.SetToolTip("Why this limit (shown in the tests file; optional)")
        self.unit = wx.StaticText(self, label="")
        rows = [("Value", self.what), ("", self.help), ("At", self.at), ("Min", self.min), ("Max", self.max),
                ("Unit", self.unit), ("Note", self.note)]
        for lab, w in rows:
            g.AddMany([(wx.StaticText(self, label=lab), 0, wx.ALIGN_CENTER_VERTICAL), (w, 1, wx.EXPAND)])
        s.Add(g, 1, wx.EXPAND | wx.ALL, 10)
        s.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(s)
        self.what.SetSelection(self.names.index(c.what) if c.what in self.names else 0)
        self._sync()
        self.at.SetValue(c.at)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        self.check = None

    def _label(self, n):
        m = self.mod.METRICS[n]
        return f"{n}" + (f"  [{m.unit}]" if m.unit else "") + (f"  per {m.per}" if m.per else "")

    def _sync(self):
        n = self.names[self.what.GetSelection()]
        m = self.mod.METRICS[n]
        self.help.SetLabel(m.text)
        self.help.Wrap(420)
        self.unit.SetLabel(m.unit or "–")
        self.at.Enable(bool(m.per))
        keep = self.at.GetValue()
        self.at.Set(sorted(self.items.get(n, {}))) if m.per else self.at.Set([])
        self.at.SetValue(keep if m.per else "")
        self.Layout()
        self.Fit()

    def _ok(self, e):
        try:
            lo, hi = _num_or_none(self.min.GetValue()), _num_or_none(self.max.GetValue())
        except ValueError:
            return wx.MessageBox("Min and max must be numbers (or empty).", "Check", parent=self)
        if lo is not None and hi is not None and lo > hi:
            return wx.MessageBox("Min is above max.", "Check", parent=self)
        n = self.names[self.what.GetSelection()]
        self.check = Check(n, self.at.GetValue().strip() if self.mod.METRICS[n].per else "", lo, hi,
                           self.note.GetValue().strip())
        self.EndModal(wx.ID_OK)


# ---------------------------------------------------------------- tests window
class TestsFrame(wx.Frame):
    """The board's tests file: tests, their checks and outputs; run them and open the reports."""

    def __init__(self, session):
        super().__init__(None, title="Simulation tests - PCB analysis suite", size=(1350, 820))
        self.session = session
        self.tf, self.last, self._running = None, None, False
        self.CreateStatusBar()
        self._build()
        from .suite import attach_suite_menu
        attach_suite_menu(self, session, "tests")
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.reload()

    def _status(self, text):
        self.SetStatusText(text)

    def _on_close(self, e):
        try:
            self.session.unlisten(self._on_session)
        except (AttributeError, ValueError):
            pass
        e.Skip()

    def _on_session(self, what):
        if what == "board":
            self.reload()

    # ---- layout ------------------------------------------------------------
    def _build(self):
        sp = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = wx.Panel(sp)
        ls = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.path = wx.StaticText(left, label="")
        b = wx.Button(left, label="Open file…", style=wx.BU_EXACTFIT)
        b.SetToolTip("Open another tests file (.simtests.toml)")
        b.Bind(wx.EVT_BUTTON, lambda e: self._open())
        b2 = wx.Button(left, label="Reload", style=wx.BU_EXACTFIT)
        b2.SetToolTip("Read the file again (after editing it by hand)")
        b2.Bind(wx.EVT_BUTTON, lambda e: self.reload())
        row.AddMany([(self.path, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8), (b, 0, wx.RIGHT, 4), (b2,)])
        ls.Add(row, 0, wx.EXPAND | wx.ALL, 6)
        self.list = AutoList(left, style=wx.LC_REPORT, stretch=1)
        for k, t in enumerate(("On", "Test", "Tool", "Checks", "Outputs", "Last result")):
            self.list.InsertColumn(k, t, width=80)
        self.list.SetToolTip("The tests in the file. Add tests with 'Save as test…' in a tool window (or "
                             "'python3 -m pcb_current_sim init BOARD'); select one to edit its checks and outputs")
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda e: self._show())
        self.list.Bind(wx.EVT_LIST_ITEM_DESELECTED, lambda e: self._show())
        ls.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        row = wx.WrapSizer(wx.HORIZONTAL)
        for label, tip, fn in (
                ("Rename…", "Rename the selected test", self._rename),
                ("Duplicate", "Copy the selected test (e.g. to try other limits)", self._duplicate),
                ("On / off", "Switch the selected tests on or off (--all runs only those on)", self._toggle),
                ("Delete", "Remove the selected tests from the file", self._delete),
                ("Update from its tool", "Take the tool window's current setup into the selected test (its checks "
                                         "and outputs are kept)", self._update_from_tool)):
            b = wx.Button(left, label=label, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 3)
        ls.Add(row, 0, wx.ALL, 3)
        box = wx.StaticBoxSizer(wx.VERTICAL, left, "Run")
        rb = box.GetStaticBox()
        row = wx.WrapSizer(wx.HORIZONTAL)
        self.run_sel = wx.Button(rb, label="Run selected")
        self.run_sel.SetToolTip("Run the selected tests now (results in simulations/<timestamp>/ next to the board)")
        self.run_sel.Bind(wx.EVT_BUTTON, lambda e: self._run(False))
        self.run_all = wx.Button(rb, label="Run all")
        self.run_all.SetToolTip("Run every test that is on")
        self.run_all.Bind(wx.EVT_BUTTON, lambda e: self._run(True))
        self.report_b = wx.Button(rb, label="Open report")
        self.report_b.SetToolTip("Open the last run's report.html in the browser")
        self.report_b.Bind(wx.EVT_BUTTON, lambda e: self._open_report())
        self.folder_b = wx.Button(rb, label="Open folder")
        self.folder_b.SetToolTip("Open the last run's folder (outputs, results.json, log)")
        self.folder_b.Bind(wx.EVT_BUTTON, lambda e: self._open_folder())
        for b in (self.run_sel, self.run_all, self.report_b, self.folder_b):
            row.Add(b, 0, wx.ALL, 3)
        box.Add(row, 0)
        self.cmd = wx.TextCtrl(rb, style=wx.TE_READONLY)
        self.cmd.SetToolTip("The same run from the command line (for a Makefile or a script)")
        box.Add(self.cmd, 0, wx.EXPAND | wx.ALL, 3)
        ls.Add(box, 0, wx.EXPAND | wx.ALL, 6)
        left.SetSizer(ls)

        right = scrolled(sp)
        self.right = right
        rs = wx.BoxSizer(wx.VERTICAL)
        self.title = wx.StaticText(right, label="Select a test")
        f = self.title.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        self.title.SetFont(f)
        rs.Add(self.title, 0, wx.ALL, 6)
        self.note = wx.TextCtrl(right, size=(-1, -1))
        self.note.SetToolTip("A note about the test (shown in the report)")
        self.note.Bind(wx.EVT_KILL_FOCUS, lambda e: (self._save_note(), e.Skip()))
        rs.Add(wx.StaticText(right, label="Note"), 0, wx.LEFT, 6)
        rs.Add(self.note, 0, wx.EXPAND | wx.ALL, 6)
        cb = wx.StaticBoxSizer(wx.VERTICAL, right, "Pass / fail checks")
        cbox = cb.GetStaticBox()
        self.checks = AutoList(cbox, style=wx.LC_REPORT, size=(-1, 210), stretch=4)
        for k, t in enumerate(("Value", "At", "Min", "Max", "Last value", "Note")):
            self.checks.InsertColumn(k, t, width=70)
        self.checks.SetToolTip("Each check passes when the value lies between min and max. Double-click to edit")
        self.checks.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit_check())
        cb.Add(self.checks, 1, wx.EXPAND | wx.ALL, 3)
        row = wx.BoxSizer(wx.HORIZONTAL)
        for label, tip, fn in (("Add…", "Add a check: a result value with a min and / or max", self._add_check),
                               ("Edit…", "Edit the selected check", self._edit_check),
                               ("Remove", "Remove the selected checks", self._remove_checks)):
            b = wx.Button(cbox, label=label, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 3)
        cb.Add(row)
        rs.Add(cb, 0, wx.EXPAND | wx.ALL, 6)
        ob = wx.StaticBoxSizer(wx.VERTICAL, right, "Files to keep (outputs; the report is always written)")
        self.outs = wx.CheckListBox(ob.GetStaticBox(), size=(-1, 150))
        self.outs.SetToolTip("Files written to the test's folder in each run, besides the report and results.json")
        self.outs.Bind(wx.EVT_CHECKLISTBOX, lambda e: self._outputs_changed())
        ob.Add(self.outs, 1, wx.EXPAND | wx.ALL, 3)
        rs.Add(ob, 0, wx.EXPAND | wx.ALL, 6)
        lb = wx.StaticBoxSizer(wx.VERTICAL, right, "Last run of this test")
        self.result = wx.TextCtrl(lb.GetStaticBox(), style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 240))
        self.result.SetToolTip("Its checks, key numbers and warnings from the last run")
        lb.Add(self.result, 1, wx.EXPAND | wx.ALL, 3)
        rs.Add(lb, 1, wx.EXPAND | wx.ALL, 6)
        right.SetSizer(rs)
        sp.SplitVertically(left, right, 640)
        sp.SetMinimumPaneSize(320)

    # ---- file --------------------------------------------------------------
    def reload(self, select=None):
        p = self.tf.path if self.tf is not None and self.tf.path and os.path.exists(self.tf.path) else None
        if p is None and self.session.board_path:
            p = path_for(self.session.board_path)
        try:
            if p and os.path.exists(p):
                self.tf = TestFile.load(p)
            elif self.session.board_path:
                self.tf = TestFile.new(self.session.board_path)
            else:
                self.tf = None
        except (ValueError, OSError) as e:
            self.tf = None
            self._status(f"Could not read {p}: {e}")
        self._path_label()
        self._load_last()
        self._fill(select)

    def _path_label(self):
        self.path.SetLabelText(os.path.basename(self.tf.path) if self.tf else "No board loaded")
        self.path.SetToolTip(self.tf.path if self.tf else "")

    def _open(self):
        with wx.FileDialog(self, "Open a tests file", wildcard="Tests (*.simtests.toml)|*.simtests.toml|TOML (*.toml)|*.toml",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            try:
                self.tf = TestFile.load(d.GetPath())
            except (ValueError, OSError) as e:
                return wx.MessageBox(str(e), "Open", parent=self)
        self._path_label()
        self._load_last()
        self._fill()

    def _save(self):
        try:
            self.tf.save()
        except OSError as e:
            wx.MessageBox(f"Could not save {self.tf.path}:\n{e}", "Tests", parent=self)

    def _load_last(self):
        """The latest run's results.json in the board's simulations folder ({test name: dict})."""
        self.last, self.last_folder = {}, ""
        if not self.tf:
            return
        root = os.path.join(os.path.dirname(self.tf.board_path), RN.RUNS_DIR)
        prev = RN.previous_run(root)
        if prev:
            import json
            try:
                with open(os.path.join(prev, "results.json")) as fh:
                    self.last = {t["name"]: t for t in json.load(fh).get("tests", [])}
                self.last_folder = prev
            except (OSError, ValueError):
                pass

    # ---- list --------------------------------------------------------------
    def _sel(self):
        out, k = [], self.list.GetFirstSelected()
        while k >= 0:
            out.append(k)
            k = self.list.GetNextSelected(k)
        return [i for i in out if self.tf and i < len(self.tf.tests)]

    def _test(self):
        s = self._sel()
        return self.tf.tests[s[0]] if len(s) == 1 else None

    def _fill(self, select=None):
        keep = [self.tf.tests[i].name for i in self._sel()] if self.tf else []
        if select:
            keep = [select]
        self.list.DeleteAllItems()
        for t in (self.tf.tests if self.tf else []):
            last = self.last.get(t.name)
            self.list.Append(["✓" if t.enabled else "", t.name, TOOL_TITLES.get(t.tool, t.tool), str(len(t.checks)),
                              ", ".join(t.outputs), STATUS.get(last["status"], "") if last else ""])
        for i, t in enumerate(self.tf.tests if self.tf else []):
            if t.name in keep:
                self.list.Select(i)
                self.list.EnsureVisible(i)
        self._cmd()
        self._show()

    def _cmd(self):
        if not self.tf:
            self.cmd.SetValue("")
            return
        names = [self.tf.tests[i].name for i in self._sel()]
        sel = " ".join(f"--test '{n}'" for n in names) if names else "--all"
        self.cmd.SetValue(f"python3 -m pcb_current_sim run '{self.tf.path}' {sel}")

    def _show(self):
        self._cmd()
        t = self._test()
        on = t is not None
        for w in (self.note, self.checks, self.outs):
            w.Enable(on)
        self.checks.DeleteAllItems()
        self.outs.Set([])
        self.result.SetValue("")
        if not on:
            self.title.SetLabelText("Select a test" if not self._sel() else f"{len(self._sel())} tests selected")
            self.note.SetValue("")
            return
        self.title.SetLabelText(f"{t.name}  ({TOOL_TITLES.get(t.tool, t.tool)})")
        self.note.SetValue(t.note)
        last = self.last.get(t.name) or {}
        vals = last.get("values") or {}
        for c in t.checks:
            v = vals.get(c.what)
            if isinstance(v, dict):
                v = v.get(c.at) if c.at else (max(v.values(), key=lambda x: x if x is not None else -1e300)
                                              if v and c.max is not None else
                                              min(v.values(), key=lambda x: x if x is not None else 1e300) if v else None)
            self.checks.Append([c.what, c.at or ("every item" if tool_mod(t.tool).METRICS.get(c.what) and
                                                 tool_mod(t.tool).METRICS[c.what].per else ""),
                                "" if c.min is None else f"{c.min:g}", "" if c.max is None else f"{c.max:g}",
                                "" if v is None else f"{v:.4g}", c.note])
        mod = tool_mod(t.tool)
        keys = list(mod.OUTPUTS)
        self.outs.Set([f"{k}: {mod.OUTPUTS[k]}" for k in keys])
        for i, k in enumerate(keys):
            self.outs.Check(i, k in t.outputs)
        if last:
            L = [f"{STATUS.get(last['status'], last['status'])} in {last.get('seconds', 0):.1f} s "
                 f"({os.path.basename(self.last_folder)})"]
            if last.get("error"):
                L += ["", last["error"].strip()]
            for c in last.get("checks", []):
                where = c.get("at") or c.get("item")
                val = "-" if c.get("value") is None else f"{c['value']:.4g} {c.get('unit') or ''}".strip()
                L.append(f"  {STATUS.get(c['status'], c['status'])[:1]} {c['what']}{' at ' + where if where else ''}: "
                         f"{val}{' - ' + c['text'] if c.get('text') else ''}")
            L += [""] + list(last.get("lines") or [])
            if last.get("warnings"):
                L += ["", "Warnings:"] + [f"  {w}" for w in last["warnings"]]
            self.result.SetValue("\n".join(L))
        refit(self.right)

    # ---- test edits -----------------------------------------------------------
    def _rename(self):
        t = self._test()
        if t is None:
            return
        with wx.TextEntryDialog(self, "New name:", "Rename test", t.name) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            n = d.GetValue().strip()
        if not n or n == t.name:
            return
        if self.tf.get(n) is not None:
            return wx.MessageBox(f"There is already a test '{n}'.", "Rename", parent=self)
        t.name = n
        self._save()
        self._fill(select=n)

    def _duplicate(self):
        t = self._test()
        if t is None:
            return
        c = t.copy()
        c.name = self.tf.unique_name(t.name + " copy")
        self.tf.tests.insert(self.tf.tests.index(t) + 1, c)
        self._save()
        self._fill(select=c.name)

    def _toggle(self):
        for i in self._sel():
            self.tf.tests[i].enabled = not self.tf.tests[i].enabled
        self._save()
        self._fill()

    def _delete(self):
        sel = self._sel()
        if not sel:
            return
        names = [self.tf.tests[i].name for i in sel]
        if wx.MessageBox(f"Delete {', '.join(names)} from the tests file?", "Delete", wx.YES_NO,
                         parent=self) != wx.YES:
            return
        self.tf.tests = [t for i, t in enumerate(self.tf.tests) if i not in sel]
        self._save()
        self._fill()

    def _update_from_tool(self):
        t = self._test()
        if t is None:
            return
        w = self.session.windows.get(TOOL_WINDOW.get(t.tool, t.tool))
        getter = getattr(w, "test_setup", None) if w else None
        if not getter:
            return wx.MessageBox(f"Open the {TOOL_TITLES.get(t.tool, t.tool)} window with the setup you want first.",
                                 "Update", parent=self)
        got = getter()
        if not got:
            return
        tool, setup, _name = got
        new = MK.from_session(self.session, tool, t.name, setup)
        t.setup, t.stackup, t.components = new.setup, new.stackup, new.components
        if t.components == self.tf.components:
            t.components = None
        if t.stackup == self.tf.stackup:
            t.stackup = None
        self._save()
        self._status(f"'{t.name}' now has the {TOOL_TITLES.get(t.tool, t.tool)} window's setup.")
        self._fill(select=t.name)

    def _save_note(self):
        t = self._test()
        if t is not None and self.note.GetValue().strip() != t.note:
            t.note = self.note.GetValue().strip()
            self._save()

    def _outputs_changed(self):
        t = self._test()
        if t is None:
            return
        keys = list(tool_mod(t.tool).OUTPUTS)
        t.outputs = [k for i, k in enumerate(keys) if self.outs.IsChecked(i)]
        self._save()
        self._fill(select=t.name)

    # ---- checks -----------------------------------------------------------------
    def _items(self, t):
        vals = (self.last.get(t.name) or {}).get("values") or {}
        return {k: v for k, v in vals.items() if isinstance(v, dict)}

    def _csel(self):
        out, k = [], self.checks.GetFirstSelected()
        while k >= 0:
            out.append(k)
            k = self.checks.GetNextSelected(k)
        return out

    def _add_check(self):
        t = self._test()
        if t is None:
            return
        d = CheckDialog(self, t.tool, None, self._items(t))
        if d.ShowModal() == wx.ID_OK and d.check:
            t.checks.append(d.check)
            self._save()
            self._fill(select=t.name)
        d.Destroy()

    def _edit_check(self):
        t = self._test()
        s = self._csel()
        if t is None or len(s) != 1:
            return
        d = CheckDialog(self, t.tool, t.checks[s[0]], self._items(t))
        if d.ShowModal() == wx.ID_OK and d.check:
            t.checks[s[0]] = d.check
            self._save()
            self._fill(select=t.name)
        d.Destroy()

    def _remove_checks(self):
        t = self._test()
        s = set(self._csel())
        if t is None or not s:
            return
        t.checks = [c for i, c in enumerate(t.checks) if i not in s]
        self._save()
        self._fill(select=t.name)

    # ---- running ------------------------------------------------------------------
    def _run(self, all_tests):
        if self._running or not self.tf:
            return
        names = None if all_tests else [self.tf.tests[i].name for i in self._sel()]
        if not all_tests and not names:
            return wx.MessageBox("Select the tests to run.", "Run", parent=self)
        if not self.tf.tests:
            return wx.MessageBox("The file has no tests yet: use 'Save as test…' in a tool window.", "Run", parent=self)
        self._save()
        self._running = True
        for b in (self.run_sel, self.run_all):
            b.Disable()
        board = self.session.board if (self.session.board_path and os.path.abspath(self.session.board_path)
                                       == os.path.abspath(self.tf.board_path)) else None
        path = self.tf.path

        def work():
            try:
                rr = RN.run_tests(path, names, all_tests, log=lambda m: wx.CallAfter(self._status, m.strip()),
                                  board=board)
                wx.CallAfter(self._done, rr, None)
            except (ValueError, OSError) as e:
                wx.CallAfter(self._done, None, str(e))
            except Exception:       # noqa: BLE001
                wx.CallAfter(self._done, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _done(self, rr, err):
        self._running = False
        for b in (self.run_sel, self.run_all):
            b.Enable()
        if err:
            self._status("Run failed.")
            return wx.MessageBox(err, "Run", parent=self)
        n = rr.counts
        self._status(f"Done: {', '.join(f'{v} {k}' for k, v in n.items())} - {rr.folder}")
        self._load_last()
        self._fill()

    def _open_report(self):
        if not self.last_folder:
            return wx.MessageBox("No run yet.", "Report", parent=self)
        wx.LaunchDefaultBrowser("file://" + os.path.join(self.last_folder, "report.html"))

    def _open_folder(self):
        if not self.last_folder:
            return wx.MessageBox("No run yet.", "Folder", parent=self)
        wx.LaunchDefaultApplication(self.last_folder)


TOOL_WINDOW = {"dc": "dc", "pdn": "pdn", "loop": "loop", "extract": "extract", "matching": "matching"}
