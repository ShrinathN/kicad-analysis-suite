"""Matching & Tolerance tool (wx): visual ladder editor, live nominal response, tolerance analyses."""
import csv
import json
import math
import os
import re
import threading
import traceback
from dataclasses import asdict

import numpy as np
import wx
import wx.grid
import wx.propgrid as wxpg

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import scenario as SC  # noqa: E402
from ..matching import analysis as A  # noqa: E402
from ..matching import microstrip as MS  # noqa: E402
from ..matching import model as M  # noqa: E402
from ..matching import plots as PL  # noqa: E402
from ..suite.library import C_TOL_CODES, DIELECTRICS, fmt_like, fmt_si, parse_si  # noqa: E402
from .layout import refit, scrolled  # noqa: E402
from .schematic import SchematicCanvas, make_icon  # noqa: E402
from .suite import PART_FIELDS, PART_TIPS, PartDialog, attach_suite_menu  # noqa: E402

PALETTE = (("Series", False, ("L", "C", "R", "TL", "S2P")),
           ("Shunt", True, ("L", "C", "R", "stub_open", "stub_short")))
PORT_KINDS = (("Z", "R + jX"), ("series RLC", "Series RLC"), ("parallel RLC", "Parallel RLC"),
              ("s1p", "Touchstone .s1p"))
BAND_COLS = (("Name", "name", str), ("f1", "f1", "si"), ("f2", "f2", "si"), ("RL min [dB]", "rl_min", float),
             ("IL max [dB]", "il_max", float), ("VSWR max", "vswr_max", float))


# hover help for the property grid rows (by property name; p_<attr> rows use PART_TIPS)
PG_TIPS = {
    "port_kind": "How the port impedance is described: a constant R + jX, an R-L-C in series or in parallel, "
                 "or a measured one-port file",
    "port_r": "Resistance of the port (e.g. 50 Ω for a source, an antenna's radiation resistance)",
    "port_x": "Reactance of the port at every frequency (constant)",
    "port_l": "Inductance in the port's RLC model (0 = none)",
    "port_c": "Capacitance in the port's RLC model (0 = none)",
    "name": "Name of the part in the ladder and the BOM",
    "slot": "Series: in the signal path. Shunt: from the signal path to ground",
    "value": "Nominal value; SI prefixes allowed (4n7, 1.2p)",
    "dielectric": "Capacitor dielectric: sets the temperature drift (C0G flat, X7R ±15 %, …) and DC-bias loss",
    "vdc": "DC voltage across the capacitor, for the DC-bias derating",
    "package": "Package size (text, for the BOM)",
    "line": "Ideal: give Z0 and εeff directly. Microstrip: computed from the trace width, dielectric height and εr",
    "length": "Physical length of the line or stub",
    "w": "Trace width",
    "h": "Dielectric thickness between the trace and its ground plane",
    "er": "Relative permittivity of the dielectric",
    "tand": "Dielectric loss tangent",
    "w_tol": "Etching tolerance of the trace width (± µm), varied in the analyses",
    "er_tol": "Tolerance of εr (± %), varied in the analyses",
    "z0": "Characteristic impedance of the line",
    "eeff": "Effective permittivity (sets the line's electrical length)",
    "loss_db_m": "Line loss per metre",
    "z0_tol": "Tolerance of Z0 (± %), varied in the analyses",
    "c_z0": "Computed at the band centre (read only)",
    "c_eeff": "Computed at the band centre (read only)",
    "c_deg": "Electrical length at the band centre (read only); 90° = a quarter wave",
}


def default_setup():
    s = M.Setup(f_start=1e9, f_stop=4e9, n_f=301)
    s.load = M.Port("series RLC", r=20.0, l=1.5e-9, c=3e-12)     # e.g. a small antenna
    s.bands = [M.Band("ISM 2.4", 2.4e9, 2.5e9, rl_min=10.0, il_max=1.0)]
    return s


def tol_text(p):
    return f"{p.tol:g} %" if p.tol_unit == "%" else fmt_like(p.tol, p.value, p.unit)


def parse_tol(text, kind):
    t = text.strip().replace(" ", "")
    if t.endswith("%"):
        return float(t[:-1]), "%"
    if re.fullmatch(r"[0-9.]+", t):
        return float(t), "%"
    if kind == "C" and t.upper() in C_TOL_CODES:
        return C_TOL_CODES[t.upper()], "abs"
    return parse_si(t), "abs"


class PlotPage(wx.Panel):
    def __init__(self, parent, fn):
        super().__init__(parent)
        self.fn, self.dirty = fn, True
        self.fig = Figure(figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        tb = NavigationToolbar2WxAgg(self.canvas)
        tb.Realize()
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(self.canvas, 1, wx.EXPAND)
        s.Add(tb, 0, wx.EXPAND)
        self.SetSizer(s)

    def draw(self, setup, nom, res):
        self.fig.clf()
        try:
            self.fn(self.fig, setup, nom, res)
        except Exception as e:      # noqa: BLE001 - a bad plot must not break the editor
            self.fig.clf()
            self.fig.text(0.5, 0.5, f"Plot failed: {e}", ha="center", va="center", wrap=True)
        self.canvas.draw()
        self.dirty = False


class MatchingFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="Matching & Tolerance - PCB analysis suite", size=(1500, 980))
        self.session = session
        self.file_path = None
        self.setup = self._initial_setup()
        self.nom, self.result = None, None
        self._cancel, self._running = False, False
        self._timer = None
        self._build()
        self._menus()
        attach_suite_menu(self, session, "matching")
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._sweep_to_ui()
        self._changed(rebuild=True)

    # ------------------------------------------------------------- persistence
    def _initial_setup(self):
        sp = self.session.scenario_path
        if sp:
            try:
                d = SC.section(sp, "matching")
                if d.get("setup"):
                    return M.Setup.from_dict(d["setup"])
            except (OSError, ValueError, TypeError) as e:
                wx.CallAfter(self._status, f"Could not load the saved network: {e}")
        return default_setup()

    def _target(self):
        if self.file_path:
            return os.path.basename(self.file_path)
        if self.session.scenario_path:
            return os.path.basename(self.session.scenario_path)
        return None

    def save(self, quiet=False):
        d = self.setup.to_dict()
        try:
            if self.file_path:
                with open(self.file_path, "w") as f:
                    json.dump(dict(version=1, tool="matching", setup=d), f, indent=1)
            elif self.session.scenario_path:
                SC.save_section(self.session.scenario_path, "matching", dict(setup=d))
            elif not quiet:
                return self._save_as()
            else:
                return
        except OSError as e:
            return wx.MessageBox(f"Save failed: {e}", "Matching", parent=self) if not quiet else None
        if not quiet:
            self._status(f"Saved to {self._target()}")

    def _save_as(self):
        with wx.FileDialog(self, "Save network", wildcard="Matching network (*.match.json)|*.match.json",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() == wx.ID_OK:
                p = d.GetPath()
                self.file_path = p if p.endswith(".json") else p + ".match.json"
                self.save()

    def _open(self):
        with wx.FileDialog(self, "Open network", wildcard="Matching network (*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            try:
                with open(d.GetPath()) as f:
                    raw = json.load(f)
                if "setup" not in raw:           # a board scenario file
                    raw = raw.get("matching") or {}
                self.setup = M.Setup.from_dict(raw["setup"])
            except (OSError, ValueError, KeyError, TypeError) as e:
                return wx.MessageBox(f"Could not open: {e}", "Matching", parent=self)
            self.file_path = d.GetPath() if d.GetPath().endswith(".match.json") else None
        self.canvas.sel = None
        self._sweep_to_ui()
        self._changed(rebuild=True)

    def _new(self):
        self.setup, self.result = default_setup(), None
        self.canvas.sel = None
        self._sweep_to_ui()
        self._changed(rebuild=True)

    def _on_close(self, e):
        self._cancel = True
        self.save(quiet=True)
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if what == "library" and isinstance(self.canvas.sel, int):
            wx.CallAfter(self._props)

    # ------------------------------------------------------------- layout
    def _build(self):
        p = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)
        pal = wx.BoxSizer(wx.HORIZONTAL)
        for label, shunt, kinds in PALETTE:
            pal.Add(wx.StaticText(p, label=label), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 6)
            for k in kinds:
                b = wx.BitmapButton(p, bitmap=make_icon(k, shunt))
                b.SetToolTip(f"Add {'shunt' if shunt else 'series'} {M.KINDS[k].lower()}")
                b.Bind(wx.EVT_BUTTON, lambda e, k=k, s=shunt: self._add(k, s))
                pal.Add(b, 0, wx.RIGHT, 2)
            pal.AddSpacer(14)
        for text, cmd, tip in (("Flip", self._flip, "Series ↔ shunt"), ("Duplicate", self._dup, ""),
                               ("Delete", self._delete, "Del"), ("◀", lambda: self._nudge(-1), "Move left (Ctrl+←)"),
                               ("▶", lambda: self._nudge(1), "Move right (Ctrl+→)")):
            b = wx.Button(p, label=text, style=wx.BU_EXACTFIT if len(text) == 1 else 0)
            b.SetToolTip(tip or text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            pal.Add(b, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        top.Add(pal, 0, wx.ALL, 4)
        self.canvas = SchematicCanvas(p, self.setup, self._select, self._edit, self._move, self._menu)
        top.Add(self.canvas, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
        hint = wx.StaticText(p, label="Click a part (or the source / load) to edit it on the left; drag to reorder; "
                                      "new parts go in at the ▼ marker. The plots follow every edit.")
        hint.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))
        top.Add(hint, 0, wx.LEFT | wx.TOP, 6)

        split = wx.SplitterWindow(p, style=wx.SP_LIVE_UPDATE)
        left = self.left = wx.Notebook(split)
        left.AddPage(self._props_page(left), "Properties")
        left.AddPage(self._sweep_page(left), "Sweep && specs")
        left.AddPage(self._analysis_page(left), "Analysis")
        right = self.plots_nb = wx.Notebook(split)
        self.pages = []
        for name, fn in PL.PLOTS:
            pg = PlotPage(right, fn)
            right.AddPage(pg, name)
            self.pages.append(pg)
        right.AddPage(self._results_page(right), "Worst case && BOM")
        right.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, lambda e: (self._draw_visible(), e.Skip()))
        split.SplitVertically(left, right, 440)
        split.SetMinimumPaneSize(300)
        top.Add(split, 1, wx.EXPAND | wx.ALL, 4)
        self.status = wx.StaticText(p, label="")
        top.Add(self.status, 0, wx.EXPAND | wx.LEFT | wx.BOTTOM, 6)
        p.SetSizer(top)

    def _menus(self):
        mb = wx.MenuBar()
        m = wx.Menu()
        for label, cmd in (("&New network", self._new), ("&Open…\tCtrl+O", self._open),
                           ("&Save\tCtrl+S", self.save), ("Save &as…", self._save_as), (None, None),
                           ("Export &BOM (CSV)…", self._export_bom), ("Export &response (CSV)…", self._export_resp),
                           (None, None), ("&Close\tCtrl+W", self.Close)):
            if label is None:
                m.AppendSeparator()
                continue
            it = m.Append(wx.ID_ANY, label)
            self.Bind(wx.EVT_MENU, lambda e, c=cmd: c(), it)
        mb.Append(m, "&File")
        self.SetMenuBar(mb)

    def _props_page(self, parent):
        pnl = scrolled(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        self.pg = wxpg.PropertyGrid(pnl, style=wxpg.PG_SPLITTER_AUTO_CENTER | wxpg.PG_TOOLTIPS)
        self.pg.Bind(wxpg.EVT_PG_CHANGED, self._pg_changed)
        self.pg.SetMinSize((280, 300))
        self.pg.Bind(wx.EVT_MOTION, self._pg_hover)
        s.Add(self.pg, 1, wx.EXPAND | wx.ALL, 2)
        self.pbtns = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_lib = wx.Button(pnl, label="Save part to library…")
        self.btn_lib.Bind(wx.EVT_BUTTON, lambda e: self._to_library())
        self.btn_stack = wx.Button(pnl, label="Use board stackup")
        self.btn_stack.SetToolTip("h, εr and tanδ of the dielectric under the top layer")
        self.btn_stack.Bind(wx.EVT_BUTTON, lambda e: self._from_stackup())
        self.btn_w = wx.Button(pnl, label="Width for Z0…")
        self.btn_w.Bind(wx.EVT_BUTTON, lambda e: self._width_for())
        for b in (self.btn_lib, self.btn_stack, self.btn_w):
            self.pbtns.Add(b, 0, wx.RIGHT, 4)
        s.Add(self.pbtns, 0, wx.ALL, 4)
        pnl.SetSizer(s)
        pnl.FitInside()
        return pnl

    def _sweep_page(self, parent):
        pnl = scrolled(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        g = wx.FlexGridSizer(0, 2, 4, 8)
        self.sw = {}
        for key, label in (("f_start", "Start frequency"), ("f_stop", "Stop frequency"), ("n_f", "Points"),
                           ("z_ref", "Smith chart reference [Ω]")):
            t = wx.TextCtrl(pnl, size=(110, -1))
            t.Bind(wx.EVT_TEXT, lambda e: self._sweep_from_ui())
            t.SetToolTip({"f_start": "First frequency of the sweep (2.4G, 900M, …)",
                          "f_stop": "Last frequency of the sweep",
                          "n_f": "Number of frequency points (the bands add their own points)",
                          "z_ref": "Reference impedance of the Smith chart and the S-parameters"}[key])
            self.sw[key] = t
            g.Add(wx.StaticText(pnl, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            g.Add(t)
        self.log_f = wx.CheckBox(pnl, label="Logarithmic sweep")
        self.log_f.Bind(wx.EVT_CHECKBOX, lambda e: self._sweep_from_ui())
        self.log_f.SetToolTip("Space the points logarithmically (for wide sweeps) instead of evenly")
        s.Add(g, 0, wx.ALL, 6)
        s.Add(self.log_f, 0, wx.LEFT | wx.BOTTOM, 6)
        s.Add(wx.StaticText(pnl, label="Bands and specs (0 = no limit). Frequencies accept 2.4G, 2400M, …"),
              0, wx.LEFT, 6)
        self.bgrid = wx.grid.Grid(pnl)
        self.bgrid.CreateGrid(0, len(BAND_COLS))
        for k, (t, _, _) in enumerate(BAND_COLS):
            self.bgrid.SetColLabelValue(k, t)
        self.bgrid.SetRowLabelSize(0)
        self.bgrid.Bind(wx.grid.EVT_GRID_CELL_CHANGED, self._band_changed)
        self.bgrid.GetGridWindow().SetToolTip(
            "One row per band: f1 to f2, and the spec to meet there: return loss at least RL min, insertion "
            "loss at most IL max, VSWR at most VSWR max (0 = no limit)")
        s.Add(self.bgrid, 1, wx.EXPAND | wx.ALL, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        for text, cmd in (("Add band", self._band_add), ("Delete band", self._band_del)):
            b = wx.Button(pnl, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            row.Add(b, 0, wx.RIGHT, 4)
        s.Add(row, 0, wx.ALL, 6)
        pnl.SetSizer(s)
        pnl.FitInside()
        return pnl

    def _analysis_page(self, parent):
        pnl = scrolled(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        self.chk = {}
        for key, label in (("corners", "Corners (every variable at both limits)"),
                           ("mc", "Monte Carlo"), ("worst", "Worst-case search (bounded optimiser)"),
                           ("sens", "Sensitivity (tornado + MC variance share)")):
            self.chk[key] = wx.CheckBox(pnl, label=label)
            self.chk[key].SetValue(True)
            self.chk[key].SetToolTip({
                "corners": "Evaluate every combination of each part at its minimum and maximum value",
                "mc": "Random samples of all parts within their tolerances: yield and spread",
                "worst": "Search inside the tolerances for the worst response per band (catches worst cases "
                         "that are not at a corner)",
                "sens": "Which parts matter most: one-at-a-time swings and each part's share of the variance"}[key])
            s.Add(self.chk[key], 0, wx.LEFT | wx.TOP, 6)
        g = wx.FlexGridSizer(0, 2, 4, 8)
        self.mc_n = wx.SpinCtrl(pnl, min=100, max=200000, initial=self.setup.mc_n, size=(160, -1))
        self.dist = wx.RadioBox(pnl, label="Part spread", choices=["Uniform within tolerance",
                                                                   "Gaussian, 3σ = tolerance (truncated)"],
                                majorDimension=1)
        self.seed = wx.SpinCtrl(pnl, min=0, max=10 ** 6, initial=self.setup.seed, size=(160, -1))
        self.mc_n.SetToolTip("Number of random samples; more = smoother statistics, slower")
        self.dist.SetToolTip("How each part's value is spread within its tolerance")
        self.seed.SetToolTip("Seed of the random numbers; the same seed repeats the same samples")
        g.Add(wx.StaticText(pnl, label="Monte Carlo samples"), 0, wx.ALIGN_CENTER_VERTICAL)
        g.Add(self.mc_n)
        g.Add(wx.StaticText(pnl, label="Random seed"), 0, wx.ALIGN_CENTER_VERTICAL)
        g.Add(self.seed)
        s.Add(g, 0, wx.ALL, 6)
        s.Add(self.dist, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        trow = wx.BoxSizer(wx.HORIZONTAL)
        self.temp = wx.CheckBox(pnl, label="Temperature from")
        self.tmin = wx.TextCtrl(pnl, size=(50, -1))
        self.tmax = wx.TextCtrl(pnl, size=(50, -1))
        self.temp.SetToolTip("Also vary the temperature over this range (capacitor dielectric drift)")
        self.tmin.SetToolTip("Lowest temperature [°C]")
        self.tmax.SetToolTip("Highest temperature [°C]")
        trow.AddMany([(self.temp, 0, wx.ALIGN_CENTER_VERTICAL), (self.tmin, 0, wx.LEFT, 4),
                      (wx.StaticText(pnl, label="to"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4),
                      (self.tmax,), (wx.StaticText(pnl, label="°C (capacitor dielectric drift)"), 0,
                                     wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 4)])
        s.Add(trow, 0, wx.ALL, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.run_btn = wx.Button(pnl, label="Run analysis")
        self.run_btn.Bind(wx.EVT_BUTTON, lambda e: self._run())
        self.stop_btn = wx.Button(pnl, label="Cancel")
        self.stop_btn.Bind(wx.EVT_BUTTON, lambda e: setattr(self, "_cancel", True))
        self.stop_btn.Disable()
        row.AddMany([(self.run_btn,), (self.stop_btn, 0, wx.LEFT, 4)])
        s.Add(row, 0, wx.ALL, 6)
        self.vars_lbl = wx.StaticText(pnl, label="")
        s.Add(self.vars_lbl, 0, wx.LEFT | wx.RIGHT, 6)
        self.summary = wx.TextCtrl(pnl, style=wx.TE_MULTILINE | wx.TE_READONLY)
        s.Add(self.summary, 1, wx.EXPAND | wx.ALL, 6)
        pnl.SetSizer(s)
        pnl.FitInside()
        return pnl

    def _results_page(self, parent):
        pnl = scrolled(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(wx.StaticText(pnl, label="Worst case found by the search (variable values at the worst point)"),
              0, wx.ALL, 4)
        self.wc = wx.ListCtrl(pnl, style=wx.LC_REPORT)
        s.Add(self.wc, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(pnl, label="Bill of materials"), 0, wx.ALIGN_CENTER_VERTICAL)
        b = wx.Button(pnl, label="Export CSV…")
        b.Bind(wx.EVT_BUTTON, lambda e: self._export_bom())
        row.Add(b, 0, wx.LEFT, 8)
        s.Add(row, 0, wx.ALL, 4)
        self.bom = wx.ListCtrl(pnl, style=wx.LC_REPORT)
        for k, (t, w) in enumerate((("Refs", 90), ("Qty", 40), ("Kind", 110), ("Value", 80), ("Tolerance", 80),
                                    ("Library part", 200), ("Package", 70), ("Manufacturer", 110), ("Notes", 150))):
            self.bom.InsertColumn(k, t, width=w)
        s.Add(self.bom, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        pnl.SetSizer(s)
        pnl.FitInside()
        return pnl

    def _status(self, t):
        self.status.SetLabel(t)

    # ------------------------------------------------------------- editing
    def f0(self):
        b = self.setup.bands
        return (b[0].f1 + b[0].f2) / 2 if b else math.sqrt(self.setup.f_start * self.setup.f_stop)

    def _add(self, kind, shunt):
        s = self.setup
        e = M.new_element(s, kind, shunt, stackup=self.session.stackup, f0=self.f0())
        i = self.canvas.insert_index()
        s.elements.insert(i, e)
        self.canvas.sel = i
        self._changed(rebuild=True)
        if kind == "S2P":
            wx.CallAfter(self._pick_s2p, e)

    def _pick_s2p(self, e):
        with wx.FileDialog(self, "Two-port Touchstone file", wildcard="Touchstone (*.s2p)|*.s2p;*.S2P",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                e.s2p = d.GetPath()
                self._changed(rebuild=True)

    def _sel_el(self):
        i = self.canvas.sel
        return (i, self.setup.elements[i]) if isinstance(i, int) and i < len(self.setup.elements) else (None, None)

    def _flip(self):
        i, e = self._sel_el()
        if e is None:
            return
        if e.kind in M.SERIES_ONLY | M.SHUNT_ONLY:
            return self._status(f"A {M.KINDS[e.kind].lower()} can only be {'shunt' if e.shunt else 'series'}.")
        e.shunt = not e.shunt
        self._changed(rebuild=True)

    def _dup(self):
        i, e = self._sel_el()
        if e is None:
            return
        n = M.Element(**{**asdict(e), "name": self.setup.next_name(e.kind)})
        self.setup.elements.insert(i + 1, n)
        self.canvas.sel = i + 1
        self._changed(rebuild=True)

    def _delete(self):
        i, e = self._sel_el()
        if e is None:
            return
        del self.setup.elements[i]
        n = len(self.setup.elements)
        self.canvas.sel = min(i, n - 1) if n else None
        self._changed(rebuild=True)

    def _nudge(self, d):
        i, e = self._sel_el()
        if e is not None and 0 <= i + d < len(self.setup.elements):
            self._move(i, i + d)
            self.canvas.sel = i + d
            self._changed()

    def _move(self, i, j):
        el = self.setup.elements.pop(i)
        self.setup.elements.insert(j, el)
        self._changed(rebuild=True)

    def _select(self, sel):
        self.left.SetSelection(0)
        wx.CallAfter(self._props)

    def _edit(self, sel):
        self.left.SetSelection(0)
        self._props()
        self.pg.SetFocus()

    def _menu(self, sel, pos, action=None):
        if action == "delete":
            return self._delete()
        if sel is None or pos is None:
            return
        m = wx.Menu()
        items = [("Edit", lambda: self._edit(sel))]
        if isinstance(sel, int):
            items += [("Flip series / shunt", self._flip), ("Duplicate", self._dup), ("Delete", self._delete)]
        for label, cmd in items:
            it = m.Append(wx.ID_ANY, label)
            self.Bind(wx.EVT_MENU, lambda e, c=cmd: c(), it)
        self.canvas.PopupMenu(m, pos)
        m.Destroy()

    def _changed(self, rebuild=False):
        """After any edit: redraw the schematic, drop stale analysis results, recompute the nominal response."""
        if self.result is not None:
            self.result = None
            self._status("Network changed: run the analysis again for tolerances.")
        self.canvas.f0 = self.f0()
        self.canvas.update(self.setup)
        if rebuild:
            wx.CallAfter(self._props)
        n = len(A.variables(self.setup))
        self.vars_lbl.SetLabel(f"{n} toleranced variable{'s' if n != 1 else ''}: "
                               f"{2 ** n if n <= A.MAX_CORNER_VARS else 'too many'} corners")
        refit(self.vars_lbl.GetParent())
        if self._timer is not None and self._timer.IsRunning():
            self._timer.Stop()
        self._timer = wx.CallLater(120, self._recompute)

    def _recompute(self):
        try:
            warn = A.check(self.setup)
            self.nom = A.nominal(self.setup)
        except Exception as e:      # noqa: BLE001 - show the reason, keep the editor alive
            self.nom = None
            self._status(f"Cannot evaluate: {e}")
            return
        nb = self.nom["band"]
        parts = [f"{b.name}: worst RL {d['rl_worst']:.1f} dB, IL {d['il_worst']:.2f} dB"
                 for b, d in zip(self.setup.bands, nb)]
        if self.result is None:
            self._status("Nominal  -  " + ("; ".join(parts) or "no bands defined") +
                         ("  -  " + warn[0] if warn else ""))
        for pg in self.pages:
            pg.dirty = True
        self._draw_visible()
        self._fill_bom()

    def _draw_visible(self):
        k = self.plots_nb.GetSelection()
        if 0 <= k < len(self.pages) and self.pages[k].dirty and self.nom is not None:
            self.pages[k].draw(self.setup, self.nom, self.result)

    # ------------------------------------------------------------- property grid
    def _props(self):
        if not self:
            return
        pg = self.pg
        pg.Clear()
        sel = self.canvas.sel
        self.btn_lib.Hide()
        self.btn_stack.Hide()
        self.btn_w.Hide()
        if sel in ("source", "load"):
            self._props_port(getattr(self.setup, sel), "Source (port 1)" if sel == "source" else "Load (port 2)")
        elif isinstance(sel, int) and sel < len(self.setup.elements):
            self._props_element(self.setup.elements[sel])
        else:
            pg.Append(wxpg.PropertyCategory("Nothing selected"))
            pg.Append(wxpg.StringProperty("Hint", "hint", "Click a part, the source or the load"))
            pg.SetPropertyReadOnly("hint")
        for name, tip in {**PG_TIPS, **{"p_" + k: v for k, v in PART_TIPS.items()}}.items():
            if pg.GetPropertyByName(name) and not pg.GetPropertyHelpString(name):
                pg.SetPropertyHelpString(name, tip)
        self.pbtns.Layout()
        refit(self.pg.GetParent())

    def _pg_hover(self, e):
        """The row's help string as a tooltip."""
        e.Skip()
        prop = self.pg.HitTest(e.GetPosition()).GetProperty()
        tip = prop.GetHelpString() if prop else ""
        if (self.pg.GetToolTipText() or "") != tip:
            self.pg.SetToolTip(tip)

    def _props_port(self, port, title):
        pg = self.pg
        pg.Append(wxpg.PropertyCategory(title))
        keys = [k for k, _ in PORT_KINDS]
        pg.Append(wxpg.EnumProperty("Model", "port_kind", [t for _, t in PORT_KINDS], list(range(len(keys))),
                                    keys.index(port.kind)))
        if port.kind == "s1p":
            fp = wxpg.FileProperty("One-port file (.s1p)", "port_path", port.path)
            pg.Append(fp)
            pg.SetPropertyAttribute("port_path", wxpg.PG_FILE_WILDCARD, "Touchstone (*.s1p)|*.s1p;*.S1P")
            pg.SetPropertyHelpString("port_path", "e.g. an antenna's S11 from a measurement or openEMS")
            return
        pg.Append(wxpg.StringProperty("R [Ω]", "port_r", f"{port.r:g}"))
        if port.kind == "Z":
            pg.Append(wxpg.StringProperty("X [Ω]", "port_x", f"{port.x:g}"))
        else:
            pg.Append(wxpg.StringProperty("L (0 = none)", "port_l", fmt_si(port.l, "H")))
            pg.Append(wxpg.StringProperty("C (0 = none)", "port_c", fmt_si(port.c, "F")))

    def _props_element(self, e):
        pg = self.pg
        pg.Append(wxpg.PropertyCategory(f"{e.name}: {'shunt' if e.shunt else 'series'} {M.KINDS[e.kind].lower()}"))
        pg.Append(wxpg.StringProperty("Name", "name", e.name))
        if e.kind in ("L", "C", "R"):
            self.btn_lib.Show()
            pg.Append(wxpg.EnumProperty("Slot", "slot", ["Series", "Shunt"], [0, 1], int(e.shunt)))
            p = e.get_part()
            names = ["(custom)"] + [q.name for q in self.session.library.by_kind(e.kind)]
            pg.Append(wxpg.EnumProperty("Library part", "lib", names, list(range(len(names))),
                                        names.index(e.lib) if e.lib in names else 0))
            pg.SetPropertyHelpString("lib", "Copies the library model into this slot; editing a value below "
                                            "turns it into a custom part")
            pg.Append(wxpg.PropertyCategory("Model"))
            pg.Append(wxpg.StringProperty(f"Value [{p.unit}]", "value", fmt_si(p.value, p.unit)))
            pg.Append(wxpg.StringProperty("Tolerance ±", "tol", tol_text(p)))
            pg.SetPropertyHelpString("tol", "5 % or an absolute value (0.1 pF, 0.3 nH); capacitor codes A/B/C/D")
            for attr, label, unit in PART_FIELDS[e.kind]:
                v = getattr(p, attr)
                txt = f"{v * 100:g}" if unit == "%" else (fmt_si(v, unit) if unit else f"{v:g}")
                pg.Append(wxpg.StringProperty(label, "p_" + attr, txt))
            if e.kind == "C":
                pg.Append(wxpg.EnumProperty("Dielectric", "dielectric", list(DIELECTRICS),
                                            list(range(len(DIELECTRICS))),
                                            DIELECTRICS.index(p.dielectric) if p.dielectric in DIELECTRICS else 0))
                pg.Append(wxpg.StringProperty("DC bias [V]", "vdc", f"{e.vdc:g}"))
            pg.Append(wxpg.StringProperty("Package", "package", p.package))
        elif e.kind == "S2P":
            pg.Append(wxpg.FileProperty("Two-port file (.s2p)", "s2p", e.s2p))
            pg.SetPropertyAttribute("s2p", wxpg.PG_FILE_WILDCARD, "Touchstone (*.s2p)|*.s2p;*.S2P")
            pg.SetPropertyHelpString("s2p", "Vendor S-parameters, used as measured (no tolerance)")
        else:
            self.btn_stack.Show(self.session.stackup is not None)
            pg.Append(wxpg.EnumProperty("Line model", "line", ["Ideal (Z0, εeff)", "Microstrip (w, h, εr)"],
                                        [0, 1], int(e.line == "ms")))
            pg.Append(wxpg.StringProperty("Length [mm]", "length", f"{e.length * 1e3:.5g}"))
            if e.line == "ms":
                self.btn_w.Show()
                for key, label, val in (("w", "Width [mm]", e.w * 1e3), ("h", "Dielectric h [mm]", e.h * 1e3),
                                        ("er", "εr", e.er), ("tand", "tanδ", e.tand),
                                        ("w_tol", "Width tolerance ± [µm]", e.w_tol * 1e6),
                                        ("er_tol", "εr tolerance ± [%]", e.er_tol)):
                    pg.Append(wxpg.StringProperty(label, key, f"{val:.5g}"))
            else:
                for key, label, val in (("z0", "Z0 [Ω]", e.z0), ("eeff", "εeff", e.eeff),
                                        ("loss_db_m", "Loss [dB/m]", e.loss_db_m),
                                        ("z0_tol", "Z0 tolerance ± [%]", e.z0_tol)):
                    pg.Append(wxpg.StringProperty(label, key, f"{val:.5g}"))
            f0 = self.f0()
            z0, _, eeff = e.line_params(f0)
            pg.Append(wxpg.PropertyCategory(f"At {fmt_si(f0, 'Hz')}"))
            for key, label, val in (("c_z0", "Z0", f"{float(z0):.2f} Ω"), ("c_eeff", "εeff", f"{eeff:.3f}"),
                                    ("c_deg", "Electrical length",
                                     f"{360 * f0 * e.length * math.sqrt(eeff) / M.C0:.1f}°")):
                pg.Append(wxpg.StringProperty(label, key, val))
                pg.SetPropertyReadOnly(key)

    def _pg_changed(self, evt):
        name = evt.GetPropertyName()
        val = self.pg.GetPropertyValue(name)
        sel = self.canvas.sel
        rebuild = False
        try:
            if sel in ("source", "load"):
                port = getattr(self.setup, sel)
                if name == "port_kind":
                    port.kind, rebuild = PORT_KINDS[val][0], True
                    if port.kind == "s1p" and not port.path:
                        wx.CallAfter(self._pick_s1p, port)
                elif name == "port_path":
                    port.path = val
                elif name in ("port_r", "port_x"):
                    v = parse_si(val) if name == "port_r" else float(val)
                    setattr(port, name[5:], v)
                else:
                    setattr(port, name[5:], parse_si(val))
            else:
                e = self.setup.elements[sel]
                rebuild = self._apply_element(e, name, val)
        except (ValueError, TypeError) as err:
            wx.MessageBox(f"Not accepted: {err}", "Matching", parent=self)
            rebuild = True
        self._changed(rebuild=rebuild)

    def _pick_s1p(self, port):
        with wx.FileDialog(self, "One-port Touchstone file", wildcard="Touchstone (*.s1p)|*.s1p;*.S1P",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                port.path = d.GetPath()
                self._changed(rebuild=True)

    def _apply_element(self, e, name, val):
        """Apply one property edit; True when the grid layout must be rebuilt."""
        if name == "name":
            v = str(val).strip()
            if not v or any(x is not e and x.name == v for x in self.setup.elements):
                raise ValueError("names must be unique and not empty")
            e.name = v
            return True
        if name == "slot":
            e.shunt = bool(val)
            return True
        if name == "lib":
            names = ["(custom)"] + [q.name for q in self.session.library.by_kind(e.kind)]
            if val > 0:
                q = self.session.library.get(names[val])
                e.part, e.lib = asdict(q), q.name
            else:
                e.lib = ""
            return True
        if name == "s2p":
            e.s2p = val
            return True
        if name == "line":
            e.line = "ms" if val else "ideal"
            if e.line == "ms" and self.session.stackup is not None:
                self._from_stackup(e, quiet=True)
            return True
        if name == "length":
            e.length = float(val) * 1e-3
            if e.length <= 0:
                raise ValueError("length must be > 0")
            return True
        if name in ("w", "h"):
            setattr(e, name, float(val) * 1e-3)
            return True
        if name == "w_tol":
            e.w_tol = float(val) * 1e-6
            return False
        if name in ("er", "tand", "er_tol", "z0", "eeff", "loss_db_m", "z0_tol"):
            setattr(e, name, float(val))
            return name in ("er", "z0", "eeff")
        if name == "vdc":
            e.vdc = float(val)
            return False
        # part fields: the slot becomes a custom part
        p = e.get_part()
        if name == "value":
            p.value = parse_si(val)
            if p.value <= 0:
                raise ValueError("value must be > 0")
        elif name == "tol":
            p.tol, p.tol_unit = parse_tol(val, e.kind)
        elif name == "dielectric":
            p.dielectric = DIELECTRICS[val]
        elif name == "package":
            p.package = str(val)
        elif name.startswith("p_"):
            attr = name[2:]
            unit = next(u for a, _, u in PART_FIELDS[e.kind] if a == attr)
            setattr(p, attr, parse_si(val) / (100 if unit == "%" else 1))
        else:
            return False
        p.name = f"{e.name} custom"
        e.part, e.lib = asdict(p), ""
        return True

    def _to_library(self):
        i, e = self._sel_el()
        if e is None or e.get_part() is None:
            return
        p = e.get_part()
        if not e.lib:
            p.name = ""
        with PartDialog(self, p, {q.name for q in self.session.library.parts}) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            new = d.result
        self.session.library.put(new)
        self.session.library_changed()
        e.part, e.lib = asdict(new), new.name
        self._changed(rebuild=True)

    def _from_stackup(self, e=None, quiet=False):
        if e is None:
            e = self._sel_el()[1]
        st = self.session.stackup
        if e is None or st is None or not st.dielectrics:
            return
        d = st.dielectrics[0]
        e.h, e.er, e.tand, e.line = d.t, float(d.er_tand(self.f0(), model=st.model)[0]), d.tand, "ms"
        if not quiet:
            self._status(f"{e.name}: h {d.t * 1e3:.3g} mm, εr {e.er:.3g} at {fmt_si(self.f0(), 'Hz')}, tanδ {d.tand:g} "
                         f"from {d.name}")
            self._changed(rebuild=True)

    def _width_for(self):
        i, e = self._sel_el()
        if e is None or e.line != "ms":
            return
        cur = MS.z0_eeff(e.w, e.h, e.er)[0]
        with wx.TextEntryDialog(self, "Target Z0 [Ω]", "Microstrip width", f"{cur:.1f}") as d:
            if d.ShowModal() != wx.ID_OK:
                return
            try:
                z = float(d.GetValue())
            except ValueError:
                return
        e.w = MS.width_for(z, e.h, e.er)
        self._changed(rebuild=True)

    # ------------------------------------------------------------- sweep & bands
    def _sweep_to_ui(self):
        s = self.setup
        self._loading_ui = True
        for key, v in (("f_start", fmt_si(s.f_start, "Hz")), ("f_stop", fmt_si(s.f_stop, "Hz")),
                       ("n_f", str(s.n_f)), ("z_ref", f"{s.z_ref:g}")):
            self.sw[key].ChangeValue(v)
        self.log_f.SetValue(s.log_f)
        g = self.bgrid
        if g.GetNumberRows():
            g.DeleteRows(0, g.GetNumberRows())
        g.AppendRows(len(s.bands))
        for r, b in enumerate(s.bands):
            for c, (_, attr, typ) in enumerate(BAND_COLS):
                v = getattr(b, attr)
                g.SetCellValue(r, c, fmt_si(v, "Hz") if typ == "si" else (f"{v:g}" if typ is float else v))
        g.AutoSizeColumns()
        self.mc_n.SetValue(s.mc_n)
        self.seed.SetValue(s.seed)
        self.dist.SetSelection(1 if s.dist == "gauss" else 0)
        self.temp.SetValue(s.temp)
        self.tmin.SetValue(f"{s.T_min:g}")
        self.tmax.SetValue(f"{s.T_max:g}")
        self._loading_ui = False

    def _sweep_from_ui(self):
        if getattr(self, "_loading_ui", False):
            return
        s = self.setup
        ok = True
        for key in self.sw:
            t = self.sw[key]
            try:
                v = parse_si(t.GetValue()) if key in ("f_start", "f_stop") else float(t.GetValue())
                if v <= 0:
                    raise ValueError
                setattr(s, key, int(v) if key == "n_f" else v)
                t.SetBackgroundColour(wx.NullColour)
            except ValueError:
                t.SetBackgroundColour(wx.Colour(255, 200, 200))
                ok = False
            t.Refresh()
        s.log_f = self.log_f.GetValue()
        if ok and s.f_stop > s.f_start:
            self._changed()

    def _band_changed(self, evt):
        r, c = evt.GetRow(), evt.GetCol()
        _, attr, typ = BAND_COLS[c]
        b = self.setup.bands[r]
        txt = self.bgrid.GetCellValue(r, c)
        try:
            setattr(b, attr, parse_si(txt) if typ == "si" else typ(txt))
        except ValueError:
            wx.MessageBox(f"Not a value: {txt}", "Bands", parent=self)
        self._sweep_to_ui()
        self._changed()

    def _band_add(self):
        s = self.setup
        lo, hi = s.f_start, s.f_stop
        c = math.sqrt(lo * hi)
        s.bands.append(M.Band(f"band {len(s.bands) + 1}", c * 0.98, c * 1.02, 10.0))
        self._sweep_to_ui()
        self._changed()

    def _band_del(self):
        rows = self.bgrid.GetSelectedRows() or ([self.bgrid.GetGridCursorRow()] if self.setup.bands else [])
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self.setup.bands):
                del self.setup.bands[r]
        self._sweep_to_ui()
        self._changed()

    # ------------------------------------------------------------- analysis
    def _run(self):
        if self._running:
            return
        s = self.setup
        s.mc_n, s.seed = self.mc_n.GetValue(), self.seed.GetValue()
        s.dist = "gauss" if self.dist.GetSelection() == 1 else "uniform"
        s.temp = self.temp.GetValue()
        try:
            s.T_min, s.T_max = float(self.tmin.GetValue()), float(self.tmax.GetValue())
        except ValueError:
            return wx.MessageBox("Temperatures must be numbers.", "Analysis", parent=self)
        self.save(quiet=True)
        snap = M.Setup.from_dict(s.to_dict())
        flags = {k: c.GetValue() for k, c in self.chk.items()}
        self._cancel, self._running = False, True
        self.run_btn.Disable()
        self.stop_btn.Enable()

        def work():
            try:
                res = A.run(snap, flags["corners"], flags["mc"], flags["worst"], flags["sens"],
                            progress=lambda m: wx.CallAfter(self._status, m), cancel=lambda: self._cancel)
                wx.CallAfter(self._done, snap, res, None)
            except InterruptedError:
                wx.CallAfter(self._done, snap, None, "Cancelled")
            except ValueError as e:
                wx.CallAfter(self._done, snap, None, str(e))
            except Exception:       # noqa: BLE001
                wx.CallAfter(self._done, snap, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _done(self, snap, res, err):
        if not self:
            return
        self._running = False
        self.run_btn.Enable()
        self.stop_btn.Disable()
        if err:
            self._status(err.splitlines()[-1])
            if err != "Cancelled":
                wx.MessageBox(err, "Analysis failed", style=wx.ICON_ERROR, parent=self)
            return
        if snap.to_dict() != self.setup.to_dict():
            return self._status("The network changed during the run; results discarded.")
        self.result, self.nom = res, res.nominal
        lines = A.summary_lines(self.setup, res) + [f"Warning: {w}" for w in res.warnings]
        self.summary.SetValue("\n".join(lines))
        self._status(lines[0] if lines else "Done")
        for pg in self.pages:
            pg.dirty = True
        self._draw_visible()
        self._fill_worst()

    def _fill_worst(self):
        wc, res = self.wc, self.result
        wc.ClearAll()
        if res is None or not res.worst:
            return
        cols = ["Spec", "Worst [dB]", "Margin [dB]"] + [v.label for v in res.vars]
        for k, t in enumerate(cols):
            wc.InsertColumn(k, t, width=110 if k < 3 else 95)
        nomrow = ["nominal", "", ""] + [v.text(v.nom) for v in res.vars]
        rngrow = ["range", "", ""] + [f"{v.text(v.lo)} … {v.text(v.hi)}" for v in res.vars]
        for w in res.worst:
            b = self.setup.bands[w["band"]]
            wc.Append([f"{b.name} {'RL' if w['metric'] == 'rl' else 'IL'}", f"{w['value']:.2f}",
                       f"{w['margin']:+.2f}"] + [v.text(x) for v, x in zip(res.vars, w["x"])])
        wc.Append(nomrow)
        wc.Append(rngrow)

    def _fill_bom(self):
        self.bom.DeleteAllItems()
        for r in A.bom(self.setup):
            self.bom.Append([r["refs"], str(r["qty"]), r["kind"], r["value"], r["tol"], r["part"], r["package"],
                             r["mfr"], r["notes"]])

    def _export_bom(self):
        with wx.FileDialog(self, "Export BOM", wildcard="CSV (*.csv)|*.csv", defaultFile="matching_bom.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            rows = A.bom(self.setup)
            with open(d.GetPath(), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["refs"])
                w.writeheader()
                w.writerows(rows)

    def _export_resp(self):
        if self.nom is None:
            return
        with wx.FileDialog(self, "Export response", wildcard="CSV (*.csv)|*.csv",
                           defaultFile="matching_response.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            n, mc = self.nom, (self.result.mc if self.result is not None else None)
            with open(d.GetPath(), "w", newline="") as f:
                w = csv.writer(f)
                head = ["f_Hz", "S11_dB", "S21_dB", "loss_in_parts_dB", "VSWR", "Zin_re", "Zin_im"]
                if mc is not None:
                    head += ["S11_dB_MC_max", "S11_dB_MC_p95", "S21_dB_MC_min", "S21_dB_MC_p5"]
                w.writerow(head)
                for k, fr in enumerate(n["f"]):
                    row = [fr, -n["rl"][k], -n["il"][k], n["diss"][k], n["vswr"][k], n["zin"][k].real,
                           n["zin"][k].imag]
                    if mc is not None:
                        row += [-mc.rl[:, k].min(), -float(np.percentile(mc.rl[:, k], 5)),
                                -mc.il[:, k].max(), -float(np.percentile(mc.il[:, k], 95))]
                    w.writerow(row)

