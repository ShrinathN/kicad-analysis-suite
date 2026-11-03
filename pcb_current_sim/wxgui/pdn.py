"""PDN (AC) window: a rail's impedance at its loads against the target, its decaps and regulator."""
import math
import threading
import traceback
from dataclasses import asdict

import numpy as np
import wx

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import elements as EL  # noqa: E402
from .. import scenario as SC  # noqa: E402
from ..pdn import analysis as PA  # noqa: E402
from ..pdn import model as PM  # noqa: E402
from ..pdn import regstore as RS  # noqa: E402
from ..suite.library import Part, fmt_si, parse_si, part_from_dict  # noqa: E402
from .crossprobe import CrossProbe, selected_rows  # noqa: E402
from .layout import AutoList, refit, scrolled  # noqa: E402
from .suite import attach_suite_menu  # noqa: E402

TARGET = (("v", "Rail voltage [V]", "V", "The rail's nominal voltage (also the capacitors' DC bias)"),
          ("ripple", "Allowed deviation [%]", "", "How far the rail may move from its voltage during a load step, "
                                                  "± % (the chip's supply tolerance minus the regulator's own)"),
          ("i_step", "Load step ΔI [A]", "A", "The largest change of the load current (e.g. a processor waking up)"),
          ("t_rise", "Step rise time", "s", "How fast the load current changes (10-90 %); the target must hold up "
                                            "to 0.35 / t_rise, e.g. 1 µs -> 350 kHz, 10 ns -> 35 MHz"))
OPTS = (("f_min", "f min [Hz]", "Hz", "Lowest frequency of the sweep"),
        ("f_max", "f max [Hz]", "Hz", "Highest frequency of the sweep (the mesh should be finer than λ/20 here)"),
        ("n_f", "Points", "", "Number of frequencies (log spaced)"),
        ("cells", "Mesh cells", "", "Approximate number of mesh cells of the rail's copper (more: finer, slower)"))
VRM_F = (("r_out", "Output R [Ω]", "Ω", "The regulator's output resistance at low frequency (its load "
                                        "regulation, ΔV_out / ΔI_out from the datasheet)"),
         ("l_out", "Output L [H]", "H", "How fast the regulator reacts, as an inductance: about 1 / ((2π f_bw)² C_out) "
                                        "for a loop bandwidth f_bw; e.g. 50 kHz with 44 µF -> 230 nH"))


PULSE = (("i_step", "Step [A]", "A", "How much the chip's current rises (and later falls back)"),
         ("delay", "Delay", "s", "When the step starts; stagger the loads' delays to see each one's effect alone"),
         ("t_rise", "Rise time", "s", "How fast the current rises (0-100 %)"),
         ("width", "Width", "s", "How long the higher current lasts (negative: it stays on)"),
         ("t_fall", "Fall time", "s", "How fast it falls back"),
         ("period", "Period", "s", "Repeat every … (0: once)"))


class PulseDialog(wx.Dialog):
    """A load's current over time (a PULSE as in SPICE)."""

    def __init__(self, parent, ld):
        super().__init__(parent, title=f"{ld.name}: load step")
        self.ld = ld
        g = wx.FlexGridSizer(0, 2, 4, 8)
        self.f = {}
        for key, label, unit, tip in PULSE:
            t = wx.TextCtrl(self, value=fmt_si(getattr(ld, key), unit), size=(120, -1))
            t.SetToolTip(tip)
            self.f[key] = (t, unit)
            g.AddMany([(wx.StaticText(self, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t,)])
        top = wx.BoxSizer(wx.VERTICAL)
        top.Add(g, 0, wx.ALL, 10)
        top.Add(wx.StaticText(self, label="Values like 1.5, 500m, 2u, 100n. The current rises by Step at Delay,\n"
                                          "stays for Width, falls back, and repeats every Period (0: once)."),
                0, wx.LEFT | wx.RIGHT, 10)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)

    def _ok(self, e):
        try:
            vals = {k: _num(t.GetValue(), u) for k, (t, u) in self.f.items()}
        except (ValueError, TypeError):
            return wx.MessageBox("Enter numbers (SI prefixes allowed).", "Load step", parent=self)
        if vals["t_rise"] < 0 or vals["t_fall"] < 0 or vals["delay"] < 0 or vals["period"] < 0:
            return wx.MessageBox("Times must not be negative (only Width may be).", "Load step", parent=self)
        for k, v in vals.items():
            setattr(self.ld, k, v)
        self.EndModal(wx.ID_OK)


class TouchstoneDialog(wx.Dialog):
    """What to export and the reference impedance."""

    def __init__(self, parent):
        super().__init__(parent, title="Save as Touchstone")
        s = wx.BoxSizer(wx.VERTICAL)
        self.kind = wx.RadioBox(parent=self, label="Ports", choices=list(PA.EXPORTS.values()),
                                majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.kind.SetToolTip("Loads: Z between the loads as analysed, decaps and the regulator inside. Board only: "
                             "the bare copper (no parts), with a port at every load, every decap (also those "
                             "switched off) and the regulator output, to connect your own models there")
        s.Add(self.kind, 0, wx.EXPAND | wx.ALL, 8)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.z0c = wx.TextCtrl(self, value="50", size=(80, -1))
        self.z0c.SetToolTip("Reference impedance of every port [Ω]. 50 Ω reads in any tool; PI tools often use "
                            "1 Ω or 0.1 Ω so milliohm impedances are not crowded near S11 = −1 (the numbers are "
                            "written with 13 digits, so either is exact)")
        row.AddMany([(wx.StaticText(self, label="Reference impedance [Ω]"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.z0c,)])
        s.Add(row, 0, wx.ALL, 8)
        s.Add(wx.StaticText(self, label="S-parameters, real/imaginary, Touchstone v1; port names in the file's "
                                        "comments. Ports are against the return net (ideal reference)."),
              0, wx.ALL, 8)
        s.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizerAndFit(s)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        self.z0 = 50.0

    def what(self):
        return list(PA.EXPORTS)[self.kind.GetSelection()]

    def _ok(self, e):
        try:
            self.z0 = parse_si(self.z0c.GetValue().replace("Ω", "").replace("ohm", "").strip())
            if not self.z0 > 0:
                raise ValueError
        except Exception:       # noqa: BLE001
            return wx.MessageBox("The reference impedance must be a positive number (e.g. 50, 1, 100m).",
                                 "Touchstone", parent=self)
        self.EndModal(wx.ID_OK)


def _num(text, unit=""):
    t = text.strip()
    if unit and t.endswith(unit):
        t = t[: -len(unit)].strip()
    return parse_si(t)


class PdnFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="PDN (AC) - PCB analysis suite", size=(1500, 960))
        self.session, self.board = session, session.board
        self.res, self.whatif, self._running = None, None, False
        self.mark = None            # (x, y) mm marked on the map (a layout issue)
        self.setup = self._load()
        self._build()
        attach_suite_menu(self, session, "pdn")
        self.probe = CrossProbe(self, session)
        self.probe.watch(self.dlist, lambda lc: [self.setup.decaps[k].ref for k in self._dsel()])
        self.probe.watch(self.llist, lambda lc: [self.setup.loads[i].name for i in selected_rows(lc)
                                                 if i < len(self.setup.loads)])
        self.probe.watch(self.ltab, lambda lc: [r for i in selected_rows(lc) if self.res is not None
                                                and i < len(self.res.layout) for r in self.res.layout[i].refs])
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._to_ui()

    # ------------------------------------------------------------- persistence
    def _load(self):
        try:
            d = SC.section(self.session.scenario_path, "pdn") if self.session.scenario_path else {}
        except (OSError, ValueError):
            d = {}
        if d.get("setup"):
            try:
                return PM.Setup.from_dict(d["setup"])
            except (TypeError, ValueError):
                pass
        try:
            s = PM.detect(self.board, None, self.session.components)
        except ValueError:
            return PM.Setup()
        RS.apply(self.board, s, RS.load(RS.path_for(self.session.library.path)))
        return s

    def save(self):
        if self.session.scenario_path:
            try:
                SC.save_section(self.session.scenario_path, "pdn", dict(setup=self.setup.to_dict()))
            except OSError as e:
                self._status(f"Could not save the setup: {e}")

    def _on_close(self, e):
        self._from_ui(quiet=True)
        self.save()
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if what == "board" and self.session.board is not None:
            self.board = self.session.board
            wx.CallAfter(self._to_ui)
        elif what == "components" and self.session.sharing != "pdn":
            changed = False
            for d in self.setup.decaps:
                e = self.session.components.get(d.ref)
                if e is not None and e.model == "C" and (e.part or e.C > 0):
                    from ..components import cap_model
                    m = cap_model(e, d.part)
                    if m != d.part:
                        d.part, changed = m, True
            if changed:
                wx.CallAfter(self._to_ui)

    def _status(self, text):
        self.SetStatusText(text)

    # ------------------------------------------------------------- layout
    def _build(self):
        self.CreateStatusBar()
        sp = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = self.left = scrolled(sp)
        ls = wx.BoxSizer(wx.VERTICAL)

        def box(title):
            b = wx.StaticBoxSizer(wx.VERTICAL, left, title)
            ls.Add(b, 0, wx.EXPAND | wx.ALL, 5)
            return b, b.GetStaticBox()

        # rail
        rb, rbox = box("Rail")
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.railc = wx.Choice(rbox)
        self.railc.SetToolTip("The power net to analyse (nets with decoupling capacitors first)")
        self.railc.Bind(wx.EVT_CHOICE, lambda e: self._detect())
        g.AddMany([(wx.StaticText(rbox, label="Power net"), 0, wx.ALIGN_CENTER_VERTICAL), (self.railc, 1, wx.EXPAND)])
        rb.Add(g, 0, wx.EXPAND | wx.ALL, 4)
        rb.Add(wx.StaticText(rbox, label="Return (ground) nets:"), 0, wx.LEFT, 4)
        self.rets = wx.CheckListBox(rbox, size=(-1, 70))
        self.rets.SetToolTip("The nets the rail's current returns in (its reference planes); ideal in this version: "
                             "no ground bounce between distant ground pins")
        rb.Add(self.rets, 0, wx.EXPAND | wx.ALL, 4)
        b = wx.Button(rbox, label="Find the rail's parts again")
        b.SetToolTip("Detect the decaps, the regulator and the loads of this rail again (keeps the target and the "
                     "options; decaps keep their models)")
        b.Bind(wx.EVT_BUTTON, lambda e: self._detect(keep=True))
        rb.Add(b, 0, wx.ALL, 4)

        # target
        tb, tbox = box("Target (noise budget)")
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.tf = {}
        for key, label, unit, tip in TARGET:
            t = wx.TextCtrl(tbox, size=(110, -1))
            t.SetToolTip(tip)
            t.Bind(wx.EVT_KILL_FOCUS, lambda e: (e.Skip(), wx.CallAfter(self._target_label)))
            self.tf[key] = (t, unit)
            g.AddMany([(wx.StaticText(tbox, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t,)])
        tb.Add(g, 0, wx.ALL, 4)
        self.tlabel = wx.StaticText(tbox, label="")
        tb.Add(self.tlabel, 0, wx.ALL, 4)

        # regulator
        vb, vbox = box("Regulator output (VRM)")
        self.vlabel = wx.StaticText(vbox, label="")
        vb.Add(self.vlabel, 0, wx.ALL, 4)
        self.vkind = wx.RadioBox(vbox, label="Model", choices=["Output R + L", "Control loop fitted to the "
                                                                              "datasheet"],
                                 majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.vkind.SetToolTip("R + L: a simple stand-in. Fitted loop: the regulator's control loop fitted to its "
                              "datasheet's load-transient plot; it reacts to your board's capacitors as the real "
                              "loop does")
        self.vkind.Bind(wx.EVT_RADIOBOX, lambda e: self._vkind())
        vb.Add(self.vkind, 0, wx.EXPAND | wx.ALL, 4)
        b = wx.Button(vbox, label="Fit from the datasheet plot…", style=wx.BU_EXACTFIT)
        b.SetToolTip("Digitise the datasheet's load-transient plot (a screenshot) and fit the regulator's loop")
        b.Bind(wx.EVT_BUTTON, lambda e: self._regfit())
        vb.Add(b, 0, wx.ALL, 4)
        self.vmodel = wx.StaticText(vbox, label="")
        vb.Add(self.vmodel, 0, wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.vsaved = wx.Choice(vbox, size=(170, -1))
        self.vsaved.SetToolTip("Regulator fits saved by part number (from any board): pick one to use it here. "
                               "Parts on this board next to the rail are listed first, with their reference")
        self.vsaved.Bind(wx.EVT_CHOICE, lambda e: self._use_saved())
        b1 = wx.Button(vbox, label="Save fit…", style=wx.BU_EXACTFIT)
        b1.SetToolTip("Keep this rail's fitted loop under the regulator's part number, for other boards and rails: "
                      "a board with that part (its footprint's Value) then gets the fit automatically")
        b1.Bind(wx.EVT_BUTTON, lambda e: self._save_fit())
        b2 = wx.Button(vbox, label="Forget", style=wx.BU_EXACTFIT)
        b2.SetToolTip("Remove the selected saved fit from the store (this rail keeps the loop it has)")
        b2.Bind(wx.EVT_BUTTON, lambda e: self._forget_fit())
        row.AddMany([(wx.StaticText(vbox, label="Saved fits"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.vsaved, 0, wx.RIGHT, 6), (b1, 0, wx.RIGHT, 4), (b2,)])
        vb.Add(row, 0, wx.ALL, 4)
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.vf = {}
        for key, label, unit, tip in VRM_F:
            t = wx.TextCtrl(vbox, size=(110, -1))
            t.SetToolTip(tip)
            self.vf[key] = (t, unit)
            g.AddMany([(wx.StaticText(vbox, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t,)])
        vb.Add(g, 0, wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.von = wx.CheckBox(vbox, label="Include")
        self.von.SetToolTip("Include the regulator (untick: the board and decaps alone)")
        b = wx.Button(vbox, label="Pick output pad…", style=wx.BU_EXACTFIT)
        b.SetToolTip("The pad where the regulator feeds the rail: its OUT pin, or a buck's inductor pad on the rail")
        b.Bind(wx.EVT_BUTTON, lambda e: self._pick_vrm())
        row.AddMany([(self.von, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8), (b,)])
        vb.Add(row, 0, wx.ALL, 4)

        # decaps
        db, dbox = box("Decoupling capacitors")
        self.dlist = AutoList(dbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 170))
        for k, t in enumerate(("On", "Ref", "Value", "Model", "ESR", "ESL")):
            self.dlist.InsertColumn(k, t, width=60)
        self.dlist.SetToolTip("Capacitors between the rail and a return net. Their models are the board's shared "
                              "component list's (Switching Loop edits show here, and edits here show there). "
                              "Double-click to edit a model")
        self.dlist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit_cap())
        db.Add(self.dlist, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn, tip in (("Edit model…", self._edit_cap, "Value, ESR, ESL, dielectric and DC-bias loss of the "
                                                             "selected capacitor"),
                              ("From library…", self._lib_cap, "Give the selected capacitors a part from the "
                                                              "component library"),
                              ("On / off", self._toggle_caps, "Leave the selected capacitors out (what if they were "
                                                              "not fitted)")):
            b = wx.Button(dbox, label=text, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        db.Add(row, 0)

        # loads
        lb, lbox = box("Loads (where the impedance is seen)")
        self.llist = AutoList(lbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 90))
        for k, t in enumerate(("On", "Part", "Pins on the rail", "Step", "Delay", "Rise", "Width", "Fall", "Period")):
            self.llist.InsertColumn(k, t, width=60)
        self.llist.SetToolTip("Chips drawing current from the rail; each one's power pins on the rail are one port. "
                              "Step … Period: its current over time for the load-step simulation. Double-click to "
                              "edit that")
        self.llist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit_pulse())
        lb.Add(self.llist, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn, tip in (("Edit step…", self._edit_pulse, "The selected load's current over time (step, delay, "
                                                               "rise, width, fall, period)"),
                              ("Stagger", self._stagger, "Give the loads one after another the same step, each "
                                                         "starting after the previous one has settled, so their "
                                                         "effects can be told apart"),
                              ("Add part…", self._add_load, "Add a part with pins on the rail as a load"),
                              ("On / off", self._toggle_loads, "Leave the selected loads out of the analysis"),
                              ("Remove", self._del_loads, "Remove the selected loads")):
            b = wx.Button(lbox, label=text, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        lb.Add(row, 0)

        # options
        ob, obox = box("Sweep and mesh")
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.of = {}
        for key, label, unit, tip in OPTS:
            t = wx.TextCtrl(obox, size=(110, -1))
            t.SetToolTip(tip)
            self.of[key] = (t, unit)
            g.AddMany([(wx.StaticText(obox, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t,)])
        ob.Add(g, 0, wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.tran = wx.CheckBox(obox, label="Simulate the load steps")
        self.tran.SetToolTip("Also compute the rail's voltage over time while the loads' currents step (each load's "
                             "Step … Period)")
        self.tstop = wx.TextCtrl(obox, size=(90, -1))
        self.tstop.SetToolTip("Simulated time, e.g. 2m; blank: until the regulator has settled after the last step")
        row.AddMany([(self.tran, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8),
                     (wx.StaticText(obox, label="for"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4), (self.tstop,)])
        ob.Add(row, 0, wx.ALL, 4)
        self.run_b = wx.Button(left, label="Analyse")
        self.run_b.SetToolTip("Mesh the rail and compute its impedance at the loads (save the board in KiCad "
                              "first if you changed it)")
        self.run_b.Bind(wx.EVT_BUTTON, lambda e: self._run())
        ls.Add(self.run_b, 0, wx.EXPAND | wx.ALL, 8)
        from .tests import add_button
        ls.Add(add_button(left, self, self.test_setup), 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        left.SetSizer(ls)

        # results
        right = wx.Panel(sp)
        rs = wx.BoxSizer(wx.VERTICAL)
        self.nb = wx.Notebook(right)
        # impedance
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.loadc = wx.Choice(p)
        self.loadc.SetToolTip("The load whose impedance is plotted")
        self.loadc.Bind(wx.EVT_CHOICE, lambda e: self._plot_z())
        self.zinfo = wx.StaticText(p, label="")
        b = wx.Button(p, label="Touchstone…", style=wx.BU_EXACTFIT)
        b.SetToolTip("Save the rail's impedance as an S-parameter file (.sNp) for a circuit simulator or another "
                     "PI tool: between the loads with every part included, or the bare board with a port at "
                     "each load, decap and the regulator")
        b.Bind(wx.EVT_BUTTON, lambda e: self._touchstone())
        row.AddMany([(wx.StaticText(p, label="Load"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.loadc, 0, wx.RIGHT, 8), (b, 0, wx.RIGHT, 12), (self.zinfo, 1, wx.ALIGN_CENTER_VERTICAL)])
        s.Add(row, 0, wx.EXPAND | wx.ALL, 6)
        self.zfig = Figure(figsize=(8, 5), constrained_layout=True)
        self.zcanvas = FigureCanvasWxAgg(p, -1, self.zfig)
        s.Add(self.zcanvas, 1, wx.EXPAND)
        s.Add(NavigationToolbar2WxAgg(self.zcanvas), 0, wx.EXPAND)
        p.SetSizer(s)
        self.nb.AddPage(p, "Impedance")
        # decap table
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.dtab = AutoList(p, style=wx.LC_REPORT)
        self.dcols = ("Ref", "C at bias", "ESR", "ESL", "Mounting L (return)", "Resonance", "To load [mm]",
                      "Carries (max)", "Helps from – to", "How the mounting L was found")
        for k, t in enumerate(self.dcols):
            self.dtab.InsertColumn(k, t, width=90)
        s.Add(self.dtab, 1, wx.EXPAND | wx.ALL, 6)
        s.Add(wx.StaticText(p, label="Carries: the largest share of the first load's current the capacitor takes, "
                                     "and at what frequency (above 100 % near a resonance between capacitors, where "
                                     "current circulates between them). Helps: where it carries at least a quarter "
                                     "of it. Resonance: with its mounting L, at the rail voltage."), 0, wx.ALL, 6)
        p.SetSizer(s)
        self.nb.AddPage(p, "Decaps")
        # what-if
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        b = wx.Button(p, label="Remove each decap in turn")
        b.SetToolTip("Solve again without each capacitor (one at a time, coarser frequency grid): which ones "
                     "matter, and which could go")
        b.Bind(wx.EVT_BUTTON, lambda e: self._what_if())
        s.Add(b, 0, wx.ALL, 6)
        self.wtab = AutoList(p, style=wx.LC_REPORT)
        for k, t in enumerate(("Removed", "Worst |Z| rise", "At", "|Z| there", "Target still met")):
            self.wtab.InsertColumn(k, t, width=100)
        s.Add(self.wtab, 1, wx.EXPAND | wx.ALL, 6)
        p.SetSizer(s)
        self.nb.AddPage(p, "What if")
        # load steps
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.wload = wx.Choice(p)
        self.wload.SetToolTip("The load whose supply voltage is plotted (split by which load's steps cause it)")
        self.wload.Bind(wx.EVT_CHOICE, lambda e: self._plot_waves())
        b = wx.Button(p, label="Save CSV…", style=wx.BU_EXACTFIT)
        b.SetToolTip("The currents and the voltage at every load over time, with each load's contribution")
        b.Bind(wx.EVT_BUTTON, lambda e: self._waves_csv())
        self.winfo = wx.StaticText(p, label="")
        row.AddMany([(wx.StaticText(p, label="Voltage at"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.wload, 0, wx.RIGHT, 8), (b, 0, wx.RIGHT, 12), (self.winfo, 1, wx.ALIGN_CENTER_VERTICAL)])
        s.Add(row, 0, wx.EXPAND | wx.ALL, 6)
        self.wfig = Figure(figsize=(8, 5), constrained_layout=True)
        self.wcanvas = FigureCanvasWxAgg(p, -1, self.wfig)
        s.Add(self.wcanvas, 1, wx.EXPAND)
        s.Add(NavigationToolbar2WxAgg(self.wcanvas), 0, wx.EXPAND)
        p.SetSizer(s)
        self.nb.AddPage(p, "Load steps")
        # map
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.mapf = wx.ComboBox(p, size=(160, -1))
        self.mapf.SetToolTip("Frequency of the map: a resonance peak from the list, or type one (e.g. 120M)")
        b = wx.Button(p, label="Show", style=wx.BU_EXACTFIT)
        b.SetToolTip("Draw the rail's voltage per 1 A at the load at this frequency: hot spots of a plane "
                     "resonance show where a decap would help")
        b.Bind(wx.EVT_BUTTON, lambda e: self._map())
        row.AddMany([(wx.StaticText(p, label="Frequency"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.mapf, 0, wx.RIGHT, 6), (b,)])
        s.Add(row, 0, wx.ALL, 6)
        self.mfig = Figure(figsize=(8, 5), constrained_layout=True)
        self.mcanvas = FigureCanvasWxAgg(p, -1, self.mfig)
        s.Add(self.mcanvas, 1, wx.EXPAND)
        s.Add(NavigationToolbar2WxAgg(self.mcanvas), 0, wx.EXPAND)
        p.SetSizer(s)
        self.nb.AddPage(p, "Map")
        # layout checks
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.ltab = AutoList(p, style=wx.LC_REPORT, stretch=2)
        for k, t in enumerate(("", "What", "Details", "Where [mm]")):
            self.ltab.InsertColumn(k, t, width=90)
        self.ltab.SetToolTip("Select a line to highlight its part in KiCad; double-click one with a place to "
                             "mark it on the Map")
        self.ltab.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._layout_go(e.GetIndex()))
        s.Add(self.ltab, 1, wx.EXPAND | wx.ALL, 6)
        self.ltext = wx.TextCtrl(p, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 70))
        self.ltab.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda e: self.ltext.SetValue(
            self.res.layout[e.GetIndex()].text if self.res and e.GetIndex() < len(self.res.layout) else ""))
        s.Add(self.ltext, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        s.Add(wx.StaticText(p, label="What in the layout raises the rail's impedance: copper without return under "
                                     "it and voids or slots in the return (ranked by how much inductance they add, "
                                     "with 1 A at each load and every part connected), decaps with long return "
                                     "paths, loads with no decap close by. Which decaps could go: What if."),
              0, wx.ALL, 6)
        p.SetSizer(s)
        self.nb.AddPage(p, "Layout")
        # summary
        self.summary = wx.TextCtrl(self.nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.nb.AddPage(self.summary, "Summary")
        rs.Add(self.nb, 1, wx.EXPAND)
        right.SetSizer(rs)
        sp.SplitVertically(left, right, 470)
        sp.SetMinimumPaneSize(300)

    # ------------------------------------------------------------- setup <-> UI
    def _to_ui(self):
        s, b = self.setup, self.board
        cands = PM.rail_candidates(b)
        names = cands + sorted((n for n in b.nets if n and n not in cands), key=EL.natural_key)
        self.rail_names = names
        self.railc.Set([n + ("  (decaps)" if n in cands else "") for n in names])
        if s.rail and s.rail[0] in names:
            self.railc.SetSelection(names.index(s.rail[0]))
        rets = [n for n in b.nets if n and n not in s.rail]
        g = [n for n in rets if n in s.returns]
        self.ret_names = g + sorted((n for n in rets if n not in g), key=EL.natural_key)
        self.rets.Set(self.ret_names)
        for k, n in enumerate(self.ret_names):
            self.rets.Check(k, n in s.returns)
        for key, (t, unit) in self.tf.items():
            v = getattr(s.target, key)
            t.ChangeValue(fmt_si(v, unit) if unit in ("s", "A") else f"{v:g}")
        self._target_label()
        v = s.vrm
        self.vlabel.SetLabel((f"{v.ref}: {', '.join(v.pads)}" if v.pads else "None picked") + (f"  ({v.how})" if v.how else ""))
        self.vlabel.Wrap(420)
        for key, (t, unit) in self.vf.items():
            t.ChangeValue(fmt_si(getattr(v, key), unit))
        self.von.SetValue(v.enabled)
        self.vkind.SetSelection(1 if v.kind == "loop" else 0)
        self._vkind()
        self._fill_caps()
        self.llist.DeleteAllItems()
        for ld in s.loads:
            self.llist.Append(["✓" if ld.enabled else "–", ld.name, ", ".join(p.rsplit(".", 1)[1] for p in ld.pads),
                               fmt_si(ld.i_step, "A"), fmt_si(ld.delay, "s"), fmt_si(ld.t_rise, "s"),
                               "stays on" if ld.width < 0 else fmt_si(ld.width, "s"), fmt_si(ld.t_fall, "s"),
                               fmt_si(ld.period, "s") if ld.period > 0 else "once"])
        self.tran.SetValue(s.transient)
        self.tstop.ChangeValue(fmt_si(s.t_stop, "s") if s.t_stop > 0 else "")
        for key, (t, unit) in self.of.items():
            val = getattr(s, key)
            t.ChangeValue(fmt_si(val, unit) if unit else f"{val:g}")
        refit(self.left)

    def _fill_caps(self):
        keep = {self.setup.decaps[k].ref for k in self._dsel()} if self.dlist.GetItemCount() else set()
        self.dlist.DeleteAllItems()
        for d in self.setup.decaps:
            p = part_from_dict(d.part) if d.part else None
            i = self.dlist.Append(["✓" if d.enabled else "–", d.ref, fmt_si(p.value, "F") if p else "?",
                                   (p.name if p else "no model"), fmt_si(p.esr_eff(), "Ω") if p else "",
                                   fmt_si(p.esl_eff(), "H") if p else ""])
            if d.ref in keep:
                self.dlist.Select(i)

    def _vkind(self):
        v = self.setup.vrm
        loop = self.vkind.GetSelection() == 1
        if loop and not v.model:
            self.vkind.SetSelection(0)
            loop = False
            self._status("No fitted loop yet: use 'Fit from the datasheet plot…'.")
        for t, _u in self.vf.values():
            t.Enable(not loop)
        if v.model:
            from ..pdn.regulator import TYPES, RegulatorModel
            m = RegulatorModel(**v.model)
            lp = m.loop()
            self.vmodel.SetLabel(f"Fitted: {m.part or 'regulator'}, {TYPES.get(m.kind, ('?',))[0]}; crossover "
                                 f"{fmt_si(lp.f_cross(m.c_test), 'Hz')} with the datasheet's "
                                 f"{fmt_si(m.c_test, 'F')}" if lp else "")
            self.vmodel.Wrap(420)
        else:
            self.vmodel.SetLabel("")
        self._fill_saved()
        refit(self.left)

    def _store(self):
        return RS.path_for(self.session.library.path)

    def _fill_saved(self):
        models = RS.load(self._store())
        near = {k: ref for k, ref in reversed(RS.on_board(self.board, self.setup, models))}
        keys = sorted(near, key=lambda k: near[k]) + sorted((k for k in models if k not in near), key=str.lower)
        self._saved_keys = keys
        self.vsaved.Set(["—"] + [f"{k} ({near[k]})" if k in near else k for k in keys])
        cur = (self.setup.vrm.model or {}).get("part", "")
        k = next((i for i, n in enumerate(keys) if n.lower() == cur.lower()), -1)
        self.vsaved.SetSelection(k + 1)

    def _use_saved(self):
        k = self.vsaved.GetSelection() - 1
        if k < 0:
            return
        key = self._saved_keys[k]
        m = RS.find(RS.load(self._store()), key)
        if m is None:
            return
        self.setup.vrm.model = dict(m)
        self.setup.vrm.kind = "loop"
        self.res = None
        self.vkind.SetSelection(1)
        self._vkind()
        self.save()
        self._status(f"The regulator is now {key}'s saved fit: Analyse again.")

    def _save_fit(self):
        v = self.setup.vrm
        if not v.model:
            return wx.MessageBox("No fitted loop to save: use 'Fit from the datasheet plot…' first.", "Save fit",
                                 parent=self)
        fp = next((f for f in self.board.footprints if f.ref == v.ref), None)
        default = v.model.get("part") or (fp.value if fp and not v.ref.startswith("L") else "")
        with wx.TextEntryDialog(self, "Part number (as in the footprint's Value field):", "Save fit",
                                default) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            key = d.GetValue().strip()
        if not key:
            return
        path = self._store()
        if RS.find(RS.load(path), key) is not None and wx.MessageBox(
                f"Replace the saved fit of {key}?", "Save fit", wx.YES_NO, parent=self) != wx.YES:
            return
        RS.save(path, key, {**v.model, "part": key})
        v.model = {**v.model, "part": key}
        self.save()
        self._vkind()
        self._fill_saved()
        self._status(f"Saved the fit as {key} in {path}.")

    def _forget_fit(self):
        k = self.vsaved.GetSelection() - 1
        if k < 0:
            return
        key = self._saved_keys[k]
        if wx.MessageBox(f"Remove the saved fit of {key} from the store?", "Forget", wx.YES_NO,
                         parent=self) != wx.YES:
            return
        RS.remove(self._store(), key)
        self._fill_saved()
        self._status(f"Removed {key}'s saved fit (this rail keeps its loop).")

    def _regfit(self):
        from .regfit import RegFitFrame
        v = self.setup.vrm
        fp = next((f for f in self.board.footprints if f.ref == v.ref), None)
        part = fp.value if fp and not v.ref.startswith("L") else ""

        def use(model):
            self.setup.vrm.model = model.to_dict()
            self.setup.vrm.kind = "loop"
            self.res = None
            self.vkind.SetSelection(1)
            self._vkind()
            self.save()
            self._status("The regulator is now the fitted loop: Analyse again.")
        w = RegFitFrame(self, v.model or None, use, part)
        w.Show()

    def _target_label(self):
        try:
            t = PM.Target(**{k: _num(c.GetValue(), u) for k, (c, u) in self.tf.items()})
            self.tlabel.SetLabel(f"Target impedance {fmt_si(t.z, 'Ω')} up to {fmt_si(t.f_max, 'Hz')}")
        except (ValueError, ZeroDivisionError):
            self.tlabel.SetLabel("Enter numbers (SI prefixes allowed: 10m, 1u)")
        refit(self.left)

    def _from_ui(self, quiet=False):
        s = self.setup
        try:
            k = self.railc.GetSelection()
            if k >= 0:
                s.rail = [self.rail_names[k]]
            s.returns = [self.ret_names[i] for i in self.rets.GetCheckedItems()]
            for key, (c, u) in self.tf.items():
                setattr(s.target, key, _num(c.GetValue(), u))
            for key, (c, u) in self.vf.items():
                setattr(s.vrm, key, _num(c.GetValue(), u))
            s.vrm.enabled = self.von.GetValue()
            s.vrm.kind = "loop" if self.vkind.GetSelection() == 1 and s.vrm.model else "RL"
            for key, (c, u) in self.of.items():
                v = _num(c.GetValue(), u)
                setattr(s, key, int(v) if key in ("n_f", "cells") else v)
            s.transient = self.tran.GetValue()
            s.t_stop = _num(self.tstop.GetValue(), "s") if self.tstop.GetValue().strip() else 0.0
        except (ValueError, TypeError) as e:
            if not quiet:
                wx.MessageBox(f"Check the numbers: {e}", "PDN", parent=self)
            return False
        if s.f_min <= 0 or s.f_max <= s.f_min or s.n_f < 5 or s.cells < 200:
            if not quiet:
                wx.MessageBox("Need 0 < f min < f max, at least 5 points and 200 cells.", "PDN", parent=self)
            return False
        return True

    # ------------------------------------------------------------- actions
    def _detect(self, keep=False):
        k = self.railc.GetSelection()
        if k < 0:
            return
        self._from_ui(quiet=True)
        old = self.setup
        new = PM.detect(self.board, self.rail_names[k], self.session.components)
        if keep and new.rail == old.rail:
            new.decaps = PM.find_decaps(self.board, new.rail, old.returns or new.returns, self.session.components,
                                        old=old.decaps)
            new.returns = old.returns or new.returns
        new.target = old.target if keep else new.target
        for key in ("f_min", "f_max", "n_f", "cells", "pitch", "margin", "T"):
            setattr(new, key, getattr(old, key))
        if keep and new.vrm.ref == old.vrm.ref and old.vrm.model:
            new.vrm.model, new.vrm.kind = old.vrm.model, old.vrm.kind
        used = RS.apply(self.board, new, RS.load(self._store()))
        self.setup, self.res, self.whatif = new, None, None
        self._to_ui()
        self._status(f"{len(new.decaps)} decaps, {len(new.loads)} loads on {new.rail[0]}; regulator: "
                     f"{new.vrm.ref or 'not found'}" + (f", with {used}'s saved fit." if used else "."))

    def _dsel(self):
        return [i for i in selected_rows(self.dlist) if i < len(self.setup.decaps)]

    def _share(self, d):
        """A decap's model goes into the board's shared component list (Switching Loop, Trace, Components)."""
        pads = d.pads + d.ret
        if len(pads) < 2:
            return
        old = self.session.components.get(d.ref)
        if old is not None and old.model != "C":
            return
        e = EL.Element(d.ref, d.pads[0], d.ret[0], 0.0, model="C", C=float(d.part.get("value") or 0.0),
                       part=dict(d.part))
        if old is not None:
            e.a, e.b = old.a, old.b
        self.session.put_component("pdn", e)

    def _edit_cap(self):
        sel = self._dsel()
        if not sel:
            return
        d = self.setup.decaps[sel[0]]
        from .suite import PartDialog
        p = part_from_dict(d.part) if d.part else Part("C", d.ref, 1e-6)
        with PartDialog(self, p, set()) as dlg:
            if dlg.ShowModal() == wx.ID_OK and dlg.result:
                d.part = asdict(dlg.result)
                self._share(d)
                self._fill_caps()

    def _lib_cap(self):
        sel = self._dsel()
        if not sel:
            return
        caps = self.session.library.by_kind("C")
        if not caps:
            return wx.MessageBox("The component library has no capacitors.", "Library", parent=self)
        with wx.SingleChoiceDialog(self, "Library part for " + ", ".join(self.setup.decaps[k].ref for k in sel),
                                   "From library", [f"{c.name}  ({fmt_si(c.value, 'F')})" for c in caps]) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            c = caps[dlg.GetSelection()]
        for k in sel:
            self.setup.decaps[k].part = dict(asdict(c), builtin=False)
            self._share(self.setup.decaps[k])
        self._fill_caps()

    def _toggle_caps(self):
        sel = self._dsel()
        on = not all(self.setup.decaps[k].enabled for k in sel)
        for k in sel:
            self.setup.decaps[k].enabled = on
        self._fill_caps()

    def _pick_vrm(self):
        from .extract import PadDialog
        net = self.setup.rail[0] if self.setup.rail else None
        with PadDialog(self, self.board, net) as dlg:
            dlg.SetTitle("Regulator output pad")
            if dlg.ShowModal() != wx.ID_OK or not dlg.result:
                return
            pid = dlg.result
        ref = pid.rsplit(".", 1)[0]
        pads = [p.id for p in self.board.pads if p.ref == ref and p.net in self.setup.rail and p.layers]
        self.setup.vrm.ref, self.setup.vrm.pads, self.setup.vrm.how = ref, pads, "picked"
        self.setup.loads = [ld for ld in self.setup.loads if ld.name != ref]
        self._to_ui()

    def _add_load(self):
        on = sorted({p.ref for p in self.board.pads if p.net in self.setup.rail and p.layers}
                    - {ld.name for ld in self.setup.loads}, key=EL.natural_key)
        if not on:
            return wx.MessageBox("No other part has pins on the rail.", "Loads", parent=self)
        with wx.SingleChoiceDialog(self, "Part drawing current from the rail:", "Add load", on) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            ref = on[dlg.GetSelection()]
        pads = [p.id for p in self.board.pads if p.ref == ref and p.net in self.setup.rail and p.layers]
        self.setup.loads.append(PM.Load(ref, pads))
        self._to_ui()

    def _edit_pulse(self):
        sel = [i for i in selected_rows(self.llist) if i < len(self.setup.loads)]
        if sel:
            with PulseDialog(self, self.setup.loads[sel[0]]) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    self._to_ui()

    def _stagger(self):
        """The enabled loads one after another, each starting when the previous one's step is over."""
        on = [ld for ld in self.setup.loads if ld.enabled]
        if not on:
            return
        self._from_ui(quiet=True)
        t = self.setup.target
        gap = 300e-6
        for k, ld in enumerate(on):
            ld.i_step, ld.t_rise, ld.t_fall = t.i_step, t.t_rise, t.t_rise
            ld.width, ld.period = 200e-6, 0.0
            ld.delay = 20e-6 + k * (ld.width + ld.t_rise + ld.t_fall + gap)
        self._to_ui()
        self._status(f"{len(on)} loads staggered: each a {fmt_si(t.i_step, 'A')} step (the target's), 200 µs long, "
                     "500 µs apart.")

    def _toggle_loads(self):
        sel = [i for i in selected_rows(self.llist) if i < len(self.setup.loads)]
        on = not all(self.setup.loads[i].enabled for i in sel)
        for i in sel:
            self.setup.loads[i].enabled = on
        self._to_ui()

    def _del_loads(self):
        sel = set(selected_rows(self.llist))
        self.setup.loads = [ld for i, ld in enumerate(self.setup.loads) if i not in sel]
        self._to_ui()

    # ------------------------------------------------------------- run
    def test_setup(self):
        if not self._from_ui():
            return None
        if not self.setup.rail or not self.setup.returns:
            wx.MessageBox("Pick the rail and its return nets first.", "Save as test", parent=self)
            return None
        self.save()
        return "pdn", self.setup.to_dict(), f"PDN {'+'.join(self.setup.rail)}"

    def _run(self):
        if self._running or not self._from_ui():
            return
        if not self.setup.rail:
            return wx.MessageBox("Pick the rail.", "PDN", parent=self)
        if not self.setup.returns:
            return wx.MessageBox("Tick the return (ground) nets.", "PDN", parent=self)
        self.save()
        self._running = True
        self.run_b.Disable()
        board, st, s = self.board, self.session.stackup, PM.Setup.from_dict(self.setup.to_dict())

        def work():
            try:
                r = PA.run(board, st, s, progress=lambda m: wx.CallAfter(self._status, m))
                wx.CallAfter(self._done, r, None, s)
            except ValueError as e:
                wx.CallAfter(self._done, None, str(e), s)
            except Exception:       # noqa: BLE001
                wx.CallAfter(self._done, None, traceback.format_exc(), s)
        threading.Thread(target=work, daemon=True).start()

    def _done(self, res, err, s):
        self._running = False
        self.run_b.Enable()
        if err:
            self._status("Failed.")
            return wx.MessageBox(err, "PDN", parent=self)
        self.res, self.res_setup, self.whatif = res, s, None
        self._status(f"Done in {res.seconds:.1f} s ({res.nodes} nodes at {res.pitch:.2f} mm).")
        self._show()

    def _show(self):
        r = self.res
        self.loadc.Set(r.loads)
        self.loadc.SetSelection(0)
        self._plot_z()
        self.dtab.DeleteAllItems()
        for row in r.decaps:
            band = row.get("band")
            self.dtab.Append([row["ref"] + ("" if row["enabled"] else " (off)"), fmt_si(row.get("c_eff", 0), "F"),
                              fmt_si(row.get("esr", 0), "Ω"), fmt_si(row.get("esl", 0), "H"),
                              fmt_si(row.get("l_mount", 0), "H"), fmt_si(row.get("f_srf", math.nan), "Hz"),
                              f"{row['dist_mm']:.1f}" if np.isfinite(row["dist_mm"]) else "",
                              (f"{100 * row['i_peak']:.0f} % at {fmt_si(row['f_peak'], 'Hz')}"
                               if "i_peak" in row else "–"),
                              (f"{fmt_si(band[0], 'Hz')} – {fmt_si(band[1], 'Hz')}" if band else "–"),
                              row.get("mount", "")])
        self.wload.Set(r.waves.loads if r.waves else [])
        if r.waves:
            self.wload.SetSelection(0)
        self._plot_waves()
        pk = sorted({f for v in r.peaks.values() for f, _ in v})
        self.mapf.Set([fmt_si(f, "Hz") for f in pk])
        if pk:
            self.mapf.SetSelection(0)
        self.ltab.DeleteAllItems()
        for i in r.layout:
            self.ltab.Append(["⚠" if i.level == "warn" else "ⓘ", i.kind, i.text,
                              f"{i.at[0]:.1f}, {i.at[1]:.1f}" if i.at else ""])
        self.mark = None
        self.summary.SetValue(self._summary())

    def _summary(self):
        r, s = self.res, self.res_setup
        out = [f"Rail {', '.join(s.rail)} over {', '.join(s.returns)}: target {fmt_si(r.z_target, 'Ω')} "
               f"({s.target.v:g} V ± {s.target.ripple:g} % for a {s.target.i_step:g} A step) up to "
               f"{fmt_si(r.f_target, 'Hz')}.", ""]
        for c in r.checks:
            if c["ok"]:
                out.append(f"✔ {c['load']}: meets the target (worst {fmt_si(c['worst_z'], 'Ω')} at "
                           f"{fmt_si(c['worst_f'], 'Hz')}, {100 * c['ratio']:.0f} % of it).")
            else:
                dv = c["worst_z"] * s.target.i_step
                out.append(f"✖ {c['load']}: {fmt_si(c['worst_z'], 'Ω')} at {fmt_si(c['worst_f'], 'Hz')}, "
                           f"{c['ratio']:.1f}× the target: a {s.target.i_step:g} A step with energy there moves the "
                           f"rail by about {fmt_si(dv, 'V')}.")
        if r.wave_checks:
            out += ["", "Load steps over time (each load's Step … Period):"]
            for c in r.wave_checks:
                out.append(f"{'✔' if c['ok'] else '✖'} {c['load']}: dips {fmt_si(-c['dip'], 'V')}, rises "
                           f"{fmt_si(c['peak'], 'V')} (allowed ± {fmt_si(c['allowed'], 'V')}); the worst moment, at "
                           f"{fmt_si(c['t_worst'], 's')}, is {100 * c['share']:.0f} % caused by {c['worst_from']}'s step.")
        warn = [i for i in r.layout if i.level == "warn"]
        if warn:
            out += ["", f"Layout ({len(warn)}, see the Layout tab):"] + [f"⚠ {i.text}" for i in warn[:8]]
        out.append("")
        for name, pk in r.peaks.items():
            if pk:
                out.append(f"Resonance peaks at {name}: " + ", ".join(f"{fmt_si(f, 'Hz')} ({fmt_si(z, 'Ω')})"
                                                                       for f, z in pk[:4]))
        n_on = sum(1 for d in s.decaps if d.enabled)
        out += ["", f"{n_on} of {len(s.decaps)} decaps on; regulator "
                + ((f"{s.vrm.ref or 'pad'} (fitted control loop, rising-step fit)" if s.vrm.kind == "loop" else
                    f"{s.vrm.ref or 'pad'} (R {fmt_si(s.vrm.r_out, 'Ω')}, L {fmt_si(s.vrm.l_out, 'H')})")
                   if s.vrm.pads and s.vrm.enabled else "left out") + ".",
                f"Mesh: {r.nodes} nodes at {r.pitch:.2f} mm; {len(r.f)} frequencies in {r.seconds:.1f} s.",
                "The return copper is the ideal reference (no ground bounce); capacitors at the rail voltage "
                "(DC bias) and their mounting L on the return side are included."]
        if r.warnings:
            out += ["", "Warnings:"] + ["  - " + w for w in r.warnings]
        return "\n".join(out)

    def _plot_z(self):
        r = self.res
        if r is None:
            return
        j = max(self.loadc.GetSelection(), 0)
        fig = self.zfig
        fig.clf()
        ax = fig.add_subplot(111)
        for Z, lab, sty in ((r.Z_bare, "board alone", dict(color="0.6", lw=1)),
                            (r.Z_caps, "board + decaps", dict(color="tab:blue", lw=1, ls="--")),
                            (r.Z, "board + decaps + regulator", dict(color="tab:blue", lw=2))):
            ax.loglog(r.f, np.abs(Z[:, j, j]), label=lab, **sty)
        ft = min(r.f_target, r.f[-1])
        ax.hlines(r.z_target, r.f[0], ft, colors="tab:red", linestyles="-", lw=1.5,
                  label=f"target {fmt_si(r.z_target, 'Ω')}")
        ax.axvline(r.f_target, color="tab:red", ls=":", lw=1)
        c = r.checks[j]
        ax.plot([c["worst_f"]], [c["worst_z"]], "o", color="tab:red" if not c["ok"] else "tab:green", ms=7)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|Z| at the load [Ω]")
        ax.grid(True, which="both", alpha=0.3)
        lo = max(np.abs(r.Z[:, j, j]).min() / 3, 1e-5)
        ax.set_ylim(lo, max(np.abs(r.Z_bare[:, j, j]).max(), r.z_target) * 2)
        ax.legend(loc="best", fontsize=9)
        self.zcanvas.draw()
        self.zinfo.SetLabel(("✔ meets the target" if c["ok"] else
                             f"✖ {c['ratio']:.1f}× the target at {fmt_si(c['worst_f'], 'Hz')}"))

    def _plot_waves(self):
        fig = self.wfig
        fig.clf()
        r = self.res
        w = r.waves if r is not None else None
        if w is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Tick 'Simulate the load steps' and Analyse", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.wcanvas.draw()
            return
        at = w.loads[max(self.wload.GetSelection(), 0)]
        t = w.t * 1e6
        gs = fig.add_gridspec(3, 1)
        a1 = fig.add_subplot(gs[0, 0])
        for n in w.loads:
            a1.plot(t, w.i[n], lw=1.2, label=n)
        a1.set_ylabel("Load current change [A]")
        a1.legend(fontsize=8, loc="upper right")
        a1.grid(alpha=0.3)
        a2 = fig.add_subplot(gs[1:, 0], sharex=a1)
        allowed = r.wave_checks[0]["allowed"] if r.wave_checks else None
        if len(w.loads) > 1:
            for n in w.loads:
                a2.plot(t, w.parts[(at, n)] * 1e3, lw=0.9, alpha=0.8, label=f"from {n}'s steps")
        a2.plot(t, w.v[at] * 1e3, color="k", lw=1.6, label=f"at {at} (total)")
        if allowed:
            for sgn in (1, -1):
                a2.axhline(sgn * allowed * 1e3, color="tab:red", ls="--", lw=1)
        a2.set_xlabel("Time [µs]")
        a2.set_ylabel(f"ΔV at {at} [mV]")
        a2.legend(fontsize=8, loc="lower right")
        a2.grid(alpha=0.3)
        self.wcanvas.draw()
        c = next((c for c in r.wave_checks if c["load"] == at), None)
        if c:
            self.winfo.SetLabel(f"{'✔' if c['ok'] else '✖'} dips {fmt_si(-c['dip'], 'V')}, rises "
                                f"{fmt_si(c['peak'], 'V')} (allowed ± {fmt_si(c['allowed'], 'V')})"
                                + (f"; fit error {100 * w.fit_err:.1f} %" if w.fit_err > 0.02 else ""))

    def _waves_csv(self):
        w = self.res.waves if self.res is not None else None
        if w is None:
            return wx.MessageBox("Simulate the load steps first.", "CSV", parent=self)
        with wx.FileDialog(self, "Save the load-step waveforms", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            cols = [("t [s]", w.t)] + [(f"I {n} [A]", w.i[n]) for n in w.loads] + \
                [(f"V at {n} [V]", w.v[n]) for n in w.loads] + \
                [(f"V at {a} from {b} [V]", w.parts[(a, b)]) for a in w.loads for b in w.loads]
            with open(d.GetPath(), "w") as fh:
                fh.write(",".join(c for c, _ in cols) + "\n")
                for k in range(len(w.t)):
                    fh.write(",".join(f"{v[k]:.6g}" for _, v in cols) + "\n")
            self._status(f"Saved {d.GetPath()}")

    def _touchstone(self):
        if self.res is None or self._running:
            return wx.MessageBox("Analyse first.", "Touchstone", parent=self)
        d = TouchstoneDialog(self)
        if d.ShowModal() != wx.ID_OK:
            return d.Destroy()
        what, z0 = d.what(), d.z0
        d.Destroy()
        n = len(self.res.loads) if what == "loads" else None
        ext = f"s{n}p" if n else "snp"
        with wx.FileDialog(self, "Save the rail's impedance", wildcard=f"Touchstone (*.{ext})|*.{ext}",
                           defaultFile=f"{'_'.join(self.res_setup.rail)}_{what}.{ext}",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() != wx.ID_OK:
                return
            path = fd.GetPath()
        self._running = True
        board, st, s, r = self.board, self.session.stackup, self.res_setup, self.res

        def work():
            try:
                out = PA.touchstone(board, st, s, r, path, what, z0,
                                    progress=lambda m: wx.CallAfter(self._status, m))
                wx.CallAfter(self._touchstone_done, out, None)
            except Exception:       # noqa: BLE001
                wx.CallAfter(self._touchstone_done, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _touchstone_done(self, out, err):
        self._running = False
        if err:
            return wx.MessageBox(err, "Touchstone", parent=self)
        path, names = out
        self._status(f"Saved {path}: {len(names)} port{'s' * (len(names) != 1)} ({', '.join(names)})")

    def _what_if(self):
        if self.res is None or self._running:
            return wx.MessageBox("Analyse first.", "What if", parent=self)
        self._running = True
        board, st, s, r = self.board, self.session.stackup, self.res_setup, self.res

        def work():
            try:
                w = PA.what_if(board, st, s, r, progress=lambda m: wx.CallAfter(self._status, m))
                wx.CallAfter(self._what_done, w, None)
            except Exception:       # noqa: BLE001
                wx.CallAfter(self._what_done, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _what_done(self, w, err):
        self._running = False
        if err:
            return wx.MessageBox(err, "What if", parent=self)
        self.whatif = w
        self.wtab.DeleteAllItems()
        for ref, v in sorted(w.items(), key=lambda kv: -kv[1]["rise"]):
            self.wtab.Append([ref, f"×{v['rise']:.2f}", fmt_si(v["f"], "Hz"), fmt_si(v["z"], "Ω"),
                              "no" if v["over"] else "yes"])
        self._status("What-if done: capacitors whose removal barely changes |Z| (×1.0x) and keeps the target are "
                     "candidates to drop.")

    def _layout_go(self, k):
        if self.res is None or k >= len(self.res.layout) or self.res.layout[k].at is None:
            return
        self.mark = self.res.layout[k].at
        if not self.mapf.GetValue():
            self.mapf.SetValue(fmt_si(self.res.f_target, "Hz"))
        for q in range(self.nb.GetPageCount()):
            if self.nb.GetPageText(q) == "Map":
                self.nb.SetSelection(q)
        self._map()

    def _map(self):
        if self.res is None:
            return wx.MessageBox("Analyse first.", "Map", parent=self)
        try:
            f = _num(self.mapf.GetValue(), "Hz")
        except ValueError:
            return wx.MessageBox("Enter a frequency, e.g. 120M.", "Map", parent=self)
        j = max(self.loadc.GetSelection(), 0)
        v = PA.node_voltages(self.board, self.session.stackup, self.res_setup, self.res, f, j)
        m = self.res.net.mesh
        fig = self.mfig
        fig.clf()
        cnt = np.bincount(m.node_layer, minlength=len(m.layers))
        lays = [k for k in range(len(m.layers)) if cnt[k] >= max(50, 0.05 * cnt.max())]   # not via specks
        for q, k in enumerate(lays):
            ax = fig.add_subplot(1, len(lays), q + 1)
            sel = m.node_layer == k
            sc = ax.scatter(m.node_xy[sel, 0], m.node_xy[sel, 1], c=v[sel] * 1e3, s=4, cmap="inferno",
                            vmin=0, vmax=v.max() * 1e3)
            ax.set_title(m.layers[k], fontsize=9)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)
        if self.mark is not None:
            for ax in fig.axes:
                ax.plot(*self.mark, "o", ms=16, mfc="none", mec="cyan", mew=2)
        fig.colorbar(sc, ax=fig.axes, label="|V| per 1 A at the load [mV]", shrink=0.8)
        fig.suptitle(f"{self.res.loads[j]} at {fmt_si(f, 'Hz')}", fontsize=10)
        self.mcanvas.draw()
