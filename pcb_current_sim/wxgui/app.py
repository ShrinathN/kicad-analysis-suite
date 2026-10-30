"""wxPython main window (used inside KiCad, whose Python ships wx but not Tk)."""
import os
import threading
import traceback

import wx

from .. import elements as EL
from .. import solver as S
from ..scenario import load_scenario, save_scenario, scenario_path
from ..suite.session import Session
from ..tables import CHECKS, FLOAT_SETTINGS, ROW_TEXT


def _list(parent, cols):
    lc = wx.ListCtrl(parent, style=wx.LC_REPORT)
    for k, (title, w) in enumerate(cols):
        lc.InsertColumn(k, title, width=w)
    return lc


def _selected(lc):
    out, i = [], lc.GetFirstSelected()
    while i != -1:
        out.append(i)
        i = lc.GetNextSelected(i)
    return out


class RowDialog(wx.Dialog):
    """Edit one excitation row: source pads, sink pads (each side on one net), current."""

    def __init__(self, parent, board, row, default_name):
        super().__init__(parent, title="Excitation", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.board, self.result = board, None
        self.nets = [n for n in board.nets if board.pad_ids(n)]
        row = row or S.Row(default_name, [], [], 1.0)

        def net_of(ids, other=()):
            ps = board.pads_by_id((ids or other or [None])[0]) if (ids or other) else []
            return ps[0].net if ps else (self.nets[0] if self.nets else "")

        self.name = wx.TextCtrl(self, value=row.name)
        self.cur = wx.TextCtrl(self, value=f"{row.current:g}", size=(90, -1))
        self.kinds = list(S.KINDS)
        self.kind = wx.Choice(self, choices=[S.KINDS[k] for k in self.kinds])
        self.kind.SetSelection(self.kinds.index(row.kind))
        self.kind.Bind(wx.EVT_CHOICE, lambda e: self._labels())
        self.nsrc, self.nsnk = wx.Choice(self, choices=self.nets), wx.Choice(self, choices=self.nets)
        for ch, n0 in ((self.nsrc, net_of(row.src, row.snk)), (self.nsnk, net_of(row.snk, row.src))):
            if n0 in self.nets:
                ch.SetSelection(self.nets.index(n0))
        self.lsrc = wx.ListBox(self, style=wx.LB_EXTENDED, size=(220, 240))
        self.lsnk = wx.ListBox(self, style=wx.LB_EXTENDED, size=(220, 240))
        self.nsrc.Bind(wx.EVT_CHOICE, lambda e: self._fill_one(self.nsrc, self.lsrc))
        self.nsnk.Bind(wx.EVT_CHOICE, lambda e: self._fill_one(self.nsnk, self.lsnk))

        self.vlabel = wx.StaticText(self, label="Current [A]")
        self.slabel, self.klabel = wx.StaticText(self, label=""), wx.StaticText(self, label="")
        self.hint = wx.StaticText(self, label="")
        g = wx.FlexGridSizer(2, 4, 6, 8)
        g.AddMany([(wx.StaticText(self, label="Name"), 0, wx.ALIGN_CENTER_VERTICAL), (self.name, 1, wx.EXPAND),
                   (self.vlabel, 0, wx.ALIGN_CENTER_VERTICAL), (self.cur,),
                   (wx.StaticText(self, label="Kind"), 0, wx.ALIGN_CENTER_VERTICAL), (self.kind, 1, wx.EXPAND),
                   ((0, 0),), ((0, 0),)])
        g.AddGrowableCol(1)
        lists = wx.FlexGridSizer(3, 2, 4, 8)
        lists.AddMany([(self.slabel,), (self.klabel,),
                       (self.nsrc, 0, wx.EXPAND), (self.nsnk, 0, wx.EXPAND),
                       (self.lsrc, 1, wx.EXPAND), (self.lsnk, 1, wx.EXPAND)])
        lists.AddGrowableRow(2)
        lists.AddGrowableCol(0)
        lists.AddGrowableCol(1)
        top = wx.BoxSizer(wx.VERTICAL)
        top.Add(g, 0, wx.EXPAND | wx.ALL, 10)
        top.Add(lists, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        top.Add(self.hint, 0, wx.ALL, 10)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        self._fill(row.src, row.snk)
        self._labels()

    def _labels(self):
        k = self.kinds[self.kind.GetSelection()]
        a, b, unit, hint = ROW_TEXT[k]
        self.slabel.SetLabel(a)
        self.klabel.SetLabel(b)
        self.vlabel.SetLabel(unit)
        self.hint.SetLabel(hint + "\nSeveral pads on one side are tied together (equipotential). "
                                  "Ctrl/Shift-click to select several.")
        self.Layout()

    def _fill(self, src=(), snk=()):
        self._fill_one(self.nsrc, self.lsrc, src)
        self._fill_one(self.nsnk, self.lsnk, snk)

    def _fill_one(self, ch, lb, sel=()):
        k = ch.GetSelection()
        ids = self.board.pad_ids(self.nets[k]) if k != wx.NOT_FOUND else []
        lb.Set(ids)
        for i, pid in enumerate(ids):
            if pid in sel:
                lb.SetSelection(i)

    def _ok(self, e):
        src = [self.lsrc.GetString(i) for i in self.lsrc.GetSelections()]
        snk = [self.lsnk.GetString(i) for i in self.lsnk.GetSelections()]
        try:
            cur = float(self.cur.GetValue())
        except ValueError:
            return wx.MessageBox("The value must be a number.", "Excitation", parent=self)
        if not src or not snk:
            return wx.MessageBox("Select at least one source and one sink pad.", "Excitation", parent=self)
        if set(src) & set(snk):
            return wx.MessageBox("A pad cannot be both source and sink.", "Excitation", parent=self)
        self.result = S.Row(self.name.GetValue().strip() or "row", src, snk, cur,
                            self.kinds[self.kind.GetSelection()])
        self.EndModal(wx.ID_OK)





MODEL_LABELS = {"R": "Resistance", "V": "Constant voltage drop (diode, A → B)", "C": "Capacitor", "L": "Inductor"}


class ElementDialog(wx.Dialog):
    """Edit one element between two pads (any parts).  models: the kinds offered ("R" resistance,
    "V" constant drop / diode, "C", "L"); dc: show the DC & Thermal fields (rating, board heat)."""

    def __init__(self, parent, board, el, models=("R", "V"), dc=True):
        super().__init__(parent, title="Element", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.board, self.result, self.el = board, None, el
        self.models, self.dc = list(models), dc
        self.refs = sorted({p.ref for p in board.pads if p.layers}, key=EL.natural_key)
        self.part, self.pad = [], []
        g = wx.FlexGridSizer(0, 2, 6, 8)
        for k, pid in enumerate((el.a, el.b)):
            ref = pid.rsplit(".", 1)[0] if pid else (self.refs[0] if self.refs else "")
            pc = wx.Choice(self, choices=self.refs)
            if ref in self.refs:
                pc.SetSelection(self.refs.index(ref))
            pd = wx.Choice(self, size=(220, -1))
            pc.Bind(wx.EVT_CHOICE, lambda e, k=k: self._pads(k))
            self.part.append(pc)
            self.pad.append(pd)
            row = wx.BoxSizer(wx.HORIZONTAL)
            row.AddMany([(pc, 0, wx.RIGHT, 6), (pd, 1)])
            g.AddMany([(wx.StaticText(self, label=f"Pad {'AB'[k]}"), 0, wx.ALIGN_CENTER_VERTICAL), (row, 1, wx.EXPAND)])
            self._pads(k, pid)
        self.name = wx.TextCtrl(self, value=el.name)
        self.R = wx.TextCtrl(self, value=EL.fmt_ohms(el.R).replace(" ", "").replace("Ω", "")
                             if el.R or el.auto else "")
        self.P = wx.TextCtrl(self, value=f"{el.P_rated:g}" if el.P_rated else "")
        self.model = wx.RadioBox(self, label="Model", choices=[MODEL_LABELS[m] for m in self.models],
                                 majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.model.SetSelection(self.models.index(el.model) if el.model in self.models else 0)
        xv = el.C if el.model == "C" else el.L if el.model == "L" else 0.0
        from ..suite.library import fmt_si
        self.X = wx.TextCtrl(self, value=fmt_si(xv, "").replace(" ", "") if xv else "")
        self.X_label = wx.StaticText(self, label="Value")
        self.Vf = wx.TextCtrl(self, value=f"{el.Vf:g}" if el.model == "V" else "0.7")
        self.R_label = wx.StaticText(self, label="Resistance [Ω]")
        self.model.Bind(wx.EVT_RADIOBOX, lambda e: self._model())
        self.heats = wx.CheckBox(self, label="Dissipates on the board (untick for an external load or wire)")
        self.heats.SetValue(el.heats)
        g.AddMany([(wx.StaticText(self, label="Name"), 0, wx.ALIGN_CENTER_VERTICAL), (self.name, 1, wx.EXPAND),
                   (wx.StaticText(self, label="Forward drop Vf [V]"), 0, wx.ALIGN_CENTER_VERTICAL), (self.Vf, 1, wx.EXPAND),
                   (self.R_label, 0, wx.ALIGN_CENTER_VERTICAL), (self.R, 1, wx.EXPAND),
                   (self.X_label, 0, wx.ALIGN_CENTER_VERTICAL), (self.X, 1, wx.EXPAND)])
        self.P_label = wx.StaticText(self, label="Power rating [W]")
        g.AddMany([(self.P_label, 0, wx.ALIGN_CENTER_VERTICAL), (self.P, 1, wx.EXPAND)])
        g.AddGrowableCol(1)
        top = wx.BoxSizer(wx.VERTICAL)
        top.Add(self.model, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        top.Add(g, 0, wx.EXPAND | wx.ALL, 10)
        top.Add(self.heats, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        help_ = ("Values like 4k7, 0R1, 5m (milli), 1M (mega), 100n, 4u7." + (" Rating may be blank." if dc else "")
                 + "\nPick any two pads: a resistor, a FET's drain-source R_on, an inductor's\n"
                 "DCR, a fuse, or an external load/wire between two connectors.\n"
                 "Constant drop: Vf plus the series R, conducting from pad A (anode)\n"
                 "to pad B only, e.g. a diode; it blocks while V(A) − V(B) < Vf.")
        if "C" in self.models:
            help_ += "\nC and L are ideal (no ESR, ESL or SRF). A pad on a return net is the REF node."
        top.Add(wx.StaticText(self, label=help_), 0, wx.LEFT | wx.RIGHT, 10)
        if not dc:
            for w in (self.P, self.P_label, self.heats):
                w.Hide()
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)
        self._model()
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)

    def _model(self):
        m = self.models[self.model.GetSelection()]
        v = m == "V"
        self.Vf.Enable(v)
        self.R.Enable(m in ("R", "V"))
        self.X.Enable(m in ("C", "L"))
        self.X_label.SetLabel({"C": "Capacitance [F]", "L": "Inductance [H]"}.get(m, "Value"))
        self.R_label.SetLabel("Series resistance [Ω]" if v else "Resistance [Ω]")
        if v and not self.R.GetValue().strip():
            self.R.SetValue("0")
        self.Layout()

    def _pads(self, k, sel=None):
        i = self.part[k].GetSelection()
        ref = self.refs[i] if i != wx.NOT_FOUND else ""
        self.ids = getattr(self, "ids", [[], []])
        self.ids[k] = [p.id for p in self.board.pads if p.ref == ref and p.layers]
        self.ids[k] = sorted(set(self.ids[k]), key=EL.natural_key)
        nets = {p.id: p.net for p in self.board.pads}
        self.pad[k].Set([f"{pid.rsplit('.', 1)[1]}  ({nets[pid] or 'no net'})" for pid in self.ids[k]])
        if self.ids[k]:
            self.pad[k].SetSelection(self.ids[k].index(sel) if sel in self.ids[k] else min(k, len(self.ids[k]) - 1))
        if k == 0 and hasattr(self, "name") and not self.name.GetValue():
            self.name.SetValue(ref)

    def _ok(self, e):
        try:
            a, b = (self.ids[k][self.pad[k].GetSelection()] for k in (0, 1))
        except (IndexError, TypeError):
            return wx.MessageBox("Pick both pads.", "Element", parent=self)
        if a == b:
            return wx.MessageBox("The two pads must differ.", "Element", parent=self)
        model = self.models[self.model.GetSelection()]
        C = L = 0.0
        if model in ("C", "L"):
            from ..suite.library import parse_si
            try:
                xv = parse_si(self.X.GetValue())
            except (ValueError, TypeError):
                xv = -1
            if not xv or xv <= 0:
                return wx.MessageBox("Value: enter something like 100n, 4u7, 2.2p.", "Element", parent=self)
            C, L = (xv, 0.0) if model == "C" else (0.0, xv)
        R = EL.parse_ohms(self.R.GetValue()) if model in ("R", "V") else 0.0
        if R is None or R < 0:
            return wx.MessageBox("Resistance: enter a value such as 0.01, 10m, 4k7.", "Element", parent=self)
        try:
            P = float(self.P.GetValue()) if self.P.GetValue().strip() else None
        except ValueError:
            return wx.MessageBox("Power rating must be a number (or blank).", "Element", parent=self)
        Vf = 0.0
        if model == "V":
            try:
                Vf = float(self.Vf.GetValue().replace(",", "."))
            except ValueError:
                Vf = -1
            if Vf < 0:
                return wx.MessageBox("Forward drop: enter a voltage such as 0.7.", "Element", parent=self)
        name = self.name.GetValue().strip() or a.rsplit(".", 1)[0]
        heats = self.heats.GetValue()
        e = self.el
        auto = e.auto and (a, b, R, P, name, heats, model, Vf) == (e.a, e.b, e.R, e.P_rated, e.name, e.heats,
                                                                   e.model, e.Vf)
        self.result = EL.Element(name, a, b, R, P, True, auto, heats, model, Vf, C, L)
        self.EndModal(wx.ID_OK)


SETTINGS = [("pitch", "Grid pitch [mm] / auto"), ("t_plate", "Barrel plating [µm]"),
            ("T_cold", "Cu temp., no thermal [°C]"), ("T_amb", "Ambient [°C]"),
            ("h_conv", "h per face [W/m²K] (fixed)"), ("emissivity", "Emissivity (auto h)"),
            ("k_fr4", "Dielectric k [W/mK]"), ("thermal_pitch", "Thermal cell [mm] (0 = finest)"),
            ("rho20", "Cu ρ @20 °C [Ω·m]"), ("b_height", "B-map distance [mm]"),
            ("pulse_scale", "Pulse amplitude [× table]"), ("pulse_ms", "Pulse length [ms]")]



class MainFrame(wx.Frame):
    """The DC & Thermal tool. `session` is shared with the other suite windows (a private one if None)."""

    def __init__(self, board_path=None, use_kicad=False, session=None):
        super().__init__(None, title="DC & Thermal - PCB analysis suite", size=(1120, 990))
        self.session = session or Session()
        self.board, self.rows, self.result = None, [], None
        self.elements, self.saved_elements, self.el_warnings = [], [], []
        self.settings = S.Settings()
        self.viewers = []
        self._loading = False
        self._build()
        from .suite import attach_suite_menu
        attach_suite_menu(self, self.session, "dc")
        if use_kicad and self.session.kicad is None:
            kp = self.session.connect_kicad()
            board_path = board_path or kp
        if self.session.kicad is not None:
            self._kicad_ui()
        elif self.session.kicad_error and use_kicad:
            self._status(f"Not connected to KiCad ({self.session.kicad_error}); running standalone.")
        if board_path:
            self.load_board(board_path)
        elif self.session.board is not None:
            self.load_board(self.session.board_path, reparse=False)
        self._settings_to_ui()
        self.session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)

    @property
    def kicad(self):
        return self.session.kicad

    def _on_close(self, e):
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if self._loading:
            return
        if what == "board" and self.session.board is not None:
            self.load_board(self.session.board_path, load_table=False, reparse=False)
        elif what == "stackup":
            self.settings.thickness = dict(self.session.stackup.cu_override)
            self._refresh_layers()

    # ------------------------------------------------------------- layout
    def _build(self):
        p = self.panel = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.path = wx.TextCtrl(p)
        b1, b2 = wx.Button(p, label="Browse…"), wx.Button(p, label="Reload")
        b1.Bind(wx.EVT_BUTTON, self._browse)
        b2.Bind(wx.EVT_BUTTON, lambda e: self.load_board(self.path.GetValue(), load_table=False))
        row.AddMany([(wx.StaticText(p, label="Board"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4),
                     (self.path, 1, wx.EXPAND), (b1, 0, wx.LEFT, 4), (b2, 0, wx.LEFT, 4)])
        top.Add(row, 0, wx.EXPAND | wx.ALL, 8)
        self.kbar = wx.BoxSizer(wx.HORIZONTAL)
        top.Add(self.kbar, 0, wx.LEFT | wx.RIGHT, 8)
        self.info = wx.StaticText(p, label="Save the board (and refill zones) in KiCad before running: "
                                           "the simulator reads the saved .kicad_pcb file.")
        top.Add(self.info, 0, wx.EXPAND | wx.ALL, 8)

        # excitation table
        box = wx.StaticBoxSizer(wx.VERTICAL, p, "Excitations (all rows are solved together)")
        sb = box.GetStaticBox()
        self.tree = _list(sb, [("Name", 110), ("Kind", 110), ("Source / + pads", 200), ("Sink / − pads", 200),
                               ("Value", 80), ("Net", 200)])
        self.tree.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit())
        box.Add(self.tree, 1, wx.EXPAND | wx.ALL, 4)
        bcol = wx.BoxSizer(wx.HORIZONTAL)
        self.sel_btn = wx.Button(sb, label="Add from KiCad selection")
        self.sel_btn.Bind(wx.EVT_BUTTON, lambda e: self._from_selection())
        self.sel_btn.Disable()
        bcol.Add(self.sel_btn, 0, wx.RIGHT, 12)
        for text, cmd in (("Add…", self._add), ("Edit…", self._edit), ("Duplicate", self._dup),
                          ("Delete", self._delete), (None, None), ("Save table", self._save),
                          ("Load table…", self._load)):
            if text is None:
                bcol.AddStretchSpacer()
                continue
            b = wx.Button(sb, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            bcol.Add(b, 0, wx.RIGHT, 2)
        box.Add(bcol, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        self.tree.SetMinSize((-1, 110))
        top.Add(box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # resistance elements
        ebox = wx.StaticBoxSizer(wx.VERTICAL, p, "Resistance elements (resistors found on the board, plus "
                                                   "any two pads you add)")
        eb = ebox.GetStaticBox()
        self.etree = _list(eb, [("On", 40), ("In circuit", 75), ("Name", 80), ("Pad A", 90), ("Pad B", 90),
                                ("Value", 110), ("Rating [W]", 80), ("Nets", 220), ("Heats board", 85),
                                ("Source", 60)])
        self.etree.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._el_edit())
        ebox.Add(self.etree, 1, wx.EXPAND | wx.ALL, 4)
        ecol = wx.BoxSizer(wx.HORIZONTAL)
        self.esel_btn = wx.Button(eb, label="Add from KiCad selection (2 pads)")
        self.esel_btn.Bind(wx.EVT_BUTTON, lambda e: self._el_from_selection())
        self.esel_btn.Disable()
        ecol.Add(self.esel_btn, 0, wx.RIGHT, 12)
        for text, cmd in (("Add…", self._el_add), ("Edit…", self._el_edit), ("On / off", self._el_toggle),
                          ("Delete", self._el_delete)):
            b = wx.Button(eb, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            ecol.Add(b, 0, wx.RIGHT, 2)
        ebox.Add(ecol, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        self.etree.SetMinSize((-1, 110))
        top.Add(ebox, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)

        # settings
        sbox = wx.StaticBoxSizer(wx.HORIZONTAL, p, "Settings")
        ss = sbox.GetStaticBox()
        g = wx.FlexGridSizer(0, 4, 4, 8)
        self.vars = {}
        for key, label in SETTINGS:
            self.vars[key] = wx.TextCtrl(ss, size=(90, -1))
            g.Add(wx.StaticText(ss, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            g.Add(self.vars[key])
        left = wx.BoxSizer(wx.VERTICAL)
        left.Add(g)
        cg = wx.FlexGridSizer(0, 2, 4, 16)
        self.checks = {}
        for key, label in CHECKS:
            self.checks[key] = wx.CheckBox(ss, label=label)
            cg.Add(self.checks[key])
        self.autor = self.checks["auto_resistors"]
        self.autor.Bind(wx.EVT_CHECKBOX, lambda e: self._refresh_elements())
        left.Add(cg, 0, wx.TOP, 8)
        sbox.Add(left, 0, wx.ALL, 4)
        right = wx.BoxSizer(wx.VERTICAL)
        right.Add(wx.StaticText(ss, label="Copper thickness (double-click to override)"))
        self.ltree = _list(ss, [("Layer", 70), ("Board [µm]", 85), ("Used [µm]", 80)])
        self.ltree.SetMinSize((245, -1))
        self.ltree.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._edit_thickness)
        right.Add(self.ltree, 1, wx.EXPAND)
        sbox.Add(right, 1, wx.EXPAND | wx.ALL, 4)
        top.Add(sbox, 0, wx.EXPAND | wx.ALL, 8)

        # run
        rrow = wx.BoxSizer(wx.HORIZONTAL)
        self.run_btn = wx.Button(p, label="Run simulation")
        self.run_btn.Bind(wx.EVT_BUTTON, lambda e: self._run())
        b3, b4 = wx.Button(p, label="Show last result"), wx.Button(p, label="Open saved result…")
        b3.Bind(wx.EVT_BUTTON, lambda e: self._show())
        b4.Bind(wx.EVT_BUTTON, lambda e: self._open_result())
        self.status = wx.StaticText(p, label="")
        rrow.AddMany([(self.run_btn,), (b3, 0, wx.LEFT, 4), (b4, 0, wx.LEFT, 4),
                      (self.status, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)])
        top.Add(rrow, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        p.SetSizer(top)

    def _status(self, text):
        self.status.SetLabel(text)
        self.status.SetToolTip(text)

    # ------------------------------------------------------------- KiCad
    def _kicad_ui(self):
        self.sel_btn.Enable()
        self.esel_btn.Enable()
        for text, cmd in (("Save board", self._k_save), ("Refill zones + save", self._k_refill)):
            b = wx.Button(self.panel, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            self.kbar.Add(b, 0, wx.RIGHT, 4)
        self.kbar.Insert(0, wx.StaticText(self.panel, label="KiCad:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.panel.Layout()

    def _k_save(self):
        try:
            self.kicad.save()
            self.load_board(self.path.GetValue(), load_table=False)
        except Exception as e:      # noqa: BLE001
            wx.MessageBox(f"Save failed: {e}", "KiCad", parent=self)

    def _k_refill(self):
        try:
            self.kicad.refill_zones()
            self.kicad.save()
            self.load_board(self.path.GetValue(), load_table=False)
        except Exception as e:      # noqa: BLE001
            wx.MessageBox(f"Refill failed: {e}", "KiCad", parent=self)

    def _from_selection(self):
        if not self.board:
            return
        try:
            ids = self.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return wx.MessageBox(f"Could not read the selection: {e}", "KiCad", parent=self)
        if len(ids) < 2:
            return wx.MessageBox("Select at least two pads in the PCB editor (source first, then sink).\n"
                                 "Tip: use the selection filter to pick pads only, and save the board "
                                 "first if you changed it.", "KiCad selection", parent=self)
        self._add(S.Row(f"row{len(self.rows) + 1}", [ids[0]], ids[1:], 1.0))

    def _el_from_selection(self):
        if not self.board:
            return
        try:
            ids = self.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return wx.MessageBox(f"Could not read the selection: {e}", "KiCad", parent=self)
        if len(ids) != 2:
            return wx.MessageBox("Select exactly two pads in the PCB editor (the element's two terminals).",
                                 "KiCad selection", parent=self)
        ref = ids[0].rsplit(".", 1)[0]
        self._el_add(EL.Element(ref, ids[0], ids[1], 0.0))

    # ------------------------------------------------------------- board
    def _browse(self, e):
        with wx.FileDialog(self, "Open board", wildcard="KiCad PCB (*.kicad_pcb)|*.kicad_pcb|All|*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                self.load_board(d.GetPath())

    def load_board(self, path, load_table=True, reparse=True):
        """Load (or, with reparse=False, adopt the session's already parsed) board."""
        if reparse:
            self._loading = True
            try:
                self.session.load_board(path)
            except Exception as e:      # noqa: BLE001
                return wx.MessageBox(f"Could not read {path}:\n{e}", "Board", parent=self)
            finally:
                self._loading = False
        self.board = self.session.board
        path = self.session.board_path
        self.path.SetValue(path)
        b = self.board
        self.info.SetLabel(f"{len(b.copper_layers)} copper layers, {len(b.nets)} nets, {len(b.pads)} pads, "
                           f"{len(b.tracks)} tracks, {len(b.vias)} vias, {len(b.zones)} zone fills"
                           + ("" if b.stackup_found else "  -  no stackup in file: 35 µm default")
                           + ("  -  " + b.warnings[0] if b.warnings else ""))
        sp = scenario_path(path)
        if load_table and os.path.exists(sp):
            try:
                self.rows, self.settings, self.saved_elements = load_scenario(sp)
                self._settings_to_ui()
            except Exception as e:  # noqa: BLE001
                self._status(f"Could not load saved table: {e}")
        else:
            self.saved_elements = EL.to_save(self.elements)
        self.settings.thickness = dict(self.session.stackup.cu_override)
        auto, self.el_warnings = EL.detect(b)
        self.elements = EL.merge(auto, self.saved_elements)
        if self.el_warnings:
            self._status(self.el_warnings[0] + (f" (+{len(self.el_warnings) - 1} more)"
                                                if len(self.el_warnings) > 1 else ""))
        self._refresh_layers()
        self._refresh_rows()

    def _refresh_layers(self):
        self.ltree.DeleteAllItems()
        if not self.board:
            return
        for L in self.board.copper_layers:
            t0 = self.board.cu_thickness[L] * 1e6
            t = self.settings.thickness.get(L, self.board.cu_thickness[L]) * 1e6
            self.ltree.Append([L, f"{t0:.1f}", f"{t:.1f}"])

    def _edit_thickness(self, e):
        L = self.board.copper_layers[e.GetIndex()]
        cur = self.settings.thickness.get(L, self.board.cu_thickness[L]) * 1e6
        with wx.TextEntryDialog(self, f"{L} finished copper thickness [µm]\n(0 = use the board value)",
                                "Copper thickness", f"{cur:.2f}") as d:
            if d.ShowModal() != wx.ID_OK:
                return
            try:
                v = float(d.GetValue())
            except ValueError:
                return
        self.session.set_cu_override(L, v * 1e-6 if v > 0 else None)     # notifies -> _refresh_layers

    # ------------------------------------------------------------- rows
    def _refresh_rows(self):
        self.tree.DeleteAllItems()
        for r in self.rows:
            nets = []
            for ids in (r.src, r.snk):
                ps = self.board.pads_by_id(ids[0]) if self.board and ids else []
                nets.append(ps[0].net if ps else "?")
            self.tree.Append([r.name, S.KINDS[r.kind], " + ".join(r.src), " + ".join(r.snk), f"{r.current:g} {r.unit}",
                              nets[0] if nets[0] == nets[1] else f"{nets[0]} → {nets[1]}"])
        self._refresh_elements()

    # ------------------------------------------------------------- elements
    def _used_elements(self):
        """Elements passed to the solver (auto ones only if auto-detection is on)."""
        auto_on = self.autor.GetValue()
        return [e for e in self.elements if auto_on or not e.auto]

    def _refresh_elements(self):
        self.etree.DeleteAllItems()
        if not self.board:
            return
        try:
            _, used, _ = EL.active(self.board, S.nets_of_rows(self.board, self.rows), self._used_elements())
            used = {id(e) for e in used}
        except S.SimError:
            used = set()
        nets = {p.id: p.net for p in self.board.pads}
        auto_on = self.autor.GetValue()
        for e in self.elements:
            self.etree.Append(["✓" if e.enabled and (auto_on or not e.auto) else "–",
                               "yes" if id(e) in used else "", e.name, e.a, e.b, e.describe(),
                               f"{e.P_rated:g}" if e.P_rated else "", f"{nets.get(e.a, '?')} – {nets.get(e.b, '?')}",
                               "yes" if e.heats else "no (external)", "auto" if e.auto else "user"])

    def _el_add(self, preset=None):
        if not self.board:
            return wx.MessageBox("Load a board first.", "Board", parent=self)
        with ElementDialog(self, self.board, preset or EL.Element("", "", "", 0.0)) as d:
            if d.ShowModal() == wx.ID_OK and d.result:
                self.elements = [e for e in self.elements if e.name != d.result.name] + [d.result]
                self._refresh_elements()

    def _el_edit(self):
        sel = _selected(self.etree)
        if sel:
            with ElementDialog(self, self.board, self.elements[sel[0]]) as d:
                if d.ShowModal() == wx.ID_OK and d.result:
                    self.elements[sel[0]] = d.result
                    self._refresh_elements()

    def _el_toggle(self):
        for k in _selected(self.etree):
            self.elements[k].enabled = not self.elements[k].enabled
        self._refresh_elements()

    def _el_delete(self):
        sel = _selected(self.etree)
        if any(self.elements[k].auto for k in sel):
            wx.MessageBox("Resistors found on the board cannot be deleted; they are switched off instead.",
                          "Elements", parent=self)
        for k in sorted(sel, reverse=True):
            if self.elements[k].auto:
                self.elements[k].enabled = False
            else:
                del self.elements[k]
        self._refresh_elements()

    def _add(self, preset=None):
        if not self.board:
            return wx.MessageBox("Load a board first.", "Board", parent=self)
        with RowDialog(self, self.board, preset, f"row{len(self.rows) + 1}") as d:
            if d.ShowModal() == wx.ID_OK and d.result:
                self.rows.append(d.result)
                self._refresh_rows()

    def _edit(self):
        sel = _selected(self.tree)
        if sel:
            with RowDialog(self, self.board, self.rows[sel[0]], "") as d:
                if d.ShowModal() == wx.ID_OK and d.result:
                    self.rows[sel[0]] = d.result
                    self._refresh_rows()

    def _dup(self):
        for k in _selected(self.tree):
            r = self.rows[k]
            self.rows.append(S.Row(r.name + " copy", list(r.src), list(r.snk), r.current, r.kind))
        self._refresh_rows()

    def _delete(self):
        for k in sorted(_selected(self.tree), reverse=True):
            del self.rows[k]
        self._refresh_rows()

    def _save(self):
        if self.board and self._ui_to_settings():
            save_scenario(scenario_path(self.path.GetValue()), self.rows, self.settings,
                          EL.to_save(self.elements))
            self._status(f"Saved {os.path.basename(scenario_path(self.path.GetValue()))}")

    def _load(self):
        with wx.FileDialog(self, "Load table", wildcard="Scenario (*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                self.rows, self.settings, self.saved_elements = load_scenario(d.GetPath())
                if self.session.stackup is not None:
                    self.session.stackup.cu_override = dict(self.settings.thickness)
                    self.session.set_stackup(self.session.stackup)
                self.elements = EL.merge(EL.detect(self.board)[0] if self.board else [], self.saved_elements)
                self._settings_to_ui()
                self._refresh_layers()
                self._refresh_rows()

    # ------------------------------------------------------------- settings
    def _settings_to_ui(self):
        s = self.settings
        self.vars["pitch"].SetValue(str(s.pitch))
        self.vars["t_plate"].SetValue(f"{s.t_plate * 1e6:g}")
        for k in FLOAT_SETTINGS:
            self.vars[k].SetValue(f"{getattr(s, k):g}")
        for k, c in self.checks.items():
            c.SetValue(bool(getattr(s, k)))
        self._refresh_elements()

    def _ui_to_settings(self):
        s = self.settings
        try:
            p = self.vars["pitch"].GetValue().strip().lower()
            s.pitch = "auto" if p in ("", "auto") else float(p)
            s.t_plate = float(self.vars["t_plate"].GetValue()) * 1e-6
            for k in FLOAT_SETTINGS:
                setattr(s, k, float(self.vars[k].GetValue()))
        except ValueError as e:
            wx.MessageBox(f"Invalid number: {e}", "Settings", parent=self)
            return False
        for k, c in self.checks.items():
            setattr(s, k, c.GetValue())
        return True

    # ------------------------------------------------------------- run
    def _run(self):
        if not self.board or not self.rows:
            return wx.MessageBox("Load a board and add at least one excitation.", "Run", parent=self)
        if not self._ui_to_settings():
            return
        save_scenario(scenario_path(self.path.GetValue()), self.rows, self.settings, EL.to_save(self.elements))
        self.load_board(self.path.GetValue(), load_table=False)     # pick up the latest saved file
        self.run_btn.Disable()
        board, rows, s, els = self.board, list(self.rows), self.settings, self._used_elements()

        def work():
            try:
                res = S.simulate(board, rows, s, progress=lambda m: wx.CallAfter(self._status, m.strip()),
                                 elements=els)
                res.warnings += self.el_warnings
                wx.CallAfter(self._done, res, None)
            except S.SimError as e:
                wx.CallAfter(self._done, None, str(e))
            except Exception:       # noqa: BLE001
                wx.CallAfter(self._done, None, traceback.format_exc())

        threading.Thread(target=work, daemon=True).start()

    def _done(self, res, err):
        self.run_btn.Enable()
        if err:
            self._status("Failed")
            return wx.MessageBox(err, "Simulation failed", style=wx.ICON_ERROR, parent=self)
        self.result = self.session.last_result = res
        self._status(f"Done in {res.summary['seconds']:.1f} s")
        self._show()

    def _show(self):
        if self.result is None:
            return
        from .viewer import ViewerFrame
        v = ViewerFrame(self, self.result)
        v.Show()
        self.viewers.append(v)

    def _open_result(self):
        with wx.FileDialog(self, "Open result", wildcard="Result (*.npz)|*.npz",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                from ..results import Result
                self.result = Result.load(d.GetPath())
                self._show()


def main(board_path=None, use_kicad=False):
    """Open the DC & Thermal tool (inside the suite: its Suite menu reaches Home and the other tools)."""
    from .suite import main as suite_main
    suite_main(board_path, use_kicad, tool="dc")
