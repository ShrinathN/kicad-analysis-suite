"""Trace R/L/C → SPICE (wx): pick pads, extract the copper between them, show and export the ladder."""
import csv
import math
import os
import threading
import traceback
from dataclasses import asdict, fields

import numpy as np
import wx

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import elements as EL  # noqa: E402
from .. import scenario as SC  # noqa: E402
from ..extract import rlc as X  # noqa: E402
from ..extract import spice as SP  # noqa: E402
from ..suite.library import fmt_si, parse_si  # noqa: E402
from .schematic import colours, draw_dot, draw_ground, draw_symbol  # noqa: E402
from .suite import attach_suite_menu  # noqa: E402

OPTS = (("f_max", "f max [Hz]", "si", "Highest frequency of interest: sections are ≤ λ/10 here (≈ 0.35 / t_rise)"),
        ("sample", "Cross-section every [mm]", float, "Spacing of the 2D field solves along each track"),
        ("window", "Return search ± [mm]", float, "Return copper this far either side of a track is used"),
        ("max_section", "Max section [mm] (0 = λ/10)", float, "Extra limit on the ladder section length"),
        ("T", "Copper temperature [°C]", float, "For R"))
CHECKS = (("stubs", "Include stubs (branches to other pads)"),
          ("pad_c", "Pad capacitance"),
          ("dc_solver", "Calibrate R with the DC field solution"),
          ("dielectric_g", "Dielectric loss G (at f max)"))
SEG_COLS = (("#", "chain", 40, "{}"), ("Layer", "layer", 60, "{}"), ("w [mm]", "width_mm", 70, "{:.3g}"),
            ("l [mm]", "length_mm", 75, "{:.3g}"), ("Z0 [Ω]", "Z0", 75, "{:.1f}"), ("εeff", "eeff", 60, "{:.2f}"),
            ("L' [nH/mm]", "L_nH_mm", 100, "{:.3f}"), ("C' [pF/mm]", "C_pF_mm", 100, "{:.4f}"),
            ("R' [mΩ/mm]", "R_mOhm_mm", 100, "{:.3f}"), ("R'(f) [mΩ/mm]", "R_ac_mOhm_mm", 120, "{:.3f}"),
            ("delay [ps]", "delay_ps", 90, "{:.1f}"), ("Return", "reference", 220, "{}"))
PAIR_COLS = (("Path", "pair", 80, "{}"), ("Pads", None, 140, "{}"), ("l [mm]", "length_mm", 75, "{:.2f}"),
             ("Vias", "vias", 50, "{}"), ("R [mΩ]", "R_mOhm", 85, "{:.3f}"), ("L [nH]", "L_nH", 85, "{:.3f}"),
             ("C [pF]", "C_pF", 85, "{:.3f}"), ("Z0 avg [Ω]", "Z0_avg", 95, "{:.1f}"),
             ("Z0 min–max", None, 110, "{}"), ("Delay [ps]", "delay_ps", 95, "{:.1f}"))
UNITS = {"R": "Ω", "L": "H", "C": "F"}


def cell_label(kind, v):
    if kind == "D":
        return f"{v[0]:g} V" + (f" + {fmt_si(v[1], 'Ω')}" if v[1] else "")
    return fmt_si(v, UNITS[kind])


def _cell(fmt, v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def _list(parent, cols):
    lc = wx.ListCtrl(parent, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
    for k, (title, _, w, _) in enumerate(cols):
        lc.InsertColumn(k, title, width=w)
    return lc


class PadDialog(wx.Dialog):
    """Pick one pad: part, then pad (with its net)."""

    def __init__(self, parent, board, net=None):
        super().__init__(parent, title="Add port pad")
        self.board, self.result = board, None
        pads = [p for p in board.pads if p.layers and p.kind != "np_thru_hole" and p.net and
                (net is None or p.net == net)]
        self.refs = sorted({p.ref for p in pads}, key=EL.natural_key)
        self.pads = pads
        self.part = wx.Choice(self, choices=self.refs)
        self.pad = wx.Choice(self, size=(220, -1))
        self.part.Bind(wx.EVT_CHOICE, lambda e: self._fill())
        g = wx.FlexGridSizer(0, 2, 6, 8)
        g.AddMany([(wx.StaticText(self, label="Part"), 0, wx.ALIGN_CENTER_VERTICAL), (self.part, 1, wx.EXPAND),
                   (wx.StaticText(self, label="Pad"), 0, wx.ALIGN_CENTER_VERTICAL), (self.pad, 1, wx.EXPAND)])
        top = wx.BoxSizer(wx.VERTICAL)
        top.Add(g, 0, wx.EXPAND | wx.ALL, 10)
        if net:
            top.Add(wx.StaticText(self, label=f"Only pads on net {net} (the ports' net)."), 0, wx.LEFT | wx.RIGHT, 10)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)
        if self.refs:
            self.part.SetSelection(0)
            self._fill()
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)

    def _fill(self):
        ref = self.refs[self.part.GetSelection()]
        self.ids = sorted({p.id for p in self.pads if p.ref == ref}, key=EL.natural_key)
        nets = {p.id: p.net for p in self.pads}
        self.pad.Set([f"{i.rsplit('.', 1)[1]}  ({nets[i]})" for i in self.ids])
        if self.ids:
            self.pad.SetSelection(0)

    def _ok(self, e):
        k = self.pad.GetSelection()
        if k == wx.NOT_FOUND:
            return
        self.result = self.ids[k]
        self.EndModal(wx.ID_OK)


class LadderCanvas(wx.ScrolledWindow):
    """Read-only drawing of the extracted ladder along P1 → P2 (series R, L; shunt C, G; stubs)."""

    CW, Y0, Y1, H = 92, 60, 160, 220

    def __init__(self, parent):
        super().__init__(parent, style=wx.HSCROLL | wx.VSCROLL)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.cells, self.note = [], "Run an extraction."
        self.SetScrollRate(20, 20)
        self.Bind(wx.EVT_PAINT, self._paint)

    def show(self, cells, note=""):
        self.cells, self.note = cells, note
        self.SetVirtualSize((80 + (len(cells) + 1) * self.CW + 80, self.H + 40))
        self.Refresh()

    def _paint(self, e):
        dc = wx.AutoBufferedPaintDC(self)
        self.DoPrepareDC(dc)
        c = colours()
        dc.SetBackground(wx.Brush(c["bg"]))
        dc.Clear()
        gc = wx.GraphicsContext.Create(dc)
        gc.SetPen(wx.Pen(c["fg"], 2))
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        small, bold = wx.Font(wx.FontInfo(8)), wx.Font(wx.FontInfo(9).Bold())
        gc.SetFont(small, c["dim"])
        gc.DrawText(self.note, 10, self.H)
        if not self.cells:
            return
        CW, Y0, Y1 = self.CW, self.Y0, self.Y1
        x = 50
        gc.SetFont(bold, c["fg"])
        gc.DrawText("P1", 14, Y0 - 8)
        draw_dot(gc, x, Y0, c["fg"])
        for cell in self.cells:
            k, v = cell["kind"], cell["value"]
            if not cell["shunt"]:
                draw_symbol(gc, k, x, Y0, x + CW, Y0)
                lab, ax = cell_label(k, v), x + CW / 2
                gc.SetFont(bold, c["fg"])
                w = gc.GetFullTextExtent(cell["name"])[0]
                gc.DrawText(cell["name"], ax - w / 2, Y0 - 36)
                gc.SetFont(small, c["fg"])
                w = gc.GetFullTextExtent(lab)[0]
                gc.DrawText(lab, ax - w / 2, Y0 + 18)
            else:
                ax = x + 24
                p = gc.CreatePath()
                p.MoveToPoint(x, Y0)
                p.AddLineToPoint(x + CW, Y0)
                gc.StrokePath(p)
                draw_dot(gc, ax, Y0, c["fg"])
                if k == "branch":
                    draw_symbol(gc, "box", ax, Y0, ax, Y1 - 20)
                    lab = [f"stub", f"{v} element(s)"]
                else:
                    draw_symbol(gc, k, ax, Y0, ax, Y1)
                    draw_ground(gc, ax, Y1)
                    lab = [cell_label(k, v)]
                gc.SetFont(bold, c["fg"])
                gc.DrawText(cell["name"] if k != "branch" else "", ax + 10, Y0 + 14)
                gc.SetFont(small, c["fg"])
                for i, t in enumerate(lab):
                    gc.DrawText(t, ax + 10, Y0 + 32 + 14 * i)
            x += CW
        draw_dot(gc, x, Y0, c["fg"])
        gc.SetFont(bold, c["fg"])
        gc.DrawText("P2", x + 8, Y0 - 8)
        gc.SetFont(small, c["dim"])
        gc.DrawText("REF (return)", 14, Y1 + 18)


class ExtractFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="Trace R/L/C → SPICE - PCB analysis suite", size=(1450, 940))
        self.session = session
        self.board = session.board
        self.ports, self.opts, self.return_nets, self.elements = [], X.Options(), None, []
        self.ext, self._running = None, False
        self._load_setup()
        self._build()
        attach_suite_menu(self, session, "extract")
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._to_ui()

    # ------------------------------------------------------------- persistence
    def _load_setup(self):
        try:
            d = SC.section(self.session.scenario_path, "extract") if self.session.scenario_path else {}
        except (OSError, ValueError):
            d = {}
        known = {f.name for f in fields(X.Options)}
        self.opts = X.Options(**{k: v for k, v in (d.get("options") or {}).items() if k in known})
        ids = {p.id for p in self.board.pads}
        self.ports = [p for p in d.get("ports", []) if p in ids]
        ek = {f.name for f in fields(EL.Element)}
        self.elements = [EL.Element(**{k: v for k, v in e.items() if k in ek}) for e in d.get("elements", [])]
        self.return_nets = [n for n in d.get("return_nets", []) if n in self.board.nets] if "return_nets" in d \
            else None

    def save(self):
        if not self.session.scenario_path:
            return
        try:
            SC.save_section(self.session.scenario_path, "extract",
                            dict(ports=self.ports, return_nets=self._checked_nets(), options=asdict(self.opts),
                                 elements=[asdict(e) for e in self.elements]))
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
            ids = {p.id for p in self.board.pads}
            self.ports = [p for p in self.ports if p in ids]
            wx.CallAfter(self._to_ui)

    # ------------------------------------------------------------- UI
    def _build(self):
        sp = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = wx.ScrolledWindow(sp)
        left.SetScrollRate(0, 20)
        ls = wx.BoxSizer(wx.VERTICAL)

        pb = wx.StaticBoxSizer(wx.VERTICAL, left, "Ports (pads, in order: P1, P2, …)")
        pbox = pb.GetStaticBox()
        self.plist = wx.ListCtrl(pbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 130))
        for k, (t, w) in enumerate((("Port", 50), ("Pad", 110), ("Net", 140))):
            self.plist.InsertColumn(k, t, width=w)
        pb.Add(self.plist, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn in (("Add pad…", self._add_pad), ("Remove", self._del_pad), ("Up", lambda: self._move(-1)),
                         ("Down", lambda: self._move(1))):
            b = wx.Button(pbox, label=text, style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        if self.session.kicad is not None:
            b = wx.Button(pbox, label="From KiCad selection", style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e: self._from_kicad())
            row.Add(b, 0, wx.ALL, 2)
        pb.Add(row, 0)
        ls.Add(pb, 0, wx.EXPAND | wx.ALL, 6)

        rb = wx.StaticBoxSizer(wx.VERTICAL, left, "Return (reference) nets")
        rbox = rb.GetStaticBox()
        self.nets = wx.CheckListBox(rbox, size=(-1, 140))
        rb.Add(self.nets, 1, wx.EXPAND | wx.ALL, 4)
        b = wx.Button(rbox, label="Guess (GND-like)", style=wx.BU_EXACTFIT)
        b.Bind(wx.EVT_BUTTON, lambda e: self._guess())
        rb.Add(b, 0, wx.ALL, 2)
        rb.Add(wx.StaticText(rbox, label="Tick power planes too if decaps\ncouple them to ground."),
               0, wx.ALL, 4)
        ls.Add(rb, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        eb = wx.StaticBoxSizer(wx.VERTICAL, left, "Elements between pads (R, diode, C, L)")
        ebox = eb.GetStaticBox()
        self.elist = wx.ListCtrl(ebox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 110))
        for k, (t, w) in enumerate((("On", 35), ("Name", 55), ("Kind", 85), ("Value", 80), ("Pads", 140))):
            self.elist.InsertColumn(k, t, width=w)
        self.elist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._el_edit())
        eb.Add(self.elist, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn in (("Add…", self._el_add), ("Edit…", self._el_edit), ("On/off", self._el_toggle),
                         ("Remove", self._el_del)):
            b = wx.Button(ebox, label=text, style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        if self.session.kicad is not None:
            b = wx.Button(ebox, label="From KiCad selection", style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e: self._el_from_kicad())
            row.Add(b, 0, wx.ALL, 2)
        eb.Add(row, 0)
        ls.Add(eb, 0, wx.EXPAND | wx.ALL, 6)

        ob = wx.StaticBoxSizer(wx.VERTICAL, left, "Options")
        obox = ob.GetStaticBox()
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.fields = {}
        for key, label, kind, tip in OPTS:
            t = wx.TextCtrl(obox, size=(110, -1))
            t.SetToolTip(tip)
            self.fields[key] = (t, kind)
            g.AddMany([(wx.StaticText(obox, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t, 0)])
        ob.Add(g, 0, wx.ALL, 4)
        self.tr = wx.StaticText(obox, label="")
        ob.Add(self.tr, 0, wx.LEFT, 6)
        self.fields["f_max"][0].Bind(wx.EVT_TEXT, lambda e: self._tr())
        self.checks = {}
        for key, label in CHECKS:
            c = wx.CheckBox(obox, label=label)
            self.checks[key] = c
            ob.Add(c, 0, wx.ALL, 3)
        ls.Add(ob, 0, wx.EXPAND | wx.ALL, 6)

        self.run_b = wx.Button(left, label="Extract")
        self.run_b.Bind(wx.EVT_BUTTON, lambda e: self._run())
        ls.Add(self.run_b, 0, wx.EXPAND | wx.ALL, 6)
        ls.Add(wx.StaticText(left, label=(
            "R from the copper geometry (DC field solution); L and C from a 2D\n"
            "field solve of each cross-section against the return copper\n"
            "(microstrip, stripline, coplanar, pours). Linear, quasi-static:\n"
            "no radiation, no coupling to other signal nets. Save the board\n"
            "in KiCad first.")), 0, wx.ALL, 6)
        left.SetSizer(ls)

        nb = self.nb = wx.Notebook(sp)
        # summary
        sm = wx.Panel(nb)
        ss = wx.BoxSizer(wx.VERTICAL)
        self.pairs = _list(sm, PAIR_COLS)
        ss.Add(self.pairs, 0, wx.EXPAND | wx.ALL, 4)
        self.pairs.SetMinSize((-1, 110))
        self.info = wx.TextCtrl(sm, style=wx.TE_MULTILINE | wx.TE_READONLY)
        ss.Add(self.info, 1, wx.EXPAND | wx.ALL, 4)
        sm.SetSizer(ss)
        nb.AddPage(sm, "Summary")
        # segments
        sg = wx.Panel(nb)
        gs = wx.BoxSizer(wx.VERTICAL)
        self.segs = _list(sg, SEG_COLS)
        gs.Add(self.segs, 3, wx.EXPAND | wx.ALL, 4)
        gs.Add(wx.StaticText(sg, label="Vias, pads, zones and elements"), 0, wx.LEFT, 6)
        self.vias = wx.ListCtrl(sg, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        for k, (t, w) in enumerate((("Item", 150), ("Layers / nodes", 120), ("R [mΩ]", 70), ("L [nH]", 70),
                                    ("C [pF]", 70), ("Note", 320))):
            self.vias.InsertColumn(k, t, width=w)
        gs.Add(self.vias, 1, wx.EXPAND | wx.ALL, 4)
        b = wx.Button(sg, label="Export CSV…", style=wx.BU_EXACTFIT)
        b.Bind(wx.EVT_BUTTON, lambda e: self._csv())
        gs.Add(b, 0, wx.ALL, 4)
        sg.SetSizer(gs)
        nb.AddPage(sg, "Segments")
        # board
        bp = wx.Panel(nb)
        self.fig = Figure(figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(bp, -1, self.fig)
        tb = NavigationToolbar2WxAgg(self.canvas)
        tb.Realize()
        bs = wx.BoxSizer(wx.VERTICAL)
        bs.Add(self.canvas, 1, wx.EXPAND)
        brow = wx.BoxSizer(wx.HORIZONTAL)
        brow.Add(tb, 1, wx.EXPAND)
        if self.session.kicad is not None:
            b = wx.Button(bp, label="Select path in KiCad")
            b.Bind(wx.EVT_BUTTON, lambda e: self._select_kicad())
            brow.Add(b, 0, wx.ALL, 4)
        bs.Add(brow, 0, wx.EXPAND)
        bp.SetSizer(bs)
        nb.AddPage(bp, "Board")
        # spice
        spp = wx.Panel(nb)
        sps = wx.BoxSizer(wx.VERTICAL)
        self.spice = wx.TextCtrl(spp, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_DONTWRAP)
        self.spice.SetFont(wx.Font(wx.FontInfo(9).Family(wx.FONTFAMILY_TELETYPE)))
        sps.Add(self.spice, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        for text, fn in (("Save subcircuit…", self._save_spice), ("Copy", self._copy_spice)):
            b = wx.Button(spp, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 4)
        sps.Add(row, 0)
        spp.SetSizer(sps)
        nb.AddPage(spp, "SPICE")
        # schematic
        self.ladder = LadderCanvas(nb)
        nb.AddPage(self.ladder, "Schematic")

        sp.SplitVertically(left, nb, 470)
        sp.SetMinimumPaneSize(300)
        self.CreateStatusBar()
        self._status("Pick two or more pads on one net, tick the return nets, then Extract.")

    def _status(self, t):
        self.SetStatusText(t)

    # ------------------------------------------------------------- setup <-> UI
    def _to_ui(self):
        self.plist.DeleteAllItems()
        nets = {p.id: p.net for p in self.board.pads}
        for k, pid in enumerate(self.ports):
            self.plist.Append([f"P{k + 1}", pid, nets.get(pid, "?")])
        sig = nets.get(self.ports[0]) if self.ports else None
        names = list(self.board.nets)
        zone_nets = {z.net for z in self.board.zones}
        names.sort(key=lambda n: (n not in zone_nets, n))
        checked = self._checked_nets() if self.nets.GetCount() else self.return_nets
        if checked is None:
            checked = X.guess_return_nets(self.board, sig)
        self.nets.Set([n + ("  (zones)" if n in zone_nets else "") for n in names])
        self._net_names = names
        for k, n in enumerate(names):
            self.nets.Check(k, n in checked)
        self.elist.DeleteAllItems()
        kinds = {"R": "resistor", "V": "diode", "C": "capacitor", "L": "inductor"}
        for e in self.elements:
            self.elist.Append(["✓" if e.enabled else "–", e.name, kinds.get(e.model, e.model), e.describe(),
                               f"{e.a} – {e.b}"])
        o = self.opts
        for key, (t, kind) in self.fields.items():
            v = getattr(o, key)
            t.ChangeValue(fmt_si(v, "").strip() if kind == "si" else f"{v:g}")
        for key, c in self.checks.items():
            c.SetValue(getattr(o, key))
        self._tr()

    def _checked_nets(self):
        return [self._net_names[k] for k in self.nets.GetCheckedItems()] if hasattr(self, "_net_names") else []

    def _from_ui(self, quiet=False):
        o = X.Options(**asdict(self.opts))
        for key, (t, kind) in self.fields.items():
            try:
                v = parse_si(t.GetValue()) if kind == "si" else float(t.GetValue())
            except (ValueError, TypeError):
                if not quiet:
                    wx.MessageBox(f"{key}: not a number.", "Options", parent=self)
                return False
            setattr(o, key, v)
        for key, c in self.checks.items():
            setattr(o, key, c.GetValue())
        if o.f_max <= 0 or o.sample <= 0 or o.window <= 0:
            if not quiet:
                wx.MessageBox("f max, cross-section spacing and search width must be positive.", "Options",
                              parent=self)
            return False
        self.opts = o
        return True

    def _tr(self):
        try:
            f = parse_si(self.fields["f_max"][0].GetValue())
            self.tr.SetLabel(f"≈ rise time {fmt_si(0.35 / f, 's')}, λ/10 in FR4 ≈ "
                             f"{299792458 / (f * math.sqrt(3.3)) / 10 * 1e3:.3g} mm")
        except (ValueError, TypeError, ZeroDivisionError):
            self.tr.SetLabel("")

    # ------------------------------------------------------------- ports
    def _net(self):
        return next((p.net for p in self.board.pads if self.ports and p.id == self.ports[0]), None)

    def _add_pad(self):
        with PadDialog(self, self.board, self._net()) as d:
            if d.ShowModal() == wx.ID_OK and d.result and d.result not in self.ports:
                first = not self.ports
                self.ports.append(d.result)
                if first and not self._checked_nets():
                    self.return_nets = None
                self._to_ui()

    def _del_pad(self):
        k = self.plist.GetFirstSelected()
        if k >= 0:
            del self.ports[k]
            self._to_ui()

    def _move(self, d):
        k = self.plist.GetFirstSelected()
        j = k + d
        if k >= 0 and 0 <= j < len(self.ports):
            self.ports[k], self.ports[j] = self.ports[j], self.ports[k]
            self._to_ui()
            self.plist.Select(j)

    def _from_kicad(self):
        try:
            ids = self.session.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return wx.MessageBox(f"Could not read the selection: {e}", "KiCad", parent=self)
        if len(ids) < 2:
            return wx.MessageBox("Select two or more pads in the PCB editor (the selection filter can limit "
                                 "it to pads). Save the board first if you changed it.", "KiCad selection",
                                 parent=self)
        self.ports = ids
        self._to_ui()

    def _el_dialog(self, el):
        from .app import ElementDialog
        with ElementDialog(self, self.board, el, models=("R", "V", "C", "L"), dc=False) as d:
            if d.ShowModal() == wx.ID_OK and d.result:
                return d.result
        return None

    def _el_add(self, preset=None):
        r = self._el_dialog(preset or EL.Element("", "", "", 0.0))
        if r:
            self.elements = [e for e in self.elements if e.name != r.name] + [r]
            self._to_ui()

    def _el_edit(self):
        k = self.elist.GetFirstSelected()
        if k >= 0:
            r = self._el_dialog(self.elements[k])
            if r:
                r.enabled = self.elements[k].enabled
                self.elements[k] = r
                self._to_ui()

    def _el_toggle(self):
        k = self.elist.GetFirstSelected()
        if k >= 0:
            self.elements[k].enabled = not self.elements[k].enabled
            self._to_ui()

    def _el_del(self):
        k = self.elist.GetFirstSelected()
        if k >= 0:
            del self.elements[k]
            self._to_ui()

    def _el_from_kicad(self):
        try:
            ids = self.session.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return wx.MessageBox(f"Could not read the selection: {e}", "KiCad", parent=self)
        if len(ids) != 2:
            return wx.MessageBox("Select exactly two pads in the PCB editor (the element's terminals).",
                                 "KiCad selection", parent=self)
        self._el_add(EL.Element(ids[0].rsplit(".", 1)[0], ids[0], ids[1], 0.0))

    def _guess(self):
        g = X.guess_return_nets(self.board, self._net())
        for k, n in enumerate(self._net_names):
            self.nets.Check(k, n in g)

    def _select_kicad(self):
        if not self.ext:
            return
        try:
            n = self.session.kicad.select_items(self.ext.uuids)
            self._status(f"Selected {n} track(s)/via(s) in KiCad.")
        except Exception as e:      # noqa: BLE001
            wx.MessageBox(f"KiCad: {e}", "KiCad", parent=self)

    # ------------------------------------------------------------- run
    def _run(self):
        if self._running:
            return
        if len(self.ports) < 2:
            return wx.MessageBox("Add at least two pads.", "Ports", parent=self)
        if not self._from_ui():
            return
        nets = self._checked_nets()
        if not nets and wx.MessageBox("No return net is ticked: only R (and the partial L of the trace) can be "
                                      "extracted. Continue?", "Return nets", wx.YES_NO, self) != wx.YES:
            return
        self.save()
        board, st, ports, opts = self.board, self.session.stackup, list(self.ports), X.Options(**asdict(self.opts))
        els = [EL.Element(**asdict(e)) for e in self.elements]
        self._running = True
        self.run_b.Disable()

        def work():
            try:
                ext = X.Extractor(board, st, nets, opts, progress=lambda m: wx.CallAfter(self._status, m),
                                  elements=els).run(ports)
                wx.CallAfter(self._done, ext, None)
            except ValueError as e:
                wx.CallAfter(self._done, None, str(e))
            except Exception:      # noqa: BLE001
                wx.CallAfter(self._done, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _done(self, ext, err):
        self._running = False
        self.run_b.Enable()
        if err:
            self._status("Extraction failed.")
            return wx.MessageBox(err, "Extraction", parent=self)
        self.ext = ext
        self._show()
        s = ext.summary
        self._status(f"Done in {s['seconds']:.1f} s: {s['sections']} section(s), {s['field_solves']} field solve(s).")

    # ------------------------------------------------------------- results
    def _show(self):
        e = self.ext
        self.pairs.DeleteAllItems()
        for pr in e.pairs:
            row = []
            for title, key, _, fmt in PAIR_COLS:
                if title == "Pads":
                    row.append(f"{pr['a']} → {pr['b']}")
                elif title == "Z0 min–max":
                    row.append(f"{_cell('{:.0f}', pr['Z0_min'])}–{_cell('{:.0f}', pr['Z0_max'])}")
                else:
                    row.append(_cell(fmt, pr[key]))
            self.pairs.Append(row)
        s, o = e.summary, e.opts
        lines = [f"Net {e.net}; ports " + ", ".join(f"P{k + 1} = {p}" for k, p in enumerate(e.ports)),
                 f"Return nets: {s['return_nets']}",
                 f"Total shunt C (with stubs, pads, vias): {s['C_total_pF']:.3f} pF",
                 f"Ladder: {s['sections']} RL section(s), each ≤ λ/10 at {fmt_si(o.f_max, 'Hz')}"
                 + (f" and ≤ {o.max_section:g} mm" if o.max_section > 0 else ""),
                 f"R from ρl/(wt): {s['R_dc_formula_mOhm']:.3f} mΩ (P1→P2)"]
        if "R_dc_field_mOhm" in s:
            lines.append(f"R from the DC field solution: {s['R_dc_field_mOhm']:.3f} mΩ (pads, crowding, vias "
                         f"included); ladder R scaled × {s['R_scale']:.3f}")
        lines.append(f"Copper at {o.T:g} °C. R in the ladder is DC; the table's R'(f) is the skin-effect value "
                     f"at {fmt_si(o.f_max, 'Hz')} (current on the face towards the return).")
        if e.pads:
            lines.append("Pad C: " + ", ".join(f"{p['pad']} {p['C_pF']:.3f} pF" for p in e.pads))
        if e.warnings:
            lines += ["", "Warnings:"] + ["• " + w for w in e.warnings]
        self.info.SetValue("\n".join(lines))
        self.segs.DeleteAllItems()
        for r in e.segments:
            self.segs.Append([_cell(fmt, r.get(key)) for _, key, _, fmt in SEG_COLS])
        self.vias.DeleteAllItems()
        for v in e.vias:
            self.vias.Append([f"via ({v['x0']:.2f}, {v['y0']:.2f})", v["layer"], f"{v['R_mOhm']:.3f}",
                              f"{v['L_nH']:.3f}", f"{v['C_pF']:.3f}",
                              v["reference"] + (f"; antipad r {v['antipad_mm']:.2f} mm" if v["antipad_mm"] else
                                                "; no return plane around it")])
        for p in e.pads:
            self.vias.Append([f"pad {p['pad']}", p["port"] or "stub end", "", "", f"{p['C_pF']:.3f}",
                              p["reference"]])
        for z in e.zones:
            self.vias.Append([z["item"], f"{z['terminals']} terminals", f"{z['R_mOhm']:.3f}", f"{z['L_nH']:.3f}",
                              f"{z['C_pF']:.3f}", f"{z['area_mm2']:.0f} mm², mesh {z['pitch_mm']:.2f} mm; {z['note']}; "
                              "R and L: worst terminal pair"])
        for u in e.user:
            self.vias.Append([f"{u['model']} {u['name']}", u["nodes"], "", "", "", f"{u['value']} ({u['a']} – {u['b']})"])
        self.spice.SetValue(SP.write(e, board_name=os.path.basename(self.session.board_path or "")))
        self._plot()
        cells = X.main_path(e, "P1", "P2")
        self.ladder.show(cells, f"P1 = {e.ports[0]} → P2 = {e.ports[1]}; values as in the SPICE subcircuit"
                         + ("; other ports and stubs appear as branch boxes" if len(e.ports) > 2 or
                            any(c["kind"] == "branch" for c in cells) else ""))

    def _plot(self):
        e, b = self.ext, self.board
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ret = set(e.return_nets)
        for z in b.zones:
            if z.net in ret:
                ax.fill(z.geom.pts[:, 0], z.geom.pts[:, 1], color="0.85", lw=0, zorder=0)
        zsel = {zz["zone"] for zz in e.zones}
        for k in zsel:
            z = b.zones[k]
            ax.fill(z.geom.pts[:, 0], z.geom.pts[:, 1], color="C1", alpha=0.35, lw=0, zorder=1)
        for t in b.tracks:
            if t.net == e.net:
                ax.plot([t.start[0], t.end[0]], [t.start[1], t.end[1]], color="0.6", lw=1, zorder=1)
        segs = [[p["p0"], p["p1"]] for p in e.pieces]
        z = np.array([p["z0"] for p in e.pieces], float)
        if segs:
            ok = np.isfinite(z)
            lc = LineCollection([s for s, k in zip(segs, ok) if k], array=z[ok], cmap="viridis", linewidths=4,
                                zorder=2)
            if ok.any():
                ax.add_collection(lc)
                self.fig.colorbar(lc, ax=ax, label="Z0 [Ω]", shrink=0.8)
            bad = [s for s, k in zip(segs, ok) if not k]
            if bad:
                ax.add_collection(LineCollection(bad, colors="red", linewidths=4, linestyles="dashed", zorder=2,
                                                 label="no return in window"))
        for v in e.vias:
            ax.plot(v["x0"], v["y0"], "o", ms=7, mfc="none", mec="k", zorder=3)
        pads = {p.id: p for p in b.pads}
        for k, pid in enumerate(e.ports):
            p = pads[pid]
            ax.plot(*p.pos, "s", ms=8, color="C3", zorder=4)
            ax.annotate(f"P{k + 1} {pid}", p.pos, textcoords="offset points", xytext=(6, 6), fontsize=9,
                        weight="bold")
        for m in e.markers:
            ax.plot(*m["pos"], "x", color="C1", ms=10, mew=2, zorder=5)
            ax.annotate(m["text"], m["pos"], textcoords="offset points", xytext=(6, -12), fontsize=8, color="C1")
        zb = [b.zones[k].geom.bbox for k in zsel]
        xs = [q for s in segs for q in (s[0][0], s[1][0])] + [pads[p].pos[0] for p in e.ports] + \
            [q for bb in zb for q in (bb[0], bb[2])]
        ys = [q for s in segs for q in (s[0][1], s[1][1])] + [pads[p].pos[1] for p in e.ports] + \
            [q for bb in zb for q in (bb[1], bb[3])]
        m = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.1 + 1
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(max(ys) + m, min(ys) - m)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{e.net}: tracks coloured by Z0, extracted pours orange, return-net zones grey", fontsize=10)
        self.canvas.draw()

    # ------------------------------------------------------------- export
    def _save_spice(self):
        if not self.ext:
            return
        name = SP.subckt_name(self.ext)
        with wx.FileDialog(self, "Save SPICE subcircuit", defaultFile=name.lower() + ".lib",
                           wildcard="SPICE library (*.lib;*.sub;*.cir)|*.lib;*.sub;*.cir|All|*",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() == wx.ID_OK:
                with open(d.GetPath(), "w") as f:
                    f.write(self.spice.GetValue())
                self._status(f"Saved {d.GetPath()}")

    def _copy_spice(self):
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(self.spice.GetValue()))
            wx.TheClipboard.Close()
            self._status("Subcircuit copied.")

    def _csv(self):
        if not self.ext:
            return
        with wx.FileDialog(self, "Export segments", defaultFile="segments.csv", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            with open(d.GetPath(), "w", newline="") as f:
                w = csv.writer(f)
                keys = list(self.ext.segments[0]) if self.ext.segments else []
                w.writerow(keys)
                for r in self.ext.segments:
                    w.writerow([r.get(k) for k in keys])
                if self.ext.vias:
                    w.writerow([])
                    vk = list(self.ext.vias[0])
                    w.writerow(vk)
                    for v in self.ext.vias:
                        w.writerow([v.get(k) for k in vk])
            self._status(f"Saved {d.GetPath()}")
