"""3D extraction (wx): ports between pads, R(f) / L(f) / Q / coupling, current maps, wideband SPICE."""
import csv
import math
import threading
import traceback

import numpy as np
import wx

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import elements as EL  # noqa: E402
from .. import scenario as SC  # noqa: E402
from ..em3d import analysis as A  # noqa: E402
from ..em3d import pcb  # noqa: E402
from ..em3d import spice as SP  # noqa: E402
from ..suite.library import fmt_si, parse_si  # noqa: E402
from .crossprobe import CrossProbe, selected_rows  # noqa: E402
from .layout import AutoList, refit, scrolled  # noqa: E402
from .suite import attach_suite_menu  # noqa: E402

OPTS = (("f_min", "f min [Hz]", "si", "Lowest frequency of the results (DC is always exact)"),
        ("f_max", "f max [Hz]", "si", "Highest frequency of the results and of the wideband model"),
        ("f_disc", "Filaments for [Hz] (blank = f max)", "si?",
         "The filaments are made thin enough for the skin depth at this frequency (blank: f max). Lower it to "
         "solve faster when the top of the band matters less"),
        ("pitch", "Pour / pad cell [mm]", float,
         "Cell size of pours and pads (tracks keep their true width). Smaller: finer current maps in pours, "
         "more filaments"),
        ("margin", "Margin around the copper [mm]", float,
         "The window: copper of the ticked nets within this distance of the ports' copper is included"),
        ("n_width", "Filaments across a track (max)", int,
         "Most filaments across a track's width (graded towards the edges); fewer when the skin depth allows"),
        ("n_height", "Filaments through a track (max)", int,
         "Most filaments through a track's copper thickness (graded towards both faces)"),
        ("n_pour", "Filaments through a pour (max)", int,
         "Most filaments through the thickness of pour and pad cells (pours dominate the filament count)"),
        ("max_fils", "Filament limit", int,
         "Upper limit on the filament count: above it the filaments per track are reduced (with a warning). "
         "Memory grows as the square: about 5 GB at 12000"),
        ("points", "Frequency points", int, "Frequencies in the results (log spaced, f min to f max)"))
PORT_COLS = (("Port", 70), ("+ pads", 150), ("− pads", 150), ("Nets", 160))
SUM_COLS = ("Port", "R dc [mΩ]", "L dc [nH]", "R mid [mΩ]", "L mid [nH]", "R top [mΩ]", "L top [nH]",
            "R top / R dc", "Q max", "at [Hz]")


def _num(x, fmt="{:.4g}"):
    return fmt.format(x) if x is not None and np.isfinite(x) else "–"


class PadGroupDialog(wx.Dialog):
    """A port (name, + pads, − pads) or a link (two pad groups)."""

    def __init__(self, parent, board, title, labels, groups, name=None, tips=()):
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.board, self.groups = board, [list(g) for g in groups]
        top = wx.BoxSizer(wx.VERTICAL)
        self.name = None
        if name is not None:
            row = wx.BoxSizer(wx.HORIZONTAL)
            row.Add(wx.StaticText(self, label="Name"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
            self.name = wx.TextCtrl(self, value=name, size=(160, -1))
            self.name.SetToolTip("The port's name (in the results and the SPICE pins NAME_p / NAME_m)")
            row.Add(self.name, 0)
            top.Add(row, 0, wx.ALL, 8)
        cols = wx.BoxSizer(wx.HORIZONTAL)
        self.lists = []
        for k, label in enumerate(labels):
            box = wx.StaticBoxSizer(wx.VERTICAL, self, label)
            sb = box.GetStaticBox()
            lb = wx.ListBox(sb, size=(190, 140), style=wx.LB_EXTENDED)
            lb.SetToolTip(tips[k] if k < len(tips) else label)
            box.Add(lb, 1, wx.EXPAND | wx.ALL, 4)
            row = wx.BoxSizer(wx.HORIZONTAL)
            add = wx.Button(sb, label="Add pad…", style=wx.BU_EXACTFIT)
            add.SetToolTip("Add a pad to this side (all pads of a side are one node: a part's pins in parallel)")
            add.Bind(wx.EVT_BUTTON, lambda e, k=k: self._add(k))
            rem = wx.Button(sb, label="Remove", style=wx.BU_EXACTFIT)
            rem.SetToolTip("Remove the selected pads from this side")
            rem.Bind(wx.EVT_BUTTON, lambda e, k=k: self._rem(k))
            row.Add(add, 0, wx.ALL, 2)
            row.Add(rem, 0, wx.ALL, 2)
            box.Add(row, 0)
            cols.Add(box, 1, wx.EXPAND | wx.ALL, 4)
            self.lists.append(lb)
        top.Add(cols, 1, wx.EXPAND | wx.ALL, 4)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizerAndFit(top)
        self._fill()
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)

    def _fill(self):
        nets = {p.id: p.net for p in self.board.pads}
        for lb, g in zip(self.lists, self.groups):
            lb.Set([f"{q}  ({nets.get(q, '?')})" for q in g])

    def _add(self, k):
        from .extract import PadDialog
        with PadDialog(self, self.board) as d:
            if d.ShowModal() == wx.ID_OK and d.result and d.result not in self.groups[k]:
                self.groups[k].append(d.result)
                self._fill()

    def _rem(self, k):
        for i in sorted(self.lists[k].GetSelections(), reverse=True):
            del self.groups[k][i]
        self._fill()

    def _ok(self, e):
        if any(not g for g in self.groups):
            return wx.MessageBox("Each side needs at least one pad.", self.GetTitle(), parent=self)
        if set(self.groups[0]) & set(self.groups[1]):
            return wx.MessageBox("A pad cannot be on both sides.", self.GetTitle(), parent=self)
        if self.name is not None and not self.name.GetValue().strip():
            return wx.MessageBox("Give the port a name.", self.GetTitle(), parent=self)
        self.EndModal(wx.ID_OK)


class Em3dFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="3D extraction - PCB analysis suite", size=(1500, 960))
        self.session, self.board = session, session.board
        self.s, self.res, self._running = pcb.Setup(), None, False
        self._load()
        self._build()
        attach_suite_menu(self, session, "extract3d")
        self.probe = CrossProbe(self, session)
        ref = lambda q: q.rsplit(".", 1)[0]           # noqa: E731
        self.probe.watch(self.plist, lambda lc: [ref(q) for i in selected_rows(lc) if i < len(self.s.ports)
                                                 for q in self.s.ports[i].plus + self.s.ports[i].minus])
        self.probe.watch(self.llist, lambda lc: [ref(q) for i in selected_rows(lc) if i < len(self.s.links)
                                                 for q in self.s.links[i][0] + self.s.links[i][1]])
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._to_ui()

    # ------------------------------------------------------------- persistence
    def _load(self):
        try:
            d = SC.section(self.session.scenario_path, "extract3d") if self.session.scenario_path else {}
        except (OSError, ValueError):
            d = {}
        self.s = pcb.Setup.from_dict(d.get("setup") or {})
        self._prune()

    def _prune(self):
        ids = {p.id for p in self.board.pads}
        for p in self.s.ports:
            p.plus = [q for q in p.plus if q in ids]
            p.minus = [q for q in p.minus if q in ids]
        self.s.ports = [p for p in self.s.ports if p.plus and p.minus]
        self.s.links = [[[q for q in a if q in ids], [q for q in b if q in ids]] for a, b in self.s.links]
        self.s.links = [l for l in self.s.links if l[0] and l[1]]
        self.s.nets = [n for n in self.s.nets if n in self.board.nets]

    def save(self):
        if not self.session.scenario_path:
            return
        try:
            SC.save_section(self.session.scenario_path, "extract3d", dict(setup=self.s.to_dict()))
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
            self._prune()
            wx.CallAfter(self._to_ui)

    # ------------------------------------------------------------- UI
    def _build(self):
        sp = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = self.left = scrolled(sp)
        ls = wx.BoxSizer(wx.VERTICAL)

        pb = wx.StaticBoxSizer(wx.VERTICAL, left, "Ports (current in at + pads, out at − pads)")
        pbox = pb.GetStaticBox()
        self.plist = AutoList(pbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 120))
        for k, (t, w) in enumerate(PORT_COLS):
            self.plist.InsertColumn(k, t, width=w)
        self.plist.SetToolTip("Each port is a pair of pad groups joined by the copper: its impedance R(f) + jωL(f) "
                              "is what the tool computes; with two or more ports also their mutual inductance and "
                              "coupling. Double-click to edit")
        self.plist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._port_edit())
        pb.Add(self.plist, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn, tip in (("Add…", self._port_add, "Add a port: a name, its + pads and its − pads"),
                              ("Edit…", self._port_edit, "Edit the selected port"),
                              ("Remove", self._port_del, "Remove the selected ports")):
            b = wx.Button(pbox, label=text, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        if self.session.kicad is not None:
            b = wx.Button(pbox, label="From KiCad selection", style=wx.BU_EXACTFIT)
            b.SetToolTip("Two pads selected in the PCB editor: a new port from the first (+) to the second (−)")
            b.Bind(wx.EVT_BUTTON, lambda e: self._port_from_kicad())
            row.Add(b, 0, wx.ALL, 2)
        pb.Add(row, 0)
        ls.Add(pb, 0, wx.EXPAND | wx.ALL, 6)

        lb = wx.StaticBoxSizer(wx.VERTICAL, left, "Links (ideal shorts between pads)")
        lbox = lb.GetStaticBox()
        self.llist = AutoList(lbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 80))
        for k, (t, w) in enumerate((("Pads", 170), ("joined to", 170))):
            self.llist.InsertColumn(k, t, width=w)
        self.llist.SetToolTip("Pads joined as one node, e.g. a load's supply and ground pins, so that a port at the "
                              "source sees the whole loop out on the supply copper and back on the return. "
                              "Double-click to edit")
        self.llist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._link_edit())
        lb.Add(self.llist, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn, tip in (("Add…", self._link_add, "Join two pad groups with an ideal short"),
                              ("Edit…", self._link_edit, "Edit the selected link"),
                              ("Remove", self._link_del, "Remove the selected links")):
            b = wx.Button(lbox, label=text, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        if self.session.kicad is not None:
            b = wx.Button(lbox, label="From KiCad selection", style=wx.BU_EXACTFIT)
            b.SetToolTip("Two pads selected in the PCB editor: join them with a link")
            b.Bind(wx.EVT_BUTTON, lambda e: self._link_from_kicad())
            row.Add(b, 0, wx.ALL, 2)
        lb.Add(row, 0)
        ls.Add(lb, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        nb_ = wx.StaticBoxSizer(wx.VERTICAL, left, "Conductors (nets)")
        nbox = nb_.GetStaticBox()
        self.nets = wx.CheckListBox(nbox, size=(-1, 150))
        self.nets.SetToolTip("The nets whose copper is solved. The ports' and links' nets are always in. Tick others "
                             "to include them as passive conductors: eddy currents in a ground plane under a coil, "
                             "or in a neighbouring trace, change L and R")
        nb_.Add(self.nets, 1, wx.EXPAND | wx.ALL, 4)
        ls.Add(nb_, 0, wx.EXPAND | wx.ALL, 6)

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
        self.skin = wx.StaticText(obox, label="")
        ob.Add(self.skin, 0, wx.ALL, 4)
        for k in ("f_max", "f_disc"):
            self.fields[k][0].Bind(wx.EVT_TEXT, lambda e: self._skin())
        ls.Add(ob, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.run_b = wx.Button(left, label="Analyse")
        self.run_b.SetToolTip("Mesh the copper, solve the filament network and build the wideband model")
        self.run_b.Bind(wx.EVT_BUTTON, lambda e: self._run())
        ls.Add(self.run_b, 0, wx.EXPAND | wx.ALL, 6)
        from .tests import add_button
        ls.Add(add_button(left, self, self.test_setup), 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        ls.Add(wx.StaticText(left, label=(
            "Quasi-static 3D: copper cut into filaments (FastHenry's method),\n"
            "so skin and proximity effect come out of the solve. Valid while\n"
            "the copper is shorter than about λ/10 (no radiation, no C yet).\n"
            "No magnetic materials yet. Save the board in KiCad first.")), 0, wx.ALL, 6)
        left.SetSizer(ls)
        left.FitInside()

        nb = self.nb = wx.Notebook(sp)
        # summary
        sm = wx.Panel(nb)
        ss = wx.BoxSizer(wx.VERTICAL)
        self.sum = AutoList(sm, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 120))
        for k, t in enumerate(SUM_COLS):
            self.sum.InsertColumn(k, t, width=90)
        ss.Add(self.sum, 0, wx.EXPAND | wx.ALL, 4)
        ss.Add(wx.StaticText(sm, label="Coupling between ports"), 0, wx.LEFT, 6)
        self.coup = AutoList(sm, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 90))
        for k, t in enumerate(("Ports", "M dc [nH]", "k dc", "M mid [nH]", "k mid", "M top [nH]", "k top")):
            self.coup.InsertColumn(k, t, width=90)
        ss.Add(self.coup, 0, wx.EXPAND | wx.ALL, 4)
        self.info = wx.TextCtrl(sm, style=wx.TE_MULTILINE | wx.TE_READONLY)
        ss.Add(self.info, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(sm, label="Export CSV…")
        b.SetToolTip("R, L of every port and the couplings at every frequency")
        b.Bind(wx.EVT_BUTTON, lambda e: self._csv())
        row.Add(b, 0, wx.ALL, 4)
        ss.Add(row, 0)
        sm.SetSizer(ss)
        nb.AddPage(sm, "Summary")
        # impedance
        ip = wx.Panel(nb)
        self.zfig = Figure(figsize=(7, 6), constrained_layout=True)
        self.zcanvas = FigureCanvasWxAgg(ip, -1, self.zfig)
        ztb = NavigationToolbar2WxAgg(self.zcanvas)
        ztb.Realize()
        isz = wx.BoxSizer(wx.VERTICAL)
        isz.Add(self.zcanvas, 1, wx.EXPAND)
        isz.Add(ztb, 0, wx.EXPAND)
        ip.SetSizer(isz)
        nb.AddPage(ip, "R, L, Q (f)")
        # current map
        cp = wx.Panel(nb)
        cs = wx.BoxSizer(wx.VERTICAL)
        bar = wx.WrapSizer()
        self.cport = wx.Choice(cp)
        self.cport.SetToolTip("The port driven with 1 A (the others open)")
        self.cfreq = wx.Choice(cp)
        self.cfreq.SetToolTip("Frequency of the map: the current spreads at DC and crowds to edges, faces and "
                              "towards the return at high frequency")
        self.clayer = wx.Choice(cp)
        self.clayer.SetToolTip("Copper layer shown")
        self.clog = wx.CheckBox(cp, label="Log scale")
        self.clog.SetToolTip("Colour on a logarithmic scale (current density spans decades)")
        self.clog.SetValue(True)
        for lab, c in (("Port", self.cport), ("Frequency", self.cfreq), ("Layer", self.clayer)):
            bar.Add(wx.StaticText(cp, label=lab), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
            bar.Add(c, 0, wx.ALL, 3)
            c.Bind(wx.EVT_CHOICE, lambda e: self._map())
        bar.Add(self.clog, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 6)
        self.clog.Bind(wx.EVT_CHECKBOX, lambda e: self._map())
        cs.Add(bar, 0)
        self.cfig = Figure(figsize=(7, 6), constrained_layout=True)
        self.ccanvas = FigureCanvasWxAgg(cp, -1, self.cfig)
        ctb = NavigationToolbar2WxAgg(self.ccanvas)
        ctb.Realize()
        cs.Add(self.ccanvas, 1, wx.EXPAND)
        cs.Add(ctb, 0, wx.EXPAND)
        cs.Add(wx.StaticText(cp, label="Click a track to see the current across its cross-section."), 0, wx.ALL, 4)
        cp.SetSizer(cs)
        self.ccanvas.mpl_connect("button_press_event", self._pick)
        nb.AddPage(cp, "Current")
        # spice
        spp = wx.Panel(nb)
        sps = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.skind = wx.Choice(spp, choices=["Wideband (R and L right at every frequency)",
                                             "At one frequency (R, L, coupled K)"])
        self.skind.SetToolTip("Wideband: a Foster network fitted exactly to the solve (one port: plain R-L "
                              "branches; several: R-L modes with E / F controlled sources). One frequency: plain "
                              "R, L and K, right only near that frequency")
        self.skind.SetSelection(0)
        self.sfreq = wx.TextCtrl(spp, value="1M", size=(90, -1))
        self.sfreq.SetToolTip("Frequency of the one-frequency model [Hz]")
        self.skind.Bind(wx.EVT_CHOICE, lambda e: self._spice())
        self.sfreq.Bind(wx.EVT_TEXT, lambda e: self._spice())
        row.Add(self.skind, 0, wx.ALL, 4)
        row.Add(wx.StaticText(spp, label="f [Hz]"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        row.Add(self.sfreq, 0, wx.ALL, 4)
        sps.Add(row, 0)
        self.spice = wx.TextCtrl(spp, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_DONTWRAP)
        self.spice.SetFont(wx.Font(wx.FontInfo(9).Family(wx.FONTFAMILY_TELETYPE)))
        sps.Add(self.spice, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        for text, fn, tip in (("Save subcircuit…", self._save_spice, "Save the subcircuit as a SPICE library file"),
                              ("Copy", self._copy_spice, "Copy the subcircuit to the clipboard")):
            b = wx.Button(spp, label=text)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 4)
        sps.Add(row, 0)
        spp.SetSizer(sps)
        nb.AddPage(spp, "SPICE")

        sp.SplitVertically(left, nb, 480)
        sp.SetMinimumPaneSize(300)
        self.CreateStatusBar()
        self._status("Add a port (two pads joined by copper), tick nearby nets as passive conductors, then Analyse.")

    def _status(self, t):
        self.SetStatusText(t)

    # ------------------------------------------------------------- setup <-> UI
    def _port_nets(self):
        nets = {p.id: p.net for p in self.board.pads}
        out = []
        for p in self.s.ports:
            out += [nets.get(q) for q in p.plus + p.minus]
        for a, b in self.s.links:
            out += [nets.get(q) for q in a + b]
        return [n for n in dict.fromkeys(out) if n]

    def _to_ui(self):
        nets = {p.id: p.net for p in self.board.pads}
        self.plist.DeleteAllItems()
        for p in self.s.ports:
            ns = ", ".join(dict.fromkeys(nets.get(q, "?") for q in p.plus + p.minus))
            self.plist.Append([p.name, ", ".join(p.plus), ", ".join(p.minus), ns])
        self.llist.DeleteAllItems()
        for a, b in self.s.links:
            self.llist.Append([", ".join(a), ", ".join(b)])
        forced = set(self._port_nets())
        zone_nets = {z.net for z in self.board.zones}
        names = sorted(self.board.nets, key=lambda n: (n not in forced, n not in zone_nets, EL.natural_key(n)))
        checked = set(self._checked_nets()) if getattr(self, "_net_names", None) else set(self.s.nets)
        self._net_names = names
        self.nets.Set([n + ("  (port)" if n in forced else "  (pours)" if n in zone_nets else "") for n in names])
        for k, n in enumerate(names):
            self.nets.Check(k, n in forced or n in checked)
        for key, (t, kind) in self.fields.items():
            v = getattr(self.s, key)
            if kind in ("si", "si?"):
                t.ChangeValue("" if v is None else fmt_si(v, "").strip())
            else:
                t.ChangeValue(f"{v:g}")
        self._skin()
        refit(self.left)

    def _checked_nets(self):
        return [self._net_names[k] for k in self.nets.GetCheckedItems()] if getattr(self, "_net_names", None) else []

    def _from_ui(self, quiet=False):
        s = pcb.Setup.from_dict(self.s.to_dict())
        for key, (t, kind) in self.fields.items():
            txt = t.GetValue().strip()
            try:
                if kind == "si?" and not txt:
                    v = None
                elif kind in ("si", "si?"):
                    v = parse_si(txt)
                elif kind is int:
                    v = int(float(txt))
                else:
                    v = float(txt)
            except (ValueError, TypeError):
                if not quiet:
                    wx.MessageBox(f"{key}: not a number.", "Options", parent=self)
                return False
            setattr(s, key, v)
        if not (0 < s.f_min < s.f_max) or s.pitch <= 0 or s.margin < 0 or min(s.n_width, s.n_height, s.n_pour) < 1 \
                or s.max_fils < 10 or s.points < 2:
            if not quiet:
                wx.MessageBox("Check the options: 0 < f min < f max, a positive cell size, at least one filament.",
                              "Options", parent=self)
            return False
        forced = set(self._port_nets())
        s.nets = [n for n in self._checked_nets() if n not in forced]
        self.s = s
        return True

    def _skin(self):
        try:
            f = parse_si(self.fields["f_disc"][0].GetValue() or self.fields["f_max"][0].GetValue())
            d = math.sqrt(1.72e-8 / (math.pi * f * 4e-7 * math.pi)) * 1e6
            self.skin.SetLabel(f"Skin depth in copper at {fmt_si(f, 'Hz')}: {d:.3g} µm")
        except (ValueError, TypeError, ZeroDivisionError):
            self.skin.SetLabel("")

    # ------------------------------------------------------------- ports and links
    def _port_dialog(self, port):
        with PadGroupDialog(self, self.board, "Port", ("+ pads (current in)", "− pads (current out)"),
                            (port.plus, port.minus), name=port.name,
                            tips=("Pads where the port's current enters the copper",
                                  "Pads where it leaves (on the same copper, or joined to it by a link)")) as d:
            if d.ShowModal() == wx.ID_OK:
                return pcb.Port(d.name.GetValue().strip(), d.groups[0], d.groups[1])
        return None

    def _unique(self, name, skip=None):
        used = {p.name for k, p in enumerate(self.s.ports) if k != skip}
        base, n = name, 2
        while name in used:
            name, n = f"{base}{n}", n + 1
        return name

    def _port_add(self, plus=(), minus=()):
        self._from_ui(quiet=True)
        p = self._port_dialog(pcb.Port(self._unique(f"P{len(self.s.ports) + 1}"), list(plus), list(minus)))
        if p:
            p.name = self._unique(p.name)
            self.s.ports.append(p)
            self._changed()

    def _port_edit(self):
        k = self.plist.GetFirstSelected()
        if k < 0:
            return
        self._from_ui(quiet=True)
        p = self._port_dialog(self.s.ports[k])
        if p:
            p.name = self._unique(p.name, skip=k)
            self.s.ports[k] = p
            self._changed()

    def _port_del(self):
        sel = selected_rows(self.plist)
        if sel:
            self._from_ui(quiet=True)
            for k in sorted(sel, reverse=True):
                del self.s.ports[k]
            self._changed()

    def _kicad_pads(self, n):
        try:
            ids = self.session.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            wx.MessageBox(f"Could not read the selection: {e}", "KiCad", parent=self)
            return None
        if len(ids) != n:
            wx.MessageBox(f"Select exactly {n} pads in the PCB editor (the selection filter can limit it to pads). "
                          "Save the board first if you changed it.", "KiCad selection", parent=self)
            return None
        return ids

    def _port_from_kicad(self):
        ids = self._kicad_pads(2)
        if ids:
            self._port_add([ids[0]], [ids[1]])

    def _link_dialog(self, a, b):
        with PadGroupDialog(self, self.board, "Link", ("Pads", "joined to"), (a, b),
                            tips=("One side of the short", "The other side")) as d:
            if d.ShowModal() == wx.ID_OK:
                return [d.groups[0], d.groups[1]]
        return None

    def _link_add(self, a=(), b=()):
        self._from_ui(quiet=True)
        l = self._link_dialog(list(a), list(b))
        if l:
            self.s.links.append(l)
            self._changed()

    def _link_edit(self):
        k = self.llist.GetFirstSelected()
        if k < 0:
            return
        self._from_ui(quiet=True)
        l = self._link_dialog(*self.s.links[k])
        if l:
            self.s.links[k] = l
            self._changed()

    def _link_del(self):
        sel = selected_rows(self.llist)
        if sel:
            self._from_ui(quiet=True)
            for k in sorted(sel, reverse=True):
                del self.s.links[k]
            self._changed()

    def _link_from_kicad(self):
        ids = self._kicad_pads(2)
        if ids:
            self._link_add([ids[0]], [ids[1]])

    def _changed(self):
        self._to_ui()
        self.save()

    # ------------------------------------------------------------- run
    def test_setup(self):
        if not self.s.ports:
            wx.MessageBox("Add a port first.", "Save as test", parent=self)
            return None
        if not self._from_ui():
            return None
        self.save()
        return "extract3d", self.s.to_dict(), "3D " + ", ".join(p.name for p in self.s.ports)

    def _run(self):
        if self._running:
            return
        if not self.s.ports:
            return wx.MessageBox("Add a port: two pads (or pad groups) joined by the copper.", "Ports", parent=self)
        if not self._from_ui():
            return
        self.save()
        board, st, s = self.board, self.session.stackup, pcb.Setup.from_dict(self.s.to_dict())
        self._running = True
        self.run_b.Disable()

        def work():
            try:
                r = A.run(board, st, s, progress=lambda m: wx.CallAfter(self._status, m))
                wx.CallAfter(self._done, r, None)
            except (ValueError, MemoryError) as e:
                wx.CallAfter(self._done, None, str(e))
            except Exception:      # noqa: BLE001
                wx.CallAfter(self._done, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _done(self, r, err):
        self._running = False
        self.run_b.Enable()
        if err:
            self._status("Analysis failed.")
            return wx.MessageBox(err, "3D extraction", parent=self)
        self.res = r
        self._show()
        self._status(f"Done in {r.seconds:.1f} s: {r.n_fils} filaments, model order {r.model.order}.")

    # ------------------------------------------------------------- results
    def _show(self):
        r = self.res
        Rdc, Ldc = r.dc()
        mid = len(r.f) // 2
        self.sum.DeleteAllItems()
        for q, n in enumerate(r.names):
            Q = r.Q[:, q]
            k = int(np.nanargmax(Q))
            self.sum.Append([n, _num(Rdc[q] * 1e3), _num(Ldc[q, q] * 1e9), _num(r.R[mid, q] * 1e3),
                             _num(r.L[mid, q, q] * 1e9), _num(r.R[-1, q] * 1e3), _num(r.L[-1, q, q] * 1e9),
                             _num(r.R[-1, q] / Rdc[q], "{:.3g}"), _num(Q[k], "{:.3g}"), fmt_si(r.f[k], "")])
        self.coup.DeleteAllItems()
        L = r.L
        d0 = np.sqrt(np.abs(np.diag(Ldc)))
        for i in range(len(r.names)):
            for j in range(i + 1, len(r.names)):
                self.coup.Append([f"{r.names[i]} – {r.names[j]}", _num(Ldc[i, j] * 1e9),
                                  _num(Ldc[i, j] / (d0[i] * d0[j]), "{:.4f}"), _num(L[mid, i, j] * 1e9),
                                  _num(r.k[mid, i, j], "{:.4f}"), _num(L[-1, i, j] * 1e9),
                                  _num(r.k[-1, i, j], "{:.4f}")])
        lines = [f"{r.n_fils} filaments; window x {r.window[0]:.1f} … {r.window[2]:.1f}, "
                 f"y {r.window[1]:.1f} … {r.window[3]:.1f} mm; {r.seconds:.1f} s; wideband model of order "
                 f"{r.model.order} (exact at DC, passive).",
                 f"mid = {fmt_si(r.f[mid], 'Hz')}, top = {fmt_si(r.f[-1], 'Hz')}.", ""]
        for q, n in enumerate(r.names):
            ratio = r.R[-1, q] / Rdc[q]
            lines.append(f"{n}: R rises {ratio:.2g}x from DC to {fmt_si(r.f[-1], 'Hz')} (skin and proximity "
                         f"effect); L falls from {Ldc[q, q] * 1e9:.4g} to {r.L[-1, q, q] * 1e9:.4g} nH as the "
                         "current moves to the surfaces and towards its return.")
        if r.warnings:
            lines += ["", "Warnings:"] + [f"• {w}" for w in r.warnings]
        lines += ["", "M and k: positive when the ports' currents (+ to −) make flux in the same sense. Values "
                  "are for the copper alone: parts between pads (capacitors, resistors) are not in the loop, "
                  "use links to close a loop."]
        self.info.SetValue("\n".join(lines))
        self.cport.Set(r.names)
        self.cport.SetSelection(0)
        self.cfreq.Set([fmt_si(f, "Hz") for f in r.f_show])
        self.cfreq.SetSelection(len(r.f_show) - 1)
        b = r.bars
        used = sorted({int(x) for x in b["layer"][b["kind"] != 2]})
        self._layers = [b["layers"][k] for k in used] or list(b["layers"])
        self.clayer.Set(self._layers)
        self.clayer.SetSelection(0)
        self._cross = None
        self._plot_z()
        self._map()
        self._spice()

    def _plot_z(self):
        r = self.res
        self.zfig.clf()
        n = 4 if len(r.names) > 1 else 3
        axs = self.zfig.subplots(n, 1, sharex=True)
        for q, name in enumerate(r.names):
            axs[0].loglog(r.f, r.R[:, q] * 1e3, label=name)
            axs[1].semilogx(r.f, r.L[:, q, q] * 1e9, label=name)
            axs[2].loglog(r.f, np.maximum(r.Q[:, q], 1e-6), label=name)
        axs[0].set_ylabel("R [mΩ]")
        axs[1].set_ylabel("L [nH]")
        axs[2].set_ylabel("Q = ωL / R")
        if n == 4:
            for i in range(len(r.names)):
                for j in range(i + 1, len(r.names)):
                    axs[3].semilogx(r.f, r.k[:, i, j], label=f"{r.names[i]}–{r.names[j]}")
            axs[3].set_ylabel("coupling k")
            axs[3].legend(fontsize=8)
        axs[-1].set_xlabel("f [Hz]")
        for a in axs:
            a.grid(True, which="both", alpha=0.3)
        axs[0].legend(fontsize=8)
        self.zcanvas.draw()

    def _map(self):
        r = self.res
        if r is None or self.cport.GetSelection() < 0:
            return
        p, k = self.cport.GetSelection(), self.cfreq.GetSelection()
        layer = self._layers[self.clayer.GetSelection()]
        b = r.bars
        L_idx = b["layers"].index(layer)
        Kn, Kb = A.sheet_current(r, p, k)
        self.cfig.clf()
        gs = self.cfig.add_gridspec(1, 2, width_ratios=(3, 1))
        ax = self.cfig.add_subplot(gs[0])
        self._xax = self.cfig.add_subplot(gs[1])
        ax.set_aspect("equal")
        x0, y0, h, ny, nx = b["grid"]
        cn = b["cell_node"][layer]
        img = np.where(cn >= 0, Kn[np.maximum(cn, 0)], np.nan)
        wire = (b["kind"] == 3) & (b["layer"] == L_idx)
        vals = np.concatenate([img[np.isfinite(img) & (img > 0)], Kb[wire][Kb[wire] > 0]])
        if vals.size:
            vmax = float(vals.max())
            norm = LogNorm(max(vmax * 1e-4, vals.min()), vmax) if self.clog.GetValue() else Normalize(0, vmax)
        else:
            norm = Normalize(0, 1)
        im = ax.imshow(img, extent=(x0, x0 + nx * h, y0 + ny * h, y0), origin="upper", cmap="inferno", norm=norm,
                       interpolation="nearest")
        polys = []
        for i in np.flatnonzero(wire):
            ux, uy, ln, w = b["ux"][i], b["uy"][i], b["len"][i], b["w"][i]
            cx, cy = b["x"][i], b["y"][i]
            nx_, ny_ = -uy * w / 2, ux * w / 2
            ex, ey = ux * ln / 2, uy * ln / 2
            polys.append([(cx - ex - nx_, cy - ey - ny_), (cx + ex - nx_, cy + ey - ny_),
                          (cx + ex + nx_, cy + ey + ny_), (cx - ex + nx_, cy - ey + ny_)])
        self._wires = np.flatnonzero(wire)
        if polys:
            pc = PolyCollection(polys, array=np.maximum(Kb[wire], norm.vmin if hasattr(norm, "vmin") else 0),
                                cmap="inferno", norm=norm, edgecolors="none")
            ax.add_collection(pc)
        via = np.flatnonzero(b["kind"] == 2)
        if via.size:
            ax.scatter(b["x"][via], b["y"][via], s=18, facecolors="none", edgecolors="c", linewidths=0.8,
                       label="vias")
        pads = {q.id: q for q in self.board.pads}
        for port in self.s.ports:
            for sign, ids in (("+", port.plus), ("−", port.minus)):
                for q in ids:
                    if q in pads:
                        ax.annotate(f"{port.name}{sign}", pads[q].pos, color="c", fontsize=8, weight="bold",
                                    xytext=(4, 4), textcoords="offset points")
        w = r.window
        ax.set_xlim(w[0], w[2])
        ax.set_ylim(w[3], w[1])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{layer}: sheet current for 1 A into {r.names[p]} at {fmt_si(r.f_show[k], 'Hz')}", fontsize=10)
        self.cfig.colorbar(im, ax=ax, label="|K| [A/mm]", shrink=0.8)
        self._cross_plot()
        self.ccanvas.draw()

    def _pick(self, ev):
        r = self.res
        if r is None or ev.inaxes is None or ev.inaxes is self._xax or ev.xdata is None or not len(self._wires):
            return
        b = r.bars
        i = self._wires
        dx, dy = ev.xdata - b["x"][i], ev.ydata - b["y"][i]
        along = np.abs(dx * b["ux"][i] + dy * b["uy"][i]) - b["len"][i] / 2
        across = np.abs(-dx * b["uy"][i] + dy * b["ux"][i]) - b["w"][i] / 2
        dist = np.hypot(np.maximum(along, 0), np.maximum(across, 0))
        j = int(np.argmin(dist))
        if dist[j] > 0.5:
            return
        self._cross = int(i[j])
        self._cross_plot()
        self.ccanvas.draw()

    def _cross_plot(self):
        ax = self._xax
        ax.clear()
        r = self.res
        if self._cross is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Click a track", ha="center", va="center", transform=ax.transAxes, color="0.5")
            return
        p, k = self.cport.GetSelection(), self.cfreq.GetSelection()
        F = r.fils
        m = np.flatnonzero(F["bar"] == self._cross)
        if not m.size:
            return
        I = np.abs(r.fil_I[(p, k)][m])
        J = I / (F["w"][m] * F["h"][m]) * 1e-6          # A/mm²
        ow, oh = F["off_w"][m] * 1e3, F["off_h"][m] * 1e6
        w, h = F["w"][m] * 1e3, F["h"][m] * 1e6
        polys = [[(a - c / 2, d - e / 2), (a + c / 2, d - e / 2), (a + c / 2, d + e / 2), (a - c / 2, d + e / 2)]
                 for a, c, d, e in zip(ow, w, oh, h)]
        pc = PolyCollection(polys, array=J, cmap="inferno", edgecolors="0.3", linewidths=0.3)
        ax.add_collection(pc)
        ax.set_xlim(ow.min() - w.max(), ow.max() + w.max())
        ax.set_ylim(oh.min() - h.max(), oh.max() + h.max())
        ax.set_xlabel("across the track [mm]")
        ax.set_ylabel("through the copper [µm]")
        ax.set_title(f"Cross-section, |J| [A/mm²]\n(max / mean {J.max() / max(J.mean(), 1e-30):.2g})", fontsize=9)
        self.cfig.colorbar(pc, ax=ax, shrink=0.8)

    # ------------------------------------------------------------- exports
    def _spice(self):
        r = self.res
        if r is None:
            return
        head = [f"3D extraction of {self.session.board_path or 'the board'}",
                "ports: " + "; ".join(f"{p.name}: {'+'.join(p.plus)} -> {'+'.join(p.minus)}" for p in self.s.ports)]
        if self.skind.GetSelection() == 0:
            text = SP.foster(r, "EM3D", header=head)
        else:
            try:
                f = parse_si(self.sfreq.GetValue())
            except (ValueError, TypeError):
                return
            if f <= 0:
                return
            text = SP.fixed(r, f, "EM3D_F", header=head)
        self.spice.SetValue(text)

    def _save_spice(self):
        if not self.res:
            return
        with wx.FileDialog(self, "Save SPICE subcircuit", defaultFile="em3d.lib",
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
        r = self.res
        if not r:
            return
        with wx.FileDialog(self, "Export R, L and coupling", defaultFile="impedance.csv", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            names = r.names
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            with open(d.GetPath(), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["f_Hz"] + [f"R_{n}_ohm" for n in names] + [f"L_{n}_H" for n in names]
                           + [f"k_{names[i]}_{names[j]}" for i, j in pairs])
                L, k = r.L, r.k
                for m, fr in enumerate(r.f):
                    w.writerow([f"{fr:.6g}"] + [f"{r.R[m, q]:.6g}" for q in range(len(names))]
                               + [f"{L[m, q, q]:.6g}" for q in range(len(names))]
                               + [f"{k[m, i, j]:.6g}" for i, j in pairs])
            self._status(f"Saved {d.GetPath()}")
