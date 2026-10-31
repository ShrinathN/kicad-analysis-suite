"""PCB analysis suite, wx: Home window, the "Suite" menu, and the shared panels
(stackup & materials, component library, results browser, external tools).

All windows live in one process and share one `Session`."""
import os
import textwrap
import threading
import time

import wx
import wx.adv
import wx.grid

import numpy as np

from .. import scenario as SC
from ..suite import external, results_index
from ..suite import stackup as ST
from ..suite import tools as TL
from ..suite.library import C_TOL_CODES, DIELECTRICS, KINDS, Part, fmt_si, parse_si
from ..suite.session import Session

SHARED = (("stackup", "Stackup & materials…", "Copper, dielectric thickness, εr and tanδ against f"),
          ("library", "Component library…", "Real-part models: L, C, R with tolerance, Q, SRF, ESR, ESL"),
          ("results", "Results browser…", "Saved results of every tool"),
          ("external", "External tools…", "openEMS, FastHenry, FastCap, ParaView; Flatpak sandbox access"))


def _amp(text):
    """Escape '&' for wx labels, where it marks a mnemonic."""
    return text.replace("&", "&&")


def _note(text, width=44):
    return "\n".join(textwrap.wrap(text, width))


# ---------------------------------------------------------------- windows / menu
def open_window(session, key, parent=None):
    """Open (or raise) the window for a tool or shared panel and return it."""
    w = session.windows.get(key)
    if w:                                       # a destroyed wx window is falsy
        w.Iconize(False)
        w.Raise()
        return w
    if key == "home":
        w = HomeFrame(session)
    elif key == "dc":
        from .app import MainFrame
        w = MainFrame(session=session)
    elif key == "matching":
        from .matching import MatchingFrame
        w = MatchingFrame(session)
    elif key == "extract":
        if session.board is None:
            return wx.MessageBox("Load a board first.", "Trace R/L/C", parent=parent)
        from .extract import ExtractFrame
        w = ExtractFrame(session)
    elif key == "loop":
        if session.board is None:
            return wx.MessageBox("Load a board first.", "Switching Loop", parent=parent)
        from .loop import LoopFrame
        w = LoopFrame(session)
    elif key == "stackup":
        if session.stackup is None:
            return wx.MessageBox("Load a board first.", "Stackup", parent=parent)
        w = StackupFrame(session)
    elif key == "library":
        w = LibraryFrame(session)
    elif key == "results":
        w = ResultsFrame(session)
    elif key == "external":
        w = ExternalFrame(session)
    else:
        t = TL.get(key)
        return wx.MessageBox(f"{t.title} is planned but not built yet.\n\n{t.blurb}", t.title, parent=parent)
    session.windows[key] = w
    w.Show()
    return w


def attach_suite_menu(frame, session, current=None):
    """Add a "Suite" menu (Home, the tools, the shared panels) to a frame's menu bar."""
    mb = frame.GetMenuBar() or wx.MenuBar()
    m = wx.Menu()

    def item(label, key, enable=True, help_=""):
        it = m.Append(wx.ID_ANY, _amp(label), help_)
        it.Enable(enable)
        frame.Bind(wx.EVT_MENU, lambda e: open_window(session, key, frame), it)

    item("Home\tCtrl+H", "home", current != "home")
    m.AppendSeparator()
    for t in TL.TOOLS:
        item(t.title + ("" if t.ready else "  (planned)"), t.key, t.ready and t.key != current, t.blurb)
    m.AppendSeparator()
    for key, label, help_ in SHARED:
        item(label, key, key != current, help_)
    m.AppendSeparator()
    hl = m.AppendCheckItem(wx.ID_ANY, "Highlight selected parts in KiCad",
                           "Clicking parts in a tool's list selects them in the PCB editor (and the schematic "
                           "editor, where KiCad allows it)")
    hl.Check(getattr(session, "highlight", True))

    def toggle(e):
        session.highlight = e.IsChecked()
        for w in list(session.windows.values()):      # keep every window's menu in step
            if w and w is not frame and getattr(w, "_hl_item", None):
                try:
                    w._hl_item.Check(session.highlight)
                except RuntimeError:
                    pass
    frame.Bind(wx.EVT_MENU, toggle, hl)
    frame._hl_item = hl
    mb.Append(m, "&Suite")
    if frame.GetMenuBar() is None:
        frame.SetMenuBar(mb)
    return m


# ---------------------------------------------------------------- Home
class HomeFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="PCB analysis suite")
        self.session = session
        p = self.panel = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)

        hb = wx.StaticBoxSizer(wx.VERTICAL, p, "Board")
        hs = hb.GetStaticBox()
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.path = wx.TextCtrl(hs, style=wx.TE_READONLY)
        self.path.SetToolTip("The board all the suite's tools work on (Browse… to pick another)")
        for text, cmd in (("Browse…", self._browse), ("Reload", self._reload)):
            b = wx.Button(hs, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            row.Add(b, 0, wx.LEFT, 4)
        row.Insert(0, self.path, 1, wx.EXPAND)
        hb.Add(row, 0, wx.EXPAND | wx.ALL, 4)
        self.kbar = wx.BoxSizer(wx.HORIZONTAL)
        hb.Add(self.kbar, 0, wx.LEFT | wx.RIGHT, 4)
        self.info = wx.StaticText(hs, label="")
        self.stack = wx.StaticText(hs, label="")
        self.saved = wx.StaticText(hs, label="")
        for t in (self.info, self.stack, self.saved):
            hb.Add(t, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 4)
        top.Add(hb, 0, wx.EXPAND | wx.ALL, 8)

        groups = wx.BoxSizer(wx.HORIZONTAL)
        for g in TL.GROUPS:
            gb = wx.StaticBoxSizer(wx.VERTICAL, p, _amp(g))
            gs = gb.GetStaticBox()
            for t in (t for t in TL.TOOLS if t.group == g):
                b = wx.adv.CommandLinkButton(gs, mainLabel=_amp(t.title) + ("" if t.ready else "  (planned)"),
                                             note=_note(t.blurb, 36))
                b.Enable(t.ready)
                b.SetToolTip(t.blurb)
                b.Bind(wx.EVT_BUTTON, lambda e, k=t.key: open_window(self.session, k, self))
                gb.Add(b, 0, wx.EXPAND | wx.ALL, 4)
            groups.Add(gb, 1, wx.EXPAND | wx.RIGHT, 6)
        top.Add(groups, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        sb = wx.StaticBoxSizer(wx.HORIZONTAL, p, "Shared")
        ss = sb.GetStaticBox()
        for key, label, help_ in SHARED:
            b = wx.adv.CommandLinkButton(ss, mainLabel=_amp(label.rstrip("…")), note=_note(help_, 30))
            b.Bind(wx.EVT_BUTTON, lambda e, k=key: open_window(self.session, k, self))
            sb.Add(b, 1, wx.EXPAND | wx.ALL, 4)
        top.Add(sb, 0, wx.EXPAND | wx.ALL, 8)
        p.SetSizer(top)
        best = p.GetBestSize()
        self.SetClientSize((max(best.width, 1000), max(best.height + 60, 640)))

        attach_suite_menu(self, session, "home")
        if session.kicad is not None:
            for text, cmd in (("Save board", lambda: session.kicad.save()),
                              ("Refill zones + save", lambda: (session.kicad.refill_zones(), session.kicad.save()))):
                b = wx.Button(hs, label=text)
                b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: self._kicad(c))
                self.kbar.Add(b, 0, wx.RIGHT, 4)
            self.kbar.Insert(0, wx.StaticText(hs, label="KiCad connected:"), 0,
                             wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._refresh()

    def _on_close(self, e):
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        self._refresh()

    def _kicad(self, fn):
        try:
            fn()
            self._reload()
        except Exception as e:      # noqa: BLE001
            wx.MessageBox(f"KiCad: {e}", "KiCad", parent=self)

    def _browse(self):
        with wx.FileDialog(self, "Open board", wildcard="KiCad PCB (*.kicad_pcb)|*.kicad_pcb|All|*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                self._load(d.GetPath())

    def _reload(self):
        if self.session.board_path:
            self._load(self.session.board_path)

    def _load(self, path):
        try:
            self.session.load_board(path)
        except Exception as e:      # noqa: BLE001
            wx.MessageBox(f"Could not read {path}:\n{e}", "Board", parent=self)

    def _refresh(self):
        s = self.session
        if s.board is None:
            self.path.SetValue("")
            self.info.SetLabel("No board loaded. Browse for a .kicad_pcb"
                               + (f" (KiCad: {s.kicad_error})" if s.kicad_error else ""))
            self.stack.SetLabel("")
            self.saved.SetLabel("")
        else:
            b = s.board
            self.path.SetValue(s.board_path)
            self.info.SetLabel(f"{os.path.basename(s.board_path)}: {len(b.nets)} nets, {len(b.pads)} pads, "
                               f"{len(b.tracks)} tracks, {len(b.vias)} vias, {len(b.zones)} zone fills; "
                               f"saved {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(s.board_path)))}")
            self.stack.SetLabel("Stackup: " + s.stackup.summary())
            sp = s.scenario_path
            if os.path.exists(sp):
                try:
                    secs = [k for k, v in SC.read(sp).items() if k != "version" and v]
                except (OSError, ValueError) as e:
                    secs = [f"unreadable: {e}"]
                self.saved.SetLabel(f"Scenario {os.path.basename(sp)}: " + (", ".join(secs) or "empty"))
            else:
                self.saved.SetLabel("No saved scenario yet.")
        self.panel.Layout()


# ---------------------------------------------------------------- stackup
class StackupFrame(wx.Frame):
    """Copper thickness overrides and dielectric εr / tanδ (at f_ref), with the εr(f) / tanδ(f) model."""
    COLS = ("Layer", "Type", "Material", "Board t [µm]", "Used t [µm]", "εr", "tanδ", "f_ref [GHz]")

    def __init__(self, session):
        super().__init__(None, title="Stackup & materials", size=(900, 680))
        self.session = session
        p = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)
        self.grid = wx.grid.Grid(p)
        self.rows = []      # ("cu", layer) or ("d", index)
        self.grid.CreateGrid(0, len(self.COLS))
        for k, c in enumerate(self.COLS):
            self.grid.SetColLabelValue(k, c)
        self.grid.SetRowLabelSize(0)
        self.grid.Bind(wx.grid.EVT_GRID_CELL_CHANGED, self._changed)
        self.grid.GetGridWindow().SetToolTip(
            "Used t: copper thickness used by the tools (0 = the board's value). εr / tanδ: the dielectric's "
            "permittivity and loss tangent at f_ref (from the laminate datasheet)")
        top.Add(wx.StaticText(p, label="Grey cells are read-only. Copper: finished thickness (0 = board value). "
                                       "Dielectric: εr and tanδ at f_ref."), 0, wx.ALL, 6)
        top.Add(self.grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.model = wx.Choice(p, choices=["Wideband Djordjevic-Sarkar (causal)", "Constant εr, tanδ"])
        self.model.Bind(wx.EVT_CHOICE, lambda e: self._set_model())
        self.model.SetToolTip("Djordjevic-Sarkar: εr falls and tanδ changes with frequency, fitted at f_ref "
                              "(realistic for FR4). Constant: the values at f_ref at every frequency")
        reset = wx.Button(p, label="Reset to board values")
        reset.Bind(wx.EVT_BUTTON, lambda e: self._reset())
        row.AddMany([(wx.StaticText(p, label="Frequency model"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.model,), (reset, 0, wx.LEFT, 12)])
        top.Add(row, 0, wx.ALL, 6)

        import matplotlib
        matplotlib.use("WXAgg")
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(8, 3), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(p, -1, self.fig)
        top.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 6)
        p.SetSizer(top)
        attach_suite_menu(self, session, "stackup")
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._fill()

    def _on_close(self, e):
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if what in ("board", "stackup") and self.session.stackup is not None:
            self._fill()

    def _fill(self):
        st = self.session.stackup
        g = self.grid
        if g.GetNumberRows():
            g.DeleteRows(0, g.GetNumberRows())
        self.rows = []
        for k, L in enumerate(st.copper_layers):
            self.rows.append(("cu", L))
            if k < len(st.dielectrics):
                self.rows.append(("d", k))
        g.AppendRows(len(self.rows))
        grey = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
        for r, (typ, key) in enumerate(self.rows):
            if typ == "cu":
                vals = [key, "copper", "", f"{st.cu_board[key] * 1e6:.1f}", f"{st.cu(key) * 1e6:.1f}", "", "", ""]
                editable = {4}
            else:
                d = st.dielectrics[key]
                vals = [d.name, "dielectric" + ("" if d.from_board else " (edited)"), d.material,
                        f"{d.t * 1e6:.1f}", f"{d.t * 1e6:.1f}", f"{d.er:.3g}", f"{d.tand:.3g}", f"{d.f_ref / 1e9:g}"]
                editable = {5, 6, 7}
            for c, v in enumerate(vals):
                g.SetCellValue(r, c, v)
                g.SetReadOnly(r, c, c not in editable)
                if c not in editable:
                    g.SetCellTextColour(r, c, grey)
        g.AutoSizeColumns()
        self.model.SetSelection(0 if st.model == "ds" else 1)
        self._plot()

    def _changed(self, e):
        st = self.session.stackup
        r, c = e.GetRow(), e.GetCol()
        typ, key = self.rows[r]
        try:
            v = float(self.grid.GetCellValue(r, c))
        except ValueError:
            wx.MessageBox("Not a number.", "Stackup", parent=self)
            return self._fill()
        if typ == "cu":
            if v > 0:
                st.cu_override[key] = v * 1e-6
            else:
                st.cu_override.pop(key, None)
        else:
            d = st.dielectrics[key]
            if c == 5 and v >= 1:
                d.er = v
            elif c == 6 and 0 <= v < 1:
                d.tand = v
            elif c == 7 and v > 0:
                d.f_ref = v * 1e9
            else:
                wx.MessageBox("Out of range (εr ≥ 1, 0 ≤ tanδ < 1, f_ref > 0).", "Stackup", parent=self)
                return self._fill()
            d.from_board = False
        self.session.set_stackup(st)

    def _set_model(self):
        self.session.stackup.model = "ds" if self.model.GetSelection() == 0 else "const"
        self.session.set_stackup(self.session.stackup)

    def _reset(self):
        s = self.session
        s.set_stackup(ST.from_board(s.board))

    def _plot(self):
        st = self.session.stackup
        self.fig.clf()
        a1, a2 = self.fig.subplots(1, 2)
        f = np.logspace(6, 10.5, 200)
        seen = set()
        for d in st.dielectrics:
            key = (d.er, d.tand, d.f_ref)
            if key in seen:
                continue
            seen.add(key)
            er, td = d.er_tand(f, model=st.model)
            a1.semilogx(f / 1e9, er, label=f"{d.name} ({d.material})")
            a2.semilogx(f / 1e9, td)
        a1.set(xlabel="f [GHz]", ylabel="εr", title="Permittivity")
        a2.set(xlabel="f [GHz]", ylabel="tanδ", title="Loss tangent")
        for a in (a1, a2):
            a.grid(True, which="both", alpha=0.3)
        if seen:
            a1.legend(fontsize=7)
        self.canvas.draw()


# ---------------------------------------------------------------- component library
PART_FIELDS = {  # kind -> [(attr, label, unit or None for plain text/number)]
    "L": [("dcr", "DCR", "Ω"), ("q", "Q (0 = none)", ""), ("f_q", "at f", "Hz"), ("srf", "SRF (0 = none)", "Hz")],
    "C": [("esr", "ESR (0 = from Q)", "Ω"), ("q", "Q (0 = none)", ""), ("f_q", "at f", "Hz"),
          ("esl", "ESL (0 = from SRF)", "H"), ("srf", "SRF (0 = none)", "Hz"),
          ("v_rated", "Rated V", "V"), ("bias_loss", "C lost at rated V [%]", "%")],
    "R": [("ls", "Series L", "H"), ("cp", "Parallel C", "F")],
}
PART_TIPS = {  # hover help for PART_FIELDS (by attribute)
    "dcr": "DC resistance of the winding",
    "q": "Quality factor at the frequency below (datasheet); sets the frequency-dependent loss. 0 = no Q given",
    "f_q": "The frequency the Q is given at",
    "srf": "Self-resonant frequency (datasheet); sets the parasitic C of an inductor or the ESL of a capacitor. "
           "0 = none",
    "esr": "Equivalent series resistance; 0 = derived from Q",
    "esl": "Equivalent series inductance; 0 = derived from the SRF",
    "v_rated": "Rated voltage, used for the DC-bias derating",
    "bias_loss": "Percentage of C lost at the rated voltage (X7R / X5R: 50-80 %; C0G: 0)",
    "ls": "Parasitic series inductance of the resistor body",
    "cp": "Parasitic capacitance across the resistor",
}


class PartDialog(wx.Dialog):
    def __init__(self, parent, part, taken):
        """taken: names already used by other parts."""
        super().__init__(parent, title="Component", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.part, self.taken, self.result = part, taken, None
        g = wx.FlexGridSizer(0, 2, 4, 8)
        g.AddGrowableCol(1)
        self.labels = {}

        def add(label, ctrl):
            self.labels[ctrl] = wx.StaticText(self, label=label)
            g.Add(self.labels[ctrl], 0, wx.ALIGN_CENTER_VERTICAL)
            g.Add(ctrl, 1, wx.EXPAND)
            return ctrl
        self.kind = add("Kind", wx.Choice(self, choices=[KINDS[k][0] for k in KINDS]))
        self.kind.SetSelection(list(KINDS).index(part.kind))
        self.kind.Bind(wx.EVT_CHOICE, lambda e: self._show())
        self.kind.SetToolTip("Inductor, capacitor or resistor (sets which parasitics are asked for)")
        self.name = add("Name / part number", wx.TextCtrl(self, value=part.name))
        self.value = add("Value", wx.TextCtrl(self, value=fmt_si(part.value, part.unit)))
        tolrow = wx.BoxSizer(wx.HORIZONTAL)
        self.tol = wx.TextCtrl(self, value=part.tol_text().replace(" %", ""), size=(110, -1))
        self.tol_unit = wx.Choice(self, choices=["%", "absolute"])
        self.tol_unit.SetSelection(0 if part.tol_unit == "%" else 1)
        self.name.SetToolTip("Name or manufacturer part number (unique in the library)")
        self.value.SetToolTip("Nominal value; SI prefixes allowed (4n7, 100 nH, 2.2u)")
        self.tol.SetToolTip("Tolerance: a percentage, an absolute value (0.1 pF), or a capacitor code "
                            "A/B/C/D = ±0.05/0.1/0.25/0.5 pF")
        self.tol_unit.SetToolTip("Whether the tolerance is a percentage or an absolute value")
        tolrow.AddMany([(self.tol,), (self.tol_unit, 0, wx.LEFT, 4)])
        g.Add(wx.StaticText(self, label="Tolerance ±"), 0, wx.ALIGN_CENTER_VERTICAL)
        g.Add(tolrow)
        self.package = add("Package", wx.TextCtrl(self, value=part.package))
        self.mfr = add("Manufacturer", wx.TextCtrl(self, value=part.mfr))
        self.diel = add("Dielectric", wx.Choice(self, choices=list(DIELECTRICS)))
        self.diel.SetSelection(DIELECTRICS.index(part.dielectric) if part.dielectric in DIELECTRICS else 0)
        self.package.SetToolTip("Package size, e.g. 0402 (also hints the ESL)")
        self.mfr.SetToolTip("Manufacturer (for the BOM)")
        self.diel.SetToolTip("Capacitor dielectric: sets the temperature drift and DC-bias loss (C0G flat, X7R / "
                             "X5R lose C with temperature and voltage)")
        self.extra = {}
        for kind, flist in PART_FIELDS.items():
            for attr, label, unit in flist:
                if attr in self.extra:
                    continue
                v = getattr(part, attr)
                txt = f"{v * 100:g}" if unit == "%" else (fmt_si(v, unit) if unit else f"{v:g}")
                self.extra[attr] = (wx.StaticText(self, label=label), wx.TextCtrl(self, value=txt), unit)
                self.extra[attr][1].SetToolTip(PART_TIPS.get(attr, label))
                g.Add(self.extra[attr][0], 0, wx.ALIGN_CENTER_VERTICAL)
                g.Add(self.extra[attr][1], 1, wx.EXPAND)
        self.notes = add("Notes", wx.TextCtrl(self, value=part.notes))
        self.notes.SetToolTip("Free text (source of the data, remarks)")
        self.hint = wx.StaticText(self, label="Values accept SI prefixes: 4.7n, 4n7, 100 nH, 2.4G, 0R1. "
                                              "Capacitor tolerance codes A/B/C/D = ±0.05/0.1/0.25/0.5 pF.")
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(g, 1, wx.EXPAND | wx.ALL, 8)
        s.Add(self.hint, 0, wx.LEFT | wx.RIGHT, 8)
        bs = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        s.Add(bs, 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizerAndFit(s)
        self._show()

    def _kind(self):
        return list(KINDS)[self.kind.GetSelection()]

    def _show(self):
        k = self._kind()
        want = {a for a, _, _ in PART_FIELDS[k]}
        for attr, (lab, ctrl, _) in self.extra.items():
            lab.Show(attr in want)
            ctrl.Show(attr in want)
        self.diel.Show(k == "C")
        self.labels[self.diel].Show(k == "C")
        self.Layout()
        self.Fit()

    def _ok(self, e):
        k = self._kind()
        try:
            name = self.name.GetValue().strip()
            if not name:
                raise ValueError("give the part a name")
            if name in self.taken:
                raise ValueError(f"a part named {name!r} already exists")
            value = parse_si(self.value.GetValue())
            if value <= 0:
                raise ValueError("value must be > 0")
            tu = "%" if self.tol_unit.GetSelection() == 0 else "abs"
            t = self.tol.GetValue().strip()
            if tu == "abs" and k == "C" and t.upper() in C_TOL_CODES:
                tol = C_TOL_CODES[t.upper()]
            else:
                tol = parse_si(t)
            extra = {}
            for attr, _, unit in PART_FIELDS[k]:
                txt = self.extra[attr][1].GetValue().strip() or "0"
                extra[attr] = parse_si(txt) / (100 if unit == "%" else 1)
        except ValueError as err:
            return wx.MessageBox(str(err), "Component", parent=self)
        self.result = Part(k, name, value, tol, tu, self.package.GetValue().strip(), self.mfr.GetValue().strip(),
                           self.notes.GetValue().strip(),
                           dielectric=DIELECTRICS[self.diel.GetSelection()] if k == "C" else "C0G", **extra)
        self.EndModal(wx.ID_OK)


class LibraryFrame(wx.Frame):
    COLS = (("Kind", 50), ("Name", 230), ("Value", 80), ("Tol ±", 80), ("Package", 75), ("Parasitics", 400),
            ("Source", 75))

    def __init__(self, session):
        super().__init__(None, title="Component library", size=(1120, 720))
        self.session = session
        p = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.filter = wx.Choice(p, choices=["All"] + [KINDS[k][0] + "s" for k in KINDS])
        self.filter.SetSelection(0)
        self.filter.SetToolTip("Show only one kind of part")
        self.filter.Bind(wx.EVT_CHOICE, lambda e: self._fill())
        row.AddMany([(wx.StaticText(p, label="Show"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6), (self.filter,)])
        for text, cmd in (("Add…", self._add), ("Edit…", self._edit), ("Duplicate", self._dup),
                          ("Delete", self._delete)):
            b = wx.Button(p, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            row.Add(b, 0, wx.LEFT, 4)
        top.Add(row, 0, wx.ALL, 6)
        self.list = wx.ListCtrl(p, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for k, (t, w) in enumerate(self.COLS):
            self.list.InsertColumn(k, t, width=w)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit())
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda e: self._plot())
        top.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        self.where = wx.StaticText(p, label=f"User parts are stored in {session.library.path}. Built-in parts "
                                            "are generic typical values; duplicate one to edit it.")
        top.Add(self.where, 0, wx.ALL, 6)

        import matplotlib
        matplotlib.use("WXAgg")
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(8, 2.8), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(p, -1, self.fig)
        top.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 6)
        p.SetSizer(top)
        attach_suite_menu(self, session, "library")
        if session.library.error:
            wx.CallAfter(wx.MessageBox, session.library.error, "Component library", parent=self)
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._fill()

    def _on_close(self, e):
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if what == "library":
            self._fill()

    def _shown(self):
        k = self.filter.GetSelection()
        lib = self.session.library
        return lib.parts if k == 0 else lib.by_kind(list(KINDS)[k - 1])

    def _fill(self):
        self.list.DeleteAllItems()
        for pt in self._shown():
            if pt.kind == "L":
                par = f"DCR {fmt_si(pt.dcr, 'Ω')}, Q {pt.q:g} @ {fmt_si(pt.f_q, 'Hz')}, SRF {fmt_si(pt.srf, 'Hz')}"
            elif pt.kind == "C":
                par = (f"ESR {fmt_si(pt.esr_eff(), 'Ω')}, ESL {fmt_si(pt.esl_eff(), 'H')}, {pt.dielectric}"
                       + (f", −{pt.bias_loss * 100:g} % @ {pt.v_rated:g} V" if pt.bias_loss else ""))
            else:
                par = f"Ls {fmt_si(pt.ls, 'H')}, Cp {fmt_si(pt.cp, 'F')}"
            self.list.Append([pt.kind, pt.name, fmt_si(pt.value, pt.unit), pt.tol_text(), pt.package, par,
                              "built-in" if pt.builtin else "user"])
        self._plot()

    def _sel(self):
        i = self.list.GetFirstSelected()
        return self._shown()[i] if i >= 0 else None

    def _names(self, but=None):
        return {p.name for p in self.session.library.parts} - {but}

    def _dialog(self, part, taken=None):
        with PartDialog(self, part, self._names() if taken is None else taken) as d:
            return d.result if d.ShowModal() == wx.ID_OK else None

    def _add(self):
        k = self.filter.GetSelection()
        kind = list(KINDS)[k - 1] if k > 0 else "C"
        new = self._dialog(Part(kind, "", {"L": 10e-9, "C": 10e-12, "R": 50.0}[kind]))
        if new:
            self.session.library.put(new)
            self.session.library_changed()

    def _edit(self):
        pt = self._sel()
        if pt is None:
            return
        if pt.builtin:
            return self._dup()
        new = self._dialog(pt, self._names(but=pt.name))
        if new:
            if new.name != pt.name:
                self.session.library.remove(pt.name)
            self.session.library.put(new)
            self.session.library_changed()

    def _dup(self):
        pt = self._sel()
        if pt is None:
            return
        from dataclasses import replace
        names, n = self._names(), 2
        base = name = pt.name.replace("generic ", "my ") if pt.builtin else pt.name
        while name in names:
            name, n = f"{base} ({n})", n + 1
        new = self._dialog(replace(pt, name=name))
        if new:
            self.session.library.put(new)
            self.session.library_changed()

    def _delete(self):
        pt = self._sel()
        if pt is None or pt.builtin:
            return
        if wx.MessageBox(f"Delete {pt.name}?", "Component library", wx.YES_NO, self) == wx.YES:
            self.session.library.remove(pt.name)
            self.session.library_changed()

    def _plot(self):
        pt = self._sel()
        self.fig.clf()
        if pt is None:
            self.canvas.draw()
            return
        a1, a2 = self.fig.subplots(1, 2)
        f = np.logspace(6, 10.3, 400)
        Z = pt.Z(f)
        lo, hi = pt.value - pt.tol_abs(), pt.value + pt.tol_abs()
        for v, ls, lab in ((pt.value, "-", "nominal"), (lo, ":", "−tol"), (hi, "--", "+tol")):
            a1.loglog(f / 1e9, np.abs(pt.Z(f, value=v)), ls, label=lab)
        a1.set(xlabel="f [GHz]", ylabel="|Z| [Ω]", title=pt.label())
        a1.legend(fontsize=7)
        if pt.kind == "R":
            a2.semilogx(f / 1e9, np.degrees(np.angle(Z)))
            a2.set(ylabel="phase [°]")
        else:
            q = np.abs(Z.imag) / np.maximum(Z.real, 1e-12)
            a2.loglog(f / 1e9, q)
            a2.set(ylabel="Q = |X| / R")
        a2.set(xlabel="f [GHz]", title="Quality factor" if pt.kind != "R" else "Phase")
        for a in (a1, a2):
            a.grid(True, which="both", alpha=0.3)
        self.canvas.draw()


# ---------------------------------------------------------------- results browser
class ResultsFrame(wx.Frame):
    COLS = (("Tool", 100), ("Date", 120), ("Board", 140), ("Rows", 220), ("T_max [°C]", 95),
            ("Features", 120), ("Size", 70), ("File", 400))

    def __init__(self, session):
        super().__init__(None, title="Results browser", size=(1150, 520))
        self.session = session
        p = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.root = wx.TextCtrl(p, value=os.path.dirname(session.board_path) if session.board_path
                                else os.getcwd())
        for text, cmd in (("Folder…", self._folder), ("Refresh", self._fill), ("Open", self._open)):
            b = wx.Button(p, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            row.Add(b, 0, wx.LEFT, 4)
        self.root.SetToolTip("Folder searched (with its subfolders) for saved simulation results")
        row.Insert(0, wx.StaticText(p, label="Search in"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        row.Insert(1, self.root, 1, wx.EXPAND)
        top.Add(row, 0, wx.EXPAND | wx.ALL, 6)
        self.list = wx.ListCtrl(p, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for k, (t, w) in enumerate(self.COLS):
            self.list.InsertColumn(k, t, width=w)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._open())
        top.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        top.Add(wx.StaticText(p, label="Results saved with \"Save .npz…\" in the result viewer (or the CLI's "
                                       "npz format) are listed here; the folder is searched 3 levels deep."),
                0, wx.ALL, 6)
        p.SetSizer(top)
        attach_suite_menu(self, session, "results")
        self.items = []
        self._fill()

    def _folder(self):
        with wx.DirDialog(self, "Search results in", self.root.GetValue()) as d:
            if d.ShowModal() == wx.ID_OK:
                self.root.SetValue(d.GetPath())
                self._fill()

    def _fill(self):
        self.list.DeleteAllItems()
        self.items = []
        if self.session.last_result is not None:
            r = self.session.last_result
            self.items.append(("mem", r))
            self.list.Append(["DC & Thermal", "this session", os.path.basename(r.board_path),
                              ", ".join(x.get("name", "") for x in r.rows),
                              f"{r.summary['T_max']:.1f}" if r.summary.get("T_max") is not None else "-",
                              "", "", "(last run, not saved)"])
        for info in results_index.scan(self.root.GetValue()):
            self.items.append(("file", info))
            self.list.Append([info["tool"], info["date"], info["board"], info["rows"],
                              f"{info['T_max']:.1f}" if info["T_max"] is not None else "-", info["features"],
                              f"{info['size_kb']:.0f} kB", info["path"]])

    def _open(self):
        i = self.list.GetFirstSelected()
        if i < 0:
            return
        kind, x = self.items[i]
        from .viewer import ViewerFrame
        if kind == "file":
            from ..results import Result
            try:
                x = Result.load(x["path"])
            except Exception as e:      # noqa: BLE001
                return wx.MessageBox(f"Could not open the result:\n{e}", "Results", parent=self)
        ViewerFrame(self, x).Show()


# ---------------------------------------------------------------- external tools
class ExternalFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="External tools", size=(880, 500))
        self.session = session
        p = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)
        self.list = wx.ListCtrl(p, style=wx.LC_REPORT)
        for k, (t, w) in enumerate((("Program", 130), ("Used by", 200), ("Found", 120), ("Path / Flatpak ref", 380))):
            self.list.InsertColumn(k, t, width=w)
        top.Add(self.list, 1, wx.EXPAND | wx.ALL, 6)
        self.notes = wx.TextCtrl(p, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 90))
        top.Add(self.notes, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.again = wx.Button(p, label="Detect again")
        self.again.Bind(wx.EVT_BUTTON, lambda e: self._detect())
        copy = wx.Button(p, label="Copy sandbox-access command")
        copy.Bind(wx.EVT_BUTTON, lambda e: self._copy())
        row.AddMany([(self.again,), (copy, 0, wx.LEFT, 4)])
        top.Add(row, 0, wx.ALL, 6)
        p.SetSizer(top)
        attach_suite_menu(self, session, "external")
        self._detect()

    def _detect(self):
        self.again.Disable()
        self.notes.SetValue("Detecting…")

        def work():
            res = external.detect()
            wx.CallAfter(self._show, *res)
        threading.Thread(target=work, daemon=True).start()

    def _show(self, found, notes):
        if not self:
            return
        self.again.Enable()
        self.list.DeleteAllItems()
        for f in found:
            self.list.Append([f.label, f.used_by, f.where if f.where else "no", f.path])
        self.notes.SetValue("\n".join(notes) or "Running on the host: programs are looked up on PATH.")

    def _copy(self):
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(external.ENABLE_HOST))
            wx.TheClipboard.Close()


# ---------------------------------------------------------------- entry
def main(board_path=None, use_kicad=False, tool="home"):
    app = wx.App()
    session = Session()
    if use_kicad:
        board_path = board_path or session.connect_kicad()
    if board_path:
        try:
            session.load_board(board_path)
        except Exception as e:      # noqa: BLE001
            session.kicad_error = f"could not read {board_path}: {e}"
    open_window(session, tool)
    app.MainLoop()
