"""Regulator model from a datasheet's load-transient plot: digitise the scope screenshot, fit the loop."""
import os
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
from matplotlib.patches import Rectangle  # noqa: E402

from ..pdn import digitise as DG  # noqa: E402
from ..pdn import regulator as RG  # noqa: E402
from ..suite.library import fmt_si, parse_si  # noqa: E402
from .layout import refit, scrolled  # noqa: E402

HELP = ("1. Open (or paste) a screenshot of the datasheet's load-transient plot.  2. Click 'Mark the plot "
        "area' and click its top-left and bottom-right grid corners.  3. For each trace either click 'Pick "
        "colour' and then the trace (found automatically), or 'Click points' and click along it yourself "
        "(black-and-white plots).  4. Enter the scales and the datasheet's output capacitance for that plot, "
        "then Fit.")
BY = {"colour": "by colour", "points": "by points"}
COL = {"i": "lime", "v": "yellow"}
MODE_TEXT = {"box": "Click the plot area's top-left grid corner…", "i": "Click on the load current trace…",
             "v": "Click on the output voltage trace…",
             "pi": "Load current: click points along the trace (right click removes, Backspace undoes, Esc ends)",
             "pv": "Output voltage: click points along the trace (right click removes, Backspace undoes, Esc ends)"}
FIELDS = (  # key, label, unit, default, tooltip
    ("nx", "Divisions across", "", 10, "Grid divisions across the plot area (10 on most scopes)"),
    ("ny", "Divisions down", "", 8, "Grid divisions down the plot area (8 on most scopes)"),
    ("t_div", "Time per division", "s", 200e-6, "The plot's time base, e.g. 200 µs/div: 200u"),
    ("a_div", "Current per division", "A", 0.5, "The current trace's scale, e.g. 500 mA/div: 500m"),
    ("v_div", "Voltage per division", "V", 0.1, "The output trace's scale (AC coupled), e.g. 100 mV/div: 100m"),
    ("step", "Current step [A] (optional)", "A", None, "The load step as the plot's caption states it (e.g. 1 A "
                                                     "to 2 A: 1). Blank: as read from the current trace"),
    ("c_test", "Output C of the test [F]", "F", 44e-6, "The output capacitance the datasheet used for this plot "
                                                       "(its application circuit / test conditions, after DC-bias "
                                                       "derating if known), e.g. 2 × 22 µF: 44u"),
    ("esr_test", "Its ESR [Ω]", "Ω", 3e-3, "ESR of those output capacitors together (ceramics: a few mΩ)"),
    ("v_in", "Boost: V in [V] (optional)", "V", None, "For a boost converter: the plot's input voltage (sets the "
                                                      "right-half-plane zero). Blank for other converters"),
    ("v_out", "Boost: V out [V]", "V", None, "For a boost converter: the plot's output voltage"),
    ("i_out", "Boost: I out [A]", "A", None, "For a boost converter: the plot's mean load current"),
    ("l_ind", "Boost: inductor [H]", "H", None, "For a boost converter: the inductor of the test circuit"),
)


def _get(t, unit):
    v = t.GetValue().strip()
    if not v:
        return None
    if unit and v.endswith(unit):
        v = v[: -len(unit)].strip()
    return parse_si(v)


class RegFitFrame(wx.Frame):
    """on_use(RegulatorModel) is called with the fitted model ('Use for this rail')."""

    def __init__(self, parent, model=None, on_use=None, part=""):
        super().__init__(parent, title="Regulator model from the datasheet - PDN", size=(1450, 940))
        self.on_use = on_use
        self.img, self.img_path, self.mode = None, "", None
        self.box, self.rgb = [], {"i": None, "v": None}
        self.pts, self.by = {"i": [], "v": []}, {"i": "colour", "v": "colour"}
        self._art = []                    # overlay artists on the image (redrawn without resetting the zoom)
        self.ti = self.i = self.tv = self.v = None
        self.fits = None
        self.model = RG.RegulatorModel(**model) if model else RG.RegulatorModel(part=part)
        self.CreateStatusBar()
        self._build()
        self._restore()

    # ------------------------------------------------------------- layout
    def _build(self):
        sp = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = self.left = scrolled(sp)
        ls = wx.BoxSizer(wx.VERTICAL)
        t = wx.StaticText(left, label=HELP)
        t.Wrap(400)
        ls.Add(t, 0, wx.ALL, 6)

        def box(title):
            b = wx.StaticBoxSizer(wx.VERTICAL, left, title)
            ls.Add(b, 0, wx.EXPAND | wx.ALL, 5)
            return b, b.GetStaticBox()

        ib, ibox = box("Plot image")
        row = wx.WrapSizer()
        for text, fn, tip in (("Open image…", self._open, "A screenshot of the load-transient plot (PNG / JPEG); "
                                                           "zoom the PDF in first for a sharp image"),
                              ("Paste", self._paste, "Paste a screenshot from the clipboard"),
                              ("Import CSV…", self._csv, "Already digitised (e.g. WebPlotDigitizer): columns time "
                                                          "[s], load current [A], output voltage [V]")):
            b = wx.Button(ibox, label=text, style=wx.BU_EXACTFIT)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        ib.Add(row)
        self.src = wx.StaticText(ibox, label="No image")
        ib.Add(self.src, 0, wx.ALL, 4)

        cb, cbox = box("Calibration and traces")
        b = wx.Button(cbox, label="Mark the plot area (2 clicks)")
        b.SetToolTip("Then click the grid's top-left corner and its bottom-right corner on the image")
        b.Bind(wx.EVT_BUTTON, lambda e: self._set_mode("box"))
        cb.Add(b, 0, wx.ALL, 2)
        self.fields = {}
        g = wx.FlexGridSizer(0, 2, 4, 6)
        for key, label, unit, val, tip in FIELDS[:6]:
            self._field(cbox, g, key, label, unit, val, tip)
        cb.Add(g, 0, wx.ALL, 4)
        self.by_c, self.npts = {}, {}
        for key, label in (("i", "load current"), ("v", "output voltage")):
            row = wx.BoxSizer(wx.HORIZONTAL)
            ch = wx.Choice(cbox, choices=list(BY.values()))
            ch.SetSelection(0)
            ch.SetToolTip(f"How the {label} trace is read: found automatically by its colour, or from points you "
                          "click along it (for black-and-white plots, or where the colour search goes wrong)")
            ch.Bind(wx.EVT_CHOICE, lambda e, k=key: self._set_by(k, list(BY)[e.GetSelection()]))
            self.by_c[key] = ch
            self.npts[key] = wx.StaticText(cbox, label="")
            row.AddMany([(wx.StaticText(cbox, label=label.capitalize() + ":"), 0,
                          wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6), (ch, 0, wx.RIGHT, 8),
                         (self.npts[key], 0, wx.ALIGN_CENTER_VERTICAL)])
            cb.Add(row, 0, wx.TOP | wx.LEFT, 4)
            row = wx.WrapSizer()
            b = wx.Button(cbox, label="Pick colour", style=wx.BU_EXACTFIT)
            b.SetToolTip(f"Then click on the {label} trace in the image: its colour is used to find it")
            b.Bind(wx.EVT_BUTTON, lambda e, k=key: self._set_mode(k))
            sw = wx.Panel(cbox, size=(28, 18))
            setattr(self, f"sw_{key}", sw)
            row.AddMany([(b, 0, wx.RIGHT, 4), (sw, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)])
            rows = [row, wx.BoxSizer(wx.HORIZONTAL)]
            for n, (text, fn, tip) in enumerate((
                    ("Click points", lambda k=key: self._set_mode("p" + k),
                     f"Then click along the {label} trace, as many points as you like, in any order (left click "
                     "adds, right click removes the nearest, right-drag a box removes every point in it, Backspace "
                     "undoes, Esc or this button again finishes). Put points at the corners of each step and along "
                     "the dips and overshoots; zoom with the toolbar for accuracy"),
                    ("Edit as points", lambda k=key: self._to_points(k),
                     f"Turn the {label} trace found by its colour into points you can correct: remove the wrong "
                     "ones (where the trace ran into a label, a marker or another trace) and click the missing ones"),
                    ("Undo", lambda k=key: self._undo(k), f"Remove the last point clicked on the {label} trace"),
                    ("Clear", lambda k=key: self._clear(k), f"Remove all points of the {label} trace"))):
                b = wx.Button(cbox, label=text, style=wx.BU_EXACTFIT)
                b.SetToolTip(tip)
                b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
                rows[n != 1].Add(b, 0, wx.RIGHT, 4)
            cb.Add(rows[0], 0, wx.LEFT, 4)
            cb.Add(rows[1], 0, wx.LEFT | wx.BOTTOM | wx.TOP, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.tol = wx.SpinCtrl(cbox, min=5, max=90, initial=30, size=(90, -1))
        self.tol.SetToolTip("How close (in %) a pixel's colour must be to the picked one to count as the trace: "
                            "raise it for antialiased or faint traces, lower it if grid or text gets picked up")
        self.tol.Bind(wx.EVT_SPINCTRL, lambda e: self._extract(quiet=True))
        row.AddMany([(wx.StaticText(cbox, label="Colour tolerance [%]"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                     (self.tol,)])
        cb.Add(row, 0, wx.ALL, 4)
        self.readout = wx.StaticText(cbox, label="")
        cb.Add(self.readout, 0, wx.ALL, 4)

        tb, tbox = box("Datasheet test circuit and control type")
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.kind = wx.Choice(tbox, choices=[v[0] for v in RG.TYPES.values()] + [RG.AUTO_LABEL])
        self.kind.SetToolTip("How the regulator controls its output (the datasheet's first page or block diagram "
                             "says): sets which loop parameters are fitted. A recovery that rings for several "
                             "cycles needs 'Ringing recovery' or 'General'; 'Try every type' fits all and keeps the "
                             "best (a type with more parameters only when it is clearly better)")
        g.AddMany([(wx.StaticText(tbox, label="Control"), 0, wx.ALIGN_CENTER_VERTICAL), (self.kind, 1, wx.EXPAND)])
        for key, label, unit, val, tip in FIELDS[6:]:
            self._field(tbox, g, key, label, unit, val, tip)
        self.part = wx.TextCtrl(tbox, value=self.model.part)
        self.part.SetToolTip("The regulator's part number (for your reference)")
        g.AddMany([(wx.StaticText(tbox, label="Part number"), 0, wx.ALIGN_CENTER_VERTICAL), (self.part, 1, wx.EXPAND)])
        tb.Add(g, 0, wx.ALL, 4)

        ob, obox = box("Fit options")
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self._field(obox, g, "window", "Fit up to [s] after each edge", "s", None,
                    "Fit only this long after each load step, and as long before it (blank: up to the next step). Use it to leave out what "
                    "a linear loop can't follow, e.g. a light-load burst pattern or a drift long after the recovery")
        ob.Add(g, 0, wx.ALL, 4)
        self.fit_c = wx.CheckBox(obox, label="Also fit the test capacitance (±50 %) and its ESR")
        self.fit_c.SetToolTip("Datasheets rarely state the capacitance after DC-bias derating (a 22 µF 0603 at 3.3 V "
                              "may be 10 µF): let the fit find it within ±50 % of the value entered. The fitted "
                              "value is shown and kept with the model")
        self.peak_w = wx.CheckBox(obox, label="Match the peak deviation first")
        self.peak_w.SetToolTip("Weight the fit towards the large deviations (the dip after a rising step, the "
                               "overshoot after a falling one): the peak is what the PDN check uses. Off: every "
                               "sample counts the same")
        for cb_ in (self.fit_c, self.peak_w):
            ob.Add(cb_, 0, wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.fit_b = wx.Button(left, label="Fit")
        self.fit_b.SetToolTip("Fit the loop to each load step of the plot (rising and falling separately)")
        self.fit_b.Bind(wx.EVT_BUTTON, lambda e: self._fit())
        self.use_b = wx.Button(left, label="Use for this rail")
        self.use_b.SetToolTip("The PDN tool uses the fitted loop as the regulator (with your board's capacitors)")
        self.use_b.Bind(wx.EVT_BUTTON, lambda e: self._use())
        self.use_b.Disable()
        row.AddMany([(self.fit_b, 1, wx.RIGHT, 6), (self.use_b, 1)])
        ls.Add(row, 0, wx.EXPAND | wx.ALL, 6)
        self.result = wx.TextCtrl(left, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 170))
        ls.Add(self.result, 0, wx.EXPAND | wx.ALL, 6)
        left.SetSizer(ls)

        right = wx.Panel(sp)
        rs = wx.BoxSizer(wx.VERTICAL)
        self.nb = wx.Notebook(right)
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.ifig = Figure(figsize=(8, 5), constrained_layout=True)
        self.icanvas = FigureCanvasWxAgg(p, -1, self.ifig)
        self.icanvas.mpl_connect("button_press_event", self._click)
        self.icanvas.mpl_connect("key_press_event", self._key)
        self.icanvas.mpl_connect("button_release_event", self._release)
        self._rpress = None
        s.Add(self.icanvas, 1, wx.EXPAND)
        s.Add(NavigationToolbar2WxAgg(self.icanvas), 0, wx.EXPAND)
        p.SetSizer(s)
        self.nb.AddPage(p, "Image")
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.ffig = Figure(figsize=(8, 5), constrained_layout=True)
        self.fcanvas = FigureCanvasWxAgg(p, -1, self.ffig)
        s.Add(self.fcanvas, 1, wx.EXPAND)
        s.Add(NavigationToolbar2WxAgg(self.fcanvas), 0, wx.EXPAND)
        p.SetSizer(s)
        self.nb.AddPage(p, "Fit")
        rs.Add(self.nb, 1, wx.EXPAND)
        right.SetSizer(rs)
        sp.SplitVertically(left, right, 450)
        sp.SetMinimumPaneSize(300)

    def _field(self, parent, g, key, label, unit, val, tip):
        t = wx.TextCtrl(parent, size=(110, -1))
        t.SetToolTip(tip)
        if val is not None:
            t.ChangeValue(f"{val:g}" if not unit else fmt_si(val, unit))
        if key in ("nx", "ny", "t_div", "a_div", "v_div"):
            t.Bind(wx.EVT_KILL_FOCUS, lambda e: (e.Skip(), wx.CallAfter(self._extract, True)))
        self.fields[key] = (t, unit)
        g.AddMany([(wx.StaticText(parent, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t,)])

    def _val(self, key):
        t, unit = self.fields[key]
        return _get(t, unit)

    # ------------------------------------------------------------- image
    def _set_image(self, img, path):
        self.img, self.img_path = img, path
        self.box, self.ti = [], None
        self.ti = self.i = self.tv = self.v = None
        self.pts = {"i": [], "v": []}
        self.src.SetLabel(os.path.basename(path) if path else f"pasted image {img.shape[1]} × {img.shape[0]}")
        if DG.is_grey(img):
            for k in ("i", "v"):
                self._set_by(k, "points", extract=False)
            self.SetStatusText("The image has no colours: mark the plot area, then click points along each trace.")
        self._counts()
        self._draw()
        refit(self.left)

    def _set_by(self, k, how, extract=True):
        self.by[k] = how
        self.by_c[k].SetSelection(list(BY).index(how))
        if extract:
            self._extract(quiet=True)

    def _counts(self):
        for k in ("i", "v"):
            n = len(self.pts[k])
            self.npts[k].SetLabel(f"{n} point{'s' if n != 1 else ''}" if n else "")

    def _undo(self, k):
        if self.pts[k]:
            self.pts[k].pop()
            self._points_changed(k)

    def _clear(self, k):
        if self.pts[k]:
            self.pts[k] = []
            self._points_changed(k)

    def _points_changed(self, k):
        self._counts()
        if self.by[k] != "points":
            self._set_by(k, "points", extract=False)
        if not self._extract(quiet=True):
            self._overlay()

    def _open(self):
        with wx.FileDialog(self, "Load-transient plot", wildcard="Images (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg|"
                                                                  "All|*", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() == wx.ID_OK:
                try:
                    self._set_image(DG.load_image(d.GetPath()), d.GetPath())
                except Exception as e:      # noqa: BLE001
                    wx.MessageBox(f"Could not read the image: {e}", "Image", parent=self)

    def _paste(self):
        bmp = wx.BitmapDataObject()
        ok = wx.TheClipboard.Open()
        try:
            ok = ok and wx.TheClipboard.GetData(bmp)
        finally:
            if wx.TheClipboard.IsOpened():
                wx.TheClipboard.Close()
        if not ok:
            return wx.MessageBox("No image on the clipboard.", "Paste", parent=self)
        im = bmp.GetBitmap().ConvertToImage()
        a = np.frombuffer(bytes(im.GetData()), np.uint8).reshape(im.GetHeight(), im.GetWidth(), 3) / 255.0
        self._set_image(a, "")

    def _csv(self):
        with wx.FileDialog(self, "Digitised load transient (t, i, v)", wildcard="CSV (*.csv;*.txt)|*.csv;*.txt|All|*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as d:
            if d.ShowModal() != wx.ID_OK:
                return
            try:
                a = np.genfromtxt(d.GetPath(), delimiter=",", skip_header=0)
                a = a[np.all(np.isfinite(a[:, :3]), axis=1)]
                self.ti, self.i, self.tv, self.v = a[:, 0], a[:, 1], a[:, 0], a[:, 2]
                self.img, self.img_path = None, d.GetPath()
                self.src.SetLabel(f"{os.path.basename(d.GetPath())}: {len(a)} rows")
                self._draw()
            except Exception as e:      # noqa: BLE001
                wx.MessageBox(f"Could not read it (three numeric columns: t, i, v): {e}", "CSV", parent=self)

    def _set_mode(self, m):
        if self.img is None:
            return wx.MessageBox("Open an image first.", "Digitise", parent=self)
        if m == self.mode and m in ("pi", "pv"):     # the button again: finished
            return self._end_points()
        self.mode = m
        self.nb.SetSelection(0)
        if m in ("pi", "pv"):
            self._set_by(m[1], "points", extract=False)
            self.icanvas.SetFocus()                  # for Backspace / Esc
        self.SetStatusText(MODE_TEXT[m])
        self._overlay()

    def _end_points(self):
        k = self.mode[1]
        self.mode = None
        n = len(self.pts[k])
        self.SetStatusText(f"{n} points on the {'load current' if k == 'i' else 'output voltage'} trace."
                           + ("" if len(self.box) == 2 else "  Mark the plot area to read them."))
        self._overlay()

    def _key(self, ev):
        if self.mode not in ("pi", "pv"):
            return
        if ev.key in ("backspace", "delete", "ctrl+z"):
            self._undo(self.mode[1])
        elif ev.key in ("escape", "enter"):
            self._end_points()

    def _click(self, ev):
        if ev.xdata is None or self.img is None or self.mode is None:
            return
        tb = self.icanvas.toolbar
        if tb is not None and tb.mode:              # panning / zooming
            return
        x, y = ev.xdata, ev.ydata
        if self.mode in ("pi", "pv"):
            k = self.mode[1]
            if ev.button == 3:               # removed on release: the nearest point, or a dragged box
                self._rpress = (x, y)
                return
            if ev.button == 1:
                self.pts[k].append((float(x), float(y)))
            self._points_changed(k)
            return
        if self.mode == "box":
            self.box = (self.box if len(self.box) == 1 else []) + [(x, y)]
            if len(self.box) == 1:
                self.SetStatusText("…and now its bottom-right grid corner.")
            else:
                self.mode = None
                self.SetStatusText("Plot area marked.")
                self._extract(quiet=True)
        else:
            k = self.mode
            self.rgb[k] = DG.colour_at(self.img, x, y, 4)
            c = self.rgb[k]
            getattr(self, f"sw_{k}").SetBackgroundColour(wx.Colour(*(int(255 * q) for q in c)))
            getattr(self, f"sw_{k}").Refresh()
            self.mode = None
            self.SetStatusText("Trace colour picked." + ("  The image is black and white: if the trace is not found "
                                                         "well, use 'Click points'." if DG.is_grey(self.img) else ""))
            self._set_by(k, "colour", extract=False)
            self._extract(quiet=True)
        self._overlay()

    def _release(self, ev):
        if self._rpress is None or self.mode not in ("pi", "pv"):
            self._rpress = None
            return
        (x0, y0), self._rpress = self._rpress, None
        k = self.mode[1]
        x1, y1 = (ev.xdata, ev.ydata) if ev.xdata is not None else (x0, y0)
        if abs(x1 - x0) > 3 or abs(y1 - y0) > 3:
            xa, xb, ya, yb = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            self.pts[k] = [p for p in self.pts[k] if not (xa <= p[0] <= xb and ya <= p[1] <= yb)]
        else:
            j = DG.nearest(self.pts[k], x0, y0)
            if j is not None:
                self.pts[k].pop(j)
        self._points_changed(k)

    def _to_points(self, k):
        """The colour-found trace as editable points."""
        if self.img is None or len(self.box) != 2 or self.rgb[k] is None:
            return wx.MessageBox("Mark the plot area and pick the trace's colour first; then its trace can be turned "
                                 "into points.", "Edit as points", parent=self)
        try:
            xs, ys = DG.trace(self.img, self._calib().box, self.rgb[k], self.tol.GetValue() / 100)
        except (ValueError, TypeError) as e:
            return wx.MessageBox(str(e), "Edit as points", parent=self)
        if len(xs) < 2:
            return wx.MessageBox("No trace of that colour found.", "Edit as points", parent=self)
        if self.pts[k] and wx.MessageBox(f"Replace the {len(self.pts[k])} points already clicked?", "Edit as points",
                                         wx.YES_NO, parent=self) != wx.YES:
            return
        self.pts[k] = DG.simplify(xs, ys)
        self.mode = None
        self._set_mode("p" + k)
        self._points_changed(k)
        self.SetStatusText(f"{len(self.pts[k])} points from the colour trace: right click (or right-drag a box) to "
                           "remove wrong ones, left click to add; Esc when done.")

    def _calib(self):
        (x0, y0), (x1, y1) = self.box
        return DG.Calib((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), int(self._val("nx") or 10),
                        int(self._val("ny") or 8), self._val("t_div") or 1e-4)

    def _extract(self, quiet=False):
        if self.img is None or len(self.box) != 2:
            return False
        try:
            cal = self._calib()
        except (ValueError, TypeError) as e:
            if not quiet:
                wx.MessageBox(str(e), "Digitise", parent=self)
            return False
        tol = self.tol.GetValue() / 100
        errs = []
        for k, unit in (("i", "a_div"), ("v", "v_div")):
            per = self._val(unit) or 1.0
            try:
                if self.by[k] == "points":
                    tk, yk = DG.from_points(cal, self.pts[k], per) if len(self.pts[k]) >= 2 else (None, None)
                elif self.rgb[k] is not None:
                    tk, yk = DG.extract(self.img, cal, self.rgb[k], per, tol)
                else:
                    tk, yk = None, None
            except (ValueError, TypeError) as e:
                tk, yk = None, None
                errs.append(f"{'Load current' if k == 'i' else 'Output voltage'}: {e}")
            if k == "i":
                self.ti, self.i = tk, yk
            else:
                self.tv, self.v = tk, yk
        if errs and not quiet:
            wx.MessageBox("\n".join(errs), "Digitise", parent=self)
        parts = []
        if self.i is not None and self.ti is not None:
            parts.append(f"current step read: {fmt_si(np.percentile(self.i, 95) - np.percentile(self.i, 5), 'A')}")
        if self.v is not None and self.tv is not None:
            parts.append(f"voltage swing: {fmt_si(self.v.max() - self.v.min(), 'V')}")
        self.readout.SetLabel("\n".join(p[0].upper() + p[1:] for p in parts))
        refit(self.left)
        self._overlay()
        return not errs

    def _draw(self):
        """Full redraw (a new image or data): resets the zoom.  Marks and traces are `_overlay`."""
        fig = self.ifig
        fig.clf()
        ax = fig.add_subplot(111)
        self.iax, self._art = None, []
        if self.img is not None:
            ax.imshow(self.img, interpolation="nearest")
            ax.set_autoscale_on(False)
            ax.set_axis_off()
            self.iax = ax
            self._overlay(draw=False)
        elif self.ti is not None:
            ax.plot(self.ti * 1e6, self.i, label="current [A]")
            ax2 = ax.twinx()
            ax2.plot(self.tv * 1e6, self.v * 1e3, color="tab:red", label="voltage [mV]")
            ax.set_xlabel("t [µs]")
        else:
            ax.text(0.5, 0.5, "Open or paste a screenshot of the load-transient plot", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
        self.icanvas.draw()

    def _overlay(self, draw=True):
        """The plot-area box, the colour-found traces and the clicked points over the image (the image and the
        zoom stay as they are)."""
        ax = getattr(self, "iax", None)
        if ax is None or self.img is None:
            return
        for a in self._art:
            a.remove()
        art = self._art = []
        if len(self.box) >= 1:
            art += ax.plot(*self.box[0], "c+", ms=14, mew=2)
        cal = None
        if len(self.box) == 2:
            try:
                cal = self._calib()
            except (ValueError, TypeError):
                cal = None
        if cal is not None:
            x0, y0, x1, y1 = cal.box
            art.append(ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.5, ls="--")))
        for key in ("i", "v"):
            col = COL[key]
            if self.by[key] == "points":
                p = self.pts[key]
                if p:
                    q = np.array(p)
                    q = q[np.argsort(q[:, 0], kind="stable")]
                    active = self.mode == "p" + key
                    art += ax.plot(q[:, 0], q[:, 1], "-", color=col, lw=1.2 if active else 0.8, alpha=0.9)
                    art += ax.plot(q[:, 0], q[:, 1], "o", mfc="none", mec=col, ms=7 if active else 5, mew=1.5)
                    art += ax.plot(*p[-1], "x", color="red" if active else col, ms=8, mew=2)    # the last one
            elif cal is not None and self.rgb[key] is not None:
                try:
                    xs, ys = DG.trace(self.img, cal.box, self.rgb[key], self.tol.GetValue() / 100)
                    art += ax.plot(xs, ys, ".", color=col, ms=2, alpha=0.9)
                except ValueError:
                    pass
        if draw:
            self.icanvas.draw_idle()

    # ------------------------------------------------------------- fit
    def _kinds(self):
        return list(RG.TYPES) + [RG.AUTO]

    def _settings(self):
        kind = self._kinds()[max(self.kind.GetSelection(), 0)]
        c, esr = self._val("c_test"), self._val("esr_test") or 0.0
        if not c or c <= 0:
            raise ValueError("Enter the datasheet's output capacitance for this plot.")
        boost = [self._val(k) for k in ("v_in", "v_out", "i_out", "l_ind")]
        f_rhp = RG.rhp_zero(*boost) if all(x for x in boost) else 0.0
        return kind, c, esr, f_rhp, self._val("step")

    def _options(self):
        return dict(window=self._val("window"), fit_c=self.fit_c.GetValue(),
                    weight="peak" if self.peak_w.GetValue() else "even")

    def _fit(self):
        if self.ti is None or self.tv is None:
            if not self._extract():
                return wx.MessageBox("Mark the plot area and read both traces first (pick their colours or click "
                                     "points along them), or import a CSV.", "Fit",
                                     parent=self)
        try:
            kind, c, esr, f_rhp, step = self._settings()
            opts = self._options()
        except (ValueError, TypeError) as e:
            return wx.MessageBox(str(e), "Fit", parent=self)
        self.fit_b.Disable()
        self.SetStatusText("Fitting every control type…" if kind == RG.AUTO else "Fitting…")
        args = (self.ti, self.i, self.tv, self.v)

        def work():
            try:
                fits = RG.fit(*args, c, esr, kind, f_rhp, step, **opts)
                wx.CallAfter(self._fitted, fits, (kind, c, esr, f_rhp, step), None)
            except Exception as e:      # noqa: BLE001
                wx.CallAfter(self._fitted, None, None, f"{e}\n\n{traceback.format_exc()}")
        threading.Thread(target=work, daemon=True).start()

    def _fitted(self, fits, how, err):
        self.fit_b.Enable()
        if err:
            self.SetStatusText("Fit failed.")
            return wx.MessageBox(err, "Fit", parent=self)
        self.fits = fits
        kind, c, esr, f_rhp, step = how
        up = next((f for f in fits if f.direction > 0), fits[0])
        dn = next((f for f in fits if f.direction < 0), None)
        m = self.model
        m.part, m.kind, m.f_rhp = self.part.GetValue().strip(), up.loop.kind, f_rhp
        m.c_test, m.esr_test = (up.c, up.esr) if up.c else (c, esr)
        m.up, m.down = asdict(up.loop), (asdict(dn.loop) if dn else {})
        m.source = self._source()
        self.use_b.Enable(self.on_use is not None)
        lines = [("Every control type tried" if kind == RG.AUTO else RG.TYPES[kind][0])
                 + f", fitted to {len(fits)} edge(s):"]
        for f in fits:
            lp = f.loop
            lines.append(f"{'Rising' if f.direction > 0 else 'Falling'} load step: peak {fmt_si(f.peak, 'V')}, "
                         f"fit error {fmt_si(f.rms, 'V')} rms ({100 * f.rms / max(f.peak, 1e-12):.0f} %)")
            if kind == RG.AUTO:
                lines.append(f"   best: {RG.TYPES[lp.kind][0]}  (" + ", ".join(
                    f"{k} {100 * r / max(f.peak, 1e-12):.0f} %" for k, r in f.tried) + ")")
            lines.append(f"   K {fmt_si(lp.K, 'S')}" + (f", zero {fmt_si(lp.f_z, 'Hz')}" if lp.f_z else "")
                         + (f", 2nd zero {fmt_si(lp.f_z2, 'Hz')}" if lp.f_z2 else "")
                         + (f", pole {fmt_si(lp.f_p, 'Hz')}" if lp.f_p < 1e9 else "")
                         + (f", resonance {fmt_si(lp.f_n, 'Hz')} Q {lp.q:.2g}" if lp.f_n else "")
                         + (f", RHP zero {fmt_si(lp.f_rhp, 'Hz')}" if lp.f_rhp else "")
                         + f"; crossover with {fmt_si(f.c or c, 'F')}: {fmt_si(lp.f_cross(f.c or c), 'Hz')}")
            if f.c and abs(f.c - c) > 1e-3 * c or self.fit_c.GetValue():
                lines.append(f"   test capacitance fitted: {fmt_si(f.c, 'F')} (entered {fmt_si(c, 'F')}), "
                             f"ESR {fmt_si(f.esr, 'Ω')}")
            if f.at_limit:
                lines.append(f"   ⚠ ended at the edge of its range: {', '.join(f.at_limit)}. The plot holds something "
                             "this model can't explain (a probe spike at the edge, a burst pattern, a wrong test C or "
                             "scale): trust it less, or narrow 'Fit up to'.")
        worst = max(f.rms / max(f.peak, 1e-12) for f in fits)
        if worst > 0.2:
            lines.append("The fit is loose (> 20 %). Check, in this order: (1) the traces drawn over the image follow "
                         "the plot everywhere (where one ran into a label or another trace, use 'Edit as points' "
                         "and fix it); (2) the plot area and the scales; (3) the test capacitance (or tick 'Also "
                         "fit the test capacitance'); (4) the control type ('Try every type'). (5) A light-load plot "
                         "(the converter skips pulses: bursts on the inductor current, a saw-tooth output) is not a "
                         "linear loop: use the datasheet's heavier-load plot, or 'Fit up to' the first recovery "
                         "only.")
        self.result.SetValue("\n".join(lines))
        self.SetStatusText("Fitted.")
        self._plot_fit(c, esr)
        self.nb.SetSelection(1)

    def _source(self):
        src = dict(image=self.img_path, tol=self.tol.GetValue(), control=self._kinds()[max(self.kind.GetSelection(), 0)],
                   fit_c=self.fit_c.GetValue(), peak=self.peak_w.GetValue(),
                   c_entered=self._val("c_test"), esr_entered=self._val("esr_test"))
        if len(self.box) == 2:
            src["box"] = [list(map(float, p)) for p in self.box]
        for k in ("nx", "ny", "t_div", "a_div", "v_div", "step", "v_in", "v_out", "i_out", "l_ind", "window"):
            src[k] = self._val(k)
        src["rgb"] = {k: (list(map(float, v)) if v is not None else None) for k, v in self.rgb.items()}
        src["by"] = dict(self.by)
        src["points"] = {k: [list(map(float, p)) for p in v] for k, v in self.pts.items()}
        return src

    def _restore(self):
        m = self.model
        src = m.source or {}
        ctl = src.get("control", m.kind)
        self.kind.SetSelection(self._kinds().index(ctl) if ctl in self._kinds() else 0)
        self.fit_c.SetValue(bool(src.get("fit_c")))
        self.peak_w.SetValue(bool(src.get("peak")))
        for key, val in (("c_test", src.get("c_entered") or m.c_test), ("esr_test", src.get("esr_entered", m.esr_test))):
            t, unit = self.fields[key]
            if val is not None:
                t.ChangeValue(fmt_si(val, unit))
        for k, (t, unit) in self.fields.items():
            if src.get(k) is not None and k not in ("c_test", "esr_test"):
                t.ChangeValue(f"{src[k]:g}" if not unit else fmt_si(src[k], unit))
        if src.get("tol"):
            self.tol.SetValue(int(src["tol"]))
        if src.get("image") and os.path.exists(src["image"]) and not src["image"].lower().endswith((".csv", ".txt")):
            try:
                self._set_image(DG.load_image(src["image"]), src["image"])
                self.box = [tuple(p) for p in src.get("box", [])]
                self.rgb = {k: (np.array(v) if v is not None else None) for k, v in (src.get("rgb") or {}).items()}
                pts = src.get("points") or {}
                for k in ("i", "v"):
                    self.rgb.setdefault(k, None)
                    if self.rgb[k] is not None:
                        getattr(self, f"sw_{k}").SetBackgroundColour(wx.Colour(*(int(255 * q) for q in self.rgb[k])))
                    self.pts[k] = [tuple(p) for p in pts.get(k, [])]
                    self._set_by(k, (src.get("by") or {}).get(k, self.by[k]), extract=False)
                self._counts()
                self._extract(quiet=True)
            except Exception:       # noqa: BLE001
                pass
        if m.up:
            self.result.SetValue("A fitted model is loaded: Fit again after changing anything, or use it as it is.")
            self.use_b.Enable(self.on_use is not None)

    def _plot_fit(self, c, esr):
        fig = self.ffig
        fig.clf()
        n = len(self.fits)
        gs = fig.add_gridspec(2, max(n, 1))
        for q, f in enumerate(self.fits):
            ax = fig.add_subplot(gs[0, q])
            t = (np.array(f.t) - f.t[0]) * 1e6
            ax.plot(t, np.array(f.v_meas) * 1e3, color="0.4", lw=1, label="datasheet")
            ax.plot(t, np.array(f.v_fit) * 1e3, color="tab:red", lw=1.5, label="model")
            ax.set_title(("Rising" if f.direction > 0 else "Falling") + f" load step ({f.loop.kind}, "
                         f"{100 * f.rms / max(f.peak, 1e-12):.0f} % rms)", fontsize=9)
            ax.set_xlabel("t [µs]")
            ax.set_ylabel("ΔV out [mV]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        ax = fig.add_subplot(gs[1, :])
        fr = np.logspace(2, 8, 400)
        for f in self.fits:
            c0, e0 = (f.c, f.esr) if f.c else (c, esr)
            for cc, ls in ((c0, "-"), (10 * c0, ":")):
                z = 1 / (f.loop.Y(fr) + 1 / (e0 + 1 / (2j * np.pi * fr * cc)))
                ax.loglog(fr, np.abs(z), ls=ls, label=f"{'rising' if f.direction > 0 else 'falling'} fit, "
                                                         f"C = {fmt_si(cc, 'F')}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("|Z out| [Ω]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Output impedance with the test C, and with 10× it (the loop slows down)", fontsize=9)
        self.fcanvas.draw()

    def _use(self):
        if self.on_use and self.model.up:
            self.model.part = self.part.GetValue().strip()
            self.on_use(self.model)
            self.SetStatusText("The PDN tool now uses this model for the regulator.")
