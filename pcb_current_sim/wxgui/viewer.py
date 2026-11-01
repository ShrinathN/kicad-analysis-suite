"""wxPython result viewer: field map with cursor readout, tables and exports."""
import math
import os
import webbrowser

# WebKitGTK's DMA-BUF renderer fails inside the KiCad Flatpak ("Failed to create GBM buffer") and
# leaves the 3D tab blank; the shared-memory path works everywhere.  Must be set before WebKit starts.
os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")

import wx
try:
    import wx.html2 as html2         # embedded browser for the 3D tab (WebKitGTK on Linux)
except ImportError:                  # pragma: no cover - wx built without WebView
    html2 = None
import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import export  # noqa: E402
from ..tables import TABLES, fmt, readout, sort_key  # noqa: E402
from .layout import AutoList  # noqa: E402

REF_HINT = "ΔV reference: global ground (each net's sink).\nClick the map to set a reference point."


SIDEBAR_W = 460         # px, the tables / export sidebar's starting width
READOUT_PT = 12.5       # pt, the values box on the map


class ViewerFrame(wx.Frame):
    """Left: export buttons, then the summary and every table as tabs.  Middle: the map (2D, or the
    3D view) as large as the window allows, its controls in one bar above it, and the values under
    the pointer in a box on the map that follows it."""

    def __init__(self, parent, res):
        super().__init__(parent, title=f"Current density - {os.path.basename(res.board_path)}", size=(1500, 950))
        self.res = res
        layers = [L for L in res.layers if export._layer_has_current(res, L)] or res.layers
        self.layer = layers[0]
        self.qkeys = res.quantity_keys()
        self.ref, self.marker, self.hl = None, None, None
        self.ov, self.bg = None, None
        self.CreateStatusBar()
        self.SetStatusText(REF_HINT.replace("\n", " "))

        split = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        side = wx.Panel(split)
        self.views = wx.Notebook(split)
        top = self.top = wx.Panel(self.views)

        # ---- sidebar: exports, then the summary and the tables
        ss = wx.BoxSizer(wx.VERTICAL)
        ebox = wx.StaticBoxSizer(wx.VERTICAL, side, "Export")
        erow = wx.WrapSizer()
        tips = {"View…": "The map as shown (PNG, SVG or PDF)", "All layers…": "Every layer and quantity as images",
                "Report PDF…": "Summary, tables and maps in one PDF", "CSV…": "Every table as CSV files",
                "HTML…": "An interactive map in a web page", "ParaView…": "VTK files for ParaView",
                "Save .npz…": "The whole result, to open again later from the Results browser"}
        for text, cmd in (("View…", self._export_view), ("All layers…", self._export_all),
                          ("Report PDF…", self._export_pdf), ("CSV…", self._export_csv),
                          ("HTML…", self._export_html), ("ParaView…", self._export_vtk),
                          ("Save .npz…", self._save)):
            b = wx.Button(ebox.GetStaticBox(), label=text, style=wx.BU_EXACTFIT)
            b.SetToolTip(tips[text])
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            erow.Add(b, 0, wx.ALL, 2)
        ebox.Add(erow, 0, wx.EXPAND)
        ss.Add(ebox, 0, wx.EXPAND | wx.ALL, 4)
        # every table's name in a list (a row of tabs would not fit a sidebar), the chosen one below
        self.pick = wx.ListBox(side, style=wx.LB_SINGLE)
        self.pick.SetToolTip("The summary and the result tables: pick one to show it below")
        tabs = self.tabs = wx.Simplebook(side)
        txt = wx.TextCtrl(tabs, value=res.summary_text(), style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP)
        txt.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        txt.ShowPosition(0)
        tabs.AddPage(txt, "Summary")
        self.pick.Append("Summary")
        for title, attr, cols in TABLES:
            rows = getattr(res, attr)
            if not rows and attr not in ("rows", "pads"):
                continue                    # empty tables (no vias, no pulse...) only take room
            lc = AutoList(tabs, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
            for k, c in enumerate(cols):
                lc.InsertColumn(k, c, width=max(80, 9 * len(c)))
            lc.rows, lc.cols, lc.rev = rows, cols, False
            self._fill(lc, list(range(len(rows))))
            lc.Bind(wx.EVT_LIST_COL_CLICK, lambda e, lc=lc: self._sort(lc, e.GetColumn()))
            lc.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda e, lc=lc: self._goto(lc.rows[lc.GetItemData(e.GetIndex())]))
            lc.SetToolTip("Click a row to mark it on the map; click a column header to sort")
            tabs.AddPage(lc, f"{title} ({len(rows)})")
            self.pick.Append(f"{title} ({len(rows)})")
        self.pick.SetSelection(0)
        self.pick.Bind(wx.EVT_LISTBOX, lambda e: self.tabs.SetSelection(self.pick.GetSelection()))
        h = self.pick.GetCharHeight() + 9
        self.pick.SetMinSize((-1, h * self.pick.GetCount() + 6))
        ss.Add(self.pick, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
        ss.Add(tabs, 1, wx.EXPAND | wx.ALL, 4)
        side.SetSizer(ss)

        # ---- middle: one bar of map controls, the map, its toolbar
        bar = wx.WrapSizer(wx.HORIZONTAL, 0)      # 0: the last item keeps its own width

        def lab(t):
            bar.Add(wx.StaticText(top, label=t), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        lab("Layer")
        self.lch = wx.Choice(top, choices=res.layers)
        self.lch.SetStringSelection(self.layer)
        self.lch.Bind(wx.EVT_CHOICE, lambda e: self._set_layer(self.lch.GetStringSelection()))
        self.lch.SetToolTip("Copper layer shown in the map")
        bar.Add(self.lch, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        lab("Show")
        self.qbox = wx.Choice(top, choices=[res.QUANTITIES[k] for k in self.qkeys])
        self.qbox.SetSelection(0)
        self.qbox.Bind(wx.EVT_CHOICE, lambda e: self.redraw())
        self.qbox.SetToolTip("The quantity drawn on the map")
        bar.Add(self.qbox, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        self.log = wx.CheckBox(top, label="Log scale")
        self.log.SetValue(True)
        self.stream = wx.CheckBox(top, label="Streamlines")
        self.labels = wx.CheckBox(top, label="Pad labels")
        self.labels.SetValue(True)
        self.follow = wx.CheckBox(top, label="Values at the pointer")
        self.follow.SetValue(True)
        self.log.SetToolTip("Logarithmic colour scale for current density and power (shows the weak spreading "
                            "as well as the hot spots)")
        self.stream.SetToolTip("Draw lines along the current flow")
        self.labels.SetToolTip("Label the pads with their part and pad number")
        self.follow.SetToolTip("Show |J|, V, ΔV, power density, T and B under the mouse pointer in a box on the map "
                               "(unticked: in the top-left corner of the map)")
        for c in (self.log, self.stream, self.labels):
            c.Bind(wx.EVT_CHECKBOX, lambda e: self.redraw())
            bar.Add(c, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        bar.Add(self.follow, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        b = wx.Button(top, label="Clear ΔV reference", style=wx.BU_EXACTFIT)
        b.SetToolTip("Forget the reference point: ΔV is no longer shown (click the map to set a new one)")
        b.Bind(wx.EVT_BUTTON, lambda e: self._clear_ref())
        bar.Add(b, 0, wx.ALIGN_CENTER_VERTICAL)

        self.fig = Figure(figsize=(9, 6.5), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(top, -1, self.fig)
        self.toolbar = NavigationToolbar2WxAgg(self.canvas)
        self.toolbar.Realize()
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_press_event", self._click)
        self.canvas.mpl_connect("draw_event", self._drawn)
        self.canvas.mpl_connect("axes_leave_event", lambda e: self._show_values(None))
        self.canvas.SetToolTip("Click to set the ΔV reference point; zoom and pan with the toolbar below")
        plot = wx.BoxSizer(wx.VERTICAL)
        plot.Add(bar, 0, wx.EXPAND | wx.ALL, 4)
        plot.Add(self.canvas, 1, wx.EXPAND)
        plot.Add(self.toolbar, 0, wx.EXPAND)
        top.SetSizer(plot)
        self.views.AddPage(top, "2D map")
        self.page3d, self.web, self.html3d = None, None, None
        if res.summary.get("layer_z_mm"):
            self.page3d = wx.Panel(self.views)
            self.views.AddPage(self.page3d, "3D")
            self.views.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self._page)

        split.SplitVertically(side, self.views, SIDEBAR_W)
        split.SetMinimumPaneSize(200)
        split.SetSashGravity(0.0)            # the map takes what the window gains
        self.redraw()

    # ------------------------------------------------------------ tables
    def _fill(self, lc, order):
        lc.DeleteAllItems()
        for i in order:
            r = lc.rows[i]
            k = lc.Append([fmt(r.get(c)) for c in lc.cols])
            lc.SetItemData(k, i)

    def _sort(self, lc, col):
        c = lc.cols[col]
        order = sorted(range(len(lc.rows)), key=lambda i: sort_key(fmt(lc.rows[i].get(c))), reverse=lc.rev)
        lc.rev = not lc.rev
        self._fill(lc, order)

    def _goto(self, r):
        x, y = r.get("x"), r.get("y")
        if x is None or y is None:
            return
        if r.get("layer") in self.res.layers and r["layer"] != self.layer:
            self._set_layer(r["layer"])
        if self.hl is not None:
            try:
                self.hl.remove()
            except ValueError:
                pass
        self.hl = self.ax.plot(x, y, "o", mfc="none", mec="lime", ms=18, mew=2)[0]
        self.canvas.draw_idle()

    # ------------------------------------------------------------ drawing
    @property
    def qty(self):
        return self.qkeys[self.qbox.GetSelection()]

    def _set_layer(self, L):
        self.layer = L
        self.lch.SetStringSelection(L)
        self.redraw()

    def redraw(self):
        lims = None
        if self.fig.axes:
            a = self.fig.axes[0]
            lims = (a.get_xlim(), a.get_ylim())
        self.fig.clear()
        self.ax = self.fig.add_subplot()
        q = self.qty
        im = export.plot_field(self.ax, self.res, self.layer, q, self.log.GetValue(), self.stream.GetValue(),
                               True, self.labels.GetValue())
        if im is not None:
            self.fig.colorbar(im, ax=self.ax, label=self.res.QUANTITIES[q], shrink=0.9)
        else:
            self.ax.text(0.5, 0.5, "no data on this layer", transform=self.ax.transAxes, ha="center", color="w")
        self.ax.set_title(f"{self.res.QUANTITIES[q]}  -  {self.layer}")
        if lims:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
        self.marker = self.hl = None
        if self.ref and self.ref[0] == self.layer:
            self.marker = self.ax.plot(self.ref[1], self.ref[2], "c+", ms=14, mew=2)[0]
        # the values box: animated, so it is blitted over the saved map instead of redrawing it
        self.ov = self.ax.annotate("", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
                                   fontsize=READOUT_PT, family="monospace", color="white", zorder=20,
                                   multialignment="left",
                                   bbox=dict(boxstyle="round,pad=0.45", fc=(0.05, 0.05, 0.08, 0.90), ec="0.75"),
                                   animated=True, annotation_clip=False, visible=False)
        self.bg = None
        self.canvas.draw_idle()

    def _drawn(self, e):
        """After every full draw: keep the map as the background the values box is blitted over."""
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)

    def _show_values(self, e):
        if self.ov is None or self.bg is None or str(self.toolbar.mode):
            return                          # nothing drawn yet, or panning / zooming (the map is moving)
        if e is None or e.inaxes is not self.ax or e.xdata is None:
            self.ov.set_visible(False)
        else:
            self.ov.set_text(readout(self.res, self.layer, e.xdata, e.ydata, self.ref))
            self.ov.set_visible(True)
            if self.follow.GetValue():
                bb = self.ax.bbox          # keep the box inside the map: flip it at the right / top half
                right = e.x > bb.x0 + 0.55 * bb.width
                upper = e.y > bb.y0 + 0.5 * bb.height
                self.ov.xy = (e.xdata, e.ydata)
                self.ov.xycoords = "data"
                self.ov.set_position((-18 if right else 18, -18 if upper else 18))
                self.ov.set_ha("right" if right else "left")
                self.ov.set_va("top" if upper else "bottom")
            else:
                self.ov.xy, self.ov.xycoords = (0.01, 0.99), "axes fraction"
                self.ov.set_position((4, -4))
                self.ov.set_ha("left")
                self.ov.set_va("top")
        self.canvas.restore_region(self.bg)
        if self.ov.get_visible():
            self.ax.draw_artist(self.ov)
        self.canvas.blit(self.fig.bbox)

    def _motion(self, e):
        self._show_values(e)

    def _click(self, e):
        if e.inaxes is not self.ax or e.button != 1 or str(self.toolbar.mode):
            return
        s = self.res.sample(self.layer, e.xdata, e.ydata)
        if s.get("V") is None or not math.isfinite(s["V"]):
            return
        self.ref = (self.layer, e.xdata, e.ydata, s["V"])
        self.SetStatusText(f"ΔV reference: ({e.xdata:.2f}, {e.ydata:.2f}) mm on {self.layer}, "
                           f"V = {s['V']:.3f} mV, net {s.get('net')}. The box on the map shows ΔV against it.")
        if self.marker is not None:
            self.marker.remove()
        self.marker = self.ax.plot(e.xdata, e.ydata, "c+", ms=14, mew=2)[0]
        self.canvas.draw_idle()

    def _clear_ref(self):
        self.ref = None
        self.SetStatusText(REF_HINT.replace("\n", " "))
        self.redraw()

    # ------------------------------------------------------------ 3D
    def _page(self, e):
        e.Skip()
        if self.views.GetPage(e.GetSelection()) is self.page3d and self.web is None:
            self._build_3d()

    def _build_3d(self):
        import tempfile
        p = self.page3d
        sz = wx.BoxSizer(wx.VERTICAL)
        bar = wx.BoxSizer(wx.HORIZONTAL)
        bar.Add(wx.StaticText(p, label="Drag to rotate, scroll to zoom, right-drag to pan; the menu at the "
                                       "top left picks the quantity; click legend items to hide them."),
                1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        for text, cmd in (("Open in browser", self._3d_browser), ("Save 3D HTML…", self._3d_save)):
            b = wx.Button(p, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, c=cmd: c())
            bar.Add(b, 0, wx.LEFT, 4)
        sz.Add(bar, 0, wx.EXPAND | wx.ALL, 2)
        with wx.BusyCursor():
            fd, self.html3d = tempfile.mkstemp(suffix=".html", prefix="pcb3d_")
            os.close(fd)
            ok = export.export_html_3d(self.res, self.html3d, js="local" if export.plotly_js_path() else "inline")
        try:
            if not ok:
                raise RuntimeError("no 3D data in this result (re-run the simulation)")
            if html2 is None:
                raise RuntimeError("wxPython has no WebView")
            self.web = html2.WebView.New(p)
            self.web.LoadURL("file://" + self.html3d)
            sz.Add(self.web, 1, wx.EXPAND)
        except Exception as ex:      # noqa: BLE001 - no WebView backend: offer the browser instead
            self.web = False
            sz.Add(wx.StaticText(p, label=f"3D view not available inside this window ({ex}).\n"
                                          "Use 'Open in browser'."), 0, wx.ALL, 12)
        p.SetSizer(sz)
        p.Layout()

    def _3d_browser(self):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".html", prefix="pcb3d_")
        os.close(fd)
        if export.export_html_3d(self.res, path, js="inline"):
            webbrowser.open("file://" + path)

    def _3d_save(self):
        path = self._ask_save("Save 3D view", f"{self._stem()}_3d.html", "HTML (*.html)|*.html")
        if path:
            with wx.BusyCursor():
                export.export_html_3d(self.res, path, js="inline")

    # ------------------------------------------------------------ export
    def _stem(self):
        return os.path.splitext(os.path.basename(self.res.board_path))[0]

    def _ask_save(self, title, name, wildcard):
        with wx.FileDialog(self, title, defaultFile=name, wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as d:
            return d.GetPath() if d.ShowModal() == wx.ID_OK else None

    def _ask_dir(self, title):
        with wx.DirDialog(self, title) as d:
            return d.GetPath() if d.ShowModal() == wx.ID_OK else None

    def _export_view(self):
        p = self._ask_save("Export current view", f"{self._stem()}_{self.layer}_{self.qty}.png",
                           "PNG (*.png)|*.png|SVG (*.svg)|*.svg|PDF (*.pdf)|*.pdf")
        if p:
            self.fig.savefig(p, dpi=200)
            wx.MessageBox(f"Saved {p}", "Export", parent=self)

    def _export_all(self):
        d = self._ask_dir("Folder for images")
        if d:
            svg = wx.MessageBox("Save as SVG (vector)?\nNo = PNG", "Format", wx.YES_NO, parent=self) == wx.YES
            paths = export.export_images(self.res, d, "svg" if svg else "png", stream=self.stream.GetValue())
            wx.MessageBox(f"Saved {len(paths)} image(s) to {d}", "Export", parent=self)

    def _export_pdf(self):
        p = self._ask_save("Export report", f"{self._stem()}_current_report.pdf", "PDF (*.pdf)|*.pdf")
        if p:
            with wx.BusyCursor():
                export.export_report_pdf(self.res, p, stream=self.stream.GetValue())
            wx.MessageBox(f"Saved {p}", "Export", parent=self)

    def _export_csv(self):
        d = self._ask_dir("Folder for CSV tables")
        if d:
            paths = self.res.write_csv(d, self._stem())
            wx.MessageBox(f"Saved {len(paths)} table(s) to {d}", "Export", parent=self)

    def _export_html(self):
        p = self._ask_save("Export interactive HTML", f"{self._stem()}_current.html", "HTML (*.html)|*.html")
        if p:
            with wx.BusyCursor():
                ok = export.export_html(self.res, p)
            if ok:
                webbrowser.open("file://" + os.path.abspath(p))

    def _export_vtk(self):
        d = self._ask_dir("Folder for the ParaView files (a subfolder is created)")
        if not d:
            return
        from ..vtk_export import export_vtk
        out = os.path.join(d, f"{self._stem()}_paraview")
        try:
            with wx.BusyCursor():
                files = export_vtk(self.res, out)
        except Exception as e:      # noqa: BLE001
            return wx.MessageBox(f"ParaView export failed: {e}", "Export", parent=self)
        wx.MessageBox("Open in ParaView:\n" + "\n".join(files) + "\n\nSee README.txt in that folder for tips.",
                      "ParaView export", parent=self)

    def _save(self):
        p = self._ask_save("Save result", f"{self._stem()}_current_result.npz", "Result (*.npz)|*.npz")
        if p:
            self.res.save(p)
            wx.MessageBox(f"Saved {p} (+ .json)", "Export", parent=self)
