"""Result viewer: field map with cursor readout, tables and exports."""
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import export  # noqa: E402
from ..tables import TABLES, fmt, readout, sort_key  # noqa: E402

class Viewer(tk.Toplevel):
    def __init__(self, master, res):
        super().__init__(master)
        self.res = res
        self.title(f"Current density - {os.path.basename(res.board_path)}")
        self.geometry("1300x860")
        layers = [L for L in res.layers if export._layer_has_current(res, L)] or res.layers
        self.layer = tk.StringVar(value=layers[0])
        self.qty = tk.StringVar(value="J")
        self.log = tk.BooleanVar(value=True)
        self.stream = tk.BooleanVar(value=False)
        self.labels = tk.BooleanVar(value=True)
        self.ref = None           # (layer, x, y, V_mV)
        self.marker = None

        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True)
        top = ttk.Frame(pw)
        pw.add(top, weight=3)

        # plot
        pf = ttk.Frame(top)
        pf.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(9, 5.5), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        bar = ttk.Frame(pf)
        bar.pack(side="bottom", fill="x", pady=(0, 2))
        self.toolbar = NavigationToolbar2Tk(self.canvas, pf, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        ttk.Label(bar, text="Export:").pack(side="left", padx=(6, 2))
        for text, cmd in (("View…", self._export_view), ("All layers…", self._export_all),
                          ("Report PDF…", self._export_pdf), ("CSV…", self._export_csv),
                          ("HTML…", self._export_html), ("3D view", self._view_3d), ("ParaView…", self._export_vtk),
                          ("Save .npz…", self._save)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=1)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_press_event", self._click)

        # controls
        cf = ttk.Frame(top, padding=8, width=300)
        cf.pack(side="left", fill="y")
        ttk.Label(cf, text="Layer").pack(anchor="w")
        cb = ttk.Combobox(cf, textvariable=self.layer, values=res.layers, state="readonly", width=18)
        cb.pack(anchor="w", fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(cf, text="Quantity").pack(anchor="w", pady=(8, 0))
        for k in res.quantity_keys():
            label = res.QUANTITIES[k]
            ttk.Radiobutton(cf, text=label, value=k, variable=self.qty, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(cf, text="Log scale (|J|, power)", variable=self.log, command=self.redraw).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(cf, text="Current streamlines", variable=self.stream, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(cf, text="Pad labels", variable=self.labels, command=self.redraw).pack(anchor="w")

        ttk.Label(cf, text="Cursor", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(12, 0))
        self.readout = tk.Label(cf, text="move over the map", justify="left", anchor="nw", width=34,
                                height=9, font=("TkFixedFont", 9), relief="sunken", bg="#fafafa")
        self.readout.pack(anchor="w", fill="x")
        self.refl = ttk.Label(cf, text="ΔV reference: global ground (each net's sink)\n"
                                       "Click the map to set a reference point.", wraplength=280,
                              foreground="#555")
        self.refl.pack(anchor="w", pady=(4, 0))
        ttk.Button(cf, text="Clear reference", command=self._clear_ref).pack(anchor="w", pady=2)


        # tables
        nb = ttk.Notebook(pw)
        pw.add(nb, weight=2)
        txt = tk.Text(nb, wrap="none", font=("TkFixedFont", 9), height=12)
        txt.insert("1.0", res.summary_text())
        txt.config(state="disabled")
        nb.add(txt, text="Summary")
        for title, attr, cols in TABLES:
            rows = getattr(res, attr)
            f = ttk.Frame(nb)
            tv = ttk.Treeview(f, columns=cols, show="headings")
            for c in cols:
                tv.heading(c, text=c, command=lambda c=c, tv=tv: self._sort(tv, c))
                tv.column(c, width=max(70, 9 * len(c)), anchor="w")
            for k, r in enumerate(rows):
                tv.insert("", "end", iid=str(k), values=[fmt(r.get(c)) for c in cols])
            sb = ttk.Scrollbar(f, orient="vertical", command=tv.yview)
            tv.configure(yscrollcommand=sb.set)
            tv.pack(side="left", fill="both", expand=True)
            sb.pack(side="left", fill="y")
            tv.bind("<<TreeviewSelect>>", lambda e, tv=tv, rows=rows: self._goto(tv, rows))
            nb.add(f, text=f"{title} ({len(rows)})")
        self.redraw()

    # ------------------------------------------------------------ drawing
    def redraw(self):
        lims = None
        if self.fig.axes:
            a = self.fig.axes[0]
            lims = (a.get_xlim(), a.get_ylim())
        self.fig.clear()
        self.ax = self.fig.add_subplot()
        q = self.qty.get()
        im = export.plot_field(self.ax, self.res, self.layer.get(), q, self.log.get(),
                               self.stream.get(), True, self.labels.get())
        if im is not None:
            self.fig.colorbar(im, ax=self.ax, label=self.res.QUANTITIES[q], shrink=0.9)
        else:
            self.ax.text(0.5, 0.5, "no data on this layer", transform=self.ax.transAxes,
                         ha="center", color="w")
        self.ax.set_title(f"{self.res.QUANTITIES[q]}  -  {self.layer.get()}")
        if lims:
            self.ax.set_xlim(lims[0])
            self.ax.set_ylim(lims[1])
        self.marker = None
        if self.ref and self.ref[0] == self.layer.get():
            self.marker = self.ax.plot(self.ref[1], self.ref[2], "c+", ms=14, mew=2)[0]
        self.canvas.draw_idle()

    def _motion(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        self.readout.config(text=readout(self.res, self.layer.get(), e.xdata, e.ydata, self.ref))

    def _click(self, e):
        if e.inaxes is not self.ax or e.button != 1 or str(self.toolbar.mode):
            return
        s = self.res.sample(self.layer.get(), e.xdata, e.ydata)
        if s.get("V") is None or not math.isfinite(s["V"]):
            return
        self.ref = (self.layer.get(), e.xdata, e.ydata, s["V"])
        self.refl.config(text=f"ΔV reference: ({e.xdata:.2f}, {e.ydata:.2f}) on {self.layer.get()}, "
                              f"V = {s['V']:.3f} mV, net {s.get('net')}")
        if self.marker is not None:
            self.marker.remove()
        self.marker = self.ax.plot(e.xdata, e.ydata, "c+", ms=14, mew=2)[0]
        self.canvas.draw_idle()

    def _clear_ref(self):
        self.ref = None
        self.refl.config(text="ΔV reference: global ground (each net's sink)\nClick the map to set a reference point.")
        self.redraw()

    def _goto(self, tv, rows):
        sel = tv.selection()
        if not sel:
            return
        r = rows[int(sel[0])]
        x, y = r.get("x"), r.get("y")
        if x is None or y is None:
            return
        if r.get("layer") in self.res.layers and r["layer"] != self.layer.get():
            self.layer.set(r["layer"])
            self.redraw()
        if getattr(self, "_hl", None) is not None:
            try:
                self._hl.remove()
            except ValueError:
                pass
        self._hl = self.ax.plot(x, y, "o", mfc="none", mec="lime", ms=18, mew=2)[0]
        self.canvas.draw_idle()

    def _sort(self, tv, col):
        items = [(tv.set(k, col), k) for k in tv.get_children("")]

        rev = getattr(tv, "_rev", False)
        items.sort(key=lambda kv: sort_key(kv[0]), reverse=rev)
        tv._rev = not rev
        for i, (_, k) in enumerate(items):
            tv.move(k, "", i)

    # ------------------------------------------------------------ export
    def _stem(self):
        return os.path.splitext(os.path.basename(self.res.board_path))[0]

    def _export_view(self):
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".png",
                                         initialfile=f"{self._stem()}_{self.layer.get()}_{self.qty.get()}.png",
                                         filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if p:
            self.fig.savefig(p, dpi=200)
            messagebox.showinfo("Export", f"Saved {p}", parent=self)

    def _export_all(self):
        d = filedialog.askdirectory(parent=self, title="Folder for images")
        if d:
            fmt_ = "svg" if messagebox.askyesno("Format", "Save as SVG (vector)?\nNo = PNG", parent=self) else "png"
            paths = export.export_images(self.res, d, fmt_, stream=self.stream.get())
            messagebox.showinfo("Export", f"Saved {len(paths)} image(s) to {d}", parent=self)

    def _export_pdf(self):
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".pdf",
                                         initialfile=f"{self._stem()}_current_report.pdf",
                                         filetypes=[("PDF", "*.pdf")])
        if p:
            export.export_report_pdf(self.res, p, stream=self.stream.get())
            messagebox.showinfo("Export", f"Saved {p}", parent=self)

    def _export_csv(self):
        d = filedialog.askdirectory(parent=self, title="Folder for CSV tables")
        if d:
            paths = self.res.write_csv(d, self._stem())
            messagebox.showinfo("Export", f"Saved {len(paths)} table(s) to {d}", parent=self)

    def _export_html(self):
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".html",
                                         initialfile=f"{self._stem()}_current.html",
                                         filetypes=[("HTML", "*.html")])
        if p:
            export.export_html(self.res, p)
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(p))

    def _view_3d(self):
        """Tk has no embedded browser: open the 3D scene in the system browser."""
        import tempfile
        import webbrowser
        fd, path = tempfile.mkstemp(suffix=".html", prefix="pcb3d_")
        os.close(fd)
        if export.export_html_3d(self.res, path, js="inline"):
            webbrowser.open("file://" + path)
        else:
            messagebox.showinfo("3D view", "This result has no layer heights; re-run the simulation.", parent=self)

    def _export_vtk(self):
        d = filedialog.askdirectory(parent=self, title="Folder for the ParaView files (a subfolder is created)")
        if not d:
            return
        from ..vtk_export import export_vtk
        try:
            files = export_vtk(self.res, os.path.join(d, f"{self._stem()}_paraview"))
        except Exception as e:      # noqa: BLE001
            return messagebox.showerror("Export", f"ParaView export failed: {e}", parent=self)
        messagebox.showinfo("ParaView export", "Open in ParaView:\n" + "\n".join(files), parent=self)

    def _save(self):
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".npz",
                                         initialfile=f"{self._stem()}_current_result.npz",
                                         filetypes=[("Result", "*.npz")])
        if p:
            self.res.save(p)
            messagebox.showinfo("Export", f"Saved {p} (+ .json)", parent=self)
