"""Main window: board, settings, excitation table, run."""
import os
import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, simpledialog, ttk

from .. import board as B
from .. import elements as EL
from .. import solver as S
from ..tables import CHECKS, FLOAT_SETTINGS, ROW_TEXT
from ..scenario import load_scenario, save_scenario, scenario_path


class RowDialog(tk.Toplevel):
    """Edit one excitation row: source pads, sink pads (each side on one net), current."""

    def __init__(self, master, board, row=None):
        super().__init__(master)
        self.title("Excitation")
        self.transient(master)
        self.board, self.result = board, None
        nets = [n for n in board.nets if board.pad_ids(n)]
        row = row or S.Row(f"row{len(master.rows) + 1}", [], [], 1.0)

        def net_of(ids):
            ps = board.pads_by_id(ids[0]) if ids else []
            return ps[0].net if ps else (nets[0] if nets else "")

        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Name").grid(row=0, column=0, sticky="w")
        self.name = tk.StringVar(value=row.name)
        ttk.Entry(f, textvariable=self.name, width=24).grid(row=0, column=1, sticky="we")
        self.nsrc = tk.StringVar(value=net_of(row.src or row.snk))
        self.nsnk = tk.StringVar(value=net_of(row.snk or row.src))
        self.vlabel = ttk.Label(f, text="Current [A]")
        self.vlabel.grid(row=0, column=2, sticky="e", padx=(12, 4))
        self.cur = tk.StringVar(value=f"{row.current:g}")
        ttk.Entry(f, textvariable=self.cur, width=10).grid(row=0, column=3, sticky="w")
        self.kind = tk.StringVar(value=S.KINDS[row.kind])
        kc = ttk.Combobox(f, textvariable=self.kind, values=list(S.KINDS.values()), state="readonly", width=18)
        ttk.Label(f, text="Kind").grid(row=5, column=0, sticky="w", pady=(6, 0))
        kc.grid(row=5, column=1, sticky="w", pady=(6, 0))
        kc.bind("<<ComboboxSelected>>", lambda e: self._labels())
        self.slabel, self.klabel = ttk.Label(f), ttk.Label(f)
        self.slabel.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.klabel.grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))
        self.lsrc = tk.Listbox(f, selectmode="extended", exportselection=False, height=12, width=28)
        self.lsnk = tk.Listbox(f, selectmode="extended", exportselection=False, height=12, width=28)
        for col, var, lb in ((0, self.nsrc, self.lsrc), (2, self.nsnk, self.lsnk)):
            cb = ttk.Combobox(f, textvariable=var, values=nets, state="readonly", width=28)
            cb.grid(row=2, column=col, columnspan=2, sticky="we", padx=(8 if col else 0, 0))
            cb.bind("<<ComboboxSelected>>", lambda e, v=var, l=lb: self._fill_one(v, l))
        self.lsrc.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.lsnk.grid(row=3, column=2, columnspan=2, sticky="nsew", padx=(8, 0))
        self.hint = ttk.Label(f, foreground="#666")
        self.hint.grid(row=4, column=0, columnspan=4, sticky="w")
        b = ttk.Frame(f)
        b.grid(row=6, column=0, columnspan=4, sticky="e", pady=(8, 0))
        ttk.Button(b, text="OK", command=self._ok).pack(side="right")
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)
        f.rowconfigure(3, weight=1)
        self._fill(row.src, row.snk)
        self._labels()
        self.grab_set()
        self.wait_window()

    def _kind(self):
        return {v: k for k, v in S.KINDS.items()}[self.kind.get()]

    def _labels(self):
        a, b, unit, hint = ROW_TEXT[self._kind()]
        self.slabel.config(text=a)
        self.klabel.config(text=b)
        self.vlabel.config(text=unit)
        self.hint.config(text=hint + "\nSeveral pads on one side are tied together (equipotential).")

    def _fill(self, src=(), snk=()):
        self._fill_one(self.nsrc, self.lsrc, src)
        self._fill_one(self.nsnk, self.lsnk, snk)

    def _fill_one(self, var, lb, sel=()):
        lb.delete(0, "end")
        for k, pid in enumerate(self.board.pad_ids(var.get())):
            lb.insert("end", pid)
            if pid in sel:
                lb.selection_set(k)
                lb.see(k)

    def _ok(self):
        src = [self.lsrc.get(i) for i in self.lsrc.curselection()]
        snk = [self.lsnk.get(i) for i in self.lsnk.curselection()]
        try:
            cur = float(self.cur.get())
        except ValueError:
            return messagebox.showerror("Excitation", "The value must be a number.", parent=self)
        if not src or not snk:
            return messagebox.showerror("Excitation", "Select at least one source and one sink pad.", parent=self)
        if set(src) & set(snk):
            return messagebox.showerror("Excitation", "A pad cannot be both source and sink.", parent=self)
        self.result = S.Row(self.name.get().strip() or "row", src, snk, cur, self._kind())
        self.destroy()


class ElementDialog(tk.Toplevel):
    """Edit one resistance element: two pads (any parts), R, optional power rating."""

    def __init__(self, master, board, el):
        super().__init__(master)
        self.title("Resistance element")
        self.transient(master)
        self.board, self.result, self.el = board, None, el
        refs = sorted({p.ref for p in board.pads if p.layers}, key=EL.natural_key)
        self.nets = {p.id: p.net for p in board.pads}
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        self.ref, self.pad, self.pcb = [], [], []
        for k, pid in enumerate((el.a, el.b)):
            ttk.Label(f, text=f"Pad {'AB'[k]}").grid(row=k, column=0, sticky="w")
            rv = tk.StringVar(value=pid.rsplit(".", 1)[0] if pid else (refs[0] if refs else ""))
            pv = tk.StringVar()
            rc = ttk.Combobox(f, textvariable=rv, values=refs, state="readonly", width=10)
            pc = ttk.Combobox(f, textvariable=pv, state="readonly", width=30)
            rc.grid(row=k, column=1, sticky="w", padx=4, pady=2)
            pc.grid(row=k, column=2, sticky="we", pady=2)
            rc.bind("<<ComboboxSelected>>", lambda e, k=k: self._pads(k))
            self.ref.append(rv)
            self.pad.append(pv)
            self.pcb.append(pc)
            self._pads(k, pid)
        self.name = tk.StringVar(value=el.name)
        self.R = tk.StringVar(value=EL.fmt_ohms(el.R).replace(" ", "").replace("Ω", "") if el.R or el.auto else "")
        self.P = tk.StringVar(value=f"{el.P_rated:g}" if el.P_rated else "")
        self.heats = tk.BooleanVar(value=el.heats)
        ttk.Checkbutton(f, text="Dissipates on the board (untick for an external load or wire)",
                        variable=self.heats).grid(row=7, column=0, columnspan=3, sticky="w")
        for k, (label, var) in enumerate((("Name", self.name), ("Resistance [Ω]", self.R),
                                          ("Power rating [W]", self.P))):
            ttk.Label(f, text=label).grid(row=2 + k, column=0, sticky="w")
            ttk.Entry(f, textvariable=var, width=14).grid(row=2 + k, column=1, columnspan=2, sticky="w", padx=4,
                                                         pady=2)
        ttk.Label(f, foreground="#666", text="Values like 4k7, 0R1, 5m (milli), 1M (mega). Rating may be blank.\n"
                                             "Pick any two pads: a resistor, a FET's drain-source R_on, an "
                                             "inductor's\nDCR, a fuse, or an external load/wire between two "
                                             "connectors.").grid(row=5, column=0, columnspan=3, sticky="w", pady=6)
        b = ttk.Frame(f)
        b.grid(row=8, column=0, columnspan=3, sticky="e")
        ttk.Button(b, text="OK", command=self._ok).pack(side="right")
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        f.columnconfigure(2, weight=1)
        self.grab_set()
        self.wait_window()

    def _ids(self, k):
        return sorted({p.id for p in self.board.pads if p.ref == self.ref[k].get() and p.layers},
                      key=EL.natural_key)

    def _label(self, pid):
        return f"{pid.rsplit('.', 1)[1]}  ({self.nets[pid] or 'no net'})"

    def _pads(self, k, sel=None):
        ids = self._ids(k)
        self.pcb[k].config(values=[self._label(p) for p in ids])
        if ids:
            self.pad[k].set(self._label(sel if sel in ids else ids[min(k, len(ids) - 1)]))

    def _ok(self):
        pick = []
        for k in (0, 1):
            m = [p for p in self._ids(k) if self._label(p) == self.pad[k].get()]
            if not m:
                return messagebox.showerror("Element", "Pick both pads.", parent=self)
            pick.append(m[0])
        a, b = pick
        if a == b:
            return messagebox.showerror("Element", "The two pads must differ.", parent=self)
        R = EL.parse_ohms(self.R.get())
        if R is None or R < 0:
            return messagebox.showerror("Element", "Resistance: enter a value such as 0.01, 10m, 4k7.", parent=self)
        try:
            P = float(self.P.get()) if self.P.get().strip() else None
        except ValueError:
            return messagebox.showerror("Element", "Power rating must be a number (or blank).", parent=self)
        name = self.name.get().strip() or a.rsplit(".", 1)[0]
        e = self.el
        heats = self.heats.get()
        auto = e.auto and (a, b, R, P, name, heats) == (e.a, e.b, e.R, e.P_rated, e.name, e.heats)
        self.result = EL.Element(name, a, b, R, P, True, auto, heats)
        self.destroy()


class App(tk.Tk):
    def __init__(self, board_path=None, use_kicad=False):
        super().__init__()
        self.title("PCB current-density simulation")
        self.geometry("1120x860")
        self.board, self.rows, self.result, self.kicad = None, [], None, None
        self.elements, self.saved_elements, self.el_warnings = [], [], []
        self.settings = S.Settings()
        self.q = queue.Queue()
        self._build()
        if use_kicad:
            self._connect_kicad()
        if board_path is None and self.kicad is not None:
            board_path = self.kicad.board_path()
        if board_path:
            self.load_board(board_path)

    # ------------------------------------------------------------- layout
    def _build(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Board").pack(side="left")
        self.path = tk.StringVar()
        ttk.Entry(top, textvariable=self.path).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(top, text="Browse…", command=self._browse).pack(side="left")
        ttk.Button(top, text="Reload", command=lambda: self.load_board(self.path.get())).pack(side="left", padx=4)
        self.kbar = ttk.Frame(self, padding=(8, 4, 8, 0))
        self.kbar.pack(fill="x")
        self.info = ttk.Label(self, padding=(8, 2), foreground="#555",
                              text="Save the board (and refill zones) in KiCad before running: "
                                   "the simulator reads the saved .kicad_pcb file.")
        self.info.pack(fill="x")

        body = ttk.PanedWindow(self, orient="vertical")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # excitation table
        tf = ttk.Labelframe(body, text="Excitations (all rows are solved together)", padding=6)
        body.add(tf, weight=3)
        cols = ("name", "kind", "src", "snk", "I", "net")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=8, selectmode="extended")
        for c, t, w in zip(cols, ("Name", "Kind", "Source / + pads", "Sink / − pads", "Value", "Net"),
                           (110, 110, 200, 200, 80, 200)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._edit())
        bf = ttk.Frame(tf)
        bf.pack(side="left", fill="y", padx=(6, 0))
        for text, cmd in (("Add…", self._add), ("Edit…", self._edit), ("Duplicate", self._dup),
                          ("Delete", self._delete), (None, None), ("Save table", self._save),
                          ("Load table…", self._load)):
            if text is None:
                ttk.Separator(bf).pack(fill="x", pady=6)
            else:
                ttk.Button(bf, text=text, command=cmd, width=16).pack(fill="x", pady=1)
        self.sel_btn = ttk.Button(bf, text="Add from\nKiCad selection", command=self._from_selection,
                                  width=16, state="disabled")
        self.sel_btn.pack(fill="x", pady=(8, 1))

        # resistance elements
        ef = ttk.Labelframe(body, text="Resistance elements (resistors found on the board, plus any two pads "
                                       "you add)", padding=6)
        body.add(ef, weight=2)
        cols = ("on", "used", "name", "a", "b", "R", "P", "nets", "heats", "src")
        self.etree = ttk.Treeview(ef, columns=cols, show="headings", height=5, selectmode="extended")
        for c, t, w in zip(cols, ("On", "In circuit", "Name", "Pad A", "Pad B", "R", "Rating [W]", "Nets",
                                   "Heats board", "Source"), (40, 75, 80, 90, 90, 90, 80, 220, 85, 60)):
            self.etree.heading(c, text=t)
            self.etree.column(c, width=w, anchor="w")
        self.etree.pack(side="left", fill="both", expand=True)
        self.etree.bind("<Double-1>", lambda e: self._el_edit())
        ebf = ttk.Frame(ef)
        ebf.pack(side="left", fill="y", padx=(6, 0))
        for text, cmd in (("Add…", self._el_add), ("Edit…", self._el_edit), ("On / off", self._el_toggle),
                          ("Delete", self._el_delete)):
            ttk.Button(ebf, text=text, command=cmd, width=16).pack(fill="x", pady=1)
        self.esel_btn = ttk.Button(ebf, text="Add from\nKiCad selection", command=self._el_from_selection,
                                   width=16, state="disabled")
        self.esel_btn.pack(fill="x", pady=(8, 1))

        # settings
        sf = ttk.Labelframe(body, text="Settings", padding=6)
        body.add(sf, weight=2)
        self.vars = {}
        spec = [("pitch", "Grid pitch [mm] or auto", "auto"),
                ("t_plate", "Barrel plating [µm]", "25"),
                ("T_cold", "Copper temp. without thermal [°C]", "20"),
                ("T_amb", "Ambient [°C]", "25"),
                ("h_conv", "Heat transfer h per face, fixed [W/m²K]", "12"),
                ("emissivity", "Emissivity (auto h)", "0.9"),
                ("k_fr4", "Dielectric k [W/mK]", "0.3"),
                ("rho20", "Cu resistivity @20 °C [Ω·m]", "1.72e-8"),
                ("b_height", "B-map distance from board [mm]", "1"),
                ("pulse_scale", "Pulse amplitude [× table]", "1"),
                ("pulse_ms", "Pulse length [ms]", "10")]
        for k, (key, label, default) in enumerate(spec):
            r, c = divmod(k, 2)
            ttk.Label(sf, text=label).grid(row=r, column=2 * c, sticky="w", padx=(0, 4))
            v = tk.StringVar(value=default)
            ttk.Entry(sf, textvariable=v, width=10).grid(row=r, column=2 * c + 1, sticky="w", padx=(0, 16))
            self.vars[key] = v
        r = (len(spec) + 1) // 2
        self.checks = {}
        for k, (key, label) in enumerate(CHECKS):
            v = self.checks[key] = tk.BooleanVar(value=getattr(S.Settings(), key))
            ttk.Checkbutton(sf, text=label, variable=v,
                            command=self._refresh_elements if key == "auto_resistors" else None
                            ).grid(row=r + k // 2, column=2 * (k % 2), columnspan=2, sticky="w")
        self.autor = self.checks["auto_resistors"]
        lf = ttk.Frame(sf)
        lf.grid(row=0, column=4, rowspan=r + (len(CHECKS) + 1) // 2, sticky="nsew", padx=(8, 0))
        sf.columnconfigure(4, weight=1)
        ttk.Label(lf, text="Copper thickness (double-click to override)").pack(anchor="w")
        self.ltree = ttk.Treeview(lf, columns=("layer", "stack", "use"), show="headings", height=5)
        for c, t in zip(("layer", "stack", "use"), ("Layer", "From board [µm]", "Used [µm]")):
            self.ltree.heading(c, text=t)
            self.ltree.column(c, width=90, stretch=True)
        self.ltree.pack(fill="both", expand=True)
        self.ltree.bind("<Double-1>", self._edit_thickness)

        # run
        rf = ttk.Frame(self, padding=(8, 0, 8, 8))
        rf.pack(fill="x")
        self.run_btn = ttk.Button(rf, text="Run simulation", command=self._run)
        self.run_btn.pack(side="left")
        ttk.Button(rf, text="Show last result", command=self._show).pack(side="left", padx=4)
        ttk.Button(rf, text="Open saved result…", command=self._open_result).pack(side="left")
        self.status = ttk.Label(rf, text="", foreground="#333")
        self.status.pack(side="left", padx=12, fill="x", expand=True)

    # ------------------------------------------------------------- KiCad
    def _connect_kicad(self):
        try:
            from ..kicad_link import KiCadLink
            self.kicad = KiCadLink()
        except Exception as e:      # noqa: BLE001 - any failure means standalone mode
            self.kicad = None
            self.status.config(text=f"Not connected to KiCad ({e}); running standalone.")
            return
        self.sel_btn.config(state="normal")
        self.esel_btn.config(state="normal")
        ttk.Label(self.kbar, text="KiCad:").pack(side="left")
        ttk.Button(self.kbar, text="Save board", command=self._k_save).pack(side="left", padx=4)
        ttk.Button(self.kbar, text="Refill zones + save", command=self._k_refill).pack(side="left")

    def _k_save(self):
        try:
            self.kicad.save()
            self.load_board(self.path.get(), load_table=False)
        except Exception as e:      # noqa: BLE001
            messagebox.showerror("KiCad", f"Save failed: {e}")

    def _k_refill(self):
        try:
            self.kicad.refill_zones()
            self.kicad.save()
            self.load_board(self.path.get(), load_table=False)
        except Exception as e:      # noqa: BLE001
            messagebox.showerror("KiCad", f"Refill failed: {e}")

    def _from_selection(self):
        if not self.board:
            return
        try:
            ids = self.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return messagebox.showerror("KiCad", f"Could not read the selection: {e}")
        if len(ids) < 2:
            return messagebox.showinfo("KiCad selection",
                                       "Select at least two pads in the PCB editor (source first, then sink). "
                                       "Tip: use the selection filter to pick pads only.")
        self._add(S.Row(f"row{len(self.rows) + 1}", [ids[0]], ids[1:], 1.0))

    def _el_from_selection(self):
        if not self.board:
            return
        try:
            ids = self.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return messagebox.showerror("KiCad", f"Could not read the selection: {e}")
        if len(ids) != 2:
            return messagebox.showinfo("KiCad selection",
                                       "Select exactly two pads in the PCB editor (the element's two terminals).")
        self._el_add(EL.Element(ids[0].rsplit(".", 1)[0], ids[0], ids[1], 0.0))

    # ------------------------------------------------------------- board
    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("KiCad PCB", "*.kicad_pcb"), ("All", "*")])
        if p:
            self.load_board(p)

    def load_board(self, path, load_table=True):
        try:
            self.board = B.load(path)
        except Exception as e:      # noqa: BLE001
            return messagebox.showerror("Board", f"Could not read {path}:\n{e}")
        self.path.set(path)
        b = self.board
        self.info.config(text=f"{len(b.copper_layers)} copper layers, {len(b.nets)} nets, {len(b.pads)} pads, "
                              f"{len(b.tracks)} tracks, {len(b.vias)} vias, {len(b.zones)} zone fills"
                              + ("" if b.stackup_found else "  -  no stackup in file: 35 µm default")
                              + ("  -  " + b.warnings[0] if b.warnings else ""))
        sp = scenario_path(path)
        if load_table and os.path.exists(sp):
            try:
                self.rows, self.settings, self.saved_elements = load_scenario(sp)
                self._settings_to_ui()
            except Exception as e:  # noqa: BLE001
                self.status.config(text=f"Could not load saved table: {e}")
        else:
            self.saved_elements = EL.to_save(self.elements)
        auto, self.el_warnings = EL.detect(b)
        self.elements = EL.merge(auto, self.saved_elements)
        if self.el_warnings:
            self.status.config(text=self.el_warnings[0] + (f" (+{len(self.el_warnings) - 1} more)"
                                                           if len(self.el_warnings) > 1 else ""))
        self._refresh_layers()
        self._refresh_rows()

    def _refresh_layers(self):
        self.ltree.delete(*self.ltree.get_children())
        for L in self.board.copper_layers:
            t0 = self.board.cu_thickness[L] * 1e6
            t = self.settings.thickness.get(L, self.board.cu_thickness[L]) * 1e6
            self.ltree.insert("", "end", iid=L, values=(L, f"{t0:.1f}", f"{t:.1f}"))

    def _edit_thickness(self, e):
        L = self.ltree.identify_row(e.y)
        if not L:
            return
        cur = self.settings.thickness.get(L, self.board.cu_thickness[L]) * 1e6
        v = simpledialog.askfloat("Copper thickness", f"{L} finished copper thickness [µm]\n"
                                  "(0 = use the board value)", initialvalue=round(cur, 2), parent=self)
        if v is None:
            return
        if v <= 0:
            self.settings.thickness.pop(L, None)
        else:
            self.settings.thickness[L] = v * 1e-6
        self._refresh_layers()

    # ------------------------------------------------------------- rows
    def _refresh_rows(self):
        self.tree.delete(*self.tree.get_children())
        for k, r in enumerate(self.rows):
            nets = []
            for ids in (r.src, r.snk):
                ps = self.board.pads_by_id(ids[0]) if self.board and ids else []
                nets.append(ps[0].net if ps else "?")
            self.tree.insert("", "end", iid=str(k), values=(r.name, S.KINDS[r.kind], " + ".join(r.src),
                                                            " + ".join(r.snk), f"{r.current:g} {r.unit}",
                                                            nets[0] if nets[0] == nets[1] else f"{nets[0]} → {nets[1]}"))
        self._refresh_elements()

    # ------------------------------------------------------------- elements
    def _used_elements(self):
        """Elements passed to the solver (auto ones only if auto-detection is on)."""
        return [e for e in self.elements if self.autor.get() or not e.auto]

    def _refresh_elements(self):
        self.etree.delete(*self.etree.get_children())
        if not self.board:
            return
        try:
            _, used, _ = EL.active(self.board, S.nets_of_rows(self.board, self.rows), self._used_elements())
            used = {id(e) for e in used}
        except S.SimError:
            used = set()
        nets = {p.id: p.net for p in self.board.pads}
        for k, e in enumerate(self.elements):
            self.etree.insert("", "end", iid=str(k), values=(
                "✓" if e.enabled and (self.autor.get() or not e.auto) else "–", "yes" if id(e) in used else "",
                e.name, e.a, e.b, EL.fmt_ohms(e.R), f"{e.P_rated:g}" if e.P_rated else "",
                f"{nets.get(e.a, '?')} – {nets.get(e.b, '?')}", "yes" if e.heats else "no (external)",
                "auto" if e.auto else "user"))

    def _el_selected(self):
        return [int(i) for i in self.etree.selection()]

    def _el_add(self, preset=None):
        if not self.board:
            return messagebox.showinfo("Board", "Load a board first.")
        d = ElementDialog(self, self.board, preset or EL.Element("", "", "", 0.0))
        if d.result:
            self.elements = [e for e in self.elements if e.name != d.result.name] + [d.result]
            self._refresh_elements()

    def _el_edit(self):
        sel = self._el_selected()
        if sel:
            d = ElementDialog(self, self.board, self.elements[sel[0]])
            if d.result:
                self.elements[sel[0]] = d.result
                self._refresh_elements()

    def _el_toggle(self):
        for k in self._el_selected():
            self.elements[k].enabled = not self.elements[k].enabled
        self._refresh_elements()

    def _el_delete(self):
        sel = self._el_selected()
        if any(self.elements[k].auto for k in sel):
            messagebox.showinfo("Elements", "Resistors found on the board cannot be deleted; they are switched "
                                            "off instead.")
        for k in sorted(sel, reverse=True):
            if self.elements[k].auto:
                self.elements[k].enabled = False
            else:
                del self.elements[k]
        self._refresh_elements()

    def _selected(self):
        return [int(i) for i in self.tree.selection()]

    def _add(self, preset=None):
        if not self.board:
            return messagebox.showinfo("Board", "Load a board first.")
        d = RowDialog(self, self.board, preset)
        if d.result:
            self.rows.append(d.result)
            self._refresh_rows()

    def _edit(self):
        sel = self._selected()
        if sel:
            d = RowDialog(self, self.board, self.rows[sel[0]])
            if d.result:
                self.rows[sel[0]] = d.result
                self._refresh_rows()

    def _dup(self):
        for k in self._selected():
            r = self.rows[k]
            self.rows.append(S.Row(r.name + " copy", list(r.src), list(r.snk), r.current, r.kind))
        self._refresh_rows()

    def _delete(self):
        for k in sorted(self._selected(), reverse=True):
            del self.rows[k]
        self._refresh_rows()

    def _save(self):
        if self.board and self._ui_to_settings():
            save_scenario(scenario_path(self.path.get()), self.rows, self.settings, EL.to_save(self.elements))
            self.status.config(text=f"Saved {os.path.basename(scenario_path(self.path.get()))}")

    def _load(self):
        p = filedialog.askopenfilename(filetypes=[("Scenario", "*.json")])
        if p:
            self.rows, self.settings, self.saved_elements = load_scenario(p)
            self.elements = EL.merge(EL.detect(self.board)[0] if self.board else [], self.saved_elements)
            self._settings_to_ui()
            self._refresh_layers()
            self._refresh_rows()

    # ------------------------------------------------------------- settings
    def _settings_to_ui(self):
        s = self.settings
        self.vars["pitch"].set(str(s.pitch))
        self.vars["t_plate"].set(f"{s.t_plate * 1e6:g}")
        for k in FLOAT_SETTINGS:
            self.vars[k].set(f"{getattr(s, k):g}")
        for k, v in self.checks.items():
            v.set(bool(getattr(s, k)))
        self._refresh_elements()

    def _ui_to_settings(self):
        s = self.settings
        try:
            p = self.vars["pitch"].get().strip().lower()
            s.pitch = "auto" if p in ("", "auto") else float(p)
            s.t_plate = float(self.vars["t_plate"].get()) * 1e-6
            for k in FLOAT_SETTINGS:
                setattr(s, k, float(self.vars[k].get()))
        except ValueError as e:
            messagebox.showerror("Settings", f"Invalid number: {e}")
            return False
        for k, v in self.checks.items():
            setattr(s, k, v.get())
        return True

    # ------------------------------------------------------------- run
    def _run(self):
        if not self.board or not self.rows:
            return messagebox.showinfo("Run", "Load a board and add at least one excitation.")
        if not self._ui_to_settings():
            return
        self.load_board(self.path.get(), load_table=False)     # pick up the latest saved file
        save_scenario(scenario_path(self.path.get()), self.rows, self.settings, EL.to_save(self.elements))
        self.run_btn.config(state="disabled")
        board, rows, s, els = self.board, list(self.rows), self.settings, self._used_elements()

        def work():
            try:
                res = S.simulate(board, rows, s, progress=lambda m: self.q.put(("msg", m)), elements=els)
                res.warnings += self.el_warnings
                self.q.put(("done", res))
            except S.SimError as e:
                self.q.put(("err", str(e)))
            except Exception:       # noqa: BLE001
                self.q.put(("err", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "msg":
                    self.status.config(text=val.strip())
                elif kind == "done":
                    self.result = val
                    self.run_btn.config(state="normal")
                    self.status.config(text=f"Done in {val.summary['seconds']:.1f} s")
                    self._show()
                    return
                else:
                    self.run_btn.config(state="normal")
                    self.status.config(text="Failed")
                    messagebox.showerror("Simulation failed", val)
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _show(self):
        if self.result is None:
            return
        from .viewer import Viewer
        Viewer(self, self.result)

    def _open_result(self):
        p = filedialog.askopenfilename(filetypes=[("Result", "*.npz")])
        if p:
            from ..results import Result
            from .viewer import Viewer
            self.result = Result.load(p)
            Viewer(self, self.result)


def main(board_path=None, use_kicad=False):
    App(board_path, use_kicad).mainloop()
