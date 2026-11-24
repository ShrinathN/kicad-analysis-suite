"""Magnetic cores for the 3D extraction (wx): the core dialog with a top-view preview, the
material library and the Debye fit from datasheet points."""
import numpy as np
import wx

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .. import elements as EL  # noqa: E402
from ..em3d import cores as CR  # noqa: E402
from ..em3d import materials as MT  # noqa: E402
from ..loop import peec as P  # noqa: E402
from ..suite.library import fmt_si, parse_si  # noqa: E402
from .layout import AutoList  # noqa: E402

CUSTOM = "Custom (enter μi and f_r)"
DIM_KEYS = {"plate": ("L", "W", "T"), "E": ("A", "B", "C", "D", "E", "F"), "rod": ("F", "L"),
            "toroid": ("OD", "ID", "H")}
DIM_TIPS = {
    "L": "Length along the core's u axis (board x at rotation 0) [mm]",
    "W": "Width along v (board y at rotation 0) [mm]",
    "T": "Thickness of the plate or sheet [mm]",
    "A": "E core: overall width across the three legs [mm]",
    "B": "E core: height of one half, back plus legs [mm]",
    "C": "E core: depth of the legs and back (along v) [mm]",
    "D": "E core: window height of one half (the legs' height above the back) [mm]",
    "E": "E core: window width, between the outer legs' inner faces [mm]",
    "F": "Centre leg width, or diameter for a round leg / rod [mm]",
    "OD": "Toroid outer diameter [mm]", "ID": "Toroid inner diameter [mm]", "H": "Toroid height [mm]",
}
ROTS = (0, 90, 180, 270)


def _f(x):
    return f"{x:g}"


def draw_cores(ax, boxes, **kw):
    """Top-view outlines of core boxes (material dashed, gaps dotted) on a board-coordinate axis."""
    for b in boxes:
        x0, y0 = b.lo[0], b.lo[1]
        gap = b.mu_r <= 1
        ax.add_patch(Rectangle((x0, y0), b.hi[0] - x0, b.hi[1] - y0, fill=False,
                               ls=":" if gap else "--", lw=0.8 if gap else 1.0,
                               ec=kw.get("gap_color", "w") if gap else kw.get("color", "lime")))


# ---- materials -------------------------------------------------------------------------
class GradeDialog(wx.Dialog):
    """A user grade: name, kind, μi, f_r, notes, and a Debye fit from datasheet points."""

    def __init__(self, parent, grade=None):
        super().__init__(parent, title="Magnetic material", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        g = grade or MT.Grade("My material", 2000.0, 2e6, "MnZn ferrite")
        top = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)
        grid = wx.FlexGridSizer(0, 2, 4, 6)
        self.name = wx.TextCtrl(self, value=g.name if not g.builtin else g.name + " (copy)", size=(260, -1))
        self.name.SetToolTip("The grade's name (as its datasheet calls it); stored in materials.json next to "
                             "the component library")
        self.kind = wx.Choice(self, choices=list(MT.KINDS))
        self.kind.SetSelection(MT.KINDS.index(g.kind) if g.kind in MT.KINDS else len(MT.KINDS) - 1)
        self.kind.SetToolTip("Material family (for your reference)")
        self.mu = wx.TextCtrl(self, value=_f(g.mu_r), size=(110, -1))
        self.mu.SetToolTip("Initial (low-frequency) relative permeability μi from the datasheet")
        self.fr = wx.TextCtrl(self, value=fmt_si(g.f_r, "").strip() if g.f_r else "", size=(110, -1))
        self.fr.SetToolTip("Relaxation frequency f_r [Hz]: where μ' has fallen to about half and μ'' peaks. "
                           "Blank: lossless, constant μ (no frequency dependence)")
        self.notes = wx.TextCtrl(self, value=g.notes, size=(260, 60), style=wx.TE_MULTILINE)
        self.notes.SetToolTip("Free notes: datasheet, temperature, where the values come from")
        for lab, c in (("Name", self.name), ("Kind", self.kind), ("μi", self.mu), ("f_r [Hz]", self.fr),
                       ("Notes", self.notes)):
            grid.AddMany([(wx.StaticText(self, label=lab), 0, wx.ALIGN_CENTER_VERTICAL), (c, 0, wx.EXPAND)])
        left.Add(grid, 0, wx.ALL, 8)
        box = wx.StaticBoxSizer(wx.VERTICAL, self, "Fit from the datasheet's complex permeability")
        sb = box.GetStaticBox()
        box.Add(wx.StaticText(sb, label="One point per line: f [Hz]  μ'  [μ''], e.g. 1M 2100 150"), 0, wx.ALL, 4)
        self.pts = wx.TextCtrl(sb, size=(260, 140), style=wx.TE_MULTILINE)
        self.pts.SetToolTip("Points read off the datasheet's μ', μ'' vs f plot (SI prefixes allowed: 100k, 2.5M). "
                            "μ'' is optional; a few points around f_r fit best")
        box.Add(self.pts, 1, wx.EXPAND | wx.ALL, 4)
        b = wx.Button(sb, label="Fit μi and f_r")
        b.SetToolTip("Least-squares fit of the one-relaxation (Debye) model to the points")
        b.Bind(wx.EVT_BUTTON, lambda e: self._fit())
        box.Add(b, 0, wx.ALL, 4)
        self.fitinfo = wx.StaticText(sb, label="")
        box.Add(self.fitinfo, 0, wx.ALL, 4)
        left.Add(box, 1, wx.EXPAND | wx.ALL, 8)
        top.Add(left, 0, wx.EXPAND)
        self.fig = Figure(figsize=(4.6, 3.6), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        top.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 8)
        out = wx.BoxSizer(wx.VERTICAL)
        out.Add(top, 1, wx.EXPAND)
        out.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizerAndFit(out)
        for c in (self.mu, self.fr):
            c.Bind(wx.EVT_TEXT, lambda e: self._plot())
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        self._points = None
        self._plot()

    def _read_points(self):
        f, m1, m2 = [], [], []
        for line in self.pts.GetValue().splitlines():
            w = line.replace(",", " ").replace(";", " ").split()
            if not w:
                continue
            f.append(parse_si(w[0]))
            m1.append(float(w[1]))
            m2.append(float(w[2]) if len(w) > 2 else None)
        return np.array(f), np.array(m1), (m2 if any(v is not None for v in m2) else None)

    def _fit(self):
        try:
            f, m1, m2 = self._read_points()
            mu, fr, rms = MT.fit_debye(f, m1, m2)
        except (ValueError, IndexError) as e:
            return wx.MessageBox(f"Could not fit: {e}", "Fit", parent=self)
        self._points = (f, m1, m2)
        self.mu.ChangeValue(f"{mu:.4g}")
        self.fr.ChangeValue(fmt_si(fr, "").strip())
        self.fitinfo.SetLabel(f"μi {mu:.4g}, f_r {fmt_si(fr, 'Hz')}, rms error {100 * rms:.1f} %")
        self._plot()

    def grade(self):
        fr = self.fr.GetValue().strip()
        return MT.Grade(self.name.GetValue().strip(), float(self.mu.GetValue()), parse_si(fr) if fr else None,
                        MT.KINDS[self.kind.GetSelection()], self.notes.GetValue())

    def _plot(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        try:
            g = self.grade()
            lo = g.f_r / 300 if g.f_r else 1e3
            f = np.geomspace(lo, (g.f_r * 300) if g.f_r else 1e9, 200)
            mu = g.mu(f)
            ax.loglog(f, mu.real, label="μ'")
            if g.f_r:
                ax.loglog(f, -mu.imag, label="μ''")
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        if self._points is not None:
            f, m1, m2 = self._points
            ax.loglog(f, m1, "o", ms=4, color="C0")
            if m2 is not None:
                ax.loglog(f, [np.nan if v is None else v for v in m2], "o", ms=4, color="C1")
        ax.set_xlabel("f [Hz]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        self.canvas.draw()

    def _ok(self, e):
        try:
            g = self.grade()
        except (ValueError, TypeError):
            return wx.MessageBox("μi and f_r must be numbers.", "Material", parent=self)
        if not g.name or g.mu_r <= 1 or (g.f_r is not None and g.f_r <= 0):
            return wx.MessageBox("Give a name, μi > 1 and a positive f_r (or none).", "Material", parent=self)
        self.EndModal(wx.ID_OK)


class MaterialsDialog(wx.Dialog):
    """The material library: built-in classes and the user's grades."""

    COLS = (("Material", 280), ("Kind", 120), ("μi", 70), ("f_r", 80), ("", 60))

    def __init__(self, parent, library):
        super().__init__(parent, title="Magnetic materials", size=(940, 520),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.lib = library
        s = wx.BoxSizer(wx.VERTICAL)
        self.list = AutoList(self, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        for k, (t, w) in enumerate(self.COLS):
            self.list.InsertColumn(k, t, width=w)
        self.list.SetToolTip("Built-in grades are generic classes with typical values; add your grade from its "
                             "datasheet for a design. Double-click to edit (a built-in is copied)")
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit())
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda e: self._note())
        s.Add(self.list, 1, wx.EXPAND | wx.ALL, 6)
        self.notes = wx.StaticText(self, label="")
        s.Add(self.notes, 0, wx.EXPAND | wx.ALL, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        for text, fn, tip in (("Add…", self._add, "A new grade: μi and f_r by hand or fitted to datasheet points"),
                              ("Edit…", self._edit, "Edit the selected grade (a built-in one is saved as a copy)"),
                              ("Delete", self._del, "Delete the selected user grade")):
            b = wx.Button(self, label=text)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 4)
        s.Add(row, 0)
        s.Add(self.CreateStdDialogButtonSizer(wx.CLOSE), 0, wx.EXPAND | wx.ALL, 6)
        self.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CLOSE), id=wx.ID_CLOSE)
        self.SetSizer(s)
        self._fill()

    def _fill(self):
        self.grades = self.lib.grades
        self.list.DeleteAllItems()
        for g in self.grades:
            self.list.Append([g.name, g.kind, _f(g.mu_r), fmt_si(g.f_r, "Hz") if g.f_r else "–",
                              "built-in" if g.builtin else "yours"])

    def _sel(self):
        k = self.list.GetFirstSelected()
        return self.grades[k] if 0 <= k < len(self.grades) else None

    def _note(self):
        g = self._sel()
        self.notes.SetLabel(g.notes if g else "")
        self.notes.Wrap(max(300, self.GetClientSize().width - 20))

    def _add(self, grade=None):
        with GradeDialog(self, grade) as d:
            if d.ShowModal() == wx.ID_OK:
                g = d.grade()
                if any(x.name.lower() == g.name.lower() and not x.builtin for x in self.lib.grades) and \
                        (grade is None or grade.builtin or grade.name.lower() != g.name.lower()):
                    if wx.MessageBox(f"Replace your grade '{g.name}'?", "Material", wx.YES_NO, self) != wx.YES:
                        return
                self.lib.save(g)
                if grade is not None and not grade.builtin and grade.name.lower() != g.name.lower():
                    self.lib.remove(grade.name)
                self._fill()

    def _edit(self):
        g = self._sel()
        if g is not None:
            self._add(g)

    def _del(self):
        g = self._sel()
        if g is None:
            return
        if g.builtin:
            return wx.MessageBox("Built-in grades cannot be deleted.", "Material", parent=self)
        self.lib.remove(g.name)
        self._fill()


# ---- the core ----------------------------------------------------------------------------
class CoreDialog(wx.Dialog):
    def __init__(self, parent, board, stackup, library, spec=None):
        super().__init__(parent, title="Magnetic core", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.board, self.stackup, self.lib = board, stackup, library
        self.spec = CR.CoreSpec.from_dict(spec.to_dict()) if spec else CR.CoreSpec()
        sp = self.spec
        top = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)

        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.name = wx.TextCtrl(self, value=sp.name, size=(200, -1))
        self.name.SetToolTip("The core's name (in the lists and warnings)")
        self.shape = wx.Choice(self, choices=[CR.SHAPE_NAMES[k] for k in CR.SHAPES])
        self.shape.SetSelection(CR.SHAPES.index(sp.shape))
        self.shape.SetToolTip("Plate / sheet: on one side of the board. E core: legs through slots in the board, "
                              "window centred on it (E-E, E-I, planar, round centre leg). Rod and toroid lie on a "
                              "side. Box: any block in board coordinates (several make any shape)")
        g.AddMany([(wx.StaticText(self, label="Name"), 0, wx.ALIGN_CENTER_VERTICAL), (self.name, 0),
                   (wx.StaticText(self, label="Shape"), 0, wx.ALIGN_CENTER_VERTICAL), (self.shape, 0)])
        left.Add(g, 0, wx.ALL, 6)

        db = wx.StaticBoxSizer(wx.VERTICAL, self, "Dimensions [mm]")
        dbox = db.GetStaticBox()
        self.dgrid = wx.FlexGridSizer(0, 4, 4, 6)
        self.dlabels, self.dfields = [], []
        for _ in range(6):
            lab = wx.StaticText(dbox, label="", size=(40, -1))
            t = wx.TextCtrl(dbox, size=(70, -1))
            t.SetToolTip("A dimension of the shape [mm] (see the letter on its left)")
            self.dlabels.append(lab)
            self.dfields.append(t)
            self.dgrid.AddMany([(lab, 0, wx.ALIGN_CENTER_VERTICAL), (t, 0)])
            t.Bind(wx.EVT_TEXT, lambda e: self._preview())
        db.Add(self.dgrid, 0, wx.ALL, 4)
        self.boxrow = wx.FlexGridSizer(0, 2, 4, 6)
        self.blo = wx.TextCtrl(dbox, size=(200, -1))
        self.blo.SetToolTip("Box corner x, y, z [mm]: board x, y (as KiCad shows them) and z above the top "
                            "copper's surface (negative: into or below the board)")
        self.bhi = wx.TextCtrl(dbox, size=(200, -1))
        self.bhi.SetToolTip("The opposite corner x, y, z [mm]")
        self.boxrow.AddMany([(wx.StaticText(dbox, label="Corner"), 0, wx.ALIGN_CENTER_VERTICAL), (self.blo, 0),
                             (wx.StaticText(dbox, label="Opposite"), 0, wx.ALIGN_CENTER_VERTICAL), (self.bhi, 0)])
        for t in (self.blo, self.bhi):
            t.Bind(wx.EVT_TEXT, lambda e: self._preview())
        db.Add(self.boxrow, 0, wx.ALL, 4)
        erow = wx.WrapSizer()
        self.pair = wx.Choice(dbox, choices=["E-E pair", "E + I plate"])
        self.pair.SetSelection(0 if sp.pair else 1)
        self.pair.SetToolTip("Two E halves (window 2 x D) or an E with an I plate (window D, the I as thick as the "
                             "E's back)")
        self.round = wx.CheckBox(dbox, label="Round centre leg")
        self.round.SetValue(sp.round)
        self.round.SetToolTip("A round centre leg of diameter F (ER; EQ / PQ / RM approximately: their outer legs "
                              "are shaped, here straight), as a staircase of strips with the true area")
        erow.Add(self.pair, 0, wx.ALL, 3)
        erow.Add(self.round, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 3)
        db.Add(erow, 0)
        self.grow = wx.FlexGridSizer(0, 4, 4, 6)
        self.gc = wx.TextCtrl(dbox, value=_f(sp.gap_center), size=(60, -1))
        self.gc.SetToolTip("Air gap in the centre leg at the mating plane [mm] (a ground centre leg); 0: none")
        self.go = wx.TextCtrl(dbox, value=_f(sp.gap_outer), size=(60, -1))
        self.go.SetToolTip("Air gap in the outer legs [mm]; equal to the centre gap: a spacer between the halves")
        self.grow.AddMany([(wx.StaticText(dbox, label="Gap centre"), 0, wx.ALIGN_CENTER_VERTICAL), (self.gc, 0),
                           (wx.StaticText(dbox, label="outer"), 0, wx.ALIGN_CENTER_VERTICAL), (self.go, 0)])
        db.Add(self.grow, 0, wx.ALL, 4)
        for c in (self.pair, self.round):
            c.Bind(wx.EVT_CHOICE if isinstance(c, wx.Choice) else wx.EVT_CHECKBOX, lambda e: self._preview())
        for t in (self.gc, self.go):
            t.Bind(wx.EVT_TEXT, lambda e: self._preview())
        left.Add(db, 0, wx.EXPAND | wx.ALL, 6)

        mb = wx.StaticBoxSizer(wx.VERTICAL, self, "Material")
        mbox = mb.GetStaticBox()
        self.mat = wx.Choice(mbox, choices=[])
        self.mat.SetToolTip("A grade from the material library (Materials… to add yours), or custom values")
        mb.Add(self.mat, 0, wx.EXPAND | wx.ALL, 4)
        mrow = wx.FlexGridSizer(0, 4, 4, 6)
        self.mu = wx.TextCtrl(mbox, value=_f(sp.mu_r), size=(70, -1))
        self.mu.SetToolTip("Custom material: initial relative permeability μi")
        self.fr = wx.TextCtrl(mbox, value=fmt_si(sp.f_r, "").strip() if sp.f_r else "", size=(70, -1))
        self.fr.SetToolTip("Custom material: relaxation frequency f_r [Hz] (μ' halves, μ'' peaks); blank: lossless")
        mrow.AddMany([(wx.StaticText(mbox, label="μi"), 0, wx.ALIGN_CENTER_VERTICAL), (self.mu, 0),
                      (wx.StaticText(mbox, label="f_r [Hz]"), 0, wx.ALIGN_CENTER_VERTICAL), (self.fr, 0)])
        mb.Add(mrow, 0, wx.ALL, 4)
        b = wx.Button(mbox, label="Materials…", style=wx.BU_EXACTFIT)
        b.SetToolTip("Open the material library: add your grade, fit it to datasheet points")
        b.Bind(wx.EVT_BUTTON, lambda e: self._materials())
        mb.Add(b, 0, wx.ALL, 4)
        self.mat.Bind(wx.EVT_CHOICE, lambda e: self._mat_changed())
        left.Add(mb, 0, wx.EXPAND | wx.ALL, 6)

        pb = wx.StaticBoxSizer(wx.VERTICAL, self, "Placement")
        pbox = pb.GetStaticBox()
        pg = wx.FlexGridSizer(0, 4, 4, 6)
        self.ref = wx.Choice(pbox, choices=["(board coordinates)"] + sorted(
            {f.ref for f in getattr(board, "footprints", [])}, key=EL.natural_key))
        self.ref.SetToolTip("Place the core relative to a footprint (x, y become the offset from its anchor), or "
                            "at board coordinates")
        k = self.ref.FindString(sp.ref) if sp.ref else 0
        self.ref.SetSelection(max(k, 0))
        self.x = wx.TextCtrl(pbox, value=f"{sp.x:.4g}", size=(90, -1))
        self.x.SetToolTip("Centre x [mm] (board coordinates, or the offset from the footprint)")
        self.y = wx.TextCtrl(pbox, value=f"{sp.y:.4g}", size=(90, -1))
        self.y.SetToolTip("Centre y [mm] (board coordinates as KiCad shows them, y down; or the offset)")
        self.rot = wx.Choice(pbox, choices=[f"{r}°" for r in ROTS])
        self.rot.SetSelection(ROTS.index(int(sp.rot) % 360) if int(sp.rot) % 360 in ROTS else 0)
        self.rot.SetToolTip("Rotation in quarter turns: at 0° the core's u axis (L, or across the E's legs) is "
                            "along board x")
        self.side = wx.Choice(pbox, choices=["Top", "Bottom", "Through the board (centred)"])
        self.side.SetSelection(CR.SIDES.index(sp.side) if sp.side in CR.SIDES else 0)
        self.side.SetToolTip("Plates, rods, toroids: on the top or bottom side, or centred on the board (through a "
                             "cut-out). E cores: always through the board; top / bottom puts an E-I's E on that side")
        self.gz = wx.TextCtrl(pbox, value=_f(sp.gap_z), size=(70, -1))
        self.gz.SetToolTip("Clearance between the board's surface and a core on one side [mm] (solder mask, "
                           "adhesive, a spacer)")
        pg.AddMany([(wx.StaticText(pbox, label="Footprint"), 0, wx.ALIGN_CENTER_VERTICAL), (self.ref, 0),
                    (wx.StaticText(pbox, label="Rotation"), 0, wx.ALIGN_CENTER_VERTICAL), (self.rot, 0),
                    (wx.StaticText(pbox, label="x"), 0, wx.ALIGN_CENTER_VERTICAL), (self.x, 0),
                    (wx.StaticText(pbox, label="y"), 0, wx.ALIGN_CENTER_VERTICAL), (self.y, 0),
                    (wx.StaticText(pbox, label="Side"), 0, wx.ALIGN_CENTER_VERTICAL), (self.side, 0),
                    (wx.StaticText(pbox, label="Clearance"), 0, wx.ALIGN_CENTER_VERTICAL), (self.gz, 0)])
        pb.Add(pg, 0, wx.ALL, 4)
        self.snap_b = wx.Button(pbox, label="Snap to the board's slots", style=wx.BU_EXACTFIT)
        self.snap_b.SetToolTip("E cores: find the slots (Edge.Cuts cut-outs, also in footprints, or non-plated "
                               "holes) its legs go through and set x, y and the rotation")
        self.snap_b.Bind(wx.EVT_BUTTON, lambda e: self._snap())
        pb.Add(self.snap_b, 0, wx.ALL, 4)
        for c in (self.ref, self.rot, self.side):
            c.Bind(wx.EVT_CHOICE, lambda e: self._preview())
        for t in (self.x, self.y, self.gz, self.mu, self.fr):
            t.Bind(wx.EVT_TEXT, lambda e: self._preview())
        left.Add(pb, 0, wx.EXPAND | wx.ALL, 6)
        top.Add(left, 0, wx.EXPAND)

        right = wx.BoxSizer(wx.VERTICAL)
        self.fig = Figure(figsize=(5.2, 4.6), constrained_layout=True)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        right.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 4)
        self.msg = wx.StaticText(self, label="", size=(-1, 72))
        right.Add(self.msg, 0, wx.EXPAND | wx.ALL, 4)
        top.Add(right, 1, wx.EXPAND)
        out = wx.BoxSizer(wx.VERTICAL)
        out.Add(top, 1, wx.EXPAND)
        out.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 6)
        self.SetSizerAndFit(out)
        self.shape.Bind(wx.EVT_CHOICE, lambda e: self._shape_changed())
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        self._fill_materials()
        self._shape_ui(keep=True)
        self._preview()

    # -- fields ---------------------------------------------------------------------------
    def _fill_materials(self):
        names = self.lib.names()
        self.mat.Set(names + [CUSTOM])
        k = names.index(self.spec.material) if self.spec.material in names else len(names)
        self.mat.SetSelection(k)
        self._mat_changed()

    def _mat_changed(self):
        custom = self.mat.GetStringSelection() == CUSTOM
        self.mu.Enable(custom)
        self.fr.Enable(custom)
        if not custom:
            g = self.lib.get(self.mat.GetStringSelection())
            if g is not None:
                self.mu.ChangeValue(_f(g.mu_r))
                self.fr.ChangeValue(fmt_si(g.f_r, "").strip() if g.f_r else "")
        self._preview()

    def _materials(self):
        with MaterialsDialog(self, self.lib) as d:
            d.ShowModal()
        sel = self.mat.GetStringSelection()
        self.spec.material = "" if sel == CUSTOM else sel
        self._fill_materials()

    def _shape(self):
        return CR.SHAPES[self.shape.GetSelection()]

    def _shape_changed(self):
        self._shape_ui(keep=False)
        self._preview()

    def _shape_ui(self, keep):
        sh = self._shape()
        keys = DIM_KEYS.get(sh, ())
        dims = self.spec.dims if keep and self.spec.shape == sh else CR.DEFAULT_DIMS.get(sh, {})
        for k, (lab, t) in enumerate(zip(self.dlabels, self.dfields)):
            on = k < len(keys)
            lab.Show(on)
            t.Show(on)
            if on:
                lab.SetLabel(keys[k])
                t.ChangeValue(_f(dims.get(keys[k], CR.DEFAULT_DIMS[sh][keys[k]])))
                t.SetToolTip(DIM_TIPS[keys[k]] + (f" ({CR.DIM_TEXT[keys[k]]})" if keys[k] in CR.DIM_TEXT else ""))
        box = sh == "box"
        for c in self.boxrow.GetChildren():
            c.GetWindow().Show(box)
        if box:
            d = dims if "lo" in dims else CR.DEFAULT_DIMS["box"]
            self.blo.ChangeValue(", ".join(_f(v) for v in d["lo"]))
            self.bhi.ChangeValue(", ".join(_f(v) for v in d["hi"]))
        e = sh == "E"
        for w in (self.pair, self.round, self.gc, self.go, self.snap_b):
            w.Show(e)
        for c in self.grow.GetChildren():
            c.GetWindow().Show(e)
        for w in (self.x, self.y, self.ref, self.rot, self.side, self.gz):
            w.Enable(not box)
        self.Layout()
        self.Fit()

    def read(self):
        """The spec from the fields (ValueError on a bad number)."""
        sh = self._shape()
        s = CR.CoreSpec.from_dict(self.spec.to_dict())
        s.name = self.name.GetValue().strip() or "core"
        s.shape = sh
        if sh == "box":
            lo = [float(v) for v in self.blo.GetValue().replace(";", ",").split(",")]
            hi = [float(v) for v in self.bhi.GetValue().replace(";", ",").split(",")]
            if len(lo) != 3 or len(hi) != 3:
                raise ValueError("A box corner is x, y, z.")
            s.dims = dict(lo=lo, hi=hi)
        else:
            s.dims = {k: float(self.dfields[i].GetValue()) for i, k in enumerate(DIM_KEYS[sh])}
            if min(s.dims.values()) <= 0:
                raise ValueError("Dimensions must be positive.")
        sel = self.mat.GetStringSelection()
        s.material = "" if sel == CUSTOM else sel
        s.mu_r = float(self.mu.GetValue())
        fr = self.fr.GetValue().strip()
        s.f_r = parse_si(fr) if fr else None
        s.ref = "" if self.ref.GetSelection() <= 0 else self.ref.GetStringSelection()
        s.x, s.y = float(self.x.GetValue()), float(self.y.GetValue())
        s.rot = ROTS[self.rot.GetSelection()]
        s.side = CR.SIDES[self.side.GetSelection()]
        s.gap_z = float(self.gz.GetValue())
        s.pair = self.pair.GetSelection() == 0
        s.round = self.round.GetValue()
        s.gap_center, s.gap_outer = float(self.gc.GetValue() or 0), float(self.go.GetValue() or 0)
        return s

    def _snap(self):
        try:
            s = self.read()
            x, y, r, note = CR.snap(s, self.board)
        except (ValueError, TypeError) as e:
            return wx.MessageBox(str(e), "Snap to slots", parent=self)
        self.ref.SetSelection(0)
        self.x.ChangeValue(f"{x:.4g}")
        self.y.ChangeValue(f"{y:.4g}")
        self.rot.SetSelection(ROTS.index(r))
        self._preview(f"Snapped: {note}.")

    def _preview(self, note=""):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_aspect("equal")
        msgs = [note] if note else []
        try:
            s = self.read()
            boxes, w = CR.expand(s, self.board, self.stackup, self.lib)
            msgs += w
            if s.shape == "E":
                Ae, le = CR.magnetic_path(s)
                msgs.append(f"A_e ≈ {Ae:.3g} mm², l_e ≈ {le:.3g} mm (simple path)")
            z = [b.lo[2] for b in boxes] + [b.hi[2] for b in boxes]
            if z:
                msgs.append(f"z {min(z):.2f} … {max(z):.2f} mm (board top copper at 0, bottom at "
                            f"{CR.board_surfaces(self.stackup)[1]:.2f})")
        except (ValueError, TypeError) as e:
            boxes = []
            msgs.append(str(e))
        for loop in getattr(self.board, "outline", []) or []:
            ax.plot(*np.vstack([loop, loop[:1]]).T, color="0.4", lw=0.8)
        for t in self.board.tracks:
            line = np.array(P._polyline(t))
            ax.plot(line[:, 0], line[:, 1], color="#c87533", lw=max(0.5, t.width * 2), alpha=0.6,
                    solid_capstyle="round")
        for p in self.board.pads:
            if p.geom is not None:
                bx = p.geom.bbox
                ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], color="#c87533", alpha=0.4))
        draw_cores(ax, boxes, color="tab:green", gap_color="tab:red")
        if boxes:
            lo = np.min([b.lo for b in boxes], axis=0)
            hi = np.max([b.hi for b in boxes], axis=0)
            m = max(hi[0] - lo[0], hi[1] - lo[1]) * 0.4 + 1
            ax.set_xlim(lo[0] - m, hi[0] + m)
            ax.set_ylim(hi[1] + m, lo[1] - m)
        else:
            ax.invert_yaxis()
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Top view: core (green), gaps (red), copper", fontsize=9)
        self.canvas.draw()
        self.msg.SetLabel("\n".join(msgs))
        self.msg.Wrap(max(300, self.canvas.GetSize().width))

    def _ok(self, e):
        try:
            s = self.read()
            CR.expand(s, self.board, self.stackup, self.lib)
        except (ValueError, TypeError) as ex:
            return wx.MessageBox(str(ex), "Magnetic core", parent=self)
        self.spec = s
        self.EndModal(wx.ID_OK)


def core_text(spec, library):
    """List columns: name, shape, material, position, gaps."""
    mat = spec.material or f"μ {spec.mu_r:g}" + (f", f_r {fmt_si(spec.f_r, 'Hz')}" if spec.f_r else "")
    if spec.shape == "box":
        pos = "box " + " … ".join(", ".join(_f(v) for v in spec.dims[k]) for k in ("lo", "hi"))
    else:
        pos = (f"{spec.ref} + " if spec.ref else "") + f"({_f(spec.x)}, {_f(spec.y)})"
        pos += f", {spec.rot}°" if spec.rot else ""
        pos += "" if spec.shape == "E" else f", {spec.side}"
    gaps = ""
    if spec.shape == "E" and (spec.gap_center or spec.gap_outer):
        gaps = f"{_f(spec.gap_center)} / {_f(spec.gap_outer)} mm"
    if spec.material and library is not None and library.get(spec.material) is None:
        mat += " (missing)"
    return [spec.name, spec.label(), mat, pos, gaps, "on" if spec.enabled else "off"]

