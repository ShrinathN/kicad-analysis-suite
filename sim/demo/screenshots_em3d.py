"""Screenshots of the 3D extraction tool for docs/screenshots/ (em3d_*.png).

1. sim/demo/coil_demo.kicad_pcb: a 4-turn coil over a ground pour (GND ticked as a passive
   conductor): Summary, R/L/Q, current maps (the pour's eddy currents on B.Cu), a track's
   cross-section, SPICE.
2. The same coil with a ferrite sheet over it: the core dialog, R/L/Q and the current map.
3. sim/demo/planar_demo.kicad_pcb: a 3-turn spiral with slots in footprint T1: an E18 planar core
   snapped to the slots, 0.2 mm centre gap: the core dialog and the summary.
4. The material library and a grade fitted to datasheet points.

Works on temporary copies.  Grabs from the screen: don't use the PC while it runs.
    flatpak run --command=sh org.kicad.KiCad -c "$VENV sim/demo/screenshots_em3d.py"
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import wx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from pcb_current_sim.em3d import cores as CR  # noqa: E402
from pcb_current_sim.em3d import materials as MT  # noqa: E402
from pcb_current_sim.em3d import pcb  # noqa: E402
from pcb_current_sim.suite.session import Session  # noqa: E402
from pcb_current_sim.wxgui import suite as SU  # noqa: E402
from pcb_current_sim.wxgui.cores import CoreDialog, GradeDialog, MaterialsDialog  # noqa: E402

OUT = os.path.join(REPO, "docs", "screenshots")
TMP = tempfile.mkdtemp(prefix="em3dshots_", dir=os.path.expanduser("~"))
for f in ("coil_demo.kicad_pcb", "planar_demo.kicad_pcb"):
    shutil.copy(os.path.join(HERE, f), os.path.join(TMP, f))

app = wx.App()
wx.MessageBox = lambda *a, **k: print("MSGBOX:", str(a[0])[:200])
s = Session(library_path=os.path.join(TMP, "library.json"))
s.load_board(os.path.join(TMP, "coil_demo.kicad_pcb"))
w = SU.open_window(s, "extract3d")
w.SetPosition((0, 0))
w.SetSize((1600, 1000))
SHEET = CR.CoreSpec("sheet", "plate", dict(L=16.0, W=16.0, T=0.3), "Ferrite sheet (wireless power), μ' 700",
                    x=6.0, y=6.0, side="top", gap_z=0.1)
E18 = CR.CoreSpec("T1 core", "E", dict(A=18.0, B=2.4, C=5.0, D=1.2, E=14.0, F=4.0), "MnZn power ferrite, μi 2000",
                  gap_center=0.2)


def shot(win, name):
    win.Raise()
    win.Update()
    wx.SafeYield()
    dc = wx.ClientDC(win)
    ww, hh = win.GetClientSize()
    bmp = wx.Bitmap(ww, hh)
    m = wx.MemoryDC(bmp)
    m.Blit(0, 0, ww, hh, dc, 0, 0)
    m.SelectObject(wx.NullBitmap)
    if not any(bmp.ConvertToImage().GetData()):
        print("BLANK grab, not written:", name)
        return
    bmp.SaveFile(os.path.join(OUT, name), wx.BITMAP_TYPE_PNG)
    print("saved", name)


def tab(k):
    w.nb.SetSelection(k)
    wx.SafeYield()


queue = []


def add(fn, ms=900):
    queue.append((fn, ms))


def pump():
    if not queue:
        shutil.rmtree(TMP, ignore_errors=True)
        print("SCREENSHOTS OK")
        w.Close()
        return app.ExitMainLoop()
    fn, ms = queue.pop(0)
    try:
        r = fn()
    except Exception:
        import traceback
        traceback.print_exc()
        return app.ExitMainLoop()
    if r == "wait":                       # an analysis is running: come back
        queue.insert(0, (fn, ms))
        return wx.CallLater(500, pump)
    wx.CallLater(ms, pump)


def analyse(setup):
    def f():
        setup()
        w._to_ui()
        for k, n in enumerate(w._net_names):            # _to_ui keeps the list's ticks: set them from the setup
            if n in w.s.nets:
                w.nets.Check(k, True)
            elif n not in w._port_nets():
                w.nets.Check(k, False)
        w._run()
    return f


def done():
    return "wait" if w._running else None


state = {}


# 1. the coil over a ground pour
def coil_setup():
    w.s.ports = [pcb.Port("coil", ["L1.1"], ["L2.1"])]
    w.s.nets = ["GND"]
    w.s.f_max, w.s.pitch = 1e8, 0.6
    w.s.cores = []


def coil_shots():
    tab(0)
    shot(w, "em3d_summary.png")
    tab(1)
    shot(w, "em3d_rlq.png")
    tab(2)
    w.clayer.SetSelection(w._layers.index("F.Cu"))       # pick a coil track (on F.Cu) for the cross-section
    w._map()
    b = w.res.bars
    i = int(w._wires[len(w._wires) // 2])

    class E:
        pass
    e = E()
    e.inaxes, e.xdata, e.ydata = w.cfig.axes[0], b["x"][i], b["y"][i]
    w._pick(e)
    w.clayer.SetSelection(w._layers.index("B.Cu"))       # then the pour's eddy currents
    w._map()
    wx.SafeYield()
    shot(w, "em3d_current.png")
    tab(3)
    shot(w, "em3d_spice.png")


add(analyse(coil_setup), 500)
add(done, 500)
add(coil_shots, 1200)


# 2. a ferrite sheet over the coil
def sheet_dialog():
    d = CoreDialog(w, s.board, s.stackup, w.materials, SHEET)
    d.SetPosition((0, 0))
    d.Show()
    state["d"] = d


add(sheet_dialog, 1200)
add(lambda: (shot(state["d"], "em3d_core_dialog.png"), state["d"].Destroy()), 600)
add(analyse(lambda: setattr(w.s, "cores", [SHEET])), 500)
add(done, 500)


def sheet_shots():
    tab(1)
    shot(w, "em3d_core_rlq.png")
    tab(2)
    w.clayer.SetSelection(w._layers.index("F.Cu"))
    w._map()
    wx.SafeYield()
    shot(w, "em3d_core_current.png")
    tab(0)
    shot(w, "em3d_core_summary.png")


add(sheet_shots, 1200)


# 3. the planar E core on the board with slots
def planar_board():
    s.load_board(os.path.join(TMP, "planar_demo.kicad_pcb"))
    wx.SafeYield()
    w.s.ports = [pcb.Port("winding", ["J1.1"], ["J2.1"])]
    w.s.nets, w.s.cores = [], []
    w._to_ui()
    d = CoreDialog(w, s.board, s.stackup, w.materials, E18)
    d.SetPosition((0, 0))
    d.Show()
    d._snap()
    state["d"] = d
    state["e"] = d.read()


add(planar_board, 1500)
add(lambda: (shot(state["d"], "em3d_core_snap.png"), state["d"].Destroy()), 600)


def planar_setup():
    w.s.cores = [state["e"]]
    w.s.f_max, w.s.pitch, w.s.core_cell = 1e7, 0.4, 1.0


add(analyse(planar_setup), 500)
add(done, 500)


def planar_shots():
    tab(0)
    shot(w, "em3d_planar_summary.png")
    tab(1)
    shot(w, "em3d_planar_rlq.png")


add(planar_shots, 1200)


# 4. the material library and a fit
def materials():
    d = MaterialsDialog(w, w.materials)
    d.SetPosition((0, 0))
    d.Show()
    d.list.Select(0)
    d._note()
    state["d"] = d


add(materials, 1000)
add(lambda: (shot(state["d"], "em3d_materials.png"), state["d"].Destroy()), 600)


def grade():
    g = GradeDialog(w)
    g.SetPosition((0, 0))
    g.Show()
    g.name.SetValue("My MnZn grade")
    pts = MT.Grade("t", 2300.0, 2.5e6).mu(np.array([1e4, 1e5, 5e5, 1e6, 2e6, 3e6, 5e6, 1e7, 2e7]))
    f = [1e4, 1e5, 5e5, 1e6, 2e6, 3e6, 5e6, 1e7, 2e7]
    noise = [1.0, 1.02, 0.98, 1.03, 0.97, 1.02, 0.99, 1.04, 0.96]          # read off a plot by eye
    g.pts.SetValue("\n".join(f"{x:.3g} {m.real * n:.4g} {-m.imag * n:.4g}" for x, m, n in zip(f, pts, noise)))
    g._fit()
    state["d"] = g


add(grade, 1200)
add(lambda: (shot(state["d"], "em3d_grade_fit.png"), state["d"].Destroy()), 600)

wx.CallLater(1000, pump)
app.MainLoop()
