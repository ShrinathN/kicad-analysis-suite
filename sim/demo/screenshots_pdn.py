"""Screenshots of the PDN tool for docs/screenshots/ (pdn_*.png).

Run inside the KiCad Flatpak with the plugin venv python (see docs/DEVELOPMENT.md):
    flatpak run --command=sh org.kicad.KiCad -c "<venv>/bin/python3 sim/demo/screenshots_pdn.py BOARD.kicad_pcb"
Uses the board's saved PDN setup (rail, decaps, regulator fit, load steps) from BOARD.current_sim.json,
on a temporary copy, so the project is not touched. The docs' shots are of Magnus_Gateway's +3V3 rail
with the TPS61021A fit (a demonstration: that rail is fed by a buck). Windows must stay uncovered while it
runs (screen grabs).
"""
import os
import shutil
import sys
import tempfile

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from pcb_current_sim.suite.session import Session          # noqa: E402
from pcb_current_sim.wxgui import suite as SU              # noqa: E402

OUT = os.path.join(ROOT, "docs", "screenshots")


def shot(win, name):
    """Grab the window from the screen (as screenshots.py does)."""
    win.Raise()
    win.Update()
    wx.SafeYield()
    dc = wx.ScreenDC()
    x, y = win.GetScreenPosition()
    w, h = win.GetSize()
    bmp = wx.Bitmap(w, h)
    m = wx.MemoryDC(bmp)
    m.Blit(0, 0, w, h, dc, x, y)
    m.SelectObject(wx.NullBitmap)
    bmp.SaveFile(os.path.join(OUT, name), wx.BITMAP_TYPE_PNG)
    print("wrote", name)


board = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                        os.path.join(ROOT, "sim", "demo", "smoke_out", "pdn", "Magnus_Gateway.kicad_pcb"))
TMP = tempfile.mkdtemp(prefix="pcbsim_pdn_")
stem = os.path.splitext(board)[0]
for f in (board, stem + ".current_sim.json"):
    shutil.copy(f, TMP)

app = wx.App()
s = Session(library_path=os.path.join(TMP, "library.json"))
s.load_board(os.path.join(TMP, os.path.basename(board)))
w = SU.open_window(s, "pdn")
w.SetPosition((0, 0))
w.SetSize((1500, 900))
TABS = (("Impedance", "pdn_impedance.png"), ("Decaps", "pdn_decaps.png"), ("Load steps", "pdn_load_steps.png"),
        ("Map", "pdn_map.png"), ("Layout", "pdn_layout.png"), ("Summary", "pdn_summary.png"))


def page(name):
    for q in range(w.nb.GetPageCount()):
        if w.nb.GetPageText(q) == name:
            return q
    raise KeyError(name)


def run():
    w._run()
    wait()


def wait():
    if w._running:
        return wx.CallLater(300, wait)
    if w.res is None:
        print("analysis failed")
        return app.ExitMainLoop()
    grab(0)


def grab(k):
    if k == len(TABS):
        return regfit()
    name, fn = TABS[k]
    w.nb.SetSelection(page(name))
    if name == "Map":
        pk = sorted(w.res.peaks[w.res.loads[0]], key=lambda p: -p[1])
        hi = [f for f, _ in pk if f > 1e8]
        w.mapf.SetValue(f"{(hi or [pk[0][0]])[0] / 1e6:.4g}M")
        w._map()
    if name == "Load steps" and w.res.waves is not None:      # zoom on the staggered steps
        from pcb_current_sim.pdn import transient as TT
        ends = [e[0] for ld in w.setup.loads if ld.enabled for e in TT.pulse_edges(ld, w.res.waves.t[-1])]
        t1 = (max(ends) if ends else 1e-3) * 1.15 + 2e-4
        for ax in w.wfig.axes:
            ax.set_xlim(-0.02 * t1 * 1e6, t1 * 1e6)
        w.wcanvas.draw()
    if name == "Layout" and w.ltab.GetItemCount():
        w.ltab.Select(0)
    wx.CallLater(900, lambda: (shot(w, fn), grab(k + 1)))


def regfit():
    from pcb_current_sim.wxgui.regfit import RegFitFrame
    v = w.setup.vrm
    if not v.model:
        return finish()
    r = RegFitFrame(w, v.model, lambda m: None)
    r.SetPosition((0, 0))
    r.Show()

    def fitted():
        if not r.fit_b.IsEnabled():
            return wx.CallLater(300, fitted)
        tabs()

    def tabs(k=0):
        if k == r.nb.GetPageCount():
            r.Close()
            return finish()
        r.nb.SetSelection(k)
        fn = f"pdn_regfit_{r.nb.GetPageText(k).lower()}.png"
        wx.CallLater(1200, lambda: (shot(r, fn), tabs(k + 1)))
    wx.CallLater(1500, lambda: (r._fit(), wx.CallLater(500, fitted)))


def finish():
    print("SHOTS OK")
    w.Close()
    app.ExitMainLoop()


wx.CallLater(1000, run)
app.MainLoop()
shutil.rmtree(TMP, ignore_errors=True)
