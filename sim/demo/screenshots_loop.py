"""Screenshots of the Switching Loop tool for docs/screenshots/ (loop_*.png).

Board: sim/demo/buck_demo.kicad_pcb (a synchronous buck cell), on a temporary copy.  Analyses with
the switching edges at 10 A, keeps the result as the baseline, puts the suggested snubber in the
cell, analyses again and grabs every tab.  Grabs from the screen: don't use the PC while it runs.

    flatpak run --command=sh org.kicad.KiCad -c "$VENV sim/demo/screenshots_loop.py"
"""
import os
import shutil
import sys
import tempfile

import wx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from pcb_current_sim.suite.session import Session  # noqa: E402
from pcb_current_sim.wxgui import suite as SU  # noqa: E402

OUT = os.path.join(REPO, "docs", "screenshots")
TMP = tempfile.mkdtemp(prefix="loopshots_", dir=os.path.expanduser("~"))
shutil.copy(os.path.join(HERE, "buck_demo.kicad_pcb"), os.path.join(TMP, "buck.kicad_pcb"))

app = wx.App()
wx.MessageBox = lambda *a, **k: print("MSGBOX:", str(a[0])[:200])
s = Session(library_path=os.path.join(TMP, "library.json"))
s.load_board(os.path.join(TMP, "buck.kicad_pcb"))
w = SU.open_window(s, "loop")
w.SetPosition((0, 0))
w.SetSize((1600, 1000))


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


def tab(name):
    for i in range(w.nb.GetPageCount()):
        if w.nb.GetPageText(i) == name:
            w.nb.SetSelection(i)
            wx.SafeYield()
            return
    raise KeyError(name)


queue = []


def later(fn, ms=1200):
    queue.append((fn, ms))


def pump():
    if not queue:
        shutil.rmtree(TMP, ignore_errors=True)
        print("SCREENSHOTS OK")
        w.Close()
        return app.ExitMainLoop()
    fn, ms = queue.pop(0)
    try:
        fn()
    except Exception:
        import traceback
        traceback.print_exc()
        return app.ExitMainLoop()
    wx.CallLater(ms, pump)


phase = [0]


def start():
    w.fields["i_switch"][0].SetValue("10")
    w.fields["i_ripple"][0].SetValue("2")
    w._opts_changed()
    w._run()
    wx.CallLater(500, wait)


def wait():
    if w._running:
        return wx.CallLater(300, wait)
    if phase[0] == 0:
        phase[0] = 1
        w._keep_baseline()
        tab("Snubber")
        w._snub_suggest()
        w._snub_apply()
        w._run()
        return wx.CallLater(500, wait)
    for name, fn in (("Summary", "loop_summary.png"), ("Switching", "loop_switching.png"),
                     ("Where the L is", "loop_breakdown.png"), ("Impedance", "loop_impedance.png"),
                     ("Map", "loop_map.png"), ("Snubber", "loop_snubber.png"), ("EMI", "loop_emi.png"),
                     ("Compare", "loop_compare.png")):
        later(lambda name=name: tab(name), 400)
        later(lambda fn=fn: shot(w, fn), 900)
    pump()


wx.CallLater(1000, start)
app.MainLoop()
