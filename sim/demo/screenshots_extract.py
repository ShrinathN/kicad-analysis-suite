"""Screenshots of the Trace R/L/C -> SPICE tool for docs/screenshots/ (extract_*.png).

Run inside the KiCad Flatpak with the plugin venv python (see docs/DEVELOPMENT.md):
    flatpak run --command=sh org.kicad.KiCad -c "<venv>/bin/python3 sim/demo/screenshots_extract.py BOARD.kicad_pcb"
Uses the board's saved extraction setup (ports, return nets, elements) from BOARD.current_sim.json,
on a temporary copy, so the project is not touched. The docs' shots are of the goodtorch board
(a battery boost converter: H5.1 -> Q3.5 through the BAT_P pour, L1 and the switch-node pour).
Windows must stay uncovered while it runs (screen grabs).
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
                        os.path.expanduser("~/Documents/C/goodtorch/goodtorch.kicad_pcb"))
TMP = tempfile.mkdtemp(prefix="pcbsim_extract_")
stem = os.path.splitext(board)[0]
for f in (board, stem + ".current_sim.json"):
    shutil.copy(f, TMP)

app = wx.App()
s = Session(library_path=os.path.join(TMP, "library.json"))
s.load_board(os.path.join(TMP, os.path.basename(board)))
w = SU.open_window(s, "extract")
w.SetSize((1500, 900))
TABS = ("summary", "segments", "board", "spice", "schematic")


def run():
    w._run()
    wait()


def wait():
    if w._running:
        return wx.CallLater(300, wait)
    if w.ext is None:
        print("extraction failed")
        return app.ExitMainLoop()
    grab(0)


def grab(k):
    if k == len(TABS):
        print("SHOTS OK")
        w.Close()
        return app.ExitMainLoop()
    w.nb.SetSelection(k)
    wx.CallLater(700, lambda: (shot(w, f"extract_{TABS[k]}.png"), grab(k + 1)))


wx.CallLater(1000, run)
app.MainLoop()
shutil.rmtree(TMP, ignore_errors=True)
