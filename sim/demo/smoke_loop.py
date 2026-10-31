"""wx smoke test of the Switching Loop tool: detection, one analysis, every tab.

Run inside the KiCad Flatpak with the plugin venv python:
    <venv>/bin/python3 sim/demo/smoke_loop.py [BOARD.kicad_pcb]
Without a board it uses the synthetic buck cell sim/demo/buck_demo.kicad_pcb.  Works on a copy;
writes screenshots to sim/demo/smoke_out/."""
import os
import shutil
import sys

import wx

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", ".."))
from pcb_current_sim.suite.session import Session       # noqa: E402
from pcb_current_sim.wxgui import suite as SU          # noqa: E402

OUT = os.path.join(D, "smoke_out")
os.makedirs(OUT, exist_ok=True)
src = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(D, "buck_demo.kicad_pcb")
dst = os.path.join(OUT, "loop_" + os.path.basename(src))
shutil.copy(src, dst)
sc = os.path.splitext(dst)[0] + ".current_sim.json"
if os.path.exists(sc):
    os.remove(sc)


def shot(win, name):
    win.Update()
    dc = wx.ClientDC(win)
    w, h = win.GetClientSize()
    bmp = wx.Bitmap(w, h)
    m = wx.MemoryDC(bmp)
    m.Blit(0, 0, w, h, dc, 0, 0)
    m.SelectObject(wx.NullBitmap)
    bmp.SaveFile(os.path.join(OUT, name), wx.BITMAP_TYPE_PNG)


app = wx.App()
s = Session(library_path=os.path.join(OUT, "library.json"))
s.load_board(dst)
w = SU.open_window(s, "loop")
w.SetSize((1500, 950))
print("detected:", w.setup.sw_net, [(d.role, d.ref) for d in w.setup.devices], [g.fet for g in w.setup.drives])


def run():
    w._run()
    wait()


def wait():
    if w._running:
        return wx.CallLater(300, wait)
    assert w.res is not None, "no result"
    r = w.res
    for c in r.commutations:
        print(f"  {c['title']}: L {c['L_total_nH']:.2f} nH (board {c['L_board_nH']:.2f}), ring {c['f_ring'] / 1e6:.0f} MHz")
    for g in r.gates:
        print(f"  gate {g['fet']}: L_gate {g['L_gate_nH']:.2f} nH, L_cs {g['L_cs_nH']:.3f} nH")
    grab(0)


def grab(k):
    names = ("summary", "breakdown", "impedance", "map", "spice")
    if k == len(names):
        w.Close()
        print("SMOKE OK")
        return app.ExitMainLoop()
    w.nb.SetSelection(k)
    wx.CallLater(600, lambda: (shot(w, f"loop_{names[k]}.png"), grab(k + 1)))


wx.CallLater(800, run)
app.MainLoop()
