"""wx smoke test of the Trace R/L/C → SPICE tool on a copy of the demo board.

Run inside the KiCad Flatpak with the plugin venv python; writes screenshots to smoke_out/."""
import os
import shutil
import sys

import wx

sys.path.insert(0, "/home/shinu/Desktop/current_sim")
from pcb_current_sim.wxgui import suite as SU          # noqa: E402
from pcb_current_sim.suite.session import Session       # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "smoke_out")
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(D, "demo.kicad_pcb"), os.path.join(OUT, "extract.kicad_pcb"))


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
s.load_board(os.path.join(OUT, "extract.kicad_pcb"))
home = SU.open_window(s, "home")
steps = []


def step(fn, delay=900):
    steps.append((fn, delay))


def go():
    if not steps:
        print("SMOKE OK")
        return app.ExitMainLoop()
    fn, delay = steps.pop(0)
    fn()
    wx.CallLater(delay, go)


step(lambda: shot(home, "extract_home.png"))
step(lambda: SU.open_window(s, "extract"))


def setup():
    w = s.windows["extract"]
    w.ports = ["J2.1", "Q1.2"]
    w._to_ui()
    print("return nets:", w._checked_nets())
    w._run()


step(setup, 4000)


def check():
    w = s.windows["extract"]
    assert w.ext is not None, "no result"
    print("pairs:", w.ext.pairs)
    print("cells:", len(w.ladder.cells), "segments:", w.segs.GetItemCount())
    print(w.spice.GetValue()[:400])
    for k, name in enumerate(("summary", "segments", "board", "spice", "schematic")):
        w.nb.SetSelection(k)
        wx.SafeYield()
        wx.MilliSleep(300)
        wx.SafeYield()
        shot(w, f"extract_{name}.png")
    # elements: a cap from the load net to GND (the REF pin) and an inductor in series, then re-run
    from pcb_current_sim import elements as EL
    from pcb_current_sim.wxgui.app import ElementDialog
    d = ElementDialog(w, w.board, EL.Element("", "", "", 0.0), models=("R", "V", "C", "L"), dc=False)
    print("dialog models:", [d.model.GetString(i) for i in range(d.model.GetCount())])
    d.Destroy()
    w.elements = [EL.Element("C9", "J2.1", "J2.2", 0.0, model="C", C=10e-9),
                  EL.Element("L9", "Q1.2", "Q1.3", 0.0, model="L", L=1e-6)]
    w._to_ui()
    w._run()


step(check, 4000)


def check2():
    w = s.windows["extract"]
    print("with elements:", w.ext.pairs, [u["value"] for u in w.ext.user])
    print([c["kind"] for c in w.ladder.cells])
    for k, name in ((1, "segments_el"), (4, "schematic_el")):
        w.nb.SetSelection(k)
        wx.SafeYield()
        wx.MilliSleep(300)
        wx.SafeYield()
        shot(w, f"extract_{name}.png")
    w.Close()
    print("saved section:", __import__("json").load(open(os.path.join(OUT, "extract.current_sim.json")))["extract"])


step(check2, 600)
wx.CallLater(800, go)
app.MainLoop()
