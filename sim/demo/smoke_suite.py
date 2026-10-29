"""wx smoke test of the suite: Home, DC & Thermal, stackup, library, results, external tools.

Run inside the KiCad Flatpak with the plugin venv python; works on a copy of the demo board in
smoke_out/ and writes screenshots there."""
import os
import shutil
import sys

import wx

sys.path.insert(0, "/home/shinu/Desktop/current_sim")
from pcb_current_sim.wxgui import suite as SU          # noqa: E402
from pcb_current_sim.suite.session import Session       # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "smoke_out")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
for f in ("demo.kicad_pcb", "demo.current_sim.json"):
    shutil.copy(os.path.join(D, f), OUT)


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
s.load_board(os.path.join(OUT, "demo.kicad_pcb"))
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


step(lambda: shot(home, "home.png"))
step(lambda: SU.open_window(s, "dc"))


def check_dc():
    dc = s.windows["dc"]
    print("dc rows:", len(dc.rows), "menus:", [dc.GetMenuBar().GetMenuLabelText(i)
                                              for i in range(dc.GetMenuBar().GetMenuCount())])
    shot(dc, "dc.png")
    SU.open_window(s, "stackup")


step(check_dc)


def edit_stackup():
    st = s.windows["stackup"]
    g = st.grid
    r = [k for k, (t, _) in enumerate(st.rows) if t == "cu"][0]
    g.SetCellValue(r, 4, "52")
    st._changed(wx.grid.GridEvent(id=g.GetId(), type=wx.grid.wxEVT_GRID_CELL_CHANGED, obj=g, row=r, col=4))
    dc = s.windows["dc"]
    used = dc.ltree.GetItemText(0, 2)
    print("DC layer list after stackup edit:", used, "settings.thickness:", dc.settings.thickness)
    assert used == "52.0", used
    shot(st, "stackup.png")
    lib = SU.open_window(s, "library")
    lib.list.Select(2)


step(edit_stackup)


def lib_shot():
    shot(s.windows["library"], "library.png")
    d = SU.PartDialog(s.windows["library"], s.library.parts[6], set())
    d.Show()
    wx.CallLater(600, lambda: (shot(d, "part_dialog.png"), d.Destroy()))


step(lib_shot, 1200)
step(lambda: SU.open_window(s, "results"))
step(lambda: (shot(s.windows["results"], "results.png"), SU.open_window(s, "external")), 1500)
step(lambda: shot(s.windows["external"], "external.png"))
step(lambda: (SU.open_window(s, "home"), shot(home, "home2.png")))
wx.CallLater(1200, go)
app.MainLoop()
