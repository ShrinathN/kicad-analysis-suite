"""wx smoke test of the Matching & Tolerance tool. Run inside the KiCad Flatpak with the plugin venv python;
works on a copy of the demo board in smoke_out/ and writes screenshots there."""
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
mf = SU.open_window(s, "matching")
steps = []


def step(fn, delay=700):
    steps.append((fn, delay))


def go():
    if not steps:
        print("SMOKE OK")
        return app.ExitMainLoop()
    fn, delay = steps.pop(0)
    fn()
    wx.CallLater(delay, go)


def lib_pick(i, name):
    """Select element i and pick a library part in its property grid, as a user would."""
    mf.canvas.select(i)
    mf._props()
    e = mf.setup.elements[i]
    names = ["(custom)"] + [q.name for q in s.library.by_kind(e.kind)]
    mf._apply_element(e, "lib", names.index(name))
    mf._changed(rebuild=True)


def build():
    mf.setup.elements.clear()
    mf.canvas.sel = None
    mf._add("C", False)                     # series C
    mf._add("L", True)                      # shunt L
    mf._add("TL", False)                    # microstrip from the demo board's stackup
    mf._add("stub_open", True)
    mf._add("R", False)
    mf._add("C", True)
    lib_pick(0, "generic 0402 C0G 1p0")
    lib_pick(1, "generic 0402 RF 4n7")
    lib_pick(5, "generic 0402 C0G 1p0")
    mf.canvas.select(4)
    mf._delete()                            # drop the R again
    mf._move(3, 1)                          # drag the stub next to the source
    print("elements:", [(e.name, e.kind, e.shunt, e.lib) for e in mf.setup.elements])
    # property edits as typed into the grid
    c1, tl = mf.setup.elements[0], mf.setup.elements[3]
    mf._apply_element(c1, "value", "1p2")
    mf._apply_element(c1, "tol", "B")
    assert c1.get_part().value == 1.2e-12 and c1.get_part().tol == 0.1e-12 and c1.lib == ""
    mf._apply_element(tl, "line", 0)
    mf._apply_element(tl, "z0", "35")
    mf._apply_element(tl, "line", 1)
    mf._apply_element(tl, "w_tol", "25")
    assert tl.line == "ms" and abs(tl.w_tol - 25e-6) < 1e-12
    mf._changed(rebuild=True)


step(build, 1200)
step(lambda: (shot(mf, "match_editor.png"), mf.canvas.select(2)), 900)
step(lambda: shot(mf, "match_props_tl.png"))
step(lambda: (mf.canvas.select("load"), mf._props()))
step(lambda: (shot(mf, "match_props_load.png"), mf.left.SetSelection(2), mf._run()), 400)


def wait():
    if mf._running:
        return wx.CallLater(300, wait)
    print("summary:", mf.summary.GetValue().replace("\n", " | "))
    shots = [(k, n) for k, n in enumerate(["smith", "s11", "s21", "vswr", "hist", "sens", "worst"])]

    def nxt():
        if not shots:
            return go()
        k, n = shots.pop(0)
        mf.plots_nb.SetSelection(k)
        wx.CallLater(900, lambda: (shot(mf, f"match_{n}.png"), nxt()))
    nxt()


step(lambda: wx.CallLater(500, wait), 10 ** 7)     # wait() resumes the chain itself
wx.CallLater(1500, go)
app.MainLoop()
