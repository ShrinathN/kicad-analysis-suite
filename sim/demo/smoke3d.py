"""Open a saved result in the wx viewer, switch to the 3D tab and screenshot it (run in the KiCad Flatpak)."""
import os
import sys

import wx

sys.path.insert(0, "/home/shinu/Desktop/current_sim")
from pcb_current_sim.results import Result          # noqa: E402
from pcb_current_sim.wxgui.viewer import ViewerFrame  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))


def demo_result():
    """The demo scenario (12 V source, external 2.4 ohm load, Q1 as 20 mOhm, 2x pulse), cached in /tmp."""
    cache = "/tmp/pcb_current_sim_demo_res.npz"
    if os.path.exists(cache):
        return Result.load(cache)
    from pcb_current_sim import board as B, elements as EL, solver as S
    b = B.load(os.path.join(D, "demo.kicad_pcb"))
    els = EL.merge(EL.detect(b)[0], [EL.Element("LOAD", "J2.1", "J2.2", 2.4, heats=False),
                                     EL.Element("Q1", "Q1.3", "Q1.2", 0.02, 0.35)])
    r = S.simulate(b, [S.Row("supply", ["J1.1"], ["J1.2"], 12, "V")],
                   S.Settings(pulse=True, pulse_ms=5, pulse_scale=2), progress=lambda m: None, elements=els)
    r.save(cache)
    return r


app = wx.App()
v = ViewerFrame(None, demo_result())
v.Show()


def shot(name):
    v.Update()
    dc = wx.ScreenDC()
    x, y = v.GetScreenPosition()
    w, h = v.GetSize()
    bmp = wx.Bitmap(w, h)
    m = wx.MemoryDC(bmp)
    m.Blit(0, 0, w, h, dc, x, y)
    m.SelectObject(wx.NullBitmap)
    bmp.SaveFile(os.path.join(D, name), wx.BITMAP_TYPE_PNG)


def go():
    v.views.SetSelection(1)
    wx.CallLater(6000, lambda: (shot("viewer_3d.png"), print("web:", type(v.web).__name__), app.ExitMainLoop()))


wx.CallLater(1000, go)
app.MainLoop()
