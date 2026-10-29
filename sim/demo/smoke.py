import sys, os, wx
sys.path.insert(0, "/home/shinu/Desktop/current_sim")
from pcb_current_sim.wxgui.app import MainFrame
from pcb_current_sim import elements as EL, solver as S
D = os.path.dirname(os.path.abspath(__file__))
app = wx.App()
f = MainFrame(os.path.join(D, "demo.kicad_pcb"))
f.Show()
f.rows = [S.Row("supply", ["J1.1"], ["J1.2"], 12, "V")]
f.elements += [EL.Element("Q1", "Q1.3", "Q1.2", 0.02, 0.35), EL.Element("LOAD", "J2.1", "J2.2", 2.4, heats=False)]
f.settings.pulse, f.settings.pulse_scale, f.settings.pulse_ms, f.settings.pulse_from_steady = True, 4, 20, True
f._settings_to_ui()
f._refresh_rows()
print("element rows:", [[f.etree.GetItemText(i, c) for c in range(9)] for i in range(f.etree.GetItemCount())])
def shot(win, name):
    win.Update(); dc = wx.ClientDC(win); w, h = win.GetClientSize()
    bmp = wx.Bitmap(w, h); m = wx.MemoryDC(bmp); m.Blit(0, 0, w, h, dc, 0, 0); m.SelectObject(wx.NullBitmap)
    bmp.SaveFile(os.path.join(D, name), wx.BITMAP_TYPE_PNG)
def go():
    shot(f, "main.png")
    d = __import__("pcb_current_sim.wxgui.app", fromlist=["x"]).ElementDialog(f, f.board, f.elements[-1])
    d.Show()
    wx.CallLater(800, lambda: (shot(d, "eldlg.png"), d.Destroy(), f._run(), wait()))
    def wait():
        if f.result is None:
            return wx.CallLater(500, wait)
        print("elements in result:", [(e["name"], round(e["I_A"], 3), round(e["P_W"], 3)) for e in f.result.elements])
        nb = [c for c in f.viewers[-1].GetChildren()]
        v = f.viewers[-1]
        def b_view():
            v.qbox.SetSelection(v.qkeys.index("Bn")); v.redraw()
            wx.CallLater(1200, lambda: (shot(v, "viewer_B.png"), tp_view()))
        def tp_view():
            v.qbox.SetSelection(v.qkeys.index("Tp")); v.redraw()
            wx.CallLater(1200, lambda: (shot(v, "viewer_Tp.png"), app.ExitMainLoop()))
        print("quantities:", v.qkeys, "pages:", [v.FindWindowByName(""), None][1])
        wx.CallLater(1500, lambda: (shot(v, "viewer.png"), b_view()))
wx.CallLater(800, go)
app.MainLoop()
