"""Regenerate the README / docs screenshots in docs/screenshots/.

Run inside the KiCad Flatpak with the plugin venv python (see docs/DEVELOPMENT.md):
    flatpak run --command=sh org.kicad.KiCad -c "<venv>/bin/python3 sim/demo/screenshots.py"
Works on a temporary copy of the demo board and a temporary component library, so nothing in the
repository except docs/screenshots/ is touched. Windows must stay uncovered while it runs (screen grabs).
"""
import os
import shutil
import sys
import tempfile

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from pcb_current_sim import elements as EL                 # noqa: E402
from pcb_current_sim import solver as S                    # noqa: E402
from pcb_current_sim.matching import microstrip as MS      # noqa: E402
from pcb_current_sim.matching import model as M            # noqa: E402
from pcb_current_sim.suite.library import Part             # noqa: E402
from pcb_current_sim.suite.session import Session          # noqa: E402
from pcb_current_sim.wxgui import suite as SU              # noqa: E402

OUT = os.path.join(ROOT, "docs", "screenshots")
TMP = tempfile.mkdtemp(prefix="pcbsim_shots_")
os.makedirs(OUT, exist_ok=True)
for f in ("demo.kicad_pcb", "demo.current_sim.json"):
    shutil.copy(os.path.join(ROOT, "sim", "demo", f), TMP)


def shot(win, name):
    """Grab the window from the screen (includes the menu bar and embedded web views)."""
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


def match_example(setup):
    """A 2.45 GHz L-match from 50 Ω to a small antenna (20 Ω + 1.5 nH + 3 pF) fed by a 50 Ω microstrip."""
    setup.elements.clear()
    setup.f_start, setup.f_stop, setup.n_f = 1.5e9, 3.5e9, 301
    setup.load = M.Port("series RLC", r=20.0, l=1.5e-9, c=3e-12)
    setup.bands = [M.Band("ISM 2.4", 2.4e9, 2.5e9, rl_min=12.0, il_max=0.8)]
    tl = M.new_element(setup, "TL", f0=2.45e9)
    tl.line, tl.h, tl.er, tl.tand = "ms", 1.51e-3, 4.3, 0.02
    tl.w, tl.length, tl.w_tol = MS.width_for(50, tl.h, tl.er), 8e-3, 25e-6
    setup.elements.append(tl)
    setup.elements.append(M.new_element(setup, "C", True, Part("C", "0402 C0G 1.3 pF", 1.3e-12, 0.1e-12, "abs",
                                                               "0402", q=300, f_q=1e9, esl=0.4e-9)))
    setup.elements.append(M.new_element(setup, "L", False, Part("L", "0402 RF 1.5 nH", 1.5e-9, 0.1e-9, "abs",
                                                                "0402", dcr=0.06, q=25, f_q=2.4e9, srf=10e9)))


app = wx.App()
s = Session(library_path=os.path.join(TMP, "library.json"))
s.load_board(os.path.join(TMP, "demo.kicad_pcb"))
steps = []


def step(fn, delay=900):
    steps.append((fn, delay))


def go():
    if not steps:
        print("SCREENSHOTS OK ->", OUT)
        shutil.rmtree(TMP, ignore_errors=True)
        return app.ExitMainLoop()
    fn, delay = steps.pop(0)
    fn()
    wx.CallLater(delay, go)


def wait_for(cond, then):
    """Poll until cond() is true, then call then() and resume the step chain."""
    if not cond():
        return wx.CallLater(300, lambda: wait_for(cond, then))
    then()
    wx.CallLater(900, go)


home = SU.open_window(s, "home")
home.SetPosition((0, 0))
step(lambda: shot(home, "home.png"))

# ---- DC & Thermal: demo scenario (12 V supply, FET as 20 mΩ, external 2.4 Ω load, 4x 20 ms pulse)
state = {}


def dc_open():
    dc = state["dc"] = SU.open_window(s, "dc")
    dc.SetPosition((0, 0))
    dc.rows = [S.Row("supply", ["J1.1"], ["J1.2"], 12, "V")]
    dc.elements += [EL.Element("Q1", "Q1.3", "Q1.2", 0.02, 0.35), EL.Element("LOAD", "J2.1", "J2.2", 2.4, heats=False)]
    st = dc.settings
    st.pulse, st.pulse_scale, st.pulse_ms, st.pulse_from_steady = True, 4, 20, True
    dc._settings_to_ui()
    dc._refresh_rows()


step(dc_open, 1200)
step(lambda: shot(state["dc"], "dc_main.png"))


def dc_run():
    dc = state["dc"]
    dc.result = None
    dc._run()

    def viewer():
        v = state["v"] = dc.viewers[-1]
        v.SetPosition((0, 0))
        v.SetSize((1300, 880))
    steps.insert(0, (lambda: None, 10))
    wait_for(lambda: dc.result is not None and dc.viewers, viewer)


steps.append((dc_run, 10 ** 7))      # wait_for() resumes the chain


def quantity(key, name):
    def f():
        v = state["v"]
        v.views.SetSelection(0)
        v.qbox.SetSelection(v.qkeys.index(key))
        v.redraw()
        wx.CallLater(1500, lambda: shot(v, name))
    return f


step(quantity("J", "dc_viewer_current_density.png"), 2200)
step(quantity("T", "dc_viewer_temperature.png"), 2200)
step(quantity("Bn", "dc_viewer_bfield.png"), 2200)
step(lambda: state["v"].views.SetSelection(1), 7000)
step(lambda: (shot(state["v"], "dc_viewer_3d.png"), state["v"].Close(), state["dc"].Close(), home.Iconize()), 900)

# ---- shared panels
step(lambda: (SU.open_window(s, "stackup").SetPosition((0, 0))), 1500)
step(lambda: (shot(s.windows["stackup"], "stackup.png"), s.windows["stackup"].Close()))


def library():
    lib = SU.open_window(s, "library")
    lib.SetPosition((0, 0))
    lib.list.Select(8)              # the 100 nF X7R decap


step(library, 1500)
step(lambda: (shot(s.windows["library"], "component_library.png"), s.windows["library"].Close()))

# ---- Matching & Tolerance


def match_open():
    mf = state["mf"] = SU.open_window(s, "matching")
    mf.SetPosition((0, 0))
    match_example(mf.setup)
    mf.chk["sens"].SetValue(True)
    mf.canvas.sel = 2
    mf._sweep_to_ui()
    mf._changed(rebuild=True)


step(match_open, 1500)
step(lambda: shot(state["mf"], "matching_editor.png"))


def match_run():
    mf = state["mf"]
    mf.left.SetSelection(2)
    mf._run()
    wait_for(lambda: not mf._running and mf.result is not None, lambda: None)


steps.append((match_run, 10 ** 7))


def tab(k, name):
    def f():
        mf = state["mf"]
        mf.plots_nb.SetSelection(k)
        wx.CallLater(1300, lambda: shot(mf, name))
    return f


for k, name in enumerate(("matching_smith.png", "matching_s11.png", "matching_s21.png", None,
                          "matching_histograms.png", "matching_sensitivity.png", "matching_worst_case_bom.png")):
    if name:
        step(tab(k, name), 2200)
wx.CallLater(1500, go)
app.MainLoop()
