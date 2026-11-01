"""Shared start-up for the plugin's IPC actions: make the package importable and its deps present."""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
# the package sits next to this file in a copied install; with `install.sh --link` Python resolves the
# symlinks, so this file is the repository's kicad_plugin/ and the package is one level up
for d in (here, os.path.dirname(here)):
    if os.path.isdir(os.path.join(d, "pcb_current_sim")):
        sys.path.insert(0, d)
        break

try:
    import scipy, matplotlib, kipy  # noqa: E401, F401
except ImportError:
    # KiCad normally installs requirements.txt into this plugin's venv; do it
    # ourselves if that step did not happen.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
                           os.path.join(here, "requirements.txt")])


def run(tool):
    """Open `tool` ("home" or "dc"). KiCad's Python (e.g. the Flatpak build) ships wxPython but
    usually not Tk, so the wx suite is preferred; the Tk DC tool is the fallback."""
    try:
        import wx  # noqa: F401
        from pcb_current_sim.wxgui.suite import main
    except ImportError:
        from pcb_current_sim.gui.app import main as tk_main
        return tk_main(use_kicad=True)
    main(use_kicad=True, tool=tool)
