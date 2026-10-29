"""State shared by all suite windows in one process: board, stackup, component library, KiCad link.

Windows subscribe with `listen(fn)`; fn(what) is called with "board", "stackup" or "library"
after a change (on the GUI thread, since only GUI code changes the session).
"""
import os

from .. import board as B
from .. import scenario as SC
from . import stackup as ST
from .library import Library


class Session:
    def __init__(self, library_path=None):
        self.board_path = None
        self.board = None
        self.stackup = None
        self.library = Library(library_path) if library_path else Library()
        self.kicad = None
        self.kicad_error = None
        self.last_result = None          # most recent DC result, for the results browser
        self.windows = {}                # tool key -> open window (owned by the GUI layer)
        self._listeners = []

    # ---- events ----------------------------------------------------------
    def listen(self, fn):
        self._listeners.append(fn)

    def unlisten(self, fn):
        if fn in self._listeners:
            self._listeners.remove(fn)

    def notify(self, what):
        for fn in list(self._listeners):
            fn(what)

    # ---- KiCad -----------------------------------------------------------
    def connect_kicad(self):
        try:
            from ..kicad_link import KiCadLink
            self.kicad = KiCadLink()
        except Exception as e:      # noqa: BLE001 - any failure means standalone mode
            self.kicad, self.kicad_error = None, str(e)
            return None
        try:
            return self.kicad.board_path()
        except Exception as e:      # noqa: BLE001
            self.kicad_error = f"Could not get the board path from KiCad: {e}"
            return None

    # ---- board / stackup ---------------------------------------------------
    @property
    def scenario_path(self):
        return SC.scenario_path(self.board_path) if self.board_path else None

    def load_board(self, path):
        """Parse the board and rebuild the stackup with the saved overrides. Raises on a bad file."""
        board = B.load(path)
        self.board_path, self.board = os.path.abspath(path), board
        self.stackup = ST.from_board(board, self._saved_stackup())
        self.notify("board")
        return board

    def _saved_stackup(self):
        try:
            return SC.section(self.scenario_path, "stackup")
        except (OSError, ValueError):
            return {}

    def set_stackup(self, st, save=True):
        self.stackup = st
        if save and self.scenario_path:
            SC.save_section(self.scenario_path, "stackup", st.overrides())
        self.notify("stackup")

    def set_cu_override(self, layer, t):
        """Copper thickness override in m (None or <= 0 removes it)."""
        if self.stackup is None:
            return
        if t is None or t <= 0:
            self.stackup.cu_override.pop(layer, None)
        else:
            self.stackup.cu_override[layer] = t
        self.set_stackup(self.stackup)

    def library_changed(self):
        self.library.save()
        self.notify("library")
