"""Optional link to a running KiCad (IPC API via kicad-python / kipy).

Only used for the board path, the current pad selection, and save/refill.
The simulation itself always reads the saved .kicad_pcb file.
"""
import math
import os


class KiCadLink:
    def __init__(self):
        from kipy import KiCad          # raises ImportError when kicad-python is missing
        self.kicad = KiCad()            # uses KICAD_API_SOCKET / KICAD_API_TOKEN set by KiCad
        self.board = self.kicad.get_board()

    def board_path(self):
        b = self.board
        name = getattr(b, "name", None)
        doc = getattr(b, "document", None) or getattr(b, "_doc", None)
        if name is None and doc is not None:
            name = getattr(doc, "board_filename", None)
        if name and os.path.isabs(name):
            return name
        proj = None
        for getter in (lambda: doc.project.path, lambda: b.get_project().path):
            try:
                proj = getter()
            except Exception:   # noqa: BLE001 - API differs between kipy versions
                continue
            if proj:
                break
        if name and proj:
            return os.path.join(proj, name)
        return name

    def selected_pad_ids(self, board):
        """Map the PCB editor's selected pads to 'REF.number' ids of the parsed board, in order."""
        from kipy.board_types import Pad
        ids = []
        for item in self.board.get_selection():
            if not isinstance(item, Pad):
                continue
            x, y = item.position.x / 1e6, item.position.y / 1e6
            num = str(item.number)
            best, bd = None, 0.05
            for p in board.pads:
                d = math.dist(p.pos, (x, y))
                if p.number == num and d <= bd:
                    best, bd = p, d
            if best is not None and best.id not in ids:
                ids.append(best.id)
        return ids

    def save(self):
        self.board.save()

    def refill_zones(self):
        self.board.refill_zones()
