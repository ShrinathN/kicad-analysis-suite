"""Optional link to a running KiCad (IPC API via kicad-python / kipy).

Used for the board path, the current pad selection, save / refill, and highlighting parts in the
PCB and schematic editors.  The simulation itself always reads the saved .kicad_pcb file.
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

    def select_items(self, uuids):
        """Select the tracks and vias with these uuids in the PCB editor; returns how many."""
        uuids = set(uuids)
        items = [it for it in list(self.board.get_tracks()) + list(self.board.get_vias())
                 if str(getattr(getattr(it, "id", None), "value", "")) in uuids]
        self.board.clear_selection()
        if items:
            self.board.add_to_selection(items)
        return len(items)

    def highlight(self, refs):
        """Select the footprints (PCB editor) and symbols (schematic editor, if open and if this
        KiCad answers schematic calls) with these references.  Returns (n_footprints, n_symbols or
        None, note): None symbols means the schematic could not be reached (note says why)."""
        refs = set(refs)
        fps = [f for f in self.board.get_footprints() if _ref(f) in refs]
        self.board.clear_selection()
        if fps:
            self.board.add_to_selection(fps)
        n_sym, note = None, ""
        try:
            from kipy.proto.common.types import DocumentType
            try:
                from kipy.schematic import Schematic
            except ImportError:     # kicad-python 0.8: the schematic module is unfinished (fails to import)
                return len(fps), None, ("kicad-python cannot reach the schematic editor yet; KiCad's own "
                                        "cross-probing (PCB to schematic) can still highlight the symbols")
            docs = self.kicad.get_open_documents(DocumentType.DOCTYPE_SCHEMATIC)
            if not docs:
                note = "the schematic editor is not open"
            else:
                sch = Schematic(self.kicad._client, docs[0])
                syms = [s for s in sch.get_symbols() if _ref(s) in refs]
                sch.clear_selection()
                if syms:
                    sch.add_to_selection(syms)
                n_sym = len(syms)
        except Exception as e:      # noqa: BLE001 - older KiCad: no schematic API
            note = f"this KiCad does not answer schematic calls ({type(e).__name__})"
        return len(fps), n_sym, note

    def save(self):
        self.board.save()

    def refill_zones(self):
        self.board.refill_zones()


def _ref(item):
    """Reference designator of a kipy footprint or schematic symbol ('' if it has none)."""
    try:
        return str(item.reference_field.text.value)
    except Exception:   # noqa: BLE001
        return ""
