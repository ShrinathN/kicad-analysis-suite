"""Cross-probing: clicking parts in a tool's list selects them in KiCad's PCB and schematic editors."""
import wx


class CrossProbe:
    """Watch list controls; when their selection settles (multi-selection included), select the
    parts' footprints and symbols in KiCad.  refs_fn(listctrl) -> references of the selected rows.
    Status goes to the window's _status(text) if it has one, else its status bar."""

    def __init__(self, frame, session):
        self.frame, self.session = frame, session
        self._timer = None
        self._last = None

    def watch(self, lc, refs_fn):
        for ev in (wx.EVT_LIST_ITEM_SELECTED, wx.EVT_LIST_ITEM_DESELECTED):
            lc.Bind(ev, lambda e, lc=lc, fn=refs_fn: (e.Skip(), self._later(lc, fn)))

    def _later(self, lc, fn):
        if self._timer is not None and self._timer.IsRunning():
            self._timer.Stop()
        self._timer = wx.CallLater(150, self._send, lc, fn)

    def _say(self, text):
        say = getattr(self.frame, "_status", None)
        if callable(say):
            say(text)
        elif self.frame.GetStatusBar():
            self.frame.SetStatusText(text)

    def _send(self, lc, fn):
        link = self.session.kicad
        if link is None or not self.session.highlight or not lc:
            return
        refs = sorted({r for r in fn(lc) if r})
        if not refs or refs == self._last:
            return
        self._last = refs
        try:
            n_fp, n_sym, note = link.highlight(refs)
        except Exception as e:      # noqa: BLE001 - KiCad closed, board reopened, ...
            return self._say(f"Could not highlight in KiCad: {e}")
        what = f"{', '.join(refs)}: {n_fp} footprint{'s' if n_fp != 1 else ''} selected in the PCB editor"
        what += f", {n_sym} symbol{'s' if n_sym != 1 else ''} in the schematic" if n_sym is not None else \
            (f" (schematic: {note})" if note else "")
        self._say(what)


def selected_rows(lc):
    out, i = [], lc.GetFirstSelected()
    while i >= 0:
        out.append(i)
        i = lc.GetNextSelected(i)
    return out
