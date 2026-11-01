"""Layout helpers shared by the tool windows."""
import wx


def scrolled(parent):
    """A panel for inputs and options that scrolls both ways when its window is too small for it."""
    w = wx.ScrolledWindow(parent)
    w.SetScrollRate(20, 20)
    return w


def refit(win):
    """Re-lay out a scrolled panel after its texts or lists changed: a label set after the first
    layout otherwise keeps its old (often empty) size and runs into the controls below it."""
    if win:
        win.Layout()
        if isinstance(win, wx.ScrolledWindow):
            win.FitInside()


MAX_COL = 640           # px: a column wider than this (a long path or message) stops growing


def fit_columns(lc, stretch=None):
    """Columns of a report list as wide as their header or widest entry (capped at MAX_COL);
    column `stretch` also takes any width left over."""
    n = lc.GetColumnCount()
    lc.Freeze()
    try:
        for c in range(n):
            lc.SetColumnWidth(c, wx.LIST_AUTOSIZE)
            w = lc.GetColumnWidth(c) if lc.GetItemCount() else 0
            lc.SetColumnWidth(c, wx.LIST_AUTOSIZE_USEHEADER)
            lc.SetColumnWidth(c, min(max(w, lc.GetColumnWidth(c)) + 8, MAX_COL))
        if stretch is not None and n:
            rest = lc.GetClientSize().width - sum(lc.GetColumnWidth(c) for c in range(n) if c != stretch)
            lc.SetColumnWidth(stretch, max(rest, lc.GetColumnWidth(stretch)))
    finally:
        lc.Thaw()


class AutoList(wx.ListCtrl):
    """A report list whose columns follow their contents: any change of its rows (Append, SetItem,
    Delete...) refits the columns once the current batch of changes is done.  stretch: a column
    that also fills the spare width."""

    def __init__(self, *a, stretch=None, **k):
        super().__init__(*a, **k)
        self.stretch, self._pending = stretch, False
        self.Bind(wx.EVT_SIZE, lambda e: (e.Skip(), self._refit()) if self.stretch is not None else e.Skip())

    def _refit(self):
        if not self._pending:
            self._pending = True
            wx.CallAfter(self._do_fit)

    def _do_fit(self):
        self._pending = False
        if self:
            fit_columns(self, self.stretch)

    def InsertColumn(self, *a, **k):
        r = super().InsertColumn(*a, **k)
        self._refit()
        return r

    def Append(self, *a, **k):
        r = super().Append(*a, **k)
        self._refit()
        return r

    def InsertItem(self, *a, **k):
        r = super().InsertItem(*a, **k)
        self._refit()
        return r

    def SetItem(self, *a, **k):
        r = super().SetItem(*a, **k)
        self._refit()
        return r

    def SetStringItem(self, *a, **k):
        r = super().SetStringItem(*a, **k)
        self._refit()
        return r

    def DeleteItem(self, *a, **k):
        r = super().DeleteItem(*a, **k)
        self._refit()
        return r

    def DeleteAllItems(self, *a, **k):
        r = super().DeleteAllItems(*a, **k)
        self._refit()
        return r
