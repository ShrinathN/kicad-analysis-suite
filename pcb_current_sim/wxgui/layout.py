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
