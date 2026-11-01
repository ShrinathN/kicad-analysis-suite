"""Tk helpers shared by the windows."""
import tkinter.font as tkfont
from tkinter import ttk

MAX_COL = 640           # px: a column wider than this stops growing


def tree_frame(parent, columns, **kw):
    """(frame, treeview): a report Treeview with scroll bars both ways, so columns sized to their
    contents (fit_tree) can be wider than the window."""
    f = ttk.Frame(parent)
    tv = ttk.Treeview(f, columns=columns, show="headings", **kw)
    xs = ttk.Scrollbar(f, orient="horizontal", command=tv.xview)
    ys = ttk.Scrollbar(f, orient="vertical", command=tv.yview)
    tv.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
    tv.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    f.rowconfigure(0, weight=1)
    f.columnconfigure(0, weight=1)
    return f, tv


def fit_tree(tv):
    """Each column as wide as its heading or widest entry (capped at MAX_COL)."""
    body = tkfont.nametofont("TkDefaultFont")
    try:
        head = tkfont.nametofont("TkHeadingFont")
    except Exception:       # noqa: BLE001
        head = body
    rows = tv.get_children("")
    for c in tv["columns"]:
        w = head.measure(tv.heading(c, "text"))
        for k in rows:
            w = max(w, body.measure(str(tv.set(k, c))))
        tv.column(c, width=min(w + 24, MAX_COL), stretch=False)
