"""A live schematic of a switching cell, drawn from its setup: its kind (topology) as the title, the
supply side (rail: the input of a buck, the output of a boost, a bridge's bus) at the top,
the return at the bottom, the upper and lower switches with the switch node between them, the
hot-loop capacitors (behind their rail links), the inductor, and each FET's gate drive as a path.
A buck is drawn input-caps | switches | inductor -> output, a boost as its mirror image.

Every part is clickable (on_action(key) with the checks module's fix keys: "edit:<key>",
"drive:<fet key>", "caps", "add:<role>") and shows a tooltip.  Parts with generic values are drawn
amber, missing positions red."""
import math

import wx

from ..loop import model as M
from .schematic import colours, draw_symbol

AMBER = wx.Colour(214, 140, 0)
RED = wx.Colour(210, 50, 50)


class CellCanvas(wx.Panel):
    H = 324

    def __init__(self, parent, board, on_action):
        super().__init__(parent, size=(-1, self.H))
        self.SetMinSize((360, self.H))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.board, self.on_action, self.s = board, on_action, None
        self.hits = []          # (wx.Rect, action, tooltip)
        self._tip = None
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_SIZE, lambda e: (e.Skip(), self.Refresh()))
        self.Bind(wx.EVT_MOTION, self._motion)
        self.Bind(wx.EVT_LEFT_UP, self._click)

    def update(self, setup, board=None):
        self.s = setup
        if board is not None:
            self.board = board
        self.Refresh()

    # ------------------------------------------------------------- events
    def _hit(self, pos):
        return next(((a, t) for r, a, t in self.hits if r.Contains(pos)), (None, None))

    def _motion(self, e):
        a, t = self._hit(e.GetPosition())
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND if a else wx.CURSOR_ARROW))
        tip = t or ("The switching cell: the hot loop runs from the caps through the upper switch, the switch "
                    "node and the lower switch back to the caps. Click a part to edit it.")
        if tip != self._tip:
            self._tip = tip
            self.SetToolTip(tip)

    def _click(self, e):
        a, _ = self._hit(e.GetPosition())
        if a:
            self.on_action(a)

    # ------------------------------------------------------------- drawing
    def _inductor(self):
        s = self.s
        for fp in self.board.footprints:
            if fp.ref.startswith("L"):
                nets = {p.net for p in self.board.pads if p.ref == fp.ref and p.net}
                if s.sw_net in nets and len(nets) == 2:
                    return fp.ref, next(n for n in nets if n != s.sw_net)
        return None, None

    def _paint(self, e):
        dc = wx.AutoBufferedPaintDC(self)
        c = colours()
        dc.SetBackground(wx.Brush(c["bg"]))
        dc.Clear()
        gc = wx.GraphicsContext.Create(dc)
        self.hits = []
        if self.s is None or not self.s.sw_net:
            gc.SetFont(self.GetFont(), c["dim"])
            gc.DrawText("Pick the switch node and press Detect parts: the cell is drawn here.", 12, 12)
            return
        self._draw(gc, c)

    def _text(self, gc, txt, x, y, col, bold=False, center=False, right=False, maxw=None, small=False):
        f = self.GetFont()
        if small:
            f = f.Smaller()
        if bold:
            f = f.Bold()
        gc.SetFont(f, col)
        w, h = gc.GetFullTextExtent(txt)[:2]
        if maxw and w > maxw:
            while txt and gc.GetFullTextExtent(txt + "…")[0] > maxw:
                txt = txt[:-1]
            txt += "…"
            w = gc.GetFullTextExtent(txt)[0]
        x0 = x - w / 2 if center else x - w if right else x
        gc.DrawText(txt, x0, y - h / 2)
        return wx.Rect(int(x0), int(y - h / 2), int(w), int(h))

    def _line(self, gc, pts, col, width=2):
        gc.SetPen(wx.Pen(col, width))
        p = gc.CreatePath()
        p.MoveToPoint(*pts[0])
        for q in pts[1:]:
            p.AddLineToPoint(*q)
        gc.StrokePath(p)

    def _draw(self, gc, c):
        """Laid out as the converter is usually drawn, input on the left:
            buck-like (buck, half-bridge, a 4-switch input leg):
                hot-loop caps | switches (gate text on their left) | switch node -> inductor -> output
            boost-like (boost, a 4-switch output leg): the mirror image,
                input -> inductor -> switch node | switches (gate text on their right) | hot-loop caps
        Everything is placed in the buck-like frame and mirrored (X, rect, align) for a boost."""
        s = self.s
        W = self.GetClientSize().width
        p = self.GetParent()                # a scrolled panel may be wider than its visible part
        while p is not None and not isinstance(p, wx.ScrolledWindow):
            p = p.GetParent()
        if p is not None:
            W = min(W, p.GetClientSize().width - (self.GetScreenPosition().x - p.GetScreenPosition().x) - 8)
        W = max(W, 360)
        fg, dim = c["fg"], c["dim"]
        topo = s.resolved(self.board)
        d_, _ = s.current_dir(self.board)
        mirror = topo in ("boost", "bb_out")
        self.mirror, self.W = mirror, W
        X = (lambda x: W - x) if mirror else (lambda x: x)                     # noqa: E731
        y_rail, y_ret = 64, self.H - 40
        y_sw = (y_rail + y_ret) / 2
        xc = 56                                             # hot-loop caps
        xs = self.xs = max(220, min(W * 0.5, W - 200))      # switch column
        x_i0 = xs + 26                                      # inductor
        x_i1 = x_i0 + max(40, min(120, W - xs - 160))
        x_far = x_i1 + 18
        # title (not mirrored)
        r = self._text(gc, s.label(self.board), 12, 16, fg, bold=True, maxw=W - 24)
        self.hits.append((r, "opt:topology", "The cell's kind" + (" (detected from the parts and the board)"
                                                                  if s.topology == "auto" else "")
                          + ". It sets the drawing, which switch is hard-switched and what the nets are. Change "
                            "it with 'Topology' below."))
        T = lambda txt, x, y, col, align="l", **k: self._text(                  # noqa: E731
            gc, txt, X(x), y, col, center=align == "c", right=(align == "r") != mirror and align != "c", **k)
        # rail and return: from the caps to the switches
        links = s.by_role("link")
        side = M.TOPO_RAIL.get(topo, "supply side")
        self._line(gc, [(X(xc), y_rail), (X(xs), y_rail)], fg)
        self._line(gc, [(X(xc), y_ret), (X(xs), y_ret)], fg)
        xa, xb = xc + 34, xs - 34
        maxw_rail = (xa - 18 if links else xs - 30)
        r = T(s.rail_net or "?", 12, y_rail - 13, fg, bold=True, maxw=maxw_rail)
        r = r.Union(T(f"{side} (rail)", 12 + (r.width + 8), y_rail - 13, dim, small=True, maxw=max(maxw_rail - r.width - 8, 10)))
        self.hits.append((r, "net:rail", f"The supply side of the hot loop (the 'rail'){', here the ' + side if topo else ''}: "
                                         "VIN for a buck, VOUT for a boost, the bus for a half-bridge. Change it "
                                         "with 'Supply side' below."))
        r = T(s.ret_net or "?", 12, y_ret + 13, fg, bold=True, maxw=xs - 30)
        r = r.Union(T("return", 12 + r.width + 8, y_ret + 13, dim, small=True))
        self.hits.append((r, "net:ret", "The return of the hot loop (ground)."))
        if links:
            group = sorted({s.rail_net} | {d.nets.get("2") for d in links} - {None})
            self._line(gc, [(X(xa), y_rail), (X(xb), y_rail)], c["bg"], 4)
            gc.SetPen(wx.Pen(fg, 2))
            draw_symbol(gc, "R", X(xa), y_rail, X(xb), y_rail)
            txt = ", ".join(d.ref for d in links) + " (0 Ω)"
            T(txt, (xa + xb) / 2, y_rail - 13, fg, align="c", small=True, maxw=xb - xa)
            tip = (f"Rail link: {', '.join(d.ref for d in links)} carries the supply on to "
                   f"{', '.join(n for n in group if n != s.rail_net)}, where the caps are. Its package L is in the loop.")
            self.hits.append((self._rect(xa, y_rail - 22, xb - xa, 34), f"edit:{links[0].key}", tip))
        # hot-loop caps
        caps = s.by_role("cap")
        col = fg if caps else RED
        gc.SetPen(wx.Pen(col, 2))
        draw_symbol(gc, "C", X(xc), y_rail, X(xc), y_ret)
        if caps:
            names = [d.ref for d in caps]
            txt = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
            tip = ("Hot-loop capacitors (on): " + ", ".join(
                f"{d.ref} {fmt(d.model.get('value', 0), 'F')}" for d in caps)
                + ". The loop closes through them. Click to see them in the Parts list.")
        else:
            txt, tip = "no hot-loop cap!", "No hot-loop capacitor is on: the loop has no path back. Detect parts."
        room = xs - xc - 50
        T(txt, xc + 16, y_sw - 9, col, small=True, maxw=room)
        T(f"hot-loop caps ({side})" if topo else "hot-loop caps", xc + 16, y_sw + 8, dim, small=True, maxw=room)
        self.hits.append((self._rect(xc - 20, y_sw - 22, room + 36, 44), "caps", tip))
        # switches, their text and gate drives on the caps' side
        for pos, y0, y1 in (("upper", y_rail, y_sw), ("lower", y_sw, y_ret)):
            self._switch(gc, c, pos, s.by_role(pos), xs, y0, y1, xc, X, T, d_)
        gc.SetBrush(wx.Brush(fg))
        gc.DrawEllipse(X(xs) - 3.5, y_sw - 3.5, 7, 7)
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        # the hot loop, faintly
        hl = wx.Colour(c["hl"].Red(), c["hl"].Green(), c["hl"].Blue(), 60)
        gc.SetPen(wx.Pen(hl, 8))
        p = gc.CreatePath()
        x0, x1 = sorted((X(xc - 14), X(xs + 14)))
        p.AddRoundedRectangle(x0, y_rail - 6, x1 - x0, y_ret - y_rail + 12, 10)
        gc.StrokePath(p)
        T("hot loop", xc + 16, y_ret - 13, dim, small=True)
        # switch node, inductor and the far side (output / input / load / the other leg)
        r = T(s.sw_net, xs + 14, y_sw + 15, fg, bold=True, small=True, maxw=x_far - xs - 20)
        r = r.Union(T("switch node", xs + 14, y_sw + 30, dim, small=True))
        self.hits.append((r, "net:sw", "The switch node: between the two switches and the inductor."))
        lref, far = self._inductor()
        self._line(gc, [(X(xs), y_sw), (X(x_i0), y_sw)], fg)
        if not lref:
            T("(no inductor on the board)", x_i0 + 4, y_sw + 14, dim, small=True)
            return
        gc.SetPen(wx.Pen(fg, 2))
        draw_symbol(gc, "L", X(x_i0), y_sw, X(x_i1), y_sw)
        self._line(gc, [(X(x_i1), y_sw), (X(x_far), y_sw)], fg)
        arrow = "→" if (d_ == "out") != mirror else "←"
        cur = f" {arrow} {s.i_switch:g} A" if s.i_switch else f" {arrow}"
        r = T(lref + cur, (x_i0 + x_i1) / 2, y_sw - 15, fg, align="c", small=True)
        self.hits.append((r.Inflate(4, 8), "", f"The inductor {lref}: its current is "
                          f"{'out of' if d_ == 'out' else 'into'} the switch node at the edges (a constant current "
                          "source in the switching simulation). It is not part of the hot loop."))
        gc.SetBrush(wx.Brush(fg))
        gc.DrawEllipse(X(x_far) - 3, y_sw - 3, 6, 6)
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        what = M.TOPO_FAR.get(topo, "")
        rf = T(far, x_far + 8, y_sw - 13, fg, bold=True, small=True, maxw=W - x_far - 12)
        if what:
            rf = rf.Union(T(what, x_far + 8, y_sw + 3, dim, small=True, maxw=W - x_far - 12))
        if topo in ("buck", "boost", ""):
            # the other side's capacitor, faint: it filters, it is not in the hot loop
            soft = wx.Colour(dim.Red(), dim.Green(), dim.Blue(), 150)
            self._line(gc, [(X(xs), y_ret), (X(x_far), y_ret)], soft, 1)
            gc.SetPen(wx.Pen(soft, 1))
            yc = (y_sw + y_ret) / 2 + 10
            draw_symbol(gc, "C", X(x_far), y_sw, X(x_far), y_ret)
            other = "output" if topo in ("buck", "") else "input"
            rc = T(f"{other} cap", x_far + 8, yc, dim, small=True, maxw=W - x_far - 12)
            self.hits.append((rc.Union(rf).Inflate(4, 4), "", f"{far}: the converter's {other}. Its capacitor "
                              f"filters the {other} and is not in the hot loop (no switched current flows in it)."))
        else:
            self.hits.append((rf.Inflate(4, 4), "", f"{far}: the inductor's far end"
                              + (f", the {what}" if what else "") + "."))

    def _rect(self, x, y, w, h):
        """A hit rectangle given in the buck-like frame."""
        x0 = self.W - x - w if self.mirror else x
        return wx.Rect(int(x0), int(y), int(w), int(h))

    def _switch(self, gc, c, pos, parts, xs, y0, y1, xc, X, T, d_):
        fg, dim = c["fg"], c["dim"]
        ym = (y0 + y1) / 2
        x = X(xs)
        tx = xs - 30                       # text, right-aligned on the caps' side
        room = xs - xc - 50
        if not parts:
            self._line(gc, [(x, y0), (x, ym - 16)], RED)
            self._line(gc, [(x, ym + 16), (x, y1)], RED)
            T(f"no {pos} switch", tx, ym, RED, align="r", small=True)
            self.hits.append((self._rect(tx - 120, ym - 18, 150, 36), f"add:{pos}",
                              f"Nothing in the {pos} position. Click to add the part (or an IC's internal switch)."))
            return
        main = next((d for d in parts if d.kind == "fet"), parts[0])
        generic = any(d.model.get("assumed") for d in parts)
        col = AMBER if generic else fg
        gc.SetPen(wx.Pen(col, 2))
        if main.kind == "diode":
            draw_symbol(gc, "D", x, y1, x, y0)          # anode at the bottom, cathode up
        else:
            self._line(gc, [(x, y0), (x, ym - 18)], col)
            self._line(gc, [(x, ym + 18), (x, y1)], col)
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
            gc.DrawRoundedRectangle(x - 14, ym - 18, 28, 36, 4)
            self._text(gc, "IC" if main.internal else ("P" if main.model.get("channel") == "P" else "N"),
                       x, ym, col, center=True, small=True)
        names = ", ".join(d.key for d in parts)
        kind = ("internal switch" if main.internal else "MOSFET" if main.kind == "fet" else "diode")
        if pos == ("upper" if d_ == "out" else "lower") and any(d.kind == "fet" for d in parts):
            T("hard-switched", xs + 20, ym - 26 if pos == "upper" else ym, c["hl"], small=True)
        r = T(names, tx, ym - 20, col, align="r", bold=True, small=True, maxw=room)
        r = r.Union(T(kind + (", generic" if generic else ""), tx, ym - 4, dim, align="r", small=True, maxw=room))
        tip = (f"{pos.capitalize()} switch: {names} ({kind}). "
               + ("Generic values: click to enter its datasheet values." if generic else "Click to edit its model."))
        self.hits.append((r.Union(wx.Rect(int(x - 16), int(ym - 20), 32, 40)).Inflate(2, 2), f"edit:{main.key}", tip))
        # gate drive
        fet = next((d for d in parts if d.kind == "fet" and not d.internal and d.nets.get("G")), None)
        if fet is None:
            return
        g = next((g for g in self.s.drives if self.s.device(g.fet) is fet), None)
        self._line(gc, [(X(xs - 14), ym), (X(xs - 22), ym)], fg if g else AMBER)
        if g is None:
            r = T("no gate drive", tx, ym + 12, AMBER, align="r", small=True)
            self.hits.append((r.Inflate(4, 6), f"drive:{fet.key}",
                              f"{fet.key} has no gate drive, so its gate loop is not analysed. Click to add one."))
            return
        chain = [self.s.device(k) for k in g.chain if self.s.device(k)]
        path = " → ".join([f"{g.out[0]} {g.out[1]}"] + [d.key for d in chain]) + " → G"
        r1 = T(path, tx, ym + 12, fg, align="r", small=True, maxw=room)
        r2 = T(f"S → {g.ret[0]} {g.ret[1]}", tx, ym + 27, dim, align="r", small=True, maxw=room)
        tip = (f"Gate drive of {fet.key}: from the driver's output pin ({g.out[0]} on {g.out[1]}) through "
               f"{', '.join(d.key for d in chain) or 'no series part'} to the gate; back from the source to the "
               f"driver's return pin ({g.ret[0]} on {g.ret[1]}). Click to edit.")
        self.hits.append((r1.Union(r2).Inflate(4, 4), f"drive:{fet.key}", tip))


def fmt(v, unit):
    from ..suite.library import fmt_si
    return fmt_si(v, unit)
