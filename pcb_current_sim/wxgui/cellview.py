"""A live schematic of a switching cell, drawn from its setup: the supply side (rail) at the top,
the return at the bottom, the upper and lower switches with the switch node between them, the
hot-loop capacitors (behind their rail links), the inductor, and each FET's gate drive as a path.

Every part is clickable (on_action(key) with the checks module's fix keys: "edit:<key>",
"drive:<fet key>", "caps", "add:<role>") and shows a tooltip.  Parts with generic values are drawn
amber, missing positions red."""
import math

import wx

from .schematic import colours, draw_symbol

AMBER = wx.Colour(214, 140, 0)
RED = wx.Colour(210, 50, 50)


class CellCanvas(wx.Panel):
    H = 300

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
        s = self.s
        W = self.GetClientSize().width
        p = self.GetParent()                # a scrolled panel may be wider than its visible part
        while p is not None and not isinstance(p, wx.ScrolledWindow):
            p = p.GetParent()
        if p is not None:
            W = min(W, p.GetClientSize().width - (self.GetScreenPosition().x - p.GetScreenPosition().x) - 8)
        W = max(W, 360)
        fg, dim = c["fg"], c["dim"]
        y_rail, y_sw, y_ret = 40, self.H / 2, self.H - 40
        xs = max(190, W * 0.42)             # switch column
        xc = self.xc = min(W - 90, xs + max(150, W * 0.3))   # capacitor column
        x_l = 12                            # net labels
        links = [d for d in s.by_role("link")]
        caps = s.by_role("cap")
        # nets
        group = sorted({s.rail_net} | {d.nets.get("2") for d in links} - {None})
        r = self._text(gc, s.rail_net or "?", x_l, y_rail - 12, fg, bold=True, maxw=xs - 24)
        r = r.Union(self._text(gc, "supply side (rail)", x_l, y_rail + 11, dim, small=True))
        self.hits.append((r, "net:rail", "The supply side of the hot loop (the 'rail'): VIN for a buck, VOUT for a "
                                         "boost, the bus for a half-bridge. Change it with 'Supply side' below."))
        r = self._text(gc, s.ret_net or "?", x_l, y_ret + 12, fg, bold=True, maxw=xs - 24)
        r = r.Union(self._text(gc, "return", x_l, y_ret - 11, dim, small=True))
        self.hits.append((r, "net:ret", "The return of the hot loop (ground)."))
        r = self._text(gc, s.sw_net, x_l, y_sw - 12, fg, bold=True, maxw=xs - 24)
        r = r.Union(self._text(gc, "switch node", x_l, y_sw + 11, dim, small=True))
        self.hits.append((r, "net:sw", "The switch node: between the two switches and the inductor."))
        self._line(gc, [(x_l, y_rail), (xc, y_rail)], fg)
        self._line(gc, [(x_l, y_ret), (xc, y_ret)], fg)
        self._line(gc, [(x_l, y_sw), (xs, y_sw)], fg)
        # rail links: drawn on the rail line between the switch and the caps
        if links:
            xa, xb = xs + 30, xc - 30
            gc.SetPen(wx.Pen(c["bg"], 4))
            self._line(gc, [(xa, y_rail), (xb, y_rail)], c["bg"], 4)
            gc.SetPen(wx.Pen(fg, 2))
            draw_symbol(gc, "R", xa, y_rail, xb, y_rail)
            txt = ", ".join(d.ref for d in links) + " (0 Ω)"
            r = self._text(gc, txt, (xa + xb) / 2, y_rail - 18, fg, center=True, small=True)
            tip = (f"Rail link: {', '.join(d.ref for d in links)} carries the supply on to "
                   f"{', '.join(n for n in group if n != s.rail_net)}, where the caps are. Its package L is in the loop.")
            self.hits.append((wx.Rect(int(xa), int(y_rail - 26), int(xb - xa), 36), f"edit:{links[0].key}", tip))
        # switches
        for pos, y0, y1 in (("upper", y_rail, y_sw), ("lower", y_sw, y_ret)):
            parts = s.by_role(pos)
            self._switch(gc, c, pos, parts, xs, y0, y1)
        # dot at the switch node
        gc.SetBrush(wx.Brush(fg))
        gc.DrawEllipse(xs - 3.5, y_sw - 3.5, 7, 7)
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        # inductor
        lref, far = self._inductor()
        if lref:
            xa = xs + 10
            xb = min(xa + 110, xc - 20)
            gc.SetPen(wx.Pen(fg, 2))
            draw_symbol(gc, "L", xs, y_sw, xb, y_sw)
            d_, _ = s.current_dir(self.board)
            arrow = "→" if d_ == "out" else "←"
            cur = f" {arrow} {s.i_switch:g} A" if s.i_switch else f" {arrow}"
            self._text(gc, lref + cur, (xs + xb) / 2, y_sw - 16, fg, center=True, small=True)
            self._text(gc, f"to {far}", (xs + xb) / 2, y_sw + 16, dim, center=True, small=True, maxw=xc - xs - 24)
        # caps
        col = fg if caps else RED
        gc.SetPen(wx.Pen(col, 2))
        draw_symbol(gc, "C", xc, y_rail, xc, y_ret)
        if caps:
            names = [d.ref for d in caps]
            txt = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
            tip = ("Hot-loop capacitors (on): " + ", ".join(
                f"{d.ref} {fmt(d.model.get('value', 0), 'F')}" for d in caps)
                + ". The loop closes through them. Click to see them in the Parts list.")
        else:
            txt, tip = "no hot-loop cap!", "No hot-loop capacitor is on: the loop has no path back. Detect parts."
        r = self._text(gc, txt, xc + 18, (y_rail + y_ret) / 2 - 10, col, small=True, maxw=W - xc - 20)
        self._text(gc, "hot-loop caps", xc + 18, (y_rail + y_ret) / 2 + 8, dim, small=True, maxw=W - xc - 20)
        self.hits.append((wx.Rect(int(xc - 20), int(y_rail + 20), int(W - xc + 10), int(y_ret - y_rail - 40)),
                          "caps", tip))
        # the hot loop, faintly
        hl = wx.Colour(c["hl"].Red(), c["hl"].Green(), c["hl"].Blue(), 60)
        gc.SetPen(wx.Pen(hl, 8))
        p = gc.CreatePath()
        p.AddRoundedRectangle(xs - 14, y_rail - 6, xc - xs + 28, y_ret - y_rail + 12, 10)
        gc.StrokePath(p)
        self._text(gc, "hot loop", xc - 10, y_ret - 13, dim, right=True, small=True)

    def _switch(self, gc, c, pos, parts, x, y0, y1):
        fg = c["fg"]
        ym = (y0 + y1) / 2
        if not parts:
            gc.SetPen(wx.Pen(RED, 2))
            self._line(gc, [(x, y0), (x, ym - 16)], RED)
            self._line(gc, [(x, ym + 16), (x, y1)], RED)
            r = self._text(gc, f"no {pos} switch", x + 10, ym, RED, small=True)
            self.hits.append((wx.Rect(int(x - 20), int(ym - 18), 140, 36), f"add:{pos}",
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
        room = self.xc - x - 30
        r = self._text(gc, names, x + 22, ym - 9, col, bold=True, small=True, maxw=room)
        self._text(gc, kind + (", generic" if generic else ""), x + 22, ym + 8, c["dim"], small=True, maxw=room)
        d_, _ = self.s.current_dir(self.board)
        if pos == ("upper" if d_ == "out" else "lower"):
            self._text(gc, "hard-switched", x + 22, ym - 26 if pos == "upper" else ym + 25, c["hl"], small=True)
        tip = (f"{pos.capitalize()} switch: {names} ({kind}). "
               + ("Generic values: click to enter its datasheet values." if generic else "Click to edit its model."))
        self.hits.append((wx.Rect(int(x - 16), int(ym - 20), 200, 40), f"edit:{main.key}", tip))
        # gate drive, to the left of the switch
        fet = next((d for d in parts if d.kind == "fet" and not d.internal and d.nets.get("G")), None)
        if fet is None:
            return
        g = next((g for g in self.s.drives if self.s.device(g.fet) is fet), None)
        xg = x - 16
        if g is None:
            self._line(gc, [(xg, ym), (xg - 30, ym)], AMBER)
            r = self._text(gc, "no gate drive", xg - 34, ym, AMBER, right=True, small=True)
            self.hits.append((r.Inflate(4, 6), f"drive:{fet.key}",
                              f"{fet.key} has no gate drive, so its gate loop is not analysed. Click to add one."))
            return
        self._line(gc, [(xg, ym), (xg - 22, ym)], fg)
        chain = [self.s.device(k) for k in g.chain if self.s.device(k)]
        path = " → ".join([f"{g.out[0]} {g.out[1]}"] + [d.key for d in chain]) + " → G"
        r1 = self._text(gc, path, xg - 26, ym - 8, fg, right=True, small=True, maxw=xg - 40)
        r2 = self._text(gc, f"S → {g.ret[0]} {g.ret[1]}", xg - 26, ym + 9, c["dim"], right=True, small=True,
                        maxw=xg - 40)
        tip = (f"Gate drive of {fet.key}: from the driver's output pin ({g.out[0]} on {g.out[1]}) through "
               f"{', '.join(d.key for d in chain) or 'no series part'} to the gate; back from the source to the "
               f"driver's return pin ({g.ret[0]} on {g.ret[1]}). Click to edit.")
        self.hits.append((r1.Union(r2).Inflate(4, 4), f"drive:{fet.key}", tip))


def fmt(v, unit):
    from ..suite.library import fmt_si
    return fmt_si(v, unit)
