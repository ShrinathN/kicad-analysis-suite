"""Schematic canvas for the matching tool: draws the ladder with real symbols as it is edited.

Layout (left to right): source (generator + Zs), port 1, one cell per element, port 2, load.
Series elements sit on the top rail; shunt elements hang from it to ground.
Click selects, double-click edits, drag moves an element, right-click opens a menu.
"""
import math

import wx

from ..matching import model as M

CW = 156            # cell width
Y0 = 64             # rail
Y1 = 186            # ground connection
H = 236
SRC_W = 170
LOAD_W = 200
BODY = 48


def colours():
    ss = wx.SystemSettings
    return dict(fg=ss.GetColour(wx.SYS_COLOUR_WINDOWTEXT), bg=ss.GetColour(wx.SYS_COLOUR_WINDOW),
                hl=ss.GetColour(wx.SYS_COLOUR_HIGHLIGHT), dim=ss.GetColour(wx.SYS_COLOUR_GRAYTEXT),
                btn=ss.GetColour(wx.SYS_COLOUR_BTNFACE))


# ---------------------------------------------------------------- symbol primitives
def _body(path, kind, L, label=None):
    """Add a horizontal symbol from (0, 0) to (L, 0) to `path`; leads fill the rest."""
    a, b = (L - BODY) / 2, (L + BODY) / 2
    if kind == "L":
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(a, 0)
        r = BODY / 8
        for k in range(4):
            cx = a + r * (2 * k + 1)
            path.MoveToPoint(cx - r, 0)
            path.AddArc(cx, 0, r, math.pi, 2 * math.pi, True)
        path.MoveToPoint(b, 0)
        path.AddLineToPoint(L, 0)
    elif kind == "C":
        g, ph = 7, 28
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(L / 2 - g / 2, 0)
        path.MoveToPoint(L / 2 - g / 2, -ph / 2)
        path.AddLineToPoint(L / 2 - g / 2, ph / 2)
        path.MoveToPoint(L / 2 + g / 2, -ph / 2)
        path.AddLineToPoint(L / 2 + g / 2, ph / 2)
        path.MoveToPoint(L / 2 + g / 2, 0)
        path.AddLineToPoint(L, 0)
    elif kind == "R":
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(a + 4, 0)
        n, amp, w = 6, 8, BODY - 8
        for k in range(n):
            path.AddLineToPoint(a + 4 + w * (k + 0.5) / n, -amp if k % 2 == 0 else amp)
        path.AddLineToPoint(b - 4, 0)
        path.AddLineToPoint(L, 0)
    elif kind in ("TL", "line"):
        hh = 8
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(a, 0)
        path.MoveToPoint(a + hh / 2, -hh)
        path.AddLineToPoint(b - hh / 2, -hh)
        path.MoveToPoint(a + hh / 2, hh)
        path.AddLineToPoint(b - hh / 2, hh)
        path.AddEllipse(a, -hh, hh, 2 * hh)
        path.AddEllipse(b - hh, -hh, hh, 2 * hh)
        path.MoveToPoint(b, 0)
        path.AddLineToPoint(L, 0)
    elif kind == "box":
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(a + 4, 0)
        path.AddRectangle(a + 4, -9, BODY - 8, 18)
        path.MoveToPoint(b - 4, 0)
        path.AddLineToPoint(L, 0)
    elif kind == "D":                      # anode at 0, cathode at L
        hw = 12
        c = L / 2
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(c - hw, 0)
        path.MoveToPoint(c - hw, -hw)
        path.AddLineToPoint(c + hw, 0)
        path.AddLineToPoint(c - hw, hw)
        path.CloseSubpath()
        path.MoveToPoint(c + hw, -hw)
        path.AddLineToPoint(c + hw, hw)
        path.MoveToPoint(c + hw, 0)
        path.AddLineToPoint(L, 0)
    elif kind == "S2P":
        path.MoveToPoint(0, 0)
        path.AddLineToPoint(a, 0)
        path.AddRoundedRectangle(a, -20, BODY, 40, 4)
        path.MoveToPoint(b, 0)
        path.AddLineToPoint(L, 0)


def draw_symbol(gc, kind, x0, y0, x1, y1):
    """Stroke a symbol between two points (horizontal or vertical)."""
    L = math.hypot(x1 - x0, y1 - y0)
    gc.PushState()
    gc.Translate(x0, y0)
    gc.Rotate(math.atan2(y1 - y0, x1 - x0))
    p = gc.CreatePath()
    _body(p, kind, L)
    gc.StrokePath(p)
    gc.PopState()


def draw_ground(gc, x, y):
    p = gc.CreatePath()
    for k, w in enumerate((22, 14, 6)):
        p.MoveToPoint(x - w / 2, y + 5 * k)
        p.AddLineToPoint(x + w / 2, y + 5 * k)
    gc.StrokePath(p)


def draw_dot(gc, x, y, col):
    gc.SetBrush(wx.Brush(col))
    gc.DrawEllipse(x - 3.5, y - 3.5, 7, 7)
    gc.SetBrush(wx.TRANSPARENT_BRUSH)


def draw_generator(gc, x, y, r=17):
    p = gc.CreatePath()
    p.AddCircle(x, y, r)
    p.MoveToPoint(x - r * 0.6, y)
    for k in range(1, 25):
        t = k / 24
        p.AddLineToPoint(x - r * 0.6 + 1.2 * r * t, y - r * 0.35 * math.sin(2 * math.pi * t))
    gc.StrokePath(p)


def draw_element(gc, e_kind, shunt, left, y0=Y0, y1=Y1, cw=CW, dot_col=None):
    """One ladder cell (rail through it plus the element), without labels. Returns the symbol axis x."""
    if not shunt:
        kind = {"TL": "TL", "S2P": "S2P"}.get(e_kind, e_kind)
        draw_symbol(gc, kind, left, y0, left + cw, y0)
        return left + cw / 2
    x = left + 30
    p = gc.CreatePath()
    p.MoveToPoint(left, y0)
    p.AddLineToPoint(left + cw, y0)
    gc.StrokePath(p)
    if dot_col is not None:
        draw_dot(gc, x, y0, dot_col)
    if e_kind in ("stub_open", "stub_short"):
        end = y1 - 18 if e_kind == "stub_open" else y1
        draw_symbol(gc, "TL", x, y0, x, end)
        if e_kind == "stub_open":
            q = gc.CreatePath()
            q.AddCircle(x, end + 4, 4)
            gc.StrokePath(q)
        else:
            draw_ground(gc, x, y1)
    else:
        draw_symbol(gc, e_kind, x, y0, x, y1)
        draw_ground(gc, x, y1)
    return x


def make_icon(kind, shunt, size=(46, 40)):
    """Palette icon drawn with the same symbols as the schematic."""
    w, h = size
    bmp = wx.Bitmap(w, h)
    dc = wx.MemoryDC(bmp)
    c = colours()
    dc.SetBackground(wx.Brush(c["btn"]))
    dc.Clear()
    gc = wx.GraphicsContext.Create(dc)
    gc.SetPen(wx.Pen(c["fg"], 4))
    gc.SetBrush(wx.TRANSPARENT_BRUSH)
    s = w / 104
    gc.Scale(s, s)
    if shunt:
        draw_element(gc, kind, True, 18, y0=10, y1=(h / s) - 22, cw=68, dot_col=c["fg"])
    else:
        gc.Translate(0, (h / s) / 2 - 20)
        draw_element(gc, kind, False, 2, y0=20, y1=20, cw=100)
    del gc
    dc.SelectObject(wx.NullBitmap)
    return bmp


# ---------------------------------------------------------------- canvas
class SchematicCanvas(wx.ScrolledWindow):
    """Callbacks: on_select(sel), on_edit(sel), on_move(i, j), on_menu(sel, pos), with sel an element
    index, "source", "load" or None."""

    def __init__(self, parent, setup, on_select, on_edit, on_move, on_menu):
        super().__init__(parent, style=wx.HSCROLL | wx.BORDER_SUNKEN)
        self.setup, self.sel = setup, None
        self.on_select, self.on_edit, self.on_move, self.on_menu = on_select, on_edit, on_move, on_menu
        self._drag = None           # (index, x0) while the mouse is down on an element
        self._drop = None           # gap index while dragging
        self.f0 = None
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetScrollRate(20, 0)
        self.SetMinSize((-1, H + 4))
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._down)
        self.Bind(wx.EVT_LEFT_UP, self._up)
        self.Bind(wx.EVT_MOTION, self._motion)
        self.Bind(wx.EVT_LEFT_DCLICK, self._dclick)
        self.Bind(wx.EVT_RIGHT_DOWN, self._right)
        self.Bind(wx.EVT_KEY_DOWN, self._key)
        self.update()

    # ---- geometry
    def _x_elem(self, i):
        return SRC_W + i * CW

    def width(self):
        return SRC_W + max(len(self.setup.elements), 1) * CW + LOAD_W

    def update(self, setup=None):
        if setup is not None:
            self.setup = setup
        self.SetVirtualSize((self.width(), H))
        self.Refresh()

    def hit(self, x, y):
        n = len(self.setup.elements)
        if x < SRC_W - 20:
            return "source"
        if SRC_W <= x < SRC_W + n * CW:
            return int((x - SRC_W) // CW)
        if x >= SRC_W + max(n, 1) * CW + 10:
            return "load"
        return None

    def gap(self, x):
        n = len(self.setup.elements)
        return int(min(max(round((x - SRC_W) / CW), 0), n))

    def insert_index(self):
        return self.sel + 1 if isinstance(self.sel, int) else len(self.setup.elements)

    # ---- mouse / keys
    def _pos(self, e):
        return self.CalcUnscrolledPosition(e.GetPosition())

    def _down(self, e):
        self.SetFocus()
        p = self._pos(e)
        h = self.hit(p.x, p.y)
        self.select(h)
        if isinstance(h, int):
            self._drag = (h, p.x)
        e.Skip()

    def _motion(self, e):
        if self._drag is None or not e.LeftIsDown():
            return
        p = self._pos(e)
        if abs(p.x - self._drag[1]) > 8:
            self._drop = self.gap(p.x)
            self.SetCursor(wx.Cursor(wx.CURSOR_SIZEWE))
            self.Refresh()

    def _up(self, e):
        if self._drag is not None and self._drop is not None:
            i, j = self._drag[0], self._drop
            if j > i:
                j -= 1
            if j != i:
                self.on_move(i, j)
                self.sel = j
        self._drag = self._drop = None
        self.SetCursor(wx.NullCursor)
        self.Refresh()

    def _dclick(self, e):
        p = self._pos(e)
        h = self.hit(p.x, p.y)
        if h is not None:
            self.on_edit(h)

    def _right(self, e):
        p = self._pos(e)
        self.select(self.hit(p.x, p.y))
        self.on_menu(self.sel, e.GetPosition())

    def _key(self, e):
        k = e.GetKeyCode()
        n = len(self.setup.elements)
        if k in (wx.WXK_LEFT, wx.WXK_RIGHT) and n:
            if e.ControlDown() and isinstance(self.sel, int):
                j = self.sel + (1 if k == wx.WXK_RIGHT else -1)
                if 0 <= j < n:
                    self.on_move(self.sel, j)
                    self.select(j)
                return
            cur = self.sel if isinstance(self.sel, int) else (-1 if k == wx.WXK_RIGHT else n)
            self.select(min(max(cur + (1 if k == wx.WXK_RIGHT else -1), 0), n - 1))
        elif k in (wx.WXK_DELETE, wx.WXK_BACK) and isinstance(self.sel, int):
            self.on_menu(self.sel, None, "delete")
        elif k in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER) and self.sel is not None:
            self.on_edit(self.sel)
        else:
            e.Skip()

    def select(self, sel):
        if sel != self.sel:
            self.sel = sel
            self.on_select(sel)
        if isinstance(sel, int):
            x = self._x_elem(sel)
            vx = self.CalcScrolledPosition(x, 0)[0]
            if vx < 0 or vx + CW > self.GetClientSize().width:
                self.Scroll(max(x - 60, 0) // 20, 0)
        self.Refresh()

    # ---- painting
    def _paint(self, e):
        dc = wx.AutoBufferedPaintDC(self)
        self.DoPrepareDC(dc)
        c = colours()
        dc.SetBackground(wx.Brush(c["bg"]))
        dc.Clear()
        gc = wx.GraphicsContext.Create(dc)
        self.paint(gc, c)

    def paint(self, gc, c):
        s, n = self.setup, len(self.setup.elements)
        pen = wx.Pen(c["fg"], 2)
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        hl = wx.Colour(c["hl"].Red(), c["hl"].Green(), c["hl"].Blue(), 70)
        # selection
        if self.sel is not None:
            if isinstance(self.sel, int) and self.sel < n:
                r = (self._x_elem(self.sel) + 3, 6, CW - 6, H - 12)
            elif self.sel == "source":
                r = (4, 6, SRC_W - 28, H - 12)
            elif self.sel == "load":
                r = (SRC_W + max(n, 1) * CW + 12, 6, LOAD_W - 16, H - 12)
            else:
                r = None
            if r:
                gc.SetPen(wx.Pen(c["hl"], 1))
                gc.SetBrush(wx.Brush(hl))
                gc.DrawRoundedRectangle(*r, 6)
                gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.SetPen(pen)
        f_small = wx.Font(wx.FontInfo(8))
        f_bold = wx.Font(wx.FontInfo(9).Bold())
        # source: generator to ground, Zs box on the rail
        gx = 34
        p = gc.CreatePath()
        p.MoveToPoint(gx, Y0)
        p.AddLineToPoint(gx, (Y0 + Y1) / 2 - 17)
        p.MoveToPoint(gx, (Y0 + Y1) / 2 + 17)
        p.AddLineToPoint(gx, Y1)
        gc.StrokePath(p)
        draw_generator(gc, gx, (Y0 + Y1) / 2)
        draw_ground(gc, gx, Y1)
        draw_symbol(gc, "box", gx, Y0, SRC_W, Y0)
        self._text(gc, "Source", gx + 30, Y0 + 24, f_bold, c["fg"])
        for k, t in enumerate(("Zs " + s.source.text()).split(", ")):
            self._text(gc, t, gx + 30, Y0 + 40 + 15 * k, f_small, c["fg"], SRC_W - gx - 50)
        # elements
        if n == 0:
            p = gc.CreatePath()
            p.MoveToPoint(SRC_W, Y0)
            p.AddLineToPoint(SRC_W + CW, Y0)
            gc.StrokePath(p)
            self._text(gc, "Add parts with the", SRC_W + 6, Y0 + 24, f_small, c["dim"])
            self._text(gc, "palette above", SRC_W + 6, Y0 + 38, f_small, c["dim"])
        for i, el in enumerate(s.elements):
            left = self._x_elem(i)
            ax = draw_element(gc, el.kind, el.shunt, left, dot_col=c["fg"])
            lines = self.label_lines(el)
            if el.shunt:
                self._text(gc, el.name, ax + 14, Y0 + 16, f_bold, c["fg"])
                for k, t in enumerate(lines):
                    self._text(gc, t, ax + 14, Y0 + 34 + 15 * k, f_small, c["fg"], CW - (ax - left) - 18)
            else:
                self._text(gc, el.name, ax, Y0 - 44, f_bold, c["fg"], center=True)
                for k, t in enumerate(lines):
                    self._text(gc, t, ax, Y0 + 26 + 15 * k, f_small, c["fg"], CW - 10, center=True)
        # ports and load
        xe = SRC_W + max(n, 1) * CW
        for x, name in ((SRC_W, "P1"), (xe, "P2")):
            gc.SetPen(wx.Pen(c["dim"], 1, wx.PENSTYLE_SHORT_DASH))
            gc.StrokeLine(x, Y0 - 30, x, Y1 + 14)
            self._text(gc, name, x + 3, Y1 + 4, f_small, c["dim"])
        gc.SetPen(pen)
        lx = xe + 60
        p = gc.CreatePath()
        p.MoveToPoint(xe, Y0)
        p.AddLineToPoint(lx, Y0)
        gc.StrokePath(p)
        draw_symbol(gc, "box", lx, Y0, lx, Y1)
        draw_ground(gc, lx, Y1)
        self._text(gc, "Load", lx + 16, Y0 + 24, f_bold, c["fg"])
        for k, t in enumerate(("ZL " + s.load.text()).split(", ")):
            self._text(gc, t, lx + 16, Y0 + 40 + 15 * k, f_small, c["fg"], LOAD_W - 80)
        # insertion caret / drop marker
        gi = self._drop if self._drop is not None else self.insert_index()
        x = SRC_W + gi * CW
        gc.SetPen(wx.Pen(c["hl"], 3 if self._drop is not None else 2))
        gc.SetBrush(wx.Brush(c["hl"]))
        tri = gc.CreatePath()
        tri.MoveToPoint(x - 6, 12)
        tri.AddLineToPoint(x + 6, 12)
        tri.AddLineToPoint(x, 21)
        tri.CloseSubpath()
        gc.FillPath(tri)
        if self._drop is not None:
            gc.StrokeLine(x, 22, x, Y1 + 10)

    def label_lines(self, el):
        p = el.get_part()
        if p is not None:
            out = [el.value_text(), el.tol_text()]
            if el.lib:
                out.append(el.lib)
            elif p.kind == "L" and (p.q or p.srf):
                out.append(f"Q {p.q:g}" + (f", SRF {p.srf / 1e9:.3g} GHz" if p.srf else ""))
            return out
        if el.kind == "S2P":
            return [el.value_text(), "fixed"]
        z0, _, eeff = el.line_params(self.f0 or 1e9)
        z0 = float(z0) if el.line == "ms" else el.z0
        out = [f"{z0:.1f} Ω, {el.length * 1e3:.3g} mm"]
        if self.f0:
            out.append(f"{360 * self.f0 * el.length * math.sqrt(eeff) / M.C0:.0f}° @ {self.f0 / 1e9:.4g} GHz")
        out.append(f"µstrip w {el.w * 1e3:.3g} mm" if el.line == "ms" else f"ideal, εeff {el.eeff:g}")
        t = el.tol_text()
        if t != "exact":
            out.append(t)
        return out

    @staticmethod
    def _text(gc, text, x, y, font, col, maxw=None, center=False):
        gc.SetFont(font, col)
        w = gc.GetFullTextExtent(text)[0]
        if maxw and w > maxw:
            while text and gc.GetFullTextExtent(text + "…")[0] > maxw:
                text = text[:-1]
            text += "…"
            w = gc.GetFullTextExtent(text)[0]
        gc.DrawText(text, x - w / 2 if center else x, y)
