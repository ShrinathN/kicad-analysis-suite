"""Components window: the board's shared list of parts between pads, and which tools use each."""
import re

import wx

from .. import components as CO
from .. import elements as EL
from ..suite.library import fmt_si
from .crossprobe import CrossProbe
from .layout import AutoList

KIND = {"R": "resistor", "V": "diode (constant drop)", "C": "capacitor", "L": "inductor", "LDO": "linear regulator",
        "SMPS": "switching regulator"}
# tools with an on / off switch per component (Switching Loop's capacitors are on / off per cell)
SWITCHED = ("dc", "extract")


class ComponentsFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="Components - PCB analysis suite", size=(1180, 640))
        self.session, self.board = session, session.board
        p = wx.Panel(self)
        top = wx.BoxSizer(wx.VERTICAL)
        self.intro = wx.StaticText(p, label=(
            "Parts between pads, defined once for this board and shared by the tools: an edit here (or in any "
            "tool) changes the part everywhere, while each tool switches it on or off for itself. Resistor "
            "footprints are found on the board; add other parts (an inductor's DCR, a FET's R_on, a diode, a "
            "regulator, a capacitor) by hand. Capacitor models edited in Switching Loop are kept here as the "
            "part's real model."))
        top.Add(self.intro, 0, wx.EXPAND | wx.ALL, 8)
        self.intro.Wrap(1120)
        frow = wx.WrapSizer()
        self.find = wx.SearchCtrl(p, size=(240, -1))
        self.find.ShowCancelButton(True)
        self.find.SetDescriptiveText("Find: L1, 10u, VBUS…")
        self.find.SetToolTip("Show only the parts whose name, value, pads or nets contain one of these words "
                             "(case ignored)")
        self.find.Bind(wx.EVT_TEXT, lambda e: self._fill())
        self.find.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN, lambda e: self.find.SetValue(""))
        self.show_auto = wx.CheckBox(p, label="Show the board's resistors")
        self.show_auto.SetValue(True)
        self.show_auto.SetToolTip("List the resistor footprints found on the board, not only the parts added by "
                                  "hand")
        self.show_auto.Bind(wx.EVT_CHECKBOX, lambda e: self._fill())
        self.count = wx.StaticText(p, label="")
        for w in (self.find, self.show_auto, self.count):
            frow.Add(w, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
        top.Add(frow, 0, wx.LEFT | wx.RIGHT, 8)
        self.cols = ["Name", "Kind", "Value", "Real model", "Pads", "Nets", "Source"] + \
            [CO.TOOLS[t] for t in SWITCHED] + [CO.TOOLS["loop"], CO.TOOLS["pdn"]]
        self.list = AutoList(p, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        for k, c in enumerate(self.cols):
            self.list.InsertColumn(k, c, width=90)
        self.list.SetToolTip("The board's shared components. The tool columns show where each part is switched "
                             "on (✓); 'model' under Switching Loop or PDN: its capacitor model was edited "
                             "there (those tools switch their capacitors themselves). Double-click to edit; Shift / Ctrl-click to "
                             "select several")
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit())
        top.Add(self.list, 1, wx.EXPAND | wx.ALL, 8)
        brow = wx.WrapSizer()
        buttons = [("Add…", self._add, "Add a part between two pads: resistance, diode, capacitor or inductor "
                                       "(switched on in DC & Thermal and Trace)"),
                   ("Add regulator…", self._add_reg, "Add a linear or switching regulator (IN, OUT, GND pads) for "
                                                    "DC & Thermal"),
                   ("Edit…", self._edit, "Change the selected part (in every tool)"),
                   ("Delete", self._delete, "Delete the selected parts from every tool (resistors found on the "
                                            "board are switched off instead)")]
        buttons += [(f"On / off in {CO.TOOLS[t]}", lambda t=t: self._switch(t),
                     f"Switch the selected parts on or off in {CO.TOOLS[t]} only") for t in SWITCHED]
        for text, fn, tip in buttons:
            b = wx.Button(p, label=text.replace("&", "&&"))
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            brow.Add(b, 0, wx.RIGHT | wx.BOTTOM, 4)
        top.Add(brow, 0, wx.LEFT | wx.RIGHT, 8)
        self.status = wx.StaticText(p, label="")
        top.Add(self.status, 0, wx.ALL, 8)
        p.SetSizer(top)
        from .suite import attach_suite_menu
        attach_suite_menu(self, session, "components")
        self._rows = []
        self.probe = CrossProbe(self, session)
        self.probe.watch(self.list, lambda lc: [pid.rsplit(".", 1)[0] for e in self._sel() for pid in e.pads()])
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._fill()

    def _on_close(self, e):
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if what == "board":
            self.board = self.session.board
        if what in ("board", "components"):
            wx.CallAfter(self._fill)

    def _status(self, text):
        self.status.SetLabel(text)

    # ---- list --------------------------------------------------------------
    def _items(self):
        return self.session.components.items

    def _sel(self):
        out, k = [], self.list.GetFirstSelected()
        while k >= 0:
            if k < len(self._rows):
                out.append(self._rows[k])
            k = self.list.GetNextSelected(k)
        return out

    def _fill(self):
        if not self:
            return
        keep = {e.name for e in self._sel()}
        self.list.DeleteAllItems()
        nets = {p.id: p.net for p in self.board.pads} if self.board else {}
        words = [w for w in re.split(r"[\s,;]+", self.find.GetValue().strip().lower()) if w]
        show_auto = self.show_auto.GetValue()
        self._rows = []
        items = sorted(self._items(), key=lambda e: EL.natural_key(e.name))
        for e in items:
            if e.auto and not show_auto:
                continue
            real = e.part.get("name", "") if e.part else ""
            if e.part and e.model == "C":
                real = f"{real} (ESL {fmt_si(e.part.get('esl', 0), 'H')})" if real else \
                    f"ESL {fmt_si(e.part.get('esl', 0), 'H')}"
            pads = ", ".join(e.pads())
            text = [e.name, KIND.get(e.model, e.model), e.describe(), real, pads,
                    " – ".join(nets.get(pid, "?") or "no net" for pid in e.pads()), "board" if e.auto else "added"] + \
                ["✓" if CO.used(e, t) else "–" for t in SWITCHED] + \
                ["model" if CO.used(e, t) else "" for t in ("loop", "pdn")]
            if words and not any(w in " ".join(text[:6]).lower() for w in words):
                continue
            i = self.list.Append(text)
            self._rows.append(e)
            if e.name in keep:
                self.list.Select(i)
        n = len(self._items())
        self.count.SetLabel(f"{len(self._rows)} of {n} shown" if len(self._rows) != n else f"{n} part(s)")
        self.count.GetParent().Layout()

    def _changed(self, msg=""):
        self.session.components_changed()
        self._fill()
        self._status(msg)

    # ---- actions -------------------------------------------------------------
    def _need_board(self):
        if self.board is None:
            wx.MessageBox("Load a board first.", "Components", parent=self)
            return False
        return True

    def _add(self):
        if not self._need_board():
            return
        from .app import ElementDialog
        with ElementDialog(self, self.board, EL.Element("", "", "", 0.0), models=("R", "V", "C", "L")) as d:
            if d.ShowModal() == wx.ID_OK and d.result:
                if self.session.components.get(d.result.name) is not None and wx.MessageBox(
                        f"{d.result.name} exists already. Replace it?", "Components", wx.YES_NO, self) != wx.YES:
                    return
                self.session.components.edit(d.result.name, d.result, use={t: True for t in SWITCHED})
                self._changed(f"Added {d.result.name}, switched on in DC & Thermal and Trace.")

    def _add_reg(self):
        if not self._need_board():
            return
        from .app import RegulatorDialog
        with RegulatorDialog(self, self.board, EL.Element("", "", "", 0.0, model="LDO")) as d:
            if d.ShowModal() == wx.ID_OK and d.result:
                self.session.components.edit(d.result.name, d.result, use={"dc": True})
                self._changed(f"Added {d.result.name}, switched on in DC & Thermal.")

    def _edit(self):
        sel = self._sel()
        if not sel or not self._need_board():
            return
        e = sel[0]
        from .app import ElementDialog, RegulatorDialog
        dlg = RegulatorDialog(self, self.board, e) if e.regulator else \
            ElementDialog(self, self.board, e, models=("R", "V", "C", "L"))
        with dlg as d:
            if d.ShowModal() == wx.ID_OK and d.result:
                others = [x for x in self.session.components.users(e)]
                if d.result.name != e.name and others and wx.MessageBox(
                        f"{e.name} is used by {', '.join(others)}. Renamed, it keeps its switches here, but a "
                        "Switching Loop capacitor model is found by the part's reference. Rename it?",
                        "Components", wx.YES_NO, self) != wx.YES:
                    return
                self.session.components.edit(e.name, d.result)
                self._changed(f"{d.result.name} changed in every tool.")

    def _delete(self):
        sel = self._sel()
        if not sel:
            return
        users = [f"{e.name} ({', '.join(self.session.components.users(e))})" for e in sel
                 if not e.auto and self.session.components.users(e)]
        if users and wx.MessageBox("These parts are in use and will be deleted from every tool:\n"
                                   + "\n".join(users) + "\n\nDelete?", "Components", wx.YES_NO, self) != wx.YES:
            return
        kept = self.session.components.remove([e.name for e in sel])
        if kept:
            for t in SWITCHED:
                self.session.components.switch(kept, t, False)
        self._changed(f"Resistors found on the board are switched off instead of deleted: {', '.join(kept)}."
                      if kept else "Deleted.")

    def _switch(self, tool):
        sel = self._sel()
        if not sel:
            return
        on = self.session.components.switch([e.name for e in sel], tool)
        self._changed(f"{', '.join(e.name for e in sel)} switched {'on' if on else 'off'} in {CO.TOOLS[tool]}.")
