"""Switching Loop (wx): hot-loop and gate-loop parasitics of a switching cell, from the layout."""
import copy
import math
from dataclasses import asdict
import os
import threading
import traceback

import numpy as np
import wx

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  # noqa: E402
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg  # noqa: E402
import matplotlib.patches  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import elements as EL  # noqa: E402
from .. import scenario as SC  # noqa: E402
from ..loop import analysis as A  # noqa: E402
from ..loop import model as M  # noqa: E402
from ..suite.library import fmt_si, parse_si  # noqa: E402
from ..loop import advice as ADV  # noqa: E402
from ..loop import checks as CK  # noqa: E402
from ..loop import saved as SV  # noqa: E402
from ..loop import transient as TR  # noqa: E402
from .cellview import CellCanvas  # noqa: E402
from .crossprobe import CrossProbe, selected_rows  # noqa: E402
from .layout import refit, scrolled  # noqa: E402
from .suite import attach_suite_menu  # noqa: E402

ROLES = {"upper": "Upper switch", "lower": "Lower switch", "cap": "Hot-loop cap", "gate_r": "Gate resistor",
         "link": "Rail link (0 Ω)"}
ADD_ROLES = {**ROLES, "upper_ic": "Internal switch, upper (inside an IC)",
             "lower_ic": "Internal switch, lower (inside an IC)"}
KIND_LABEL = {"fet": "MOSFET", "diode": "Diode", "cap": "Capacitor", "res": "Resistor"}
PIN_LABEL = {"D": "Drain", "S": "Source", "G": "Gate", "A": "Anode", "K": "Cathode", "1": "Pin 1", "2": "Pin 2"}
OPTS = (("v_rail", "Voltage across the switches [V]", float,
         "The voltage across the switch that is off: the supply side's voltage (VIN for a buck, VOUT for a "
         "boost, the bus for a half-bridge). It sets the switches' and diodes' capacitance "
         "(C_oss, C_rss, C_j scaled from the datasheet's test voltage; not for capacitances entered with "
         "'at V_DS' = 0) and the hot-loop capacitors' DC-bias derating. It does not change any inductance."),
        ("margin", "Margin around the parts [mm]", float, "Copper this far beyond the parts' pads is meshed"),
        ("cells", "Mesh cells", int, "Target number of copper cells (sets the pitch); more = finer and slower"),
        ("f_min", "f min [Hz]", "si", "Start of the impedance sweep"),
        ("f_max", "f max [Hz]", "si", "End of the impedance sweep"),
        ("i_switch", "Inductor current [A]", float,
         "The inductor's (mean) current at the switching edges: the current the switches hand over. A buck: "
         "the load current; a boost: the input current (load current x V_out / V_in). Needed for the switching "
         "simulation, di/dt and the overshoot (0 = not given)"),
        ("i_ripple", "Current ripple p-p [A]", float,
         "Peak-to-peak inductor ripple: the switch turns on at I - ripple/2 and off at I + ripple/2 "
         "(0 = both edges at the mean current)"),
        ("f_sw", "Switching frequency [Hz]", "si",
         "The converter's switching frequency: switching loss = (E_on + E_off) x f_sw"))
MAIN_OPTS = ("v_rail", "i_switch", "i_ripple", "f_sw")      # the rest sit under "Advanced"
DIRS = (("auto", "Auto"), ("out", "Out of the switch node (buck, bridge sourcing)"),
        ("in", "Into the switch node (boost, bridge sinking)"))
# model fields per kind: (key, label, unit)
FIELDS = {
    # (None, "Title", None) starts a section
    "fet": ((None, "Part", None), ("name", "Part", ""),
            (None, "On-state", None), ("rds_on", "R_DS(on)", "Ω"), ("v_gs_on", "at V_GS", "V"),
            (None, "Capacitances", None), ("ciss", "C_iss", "F"), ("coss", "C_oss", "F"),
            ("crss", "C_rss", "F"), ("v_ref", "at V_DS (0 = as entered)", "V"),
            (None, "Gate", None), ("vth", "V_GS(th)", "V"), ("gfs", "g_fs", "S"), ("i_gfs", "g_fs at I_D", "A"),
            ("rg", "R_G (internal)", "Ω"),
            (None, "Body diode", None), ("vsd", "V_SD", "V"), ("i_vsd", "V_SD at I_S", "A"), ("qrr", "Q_rr", "C"),
            ("i_qrr", "Q_rr at I_F", "A"),
            (None, "Package inductance", None), ("ld", "L drain", "H"), ("ls", "L source", "H"),
            ("lg", "L gate", "H"), ("lks", "L Kelvin pin", "H"),
            (None, "Switching times", None), ("t_rise", "Rise time t_r", "s"), ("t_fall", "Fall time t_f", "s"),
            ("t_d_on", "Turn-on delay", "s"), ("t_d_off", "Turn-off delay", "s")),
    "diode": ((None, "Part", None), ("name", "Part", ""),
              (None, "Forward", None), ("vf", "V_f", "V"), ("i_vf", "V_f at I_F", "A"), ("rd", "R_d (on)", "Ω"),
              (None, "Capacitance and recovery", None), ("cj", "C_j", "F"), ("v_ref", "at V_R", "V"),
              ("qrr", "Q_rr (0 = Schottky)", "C"), ("i_qrr", "Q_rr at I_F", "A"),
              (None, "Package", None), ("l_pkg", "L (package)", "H")),
    "cap": (("name", "Part", ""), ("value", "C", "F"), ("esr", "ESR", "Ω"), ("esl", "ESL", "H"),
            ("v_rated", "Rated V", "V"), ("bias_loss", "C lost at rated V (0-1)", "")),
    "res": (("name", "Part", ""), ("r", "R", "Ω"), ("ls", "L (package)", "H")),
}
# hover help for the model fields (same keys as FIELDS; the diode's and resistor's own meanings first)
FIELD_TIPS = {
    "name": "Part number or a note, shown in the Parts list (not used in the calculation)",
    "rds_on": "On-resistance of the switch (datasheet R_DS(on) at your gate voltage and temperature). "
              "Sets the loop's damping (Q of the ringing)",
    "ciss": "Input capacitance C_iss = C_gs + C_gd (datasheet, at the V_DS below). Sets the gate loop's ringing",
    "coss": "Output capacitance C_oss = C_ds + C_gd (datasheet, at the V_DS below). The off switch rings with "
            "this and the loop inductance",
    "crss": "Reverse transfer (Miller) capacitance C_rss = C_gd (datasheet, at the V_DS below). Couples the "
            "switch node's dv/dt into the gate",
    "v_ref": "The drain-source voltage the datasheet capacitances are given at; they are scaled as √(V_ref / V) "
             "to the rail voltage. 0 = use the values as entered (no scaling)",
    "rg": "Internal gate resistance of the FET (datasheet R_G), in series with the gate drive's resistors",
    "ld": "Package inductance of the drain pin(s), inside the part (about 0.1-0.5 nH for power SMD packages, "
          "several nH for TO-220)",
    "ls": "Package inductance of the source pin(s); it is common to the power and gate loops (common-source "
          "inductance) unless a Kelvin pin is used",
    "lg": "Package inductance of the gate pin, in the gate loop",
    "lks": "Package inductance of the Kelvin-source pin (only used when Kelvin pads are ticked)",
    "t_rise": "Drain voltage / current rise time from the datasheet (0 = unknown). With the switched current it "
              "gives di/dt and the overshoot",
    "t_fall": "Fall time from the datasheet (0 = unknown). The current falls in this time when this switch turns "
              "off: di/dt = I / t_f, overshoot = L_loop · di/dt",
    "t_d_on": "Turn-on delay from the datasheet (for reference: the simulation times the edge from the gate "
              "drive itself)",
    "t_d_off": "Turn-off delay from the datasheet (for reference: the simulation times the edge from the gate "
               "drive itself)",
    "v_gs_on": "The gate-source voltage R_DS(on) is given at (e.g. 10 V or 4.5 V); the channel's on-resistance "
               "scales with the gate overdrive above V_GS(th)",
    "vth": "Gate threshold voltage V_GS(th) (datasheet, typical). The switching simulation compares the off "
           "FET's gate bounce with it: a bounce above V_th turns it on by mistake (shoot-through)",
    "gfs": "Forward transconductance g_fs (datasheet, at the drain current below): how much drain current per volt "
           "of gate overdrive. With V_th it sets the Miller plateau and so the switching speed",
    "i_gfs": "The drain current g_fs is given at",
    "vsd": "Body-diode forward voltage V_SD (datasheet, at the current below)",
    "i_vsd": "The current V_SD is given at",
    "qrr": "Reverse-recovery charge Q_rr (datasheet, at the current below): charge the diode gives back when it is "
           "turned off. It adds a current spike and loss at turn-on of the other switch. 0 = none (Schottky)",
    "i_qrr": "The forward current Q_rr is given at (stored charge is taken as proportional to it)",
    "i_vf": "The current V_f is given at",
    "cj": "Junction capacitance of the diode at the reverse voltage below (datasheet C_j or C_T). The diode "
          "rings with this when it blocks",
    "rd": "Dynamic (on) resistance of the conducting diode",
    "vf": "Forward voltage at the current below (datasheet). Used by the switching simulation; it does not change "
          "the inductance or the ringing",
    "l_pkg": "Total package inductance of the diode (both leads), about 1-3 nH",
    "value": "Capacitance (nominal, before DC-bias derating)",
    "esr": "Equivalent series resistance at high frequency (damps the hot loop)",
    "esl": "Equivalent series inductance of the capacitor body (about 0.4 / 0.6 / 0.8 / 1 nH for 0402 / 0603 / "
           "0805 / 1206); the mounting (pads, vias) comes from the copper",
    "v_rated": "Rated voltage, used for the DC-bias derating at the rail voltage",
    "bias_loss": "Fraction of C lost at the rated voltage (X7R / X5R: 0.5-0.8; C0G and film: 0); scaled to the "
                 "rail voltage",
    "r": "Resistance of the part",
}
FIELD_TIPS_KIND = {("diode", "v_ref"): "The reverse voltage C_j is given at; scaled as √(V_ref / V) to the rail "
                                       "voltage. 0 = as entered",
                   ("res", "ls"): "Package inductance of the resistor body (about 0.3-1 nH for SMD)"}


def _num(v, unit):
    return fmt_si(v, unit) if unit else (f"{v:g}" if isinstance(v, float) else str(v))


def _parse(txt, unit):
    """A value as shown by _num ('15 V', '5 ns', '1.2 nH'): the field's own unit is dropped first."""
    t = txt.strip()
    if unit and unit not in ("Ω", "F", "H") and t.endswith(unit):
        t = t[:-len(unit)]
    return parse_si(t)


def drive_path(s, gd):
    """'U2 HO → R1 → gate · source → U2 VS' for the Gate drives list."""
    chain = [s.device(k).key for k in gd.chain if s.device(k)]
    return " → ".join([f"{gd.out[0]} {gd.out[1]}"] + chain + [f"{gd.fet} gate"]) + \
        f"  ·  {gd.fet} source → {gd.ret[0]} {gd.ret[1]}"


def pins_text(board, d):
    """'D SW, S GND (pads 1 2 3), G G1, Kelvin pad 4' for the Parts list."""
    out = []
    for pin, pads in d.pins(board).items():
        nums = sorted({board.pads[k].number for k in pads}, key=EL.natural_key)
        if pin == "KS":
            out.append("Kelvin pad " + " ".join(nums))
            continue
        txt = f"{pin} {d.nets[pin]}"
        if len(nums) > 1:
            txt += f" (pads {' '.join(nums)})"
        out.append(txt)
    return ", ".join(out)


class ModelDialog(wx.Dialog):
    """A part's pins (which net each pin is on, N / P channel, Kelvin pads) and its model: datasheet
    values (SI prefixes accepted), or a library capacitor."""

    def __init__(self, parent, dev, library, board, store=None):
        kind = "Internal switch" if dev.internal else KIND_LABEL[dev.kind]
        super().__init__(parent, title=f"{dev.key}: {kind}", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.dev, self.lib, self.board, self.result = dev, library, board, None
        self.store = store
        top = wx.BoxSizer(wx.VERTICAL)
        cols = wx.BoxSizer(wx.HORIZONTAL)
        # pins
        pb = wx.StaticBoxSizer(wx.VERTICAL, self, "Pins (by net: all pads of the part on a net are one pin)")
        pbox = pb.GetStaticBox()
        pg = wx.FlexGridSizer(0, 2, 5, 8)
        self.fp_nets = sorted({p.net for p in board.pads if p.ref == dev.ref and p.net})
        self.pin_ch = {}
        if dev.kind == "fet" and not dev.internal:
            self.channel = wx.Choice(pbox, choices=["N-channel", "P-channel"])
            self.channel.SetSelection(1 if dev.model.get("channel", "N") == "P" else 0)
            self.channel.SetToolTip("N-channel: drain at the higher voltage, current drain -> source when on. "
                                    "P-channel: source at the higher voltage. Changing it swaps the drain and "
                                    "source nets")
            self.channel.Bind(wx.EVT_CHOICE, lambda e: self._swap_ds())
            pg.AddMany([(wx.StaticText(pbox, label="Channel"), 0, wx.ALIGN_CENTER_VERTICAL), (self.channel, 1, wx.EXPAND)])
        for pin in dev.nets:
            if dev.internal and pin == "G":
                continue
            if dev.pads.get(pin):        # an element between two pads: its pins are those pads
                pg.AddMany([(wx.StaticText(pbox, label=PIN_LABEL.get(pin, pin)), 0, wx.ALIGN_CENTER_VERTICAL),
                            (wx.StaticText(pbox, label=f"pad {dev.pads[pin]} ({dev.nets[pin]})"), 0)])
                continue
            choices = (["(none)"] if pin == "G" else []) + self.fp_nets
            c = wx.Choice(pbox, choices=choices)
            c.SetToolTip(f"The net of the part's {PIN_LABEL.get(pin, pin).lower()}: all the part's pads on this net "
                         "form this pin" + (" ('(none)' = no gate network)" if pin == "G" else ""))
            cur = dev.nets.get(pin) or ("(none)" if pin == "G" else "")
            if cur in choices:
                c.SetSelection(choices.index(cur))
            self.pin_ch[pin] = (c, choices)
            pg.AddMany([(wx.StaticText(pbox, label=PIN_LABEL.get(pin, pin)), 0, wx.ALIGN_CENTER_VERTICAL),
                        (c, 1, wx.EXPAND)])
        pb.Add(pg, 0, wx.ALL, 6)
        self.kel = None
        if dev.kind == "fet" and not dev.internal:
            pb.Add(wx.StaticText(pbox, label="Kelvin-source pads (tick the pad numbers of a separate\n"
                                             "source pin for the gate driver's return):"), 0, wx.LEFT | wx.TOP, 6)
            self.kel = wx.CheckListBox(pbox, size=(-1, 90))
            self.kel.SetToolTip("Pads on the source net that only carry the gate driver's return (a Kelvin-source "
                                "pin). Ticked pads are left out of the power loop, which lowers the common-source "
                                "inductance. Leave all unticked if the part has no Kelvin pin")
            pb.Add(self.kel, 1, wx.EXPAND | wx.ALL, 6)
            self.pin_ch["S"][0].Bind(wx.EVT_CHOICE, lambda e: self._fill_kelvin())
            self._fill_kelvin(initial=True)
        if dev.internal:
            pb.Add(wx.StaticText(pbox, label="A switch inside the IC: its drain and source are\n"
                                             "the IC's pins (all pins on each net); no gate pin."), 0, wx.ALL, 6)
        cols.Add(pb, 0, wx.EXPAND | wx.ALL, 6)
        # model
        mb = wx.StaticBoxSizer(wx.VERTICAL, self, "Model")
        mbox = mb.GetStaticBox()
        g = wx.FlexGridSizer(0, 2, 4, 8)
        self.ctrls = {}
        if dev.kind in ("fet", "diode") and store:
            row = wx.BoxSizer(wx.HORIZONTAL)
            self.savc = wx.Choice(mbox, size=(230, -1))
            self.savc.SetToolTip("Models you saved before (by part number): pick one to fill the values below. A "
                                 "saved model is also applied automatically when Detect parts finds a part with that "
                                 "value on any board")
            self.savc.Bind(wx.EVT_CHOICE, lambda e: self._from_saved())
            b = wx.Button(mbox, label="Save as…", style=wx.BU_EXACTFIT)
            b.SetToolTip("Save these values under a part number (the footprint's value is suggested), so the next "
                         "board with this part gets them automatically")
            b.Bind(wx.EVT_BUTTON, lambda e: self._save_model())
            row.AddMany([(wx.StaticText(mbox, label="Saved models"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6),
                         (self.savc, 0), (b, 0, wx.LEFT, 6)])
            mb.Add(row, 0, wx.ALL, 6)
            self._fill_saved()
        if dev.kind == "cap":
            self.libc = wx.Choice(mbox, choices=["(values below)"] + [p.name for p in library.by_kind("C")])
            self.libc.SetSelection(0)
            self.libc.SetToolTip("Copy C, ESR, ESL and the DC-bias data from a part of the component library")
            self.libc.Bind(wx.EVT_CHOICE, lambda e: self._from_lib())
            g.AddMany([(wx.StaticText(mbox, label="Library part"), 0, wx.ALIGN_CENTER_VERTICAL), (self.libc, 1, wx.EXPAND)])
        skip = {"ciss", "crss", "rg", "lg", "lks", "vth", "gfs", "i_gfs", "v_gs_on"} if dev.internal else set()
        defaults = asdict(M.FetModel() if dev.kind == "fet" else M.DiodeModel()) if dev.kind in ("fet", "diode") \
            else {}
        cur = ("", [])                  # fields before any section title
        sections = [cur]
        for key, label, unit in FIELDS[dev.kind]:
            if key is None:
                cur = (label, [])
                sections.append(cur)
            elif key not in skip:
                cur[1].append((key, label, unit))
        sections = [x for x in sections if x[1]]
        mb.Add(g, 0, wx.ALL, 6)
        colsz = [wx.BoxSizer(wx.VERTICAL), wx.BoxSizer(wx.VERTICAL)]
        rows = [0, 0]
        total = sum(len(f) + 1 for _, f in sections)
        for title, flds in sections:
            k = 0 if rows[0] < total / 2 or dev.kind not in ("fet", "diode") else 1     # fill the left first
            if title:
                head = wx.StaticText(mbox, label=title)
                head.SetFont(head.GetFont().Bold())
                colsz[k].Add(head, 0, wx.TOP | wx.BOTTOM, 4)
            sg = wx.FlexGridSizer(0, 2, 4, 8)
            for key, label, unit in flds:
                v = dev.model.get(key, defaults.get(key, ""))
                t = wx.TextCtrl(mbox, value=_num(v, unit) if v != "" else "", size=(130, -1))
                tip = FIELD_TIPS_KIND.get((dev.kind, key)) or FIELD_TIPS.get(key, "")
                t.SetToolTip(tip + (f" [{unit}]; SI prefixes allowed (5n, 2.2m)" if unit else ""))
                self.ctrls[key] = (t, unit)
                sg.AddMany([(wx.StaticText(mbox, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t, 1, wx.EXPAND)])
            colsz[k].Add(sg, 0, wx.LEFT, 8)
            rows[k] += len(flds) + 1
        two = wx.BoxSizer(wx.HORIZONTAL)
        two.Add(colsz[0], 0, wx.ALL, 6)
        two.Add(colsz[1], 0, wx.ALL, 6)
        mb.Add(two, 0)
        cols.Add(mb, 1, wx.EXPAND | wx.ALL, 6)
        top.Add(cols, 1, wx.EXPAND)
        self.spf = None
        if dev.kind == "fet" and not dev.internal:
            vb = wx.StaticBoxSizer(wx.VERTICAL, self, "Vendor SPICE model (optional: simulated with ngspice)")
            vbox = vb.GetStaticBox()
            vg = wx.FlexGridSizer(0, 2, 4, 8)
            self.spf = wx.FilePickerCtrl(vbox, path=dev.model.get("spice_file", ""), message="SPICE model file",
                                         wildcard="SPICE models (*.lib;*.mod;*.cir;*.sp;*.spi;*.txt)|"
                                                  "*.lib;*.mod;*.cir;*.sp;*.spi;*.txt;*.LIB;*.MOD|All files|*",
                                         style=wx.FLP_OPEN | wx.FLP_USE_TEXTCTRL | wx.FLP_FILE_MUST_EXIST)
            tip = ("The manufacturer's SPICE model file of this MOSFET (from its product page). With a model the "
                   "switching simulation runs in ngspice (bundled with KiCad) and uses it instead of the datasheet "
                   "values above. Empty = datasheet values")
            self.spf.SetToolTip(tip)
            for w in self.spf.GetChildren():
                w.SetToolTip(tip)
            self.sps = wx.Choice(vbox, size=(220, -1))
            self.sps.SetToolTip("The subcircuit (.SUBCKT) in the file to use: the part number")
            self.spp = wx.TextCtrl(vbox, value=dev.model.get("spice_pins", ""), size=(220, -1))
            self.spp.SetToolTip("The subcircuit's pins in order, as D (drain), G (gate), S (source), T (a temperature "
                                "pin, held at 25 °C) or NC, e.g. 'D G S' or 'D G S T'. Guessed from the pin names; "
                                "check the comment lines above .SUBCKT in the file")
            self.spk = wx.CheckBox(vbox, label="The model includes the package (lead inductance and R_G)")
            self.spk.SetToolTip("Ticked: the package inductances and R_G above are left out (the model has its "
                                "own). Untick for a bare-die model")
            self.spk.SetValue(bool(dev.model.get("spice_pkg", True)))
            self.spf.Bind(wx.EVT_FILEPICKER_CHANGED, lambda e: self._spice_file())
            self.sps.Bind(wx.EVT_CHOICE, lambda e: self._spice_sub())
            vg.AddMany([(wx.StaticText(vbox, label="File"), 0, wx.ALIGN_CENTER_VERTICAL), (self.spf, 1, wx.EXPAND),
                        (wx.StaticText(vbox, label="Subcircuit"), 0, wx.ALIGN_CENTER_VERTICAL), (self.sps, 0),
                        (wx.StaticText(vbox, label="Pin order"), 0, wx.ALIGN_CENTER_VERTICAL), (self.spp, 0)])
            vg.AddGrowableCol(1)
            vb.Add(vg, 0, wx.EXPAND | wx.ALL, 4)
            vb.Add(self.spk, 0, wx.ALL, 4)
            top.Add(vb, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
            self._spice_file(initial=True)
        note = {"fet": "From the datasheet: capacitances at the stated V_DS (scaled as 1/√V to the rail); V_GS(th),\n"
                       "g_fs and the body diode's Q_rr drive the switching simulation. Package L: about 0.1-0.5 nH\n"
                       "per pin for power packages, more for TO-220.",
                "diode": "C_j at the stated reverse voltage (scaled to the rail); package L about 1-3 nH.\n"
                         "V_f is kept for the switching transient (it does not change L or the ringing).",
                "cap": "ESL is about 0.4 / 0.6 / 0.8 / 1 nH for 0402 / 0603 / 0805 / 1206 ceramics.",
                "res": "A rail link: a 0 Ω resistor, jumper or net tie carrying the rail to the caps' net."
                       if dev.role == "link" else "The gate resistor in the gate loop."}[dev.kind]
        if dev.internal:
            note = ("From the regulator's datasheet: R_DS(on) of the switch, the switch-node capacitance if given\n"
                    "(C_oss; 'at V_DS' 0 = used as entered, no voltage scaling; C_oss 0 = left out), rise / fall\n"
                    "times and delays. Package L: pin + bond wire, about 0.3-1 nH per pin group.")
        top.Add(wx.StaticText(self, label=note), 0, wx.LEFT | wx.RIGHT, 12)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)

    def _fill_saved(self):
        self._saved = {k: v for k, v in SV.load(self.store).items() if v.get("kind") == self.dev.kind}
        names = sorted(self._saved, key=str.lower)
        self.savc.Set(["(choose a saved model)"] + names)
        self.savc.SetSelection(0)

    def _from_saved(self):
        k = self.savc.GetSelection()
        if k <= 0:
            return
        m = self._saved[self.savc.GetString(k)]["model"]
        for key, (t, unit) in self.ctrls.items():
            if key in m:
                t.SetValue(_num(m[key], unit))
        if self.spf is not None:
            self.spf.SetPath(m.get("spice_file", ""))
            self._spice_file()
            names = [self.sps.GetString(i) for i in range(self.sps.GetCount())]
            if m.get("spice_subckt") in names:
                self.sps.SetSelection(names.index(m["spice_subckt"]))
            self.spp.SetValue(m.get("spice_pins", self.spp.GetValue()))
            self.spk.SetValue(bool(m.get("spice_pkg", True)))

    def _save_model(self):
        key = SV.key_of(self.board, self.dev)
        with wx.TextEntryDialog(self, "Part number to save these values under:", "Save model", key) as dlg:
            if dlg.ShowModal() != wx.ID_OK or not dlg.GetValue().strip():
                return
            key = dlg.GetValue().strip()
        m = {}
        try:
            for k_, (t, unit) in self.ctrls.items():
                txt = t.GetValue().strip()
                m[k_] = txt if k_ == "name" else _parse(txt, unit)
        except ValueError as ex:
            return wx.MessageBox(str(ex), "Value", parent=self)
        if self.dev.kind == "fet" and not self.dev.internal:
            m["channel"] = "P" if self.channel.GetSelection() == 1 else "N"
        if self.spf is not None and self.spf.GetPath().strip() and self.sps.GetSelection() >= 0:
            m.update(spice_file=self.spf.GetPath().strip(), spice_subckt=self.sps.GetString(self.sps.GetSelection()),
                     spice_pins=self.spp.GetValue().strip(), spice_pkg=self.spk.GetValue())
        SV.save(self.store, key, self.dev.kind, m)
        self._fill_saved()
        wx.MessageBox(f"Saved as {key}: Detect parts now applies it to every {key} it finds.", "Save model",
                      parent=self)

    def _spice_file(self, initial=False):
        from ..loop import spice_tran as SP
        self._subs = SP.subckts(self.spf.GetPath()) if self.spf.GetPath() else {}
        names = list(self._subs)
        self.sps.Set(names)
        cur = self.dev.model.get("spice_subckt", "") if initial else ""
        if cur in names:
            self.sps.SetSelection(names.index(cur))
        elif names:
            self.sps.SetSelection(0)
            self._spice_sub()
        if self.spf.GetPath() and not names:
            self.spp.SetValue("")
            wx.CallAfter(wx.MessageBox, "No .SUBCKT found in that file.", "SPICE model", parent=self)

    def _spice_sub(self):
        from ..loop import spice_tran as SP
        k = self.sps.GetSelection()
        if k >= 0:
            self.spp.SetValue(SP.guess_pins(self._subs[self.sps.GetString(k)]))

    def _net(self, pin):
        c, choices = self.pin_ch[pin]
        k = c.GetSelection()
        v = choices[k] if k >= 0 else ""
        return "" if v == "(none)" else v

    def _swap_ds(self):
        """N <-> P: the source and drain change places in the cell."""
        (cd, chd), (cs, chs) = self.pin_ch["D"], self.pin_ch["S"]
        d, s_ = cd.GetSelection(), cs.GetSelection()
        cd.SetSelection(s_)
        cs.SetSelection(d)
        self._fill_kelvin()

    def _fill_kelvin(self, initial=False):
        if self.kel is None:
            return
        net = self._net("S")
        nums = sorted({p.number for p in self.board.pads if p.ref == self.dev.ref and p.net == net and p.number},
                      key=EL.natural_key)
        ticked = set(self.dev.kelvin) if initial else {self.kel.GetString(i) for i in self.kel.GetCheckedItems()}
        self.kel.Set(nums)
        self.kel.SetCheckedStrings([n for n in nums if n in ticked])

    def _from_lib(self):
        k = self.libc.GetSelection()
        if k <= 0:
            return
        p = self.lib.get(self.libc.GetString(k))
        for key, (t, unit) in self.ctrls.items():
            v = p.name if key == "name" else (p.esr_eff() if key == "esr" else p.esl_eff() if key == "esl"
                                               else getattr(p, key, ""))
            t.SetValue(_num(v, unit))

    def _ok(self, e):
        m = dict(self.dev.model)
        try:
            for key, (t, unit) in self.ctrls.items():
                txt = t.GetValue().strip()
                m[key] = txt if key == "name" else _parse(txt, unit)
        except ValueError as ex:
            return wx.MessageBox(str(ex), "Value", parent=self)
        nets = {pin: self._net(pin) for pin in self.pin_ch}
        nets.update({pin: self.dev.nets[pin] for pin in self.dev.pads if pin in self.dev.nets})
        if self.dev.internal and "G" in self.dev.nets:
            nets["G"] = ""
        need = [p for p in nets if p != "G" and not nets[p]]
        if need:
            return wx.MessageBox("Choose a net for " + ", ".join(PIN_LABEL.get(p, p) for p in need) + ".", "Pins",
                                 parent=self)
        kelvin = []
        if self.kel is not None:
            kelvin = [self.kel.GetString(i) for i in self.kel.GetCheckedItems()]
            left = [p for p in self.board.pads if p.ref == self.dev.ref and p.net == nets["S"]
                    and p.number not in kelvin]
            if kelvin and not left:
                return wx.MessageBox("Leave at least one pad as the power source pin.", "Kelvin", parent=self)
            m["channel"] = "P" if self.channel.GetSelection() == 1 else "N"
        if self.dev.kind in ("fet", "diode"):
            m["assumed"] = False
        if self.dev.kind == "cap":
            k = self.libc.GetSelection()
            if k > 0:
                from dataclasses import asdict
                p = self.lib.get(self.libc.GetString(k))
                m = {**asdict(p), "value": m["value"], "esr": m["esr"], "esl": m["esl"]}
                m.pop("builtin", None)
        if self.spf is not None:
            path = self.spf.GetPath().strip()
            k_ = self.sps.GetSelection()
            if path and k_ < 0:
                return wx.MessageBox("Choose the subcircuit in the SPICE file (or clear the file).", "SPICE model",
                                     parent=self)
            pins = self.spp.GetValue().split()
            if path and sorted(p for p in (x.upper() for x in pins) if p in ("D", "G", "S")) != ["D", "G", "S"]:
                return wx.MessageBox("The pin order needs D, G and S once each (e.g. 'D G S').", "SPICE model",
                                     parent=self)
            m.update(spice_file=path, spice_subckt=self.sps.GetString(k_) if path else "",
                     spice_pins=" ".join(pins), spice_pkg=self.spk.GetValue())
        self.result = (m, nets, kelvin)
        self.EndModal(wx.ID_OK)


class AddPartDialog(wx.Dialog):
    def __init__(self, parent, board, role=None):
        super().__init__(parent, title="Add a part")
        refs = sorted({f.ref for f in board.footprints if any(p.ref == f.ref and p.net for p in board.pads)},
                      key=EL.natural_key)
        self.refs = refs
        self.ref = wx.Choice(self, choices=refs)
        self.role = wx.Choice(self, choices=list(ADD_ROLES.values()))
        self.role.SetSelection(list(ADD_ROLES).index(role) if role in ADD_ROLES else 2)
        self.ref.SetToolTip("The footprint to add to the cell")
        self.role.SetToolTip("Upper / lower switch: the FET or diode between the rail and the switch node, or "
                             "between the switch node and the return. Hot-loop cap: a capacitor across rail and "
                             "return. Gate resistor: a resistor in a gate drive. Rail link: a 0 Ω resistor / "
                             "jumper that joins the rail to the net its caps are on. Internal switch: a switch "
                             "inside a regulator IC (the IC's pins are its drain and source)")
        g = wx.FlexGridSizer(0, 2, 6, 8)
        g.AddMany([(wx.StaticText(self, label="Part"), 0, wx.ALIGN_CENTER_VERTICAL), (self.ref, 1, wx.EXPAND),
                   (wx.StaticText(self, label="Role"), 0, wx.ALIGN_CENTER_VERTICAL), (self.role, 1, wx.EXPAND)])
        top = wx.BoxSizer(wx.VERTICAL)
        top.Add(g, 0, wx.ALL, 10)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)

    def value(self):
        k = self.ref.GetSelection()
        return (self.refs[k] if k >= 0 else None), list(ADD_ROLES)[self.role.GetSelection()]


class DriveDialog(wx.Dialog):
    """A FET's gate drive: the driver's output and return pins, its output resistance, and the series
    parts between the driver pin and the gate: resistors / beads on the board, or elements between
    any two pads (as in DC & Thermal).  Their order does not matter: the copper links them."""

    def __init__(self, parent, board, setup, drive, library):
        super().__init__(parent, title="Gate drive", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.board, self.setup, self.lib = board, setup, library
        self.drive = M.GateDrive(drive.fet, list(drive.out), list(drive.ret), list(drive.chain), drive.rout,
                                 drive.rout_off, drive.v_drive)
        self.parts = {k: copy.deepcopy(setup.device(k)) for k in drive.chain if setup.device(k)}
        self.result = None
        top = wx.BoxSizer(wx.VERTICAL)
        g = wx.FlexGridSizer(0, 2, 6, 8)
        fets = [d.key for d in setup.devices if d.kind == "fet" and not d.internal and d.nets.get("G")]
        self.fet = wx.Choice(self, choices=fets)
        if drive.fet in fets:
            self.fet.SetSelection(fets.index(drive.fet))
        self.fet.SetToolTip("The FET this drive switches (its gate is the end of the gate loop)")
        self.fets = fets
        self.refs = sorted({p.ref for p in board.pads if p.net}, key=EL.natural_key)
        self.pin_ctrls = []
        for label, cur in (("Driver output pin", self.drive.out), ("Driver return pin", self.drive.ret)):
            pc = wx.Choice(self, choices=self.refs, size=(90, -1))
            nc = wx.Choice(self, size=(200, -1))
            if label.startswith("Driver output"):
                pc.SetToolTip("The driver IC (or the part) whose output pin drives the gate, e.g. U2 for an "
                              "IR2104's HO / LO pin")
                nc.SetToolTip("The net of the driver's output pin (HO, LO, OUT…). The gate current leaves the "
                              "driver here and goes through the series parts to the gate")
            else:
                pc.SetToolTip("The part whose pin takes the gate current back: normally the same driver IC")
                nc.SetToolTip("The net of the driver's return pin: the FET's source net (VS / switch node for a "
                              "high-side driver, COM / GND for a low-side one). The gate loop closes from the "
                              "FET's source back to this pin")
            pc.Bind(wx.EVT_CHOICE, lambda e, pc=pc, nc=nc: self._nets(pc, nc))
            if cur and cur[0] in self.refs:
                pc.SetSelection(self.refs.index(cur[0]))
            self._nets(pc, nc, cur[1] if cur else None)
            row = wx.BoxSizer(wx.HORIZONTAL)
            row.AddMany([(pc, 0, wx.RIGHT, 6), (nc, 1)])
            g.AddMany([(wx.StaticText(self, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (row, 1, wx.EXPAND)])
            self.pin_ctrls.append((pc, nc))
        self.rout = wx.TextCtrl(self, value=fmt_si(self.drive.rout, "Ω"), size=(120, -1))
        self.rout.SetToolTip("The driver's own output resistance, inside the IC, in series with the gate loop. "
                             "From the datasheet: the output's pull-up / pull-down (source / sink) resistance, or "
                             "the supply voltage / peak output current (e.g. 12 V / 2 A = 6 Ω)")
        g.AddMany([(wx.StaticText(self, label="FET"), 0, wx.ALIGN_CENTER_VERTICAL), (self.fet, 1, wx.EXPAND)])
        self.rout_off = wx.TextCtrl(self, value=fmt_si(self.drive.rout_off, "Ω") if self.drive.rout_off else "",
                                    size=(120, -1))
        self.rout_off.SetToolTip("The driver's output resistance when it pulls the gate down (sink), if the "
                                 "datasheet gives a different value from the pull-up. Empty = the same as R out")
        self.vdrv = wx.TextCtrl(self, value=f"{self.drive.v_drive:g}", size=(120, -1))
        self.vdrv.SetToolTip("The driver's output swing: its supply voltage (VCC / VDD of a low-side driver, the "
                             "bootstrap voltage of a high-side one), e.g. 12 V, 10 V or 5 V. Used by the switching "
                             "simulation")
        g.Add(wx.StaticText(self, label="Driver R out (turning on)"), 0, wx.ALIGN_CENTER_VERTICAL)
        g.Add(self.rout, 0)
        g.Add(wx.StaticText(self, label="Driver R out (turning off)"), 0, wx.ALIGN_CENTER_VERTICAL)
        g.Add(self.rout_off, 0)
        g.Add(wx.StaticText(self, label="Drive voltage [V]"), 0, wx.ALIGN_CENTER_VERTICAL)
        g.Add(self.vdrv, 0)
        top.Add(g, 0, wx.EXPAND | wx.ALL, 10)
        sb = wx.StaticBoxSizer(wx.VERTICAL, self, "Series parts between the driver pin and the gate")
        sbox = sb.GetStaticBox()
        self.lst = wx.ListCtrl(sbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(560, 150))
        for k, (t, w) in enumerate((("Part", 80), ("Kind", 110), ("Between", 220), ("R", 70), ("L", 70))):
            self.lst.InsertColumn(k, t, width=w)
        self.lst.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit())
        self.lst.SetToolTip("Every part the gate current flows through between the driver pin and the gate "
                            "(double-click to edit)")
        sb.Add(self.lst, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.WrapSizer()
        for text, fn in (("Add resistor…", self._add_res), ("Add element between pads…", self._add_el),
                         ("Edit…", self._edit), ("Remove", self._remove), ("Find from the gate", self._find)):
            b = wx.Button(sbox, label=text, style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        sb.Add(row, 0)
        sb.Add(wx.StaticText(sbox, label="A resistor or bead footprint takes its value from the board; an element "
                                          "between two pads\nis any R you place between any pads. The order does "
                                          "not matter: the copper links them."), 0, wx.ALL, 4)
        top.Add(sb, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        top.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(top)
        self.Bind(wx.EVT_BUTTON, self._ok, id=wx.ID_OK)
        self._fill()

    def _nets(self, pc, nc, cur=None):
        k = pc.GetSelection()
        nets = sorted({p.net for p in self.board.pads if k >= 0 and p.ref == self.refs[k] and p.net})
        nc.Set(nets)
        if cur in nets:
            nc.SetSelection(nets.index(cur))
        elif nets:
            nc.SetSelection(0)

    def _fill(self):
        self.lst.DeleteAllItems()
        for k in self.drive.chain:
            d = self.parts.get(k)
            if d is None:
                continue
            between = (f"pad {d.pads['1']} – pad {d.pads['2']}" if d.pads else f"{d.nets.get('1')} – {d.nets.get('2')}")
            self.lst.Append([d.key, "element (pads)" if d.pads else "resistor / bead", between,
                             fmt_si(d.model.get("r", 0.0), "Ω"), fmt_si(d.model.get("ls", 0.0), "H")])

    def _put(self, d):
        self.parts[d.key] = d
        if d.key not in self.drive.chain:
            self.drive.chain.append(d.key)
        self._fill()

    def _add_res(self):
        two = sorted({f.ref for f in self.board.footprints if f.ref.startswith(M.SERIES_PREFIXES)
                      and len({p.net for p in self.board.pads if p.ref == f.ref and p.net}) == 2},
                     key=EL.natural_key)
        if not two:
            return wx.MessageBox("No two-pin resistor or bead footprints on the board.", "Add resistor", parent=self)
        with wx.SingleChoiceDialog(self, "Resistor or bead footprint (two pins):", "Add resistor", two) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            ref = two[dlg.GetSelection()]
        nets = sorted({p.net for p in self.board.pads if p.ref == ref and p.net})
        self._put(M.series_part(self.board, ref, nets[0], nets[1]))

    def _add_el(self, d=None):
        from .app import ElementDialog
        el = EL.Element(d.key if d else f"Rg{len(self.drive.chain) + 1}", d.pads["1"] if d else "",
                        d.pads["2"] if d else "", d.model.get("r", 0.0) if d else 0.0)
        with ElementDialog(self, self.board, el, models=("R",), dc=False) as dlg:
            if dlg.ShowModal() != wx.ID_OK or not dlg.result:
                return
            r = dlg.result
        if d is not None and d.key != r.name:
            self.parts.pop(d.key, None)
            self.drive.chain = [r.name if k == d.key else k for k in self.drive.chain]
        self._put(M.element_part(self.board, r.name, r.a, r.b, r.R, d.model.get("ls", 0.5e-9) if d else 0.5e-9))

    def _sel(self):
        i = self.lst.GetFirstSelected()
        return self.drive.chain[i] if 0 <= i < len(self.drive.chain) else None

    def _edit(self):
        k = self._sel()
        if k is None:
            return
        d = self.parts[k]
        if d.pads:
            return self._add_el(d)
        with ModelDialog(self, d, self.lib, self.board) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                d.model, d.nets, d.kelvin = dlg.result
                self._fill()

    def _remove(self):
        k = self._sel()
        if k is not None:
            self.drive.chain.remove(k)
            self.parts.pop(k, None)
            self._fill()

    def _find(self):
        k = self.fet.GetSelection()
        fet = self.setup.device(self.fets[k]) if k >= 0 else None
        if fet is None:
            return
        tmp = M.Setup(sw_net=self.setup.sw_net, rail_net=self.setup.rail_net, ret_net=self.setup.ret_net,
                      devices=[fet])
        g = M.drive_from_gate(self.board, tmp, fet)
        if g is None:
            return wx.MessageBox("No driver IC pin reached from the gate through resistors / beads.", "Find",
                                 parent=self)
        self.drive.out, self.drive.ret, self.drive.chain = g.out, g.ret, list(g.chain)
        self.parts = {d.key: d for d in tmp.devices if d.kind == "res"}
        for (pc, nc), cur in zip(self.pin_ctrls, (g.out, g.ret)):
            pc.SetSelection(self.refs.index(cur[0]))
            self._nets(pc, nc, cur[1])
        self._fill()

    def _ok(self, e):
        k = self.fet.GetSelection()
        if k < 0:
            return wx.MessageBox("Choose the FET.", "Gate drive", parent=self)
        pins = []
        for pc, nc in self.pin_ctrls:
            if pc.GetSelection() < 0 or nc.GetSelection() < 0:
                return wx.MessageBox("Choose the driver's output and return pins.", "Gate drive", parent=self)
            pins.append([self.refs[pc.GetSelection()], nc.GetString(nc.GetSelection())])
        try:
            rout = _parse(self.rout.GetValue(), "Ω")
            rout_off = _parse(self.rout_off.GetValue(), "Ω") if self.rout_off.GetValue().strip() else 0.0
            v_drive = float(self.vdrv.GetValue().replace("V", "").strip())
        except ValueError as ex:
            return wx.MessageBox(str(ex), "Gate drive", parent=self)
        self.drive.fet, self.drive.out, self.drive.ret, self.drive.rout = self.fets[k], pins[0], pins[1], rout
        self.drive.rout_off, self.drive.v_drive = rout_off, v_drive
        self.result = (self.drive, [self.parts[k_] for k_ in self.drive.chain if k_ in self.parts])
        self.EndModal(wx.ID_OK)


class LoopFrame(wx.Frame):
    def __init__(self, session):
        super().__init__(None, title="Switching Loop - PCB analysis suite", size=(1500, 960))
        self.session = session
        self.board = session.board
        self.results, self._running, self._run_ci = {}, False, 0
        self.cells, self.ci = self._load_cells()
        self._build()
        attach_suite_menu(self, session, "loop")
        self.probe = CrossProbe(self, session)
        self.probe.watch(self.plist, lambda lc: [self.setup.devices[self._rows[i]].ref for i in selected_rows(lc)
                                                 if i < len(self._rows)])
        self.probe.watch(self.glist, lambda lc: [r for i in selected_rows(lc) if i < len(self.setup.drives)
                                                 for r in self._drive_refs(self.setup.drives[i])])
        session.listen(self._on_session)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.CreateStatusBar()
        self._to_ui()
        if not self.setup.devices:
            self._status("No switch node found automatically: pick one and press Detect.")

    # ------------------------------------------------------------- persistence
    # a board may have several switching cells (a 4-switch buck-boost, a full bridge, several converters)
    @property
    def setup(self):
        return self.cells[self.ci]

    @setup.setter
    def setup(self, value):
        self.cells[self.ci] = value

    @property
    def res(self):
        return self.results.get(self.ci)

    def _load_cells(self):
        d = {}
        try:
            if self.session.scenario_path:
                d = SC.section(self.session.scenario_path, "loop")
        except (OSError, ValueError):
            d = {}
        if d.get("cells"):
            cells = [M.Setup.from_dict(c) for c in d["cells"]]
            return cells, min(max(int(d.get("current", 0)), 0), len(cells) - 1)
        if d.get("setup"):
            return [M.Setup.from_dict(d["setup"])], 0
        try:
            first = M.detect(self.board)
            self._apply_saved(first)
            return [first], 0
        except ValueError:
            return [M.Setup()], 0

    def save(self):
        if not self.session.scenario_path:
            return
        try:
            SC.save_section(self.session.scenario_path, "loop",
                            dict(cells=[c.to_dict() for c in self.cells], current=self.ci,
                                 setup=self.setup.to_dict()))
        except OSError as e:
            self._status(f"Could not save the setup: {e}")

    def _store(self):
        return SV.path_for(self.session.library.path)

    def _apply_saved(self, s):
        done = SV.apply(self.board, s, SV.load(self._store()))
        return f" Saved models applied to {', '.join(done)}." if done else ""

    def _cell_name(self, c):
        return c.name or c.sw_net or "(new cell)"

    def _on_close(self, e):
        self._from_ui(quiet=True)
        self.save()
        self.session.unlisten(self._on_session)
        e.Skip()

    def _on_session(self, what):
        if what == "board" and self.session.board is not None:
            self.board = self.session.board
            wx.CallAfter(self._to_ui)

    # ------------------------------------------------------------- UI
    def _box(self, parent, sizer, title, grow=0):
        """A titled box; grow > 0 gives it that share of the spare height."""
        b = wx.StaticBoxSizer(wx.VERTICAL, parent, title)
        sizer.Add(b, grow, wx.EXPAND | wx.ALL, 6)
        return b, b.GetStaticBox()

    @staticmethod
    def _fit_columns(lc, stretch=None):
        """Columns as wide as their header or widest entry; column `stretch` takes what is left."""
        n = lc.GetColumnCount()
        for c in range(n):
            lc.SetColumnWidth(c, wx.LIST_AUTOSIZE)
            w = lc.GetColumnWidth(c)
            lc.SetColumnWidth(c, wx.LIST_AUTOSIZE_USEHEADER)
            lc.SetColumnWidth(c, max(w, lc.GetColumnWidth(c)) + 6)
        if stretch is not None:     # the stretch column takes the rest of the width (narrower if need be)
            rest = lc.GetClientSize().width - sum(lc.GetColumnWidth(c) for c in range(n) if c != stretch)
            lc.SetColumnWidth(stretch, max(rest, 70))

    def _stretchy(self, lc, stretch):
        lc.Bind(wx.EVT_SIZE, lambda e: (e.Skip(), wx.CallAfter(self._fit_columns, lc, stretch)))

    def _buttons(self, box, sizer, items):
        row = wx.WrapSizer()
        for text, fn in items:
            b = wx.Button(box, label=text, style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 2)
        sizer.Add(row, 0)

    def _build(self):
        sp = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        left = self.left = scrolled(sp)
        ls = wx.BoxSizer(wx.VERTICAL)

        cb, cbox = self._box(left, ls, "Switching cell")
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.cellc = wx.Choice(cbox, size=(200, -1))
        self.cellc.SetToolTip("A board may have several switching cells (a 4-switch buck-boost has two switch "
                              "nodes, a full bridge two legs, a board several converters): each is analysed "
                              "on its own")
        self.cellc.Bind(wx.EVT_CHOICE, lambda e: self._switch_cell(self.cellc.GetSelection()))
        row.Add(wx.StaticText(cbox, label="Cell"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        row.Add(self.cellc, 1, wx.RIGHT, 4)
        for text, fn in (("New cell…", self._new_cell), ("Remove", self._del_cell)):
            b = wx.Button(cbox, label=text, style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.LEFT, 2)
        cb.Add(row, 0, wx.EXPAND | wx.ALL, 4)
        self.canvas = CellCanvas(cbox, self.board, self._action)
        cb.Add(self.canvas, 0, wx.EXPAND | wx.ALL, 4)
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.sw = wx.Choice(cbox, size=(230, -1))
        self.sw.SetToolTip("The switch-node net: the node between the two switches and the inductor. Changing "
                           "it detects the cell's parts again")
        self.topo = wx.Choice(cbox, choices=list(M.TOPOLOGIES.values()))
        self.topo.SetToolTip("The converter type; it names the switch positions (which part is the upper / lower "
                             "switch)")
        self.railc = wx.Choice(cbox, size=(230, -1))
        self.retc = wx.Choice(cbox, size=(230, -1))
        self.railc.SetToolTip("The supply side of the hot loop (also called the rail): the net the loop closes "
                              "through opposite the return. VIN for a buck, VOUT for a boost, the bus for a "
                              "half-bridge. Changing it moves the switches' pins and searches the caps again")
        self.retc.SetToolTip("The return net of the hot loop (ground)")
        self.railc.Bind(wx.EVT_CHOICE, lambda e: self._cell_nets())
        self.retc.Bind(wx.EVT_CHOICE, lambda e: self._cell_nets())
        self.cell_info = wx.StaticText(cbox, label="")
        g.AddMany([(wx.StaticText(cbox, label="Switch node"), 0, wx.ALIGN_CENTER_VERTICAL), (self.sw, 1, wx.EXPAND),
                   (wx.StaticText(cbox, label="Supply side"), 0, wx.ALIGN_CENTER_VERTICAL), (self.railc, 1, wx.EXPAND),
                   (wx.StaticText(cbox, label="Return"), 0, wx.ALIGN_CENTER_VERTICAL), (self.retc, 1, wx.EXPAND),
                   (wx.StaticText(cbox, label="Topology"), 0, wx.ALIGN_CENTER_VERTICAL), (self.topo, 1, wx.EXPAND)])
        cb.Add(g, 0, wx.ALL, 4)
        cb.Add(self.cell_info, 0, wx.ALL, 4)
        items = [("Detect parts", self._detect)]
        if self.session.kicad is not None:
            items.append(("Switch node from KiCad selection", self._from_kicad))
        self._buttons(cbox, cb, items)

        pb, pbox = self._box(left, ls, "Parts", grow=3)
        self.plist = wx.ListCtrl(pbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 170))
        for k, (t, w) in enumerate((("On", 32), ("Role", 100), ("Part", 60), ("Kind", 75), ("Pins", 200),
                                    ("Model", 200), ("mm", 40))):
            self.plist.InsertColumn(k, t, width=w)
        self.plist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit())
        self.plist.Bind(wx.EVT_LIST_KEY_DOWN, lambda e: self._toggle() if e.GetKeyCode() == wx.WXK_SPACE else e.Skip())
        self.plist.SetToolTip("Shift / Ctrl-click to select several parts; On/off (or Space) switches them all. "
                              "mm: distance from the middle of the switches")
        self.plist.SetMinSize((200, 150))
        self._stretchy(self.plist, 5)
        pb.Add(self.plist, 1, wx.EXPAND | wx.ALL, 4)
        self._buttons(pbox, pb, [("Edit model…", self._edit), ("On/off", self._toggle), ("Add…", self._add),
                                 ("Remove", self._remove)])

        gb, gbox = self._box(left, ls, "Gate drives", grow=1)
        self.glist = wx.ListCtrl(gbox, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 90))
        for k, (t, w) in enumerate((("FET", 45), ("Gate loop", 300), ("Σ R", 65), ("R out", 65))):
            self.glist.InsertColumn(k, t, width=w)
        self.glist.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._edit_drive())
        self.glist.SetToolTip("One row per FET gate: driver output pin -> series parts -> gate, back from the "
                              "source to the driver's return pin. Double-click to edit")
        self.glist.SetMinSize((200, 85))
        self._stretchy(self.glist, 1)
        gb.Add(self.glist, 1, wx.EXPAND | wx.ALL, 4)
        self._buttons(gbox, gb, [("Edit drive…", self._edit_drive), ("Add drive…", self._add_drive),
                                 ("Remove", self._del_drive)])

        ob, obox = self._box(left, ls, "Operating point")
        g = wx.FlexGridSizer(0, 2, 4, 6)
        self.fields = {}
        adv = wx.CollapsiblePane(obox, label="Advanced (mesh and frequency range)",
                                 style=wx.CP_DEFAULT_STYLE | wx.CP_NO_TLW_RESIZE)
        adv.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, lambda e: self._relayout())
        self.adv = adv
        ap = adv.GetPane()
        ag = wx.FlexGridSizer(0, 2, 4, 6)
        for key, label, kind, tip in OPTS:
            main = key in MAIN_OPTS
            par, grid = (obox, g) if main else (ap, ag)
            t = wx.TextCtrl(par, size=(110, -1))
            t.SetToolTip(tip)
            t.Bind(wx.EVT_KILL_FOCUS, lambda e: (e.Skip(), wx.CallAfter(self._opts_changed)))
            self.fields[key] = (t, kind)
            grid.AddMany([(wx.StaticText(par, label=label), 0, wx.ALIGN_CENTER_VERTICAL), (t, 0)])
        self.dirc = wx.Choice(obox, choices=[t for _, t in DIRS])
        self.dirc.SetToolTip("Which way the inductor current flows at the switch node. Out (a buck, a bridge leg "
                             "sourcing current): the upper switch is the hard-switched one. In (a boost, a bridge "
                             "sinking): the lower. Auto decides from the topology, the diodes, or where the "
                             "feedback divider is")
        self.dirc.Bind(wx.EVT_CHOICE, lambda e: self._opts_changed())
        g.AddMany([(wx.StaticText(obox, label="Inductor current flows"), 0, wx.ALIGN_CENTER_VERTICAL),
                   (self.dirc, 0)])
        ob.Add(g, 0, wx.ALL, 4)
        self.tran = wx.CheckBox(obox, label="Simulate the switching edges (turn-on and turn-off)")
        self.tran.SetToolTip("After the inductance analysis, simulate one turn-on and one turn-off of the hard-"
                             "switched switch in time: waveforms, overshoot, ringing, dv/dt, di/dt, switching "
                             "loss and how far the other FET's gate is pushed. Needs the inductor current")
        self.tran.Bind(wx.EVT_CHECKBOX, lambda e: self._opts_changed())
        ob.Add(self.tran, 0, wx.ALL, 4)
        self.adaptive = wx.CheckBox(ap, label="Adaptive mesh: fine where the loop current crowds")
        self.adaptive.SetToolTip("Two passes: a coarse solve finds where the loop's energy sits; there (and at "
                                 "the parts' pads) the cells are fine, 2x and 4x coarser further away, for the "
                                 "same number of cells")
        self.simc = wx.Choice(ap, choices=["Auto (ngspice when a FET has a vendor model)", "Built-in",
                                           "ngspice (bundled with KiCad)"])
        self.simc.SetToolTip("The simulator of the switching edges. Built-in: fast, datasheet models. ngspice: "
                             "needed for vendor SPICE models; with datasheet models it gives the same results "
                             "(same equations). Auto picks ngspice when a FET has a vendor model")
        self.simc.Bind(wx.EVT_CHOICE, lambda e: self._opts_changed())
        ag.AddMany([(wx.StaticText(ap, label="Switching simulator"), 0, wx.ALIGN_CENTER_VERTICAL), (self.simc, 0)])
        aps = wx.BoxSizer(wx.VERTICAL)
        aps.Add(ag, 0, wx.ALL, 4)
        aps.Add(self.adaptive, 0, wx.ALL, 4)
        ap.SetSizer(aps)
        ob.Add(adv, 0, wx.EXPAND | wx.ALL, 2)

        kb, kbox = self._box(left, ls, "Checks")
        self.kbox, self.ksizer = kbox, wx.BoxSizer(wx.VERTICAL)
        kb.Add(self.ksizer, 0, wx.EXPAND | wx.ALL, 2)

        self.run_b = wx.Button(left, label="Analyse")
        self.run_b.Bind(wx.EVT_BUTTON, lambda e: self._run())
        ls.Add(self.run_b, 0, wx.EXPAND | wx.ALL, 6)
        ls.Add(wx.StaticText(left, label=(
            "The copper of the cell's nets is meshed (every net a conductor,\n"
            "ground included) and its partial inductances solved (PEEC).\n"
            "Parts: datasheet models (package L, capacitances). Results are\n"
            "small-signal, at the operating point. Save the board in KiCad first.")), 0, wx.ALL, 6)
        left.SetSizer(ls)
        left.FitInside()

        nb = self.nb = wx.Notebook(sp)
        sm = wx.Panel(nb)
        ss = wx.BoxSizer(wx.VERTICAL)
        vrow = wx.BoxSizer(wx.HORIZONTAL)
        vrow.Add(wx.StaticText(sm, label="What it means"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        if self.session.kicad is not None:
            b = wx.Button(sm, label="Select the loop's parts in KiCad", style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_BUTTON, lambda e: self._select_loop())
            vrow.AddStretchSpacer()
            vrow.Add(b, 0, wx.ALL, 2)
        ss.Add(vrow, 0, wx.EXPAND)
        self.verdict = wx.TextCtrl(sm, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 190))
        self.verdict.SetToolTip("A plain reading of the results: rough rules of thumb for SMD power stages "
                                "(hot loop under 1 nH excellent, under 2.5 nH good, under 5 nH fair)")
        ss.Add(self.verdict, 0, wx.EXPAND | wx.ALL, 4)
        ss.Add(wx.StaticText(sm, label="Hot loop, per commutation (the part that turns off rings with its capacitance)"),
               0, wx.ALL, 4)
        self.clist = wx.ListCtrl(sm, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 100))
        for k, (t, w) in enumerate((("Commutation", 170), ("Active", 55), ("L loop [nH]", 85), ("board", 65),
                                    ("parts", 65), ("R [mΩ]", 65), ("Ringing", 90), ("Q", 50), ("C [pF]", 65),
                                    ("di/dt [A/ns]", 85), ("Overshoot [V]", 95))):
            self.clist.InsertColumn(k, t, width=w)
        ss.Add(self.clist, 0, wx.EXPAND | wx.ALL, 4)
        ss.Add(wx.StaticText(sm, label="Gate loops"), 0, wx.ALL, 4)
        self.gres = wx.ListCtrl(sm, style=wx.LC_REPORT | wx.BORDER_SUNKEN, size=(-1, 100))
        for k, (t, w) in enumerate((("FET", 50), ("Driver", 60), ("L gate [nH]", 85), ("L common-source [nH]", 140),
                                    ("R gate [Ω]", 80), ("Gate ringing", 95), ("Q", 50))):
            self.gres.InsertColumn(k, t, width=w)
        ss.Add(self.gres, 0, wx.EXPAND | wx.ALL, 4)
        self.info = wx.TextCtrl(sm, style=wx.TE_MULTILINE | wx.TE_READONLY)
        ss.Add(self.info, 1, wx.EXPAND | wx.ALL, 4)
        sm.SetSizer(ss)
        nb.AddPage(sm, "Summary")

        sw = wx.Panel(nb)
        sws = wx.BoxSizer(wx.VERTICAL)
        self.elist = wx.ListCtrl(sw, style=wx.LC_REPORT | wx.BORDER_SUNKEN | wx.LC_SINGLE_SEL, size=(-1, 105))
        for k, (t, w) in enumerate((("Edge", 190), ("I [A]", 55), ("Peak [V]", 70), ("Overshoot [V]", 95),
                                    ("dv/dt [V/ns]", 90), ("di/dt [A/ns]", 90), ("Ringing", 85), ("Q", 45),
                                    ("Energy [µJ]", 85), ("Other gate peak [V]", 135), ("Recovery [A]", 95))):
            self.elist.InsertColumn(k, t, width=w)
        self.elist.SetToolTip(
            "One row per simulated edge of the hard-switched switch. Peak: the highest voltage across the switch "
            "that blocks after the edge; overshoot: above the supply. dv/dt and di/dt: the fastest slopes. Ringing "
            "and Q: of the switch-node ring. Energy: lost in the switch during the edge. Other gate peak: how high "
            "the other FET's gate is pushed (compare with its V_th). Recovery: the reverse current of the other "
            "device's diode at turn-on")
        self.elist.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda e: self._plot_edge(e.GetIndex()))
        sws.Add(self.elist, 0, wx.EXPAND | wx.ALL, 4)
        self.tinfo = wx.StaticText(sw, label="")
        sws.Add(self.tinfo, 0, wx.ALL, 4)
        trow = wx.BoxSizer(wx.HORIZONTAL)
        self.tsel = wx.Choice(sw, choices=[])
        self.tsel.SetToolTip("Which edge to plot")
        self.tsel.Bind(wx.EVT_CHOICE, lambda e: self._plot_edge(self.tsel.GetSelection()))
        b = wx.Button(sw, label="Export waveforms (CSV)…", style=wx.BU_EXACTFIT)
        b.Bind(wx.EVT_BUTTON, lambda e: self._export_edges())
        trow.AddMany([(wx.StaticText(sw, label="Edge"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6), (self.tsel, 0),
                      (b, 0, wx.LEFT, 12)])
        sws.Add(trow, 0, wx.ALL, 4)
        self.tfig = Figure(figsize=(7, 6), constrained_layout=True)
        self.tcanvas = FigureCanvasWxAgg(sw, -1, self.tfig)
        ttb = NavigationToolbar2WxAgg(self.tcanvas)
        ttb.Realize()
        sws.Add(self.tcanvas, 1, wx.EXPAND)
        sws.Add(ttb, 0, wx.EXPAND)
        sw.SetSizer(sws)
        nb.AddPage(sw, "Switching")

        def plot_page(title, controls=()):
            p = wx.Panel(nb)
            fig = Figure(figsize=(7, 5), constrained_layout=True)
            canvas = FigureCanvasWxAgg(p, -1, fig)
            tb = NavigationToolbar2WxAgg(canvas)
            tb.Realize()
            s = wx.BoxSizer(wx.VERTICAL)
            if controls:
                row = wx.BoxSizer(wx.HORIZONTAL)
                for c in controls(p):
                    row.Add(c, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
                s.Add(row, 0)
            s.Add(canvas, 1, wx.EXPAND)
            s.Add(tb, 0, wx.EXPAND)
            p.SetSizer(s)
            nb.AddPage(p, title)
            return fig, canvas

        def bctrls(p):
            self.bsel = wx.Choice(p, size=(380, -1))
            self.bsel.Bind(wx.EVT_CHOICE, lambda e: self._plot_breakdown())
            self.bsel.SetToolTip("Which switching edge (which part turns off) to break down")
            return [wx.StaticText(p, label="Commutation"), self.bsel]
        self.bfig, self.bcanvas = plot_page("Where the L is", bctrls)
        self.zfig, self.zcanvas = plot_page("Impedance")

        def mctrls(p):
            self.msel = wx.Choice(p, size=(330, -1))
            self.mlay = wx.Choice(p)
            self.mq = wx.Choice(p, choices=["Current density |J|", "Inductance density"])
            self.mq.SetSelection(0)
            self.msel.SetToolTip("Which switching edge's high-frequency current to show")
            self.mlay.SetToolTip("Copper layer to show")
            self.mq.SetToolTip("|J|: where the loop current flows. Inductance density: where the loop's "
                               "magnetic energy (its inductance) sits")
            for c in (self.msel, self.mlay, self.mq):
                c.Bind(wx.EVT_CHOICE, lambda e: self._plot_map())
            return [wx.StaticText(p, label="Commutation"), self.msel, wx.StaticText(p, label="Layer"), self.mlay,
                    self.mq]
        self.mfig, self.mcanvas = plot_page("Map", mctrls)

        sp_ = wx.Panel(nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.spice = wx.TextCtrl(sp_, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.spice.SetFont(wx.Font(wx.FontInfo(9).Family(wx.FONTFAMILY_TELETYPE)))
        s.Add(self.spice, 1, wx.EXPAND | wx.ALL, 4)
        row = wx.BoxSizer(wx.HORIZONTAL)
        for text, fn in (("Save subcircuit…", self._save_spice), ("Copy", self._copy_spice)):
            b = wx.Button(sp_, label=text)
            b.Bind(wx.EVT_BUTTON, lambda e, f=fn: f())
            row.Add(b, 0, wx.ALL, 4)
        s.Add(row, 0)
        sp_.SetSizer(s)
        nb.AddPage(sp_, "SPICE")

        sp.SplitVertically(left, nb, 640)
        left.Bind(wx.EVT_SIZE, lambda e: (e.Skip(), self.canvas.Refresh()))
        sp.SetMinimumPaneSize(380)

    def _drive_refs(self, gd):
        """The parts of a gate drive: the FET, the driver IC and the series parts (both pads' parts of
        a pad-to-pad element)."""
        fet = self.setup.device(gd.fet)
        refs = [fet.ref if fet else gd.fet, gd.out[0]]
        for k in gd.chain:
            d = self.setup.device(k)
            if d is not None:
                refs += [pid.rsplit(".", 1)[0] for pid in d.pads.values()] if d.pads else [d.ref]
        return refs

    def _relayout(self):
        """Re-fit the left panel after its texts or lists change (labels set after the first layout
        would otherwise keep their old, empty size and run into the buttons below them)."""
        self.cell_info.Wrap(max(self.left.GetClientSize().width - 40, 300))
        refit(self.left)

    def _status(self, t):
        self.SetStatusText(t)

    # ------------------------------------------------------------- setup <-> UI
    def _to_ui(self):
        s = self.setup
        cands = M.switch_nets(self.board)
        names = cands + sorted(n for n in self.board.nets if n not in cands)
        self.sw_names = names
        self.sw.Set(names)
        if s.sw_net in names:
            self.sw.SetSelection(names.index(s.sw_net))
        self.topo.SetSelection(list(M.TOPOLOGIES).index(s.topology if s.topology in M.TOPOLOGIES else "auto"))
        self.cellc.Set([self._cell_name(c) for c in self.cells])
        self.cellc.SetSelection(self.ci)
        all_nets = sorted(self.board.nets)
        self.net_names = all_nets
        for ch, cur in ((self.railc, s.rail_net), (self.retc, s.ret_net)):
            ch.Set(all_nets)
            if cur in all_nets:
                ch.SetSelection(all_nets.index(cur))
        self.cell_info.SetLabel(s.label() if s.devices else "Pick the switch node and press Detect parts.")
        self.plist.DeleteAllItems()
        order = {"upper": 0, "lower": 1, "link": 2, "cap": 3, "gate_r": 4}
        self._rows = sorted(range(len(s.devices)), key=lambda k: (order.get(s.devices[k].role, 9),
                                                                  EL.natural_key(s.devices[k].ref)))
        for k in self._rows:
            d = s.devices[k]
            mm = "" if d.role in ("upper", "lower") or d.pads else f"{CK._cap_mm(self.board, s, d.ref):.0f}"
            self.plist.Append(["✓" if d.enabled else "", ROLES.get(d.role, d.role), d.key,
                               "Internal switch" if d.internal else KIND_LABEL[d.kind],
                               pins_text(self.board, d), self._model_text(d), mm])
        self.glist.DeleteAllItems()
        for gd in s.drives:
            parts = [s.device(k) for k in gd.chain if s.device(k)]
            self.glist.Append([gd.fet, drive_path(s, gd),
                               fmt_si(sum(p.model.get("r", 0.0) for p in parts), "Ω") if parts else "-",
                               fmt_si(gd.rout, "Ω")])
        self._fit_columns(self.plist, 5)
        self._fit_columns(self.glist, 1)
        self.canvas.update(s, self.board)
        self._refresh_checks()
        self.adaptive.SetValue(s.adaptive)
        self.tran.SetValue(s.transient)
        self.simc.SetSelection(["auto", "builtin", "ngspice"].index(s.simulator)
                               if s.simulator in ("auto", "builtin", "ngspice") else 0)
        keys = [k for k, _ in DIRS]
        self.dirc.SetSelection(keys.index(s.i_dir) if s.i_dir in keys else 0)
        self._dir_label()
        for key, (t, kind) in self.fields.items():
            v = getattr(s, key)
            t.SetValue(fmt_si(v) if kind == "si" else f"{v:g}")
        self._relayout()

    def _refresh_checks(self):
        """The Checks box: one line per issue, worst first, each with its fix button."""
        self.ksizer.Clear(delete_windows=True)
        issues = CK.check(self.board, self.setup)
        if not issues:
            issues = [CK.Issue("ok", "Ready: nothing missing.")]
        sym = {"error": ("✖", wx.Colour(210, 50, 50)), "warn": ("⚠", wx.Colour(214, 140, 0)),
               "info": ("ℹ", wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)),
               "ok": ("✔", wx.Colour(40, 150, 60))}
        width = max(self.left.GetClientSize().width - 60, 260)
        for i in issues:
            row = wx.BoxSizer(wx.HORIZONTAL)
            mark, col = sym[i.level]
            m = wx.StaticText(self.kbox, label=mark)
            m.SetForegroundColour(col)
            row.Add(m, 0, wx.RIGHT, 6)
            col_ = wx.BoxSizer(wx.VERTICAL)
            t = wx.StaticText(self.kbox, label=i.text)
            t.Wrap(width)
            col_.Add(t, 0)
            if i.fix:
                b = wx.Button(self.kbox, label=i.fix_label or "Fix", style=wx.BU_EXACTFIT)
                b.Bind(wx.EVT_BUTTON, lambda e, a=i.fix: self._action(a))
                col_.Add(b, 0, wx.TOP, 2)
            row.Add(col_, 1)
            self.ksizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)
        self.run_b.SetLabel("Analyse" if not any(i.level == "error" for i in issues)
                            else "Analyse (fix the ✖ checks first)")
        self._relayout()

    def _dir_label(self):
        s = self.setup
        txt = "Auto"
        if s.sw_net and s.i_dir == "auto":
            d, why = s.current_dir(self.board)
            txt = f"Auto: {'out of' if d == 'out' else 'into'} the switch node ({why})"
        self.dirc.SetString(0, txt)
        self.dirc.SetSelection([k for k, _ in DIRS].index(s.i_dir) if s.i_dir in dict(DIRS) else 0)

    def _opts_changed(self):
        if self and self._from_ui(quiet=True):
            self._dir_label()
            self.canvas.update(self.setup, self.board)
            self._refresh_checks()

    def _action(self, a):
        """A fix from the Checks box or a click in the cell diagram (keys as in loop.checks)."""
        kind, _, arg = a.partition(":")
        s = self.setup
        if kind == "detect":
            self._detect()
        elif kind == "edit":
            d = s.device(arg)
            if d is not None:
                self._select_keys([d.key])
                self._edit(d)
        elif kind == "off":
            for k in arg.split(","):
                d = s.device(k)
                if d is not None:
                    d.enabled = False
            self.results.pop(self.ci, None)
            self._to_ui()
        elif kind == "drive":
            i = next((i for i, g in enumerate(s.drives) if s.device(g.fet) is s.device(arg)), None)
            if i is not None:
                self._drive_dialog(s.drives[i], i)
            else:
                self._drive_dialog(M.GateDrive(arg, ["", ""], ["", ""], [], 2.0))
        elif kind == "caps":
            self._select_keys([d.key for d in s.devices if d.kind == "cap"])
        elif kind == "add":
            self._add(role=arg)
        elif kind == "opt" and arg in ("i_dir", "transient"):
            (self.dirc if arg == "i_dir" else self.tran).SetFocus()
        elif kind == "opt":
            t = self.fields.get(arg, (None,))[0]
            if t is not None:
                if arg not in MAIN_OPTS:
                    self.adv.Expand()
                    self._relayout()
                t.SetFocus()
                t.SelectAll()
        elif kind == "net":
            {"rail": self.railc, "ret": self.retc, "sw": self.sw}[arg].SetFocus()

    def _select_keys(self, keys):
        for i, k in enumerate(self._rows):
            self.plist.Select(i, self.setup.devices[k].key in keys)
        first = next((i for i, k in enumerate(self._rows) if self.setup.devices[k].key in keys), None)
        if first is not None:
            self.plist.EnsureVisible(first)

    def _select_loop(self):
        s = self.setup
        refs = sorted({d.ref for d in s.devices if d.enabled and d.role != "gate_r"})
        try:
            self.session.kicad.highlight(refs)
            self._status("Selected in KiCad: " + ", ".join(refs))
        except Exception as e:      # noqa: BLE001
            self._status(f"Could not select in KiCad: {e}")

    def _model_text(self, d):
        m = d.model
        if d.kind == "fet":
            t = f", t_f {fmt_si(m['t_fall'], 's')}" if m.get("t_fall") else ""
            ch = " P-ch" if m.get("channel") == "P" else ""
            sp = f", SPICE model {m['spice_subckt']}" if m.get("spice_file") and m.get("spice_subckt") else ""
            return (f"{m.get('name', '')}{ch}: R_on {fmt_si(m.get('rds_on', 0), 'Ω')}, Coss "
                    f"{fmt_si(m.get('coss', 0), 'F')}{t}{sp}" + (" (generic)" if m.get("assumed") else ""))
        if d.kind == "diode":
            return f"{m.get('name', '')}: Cj {fmt_si(m.get('cj', 0), 'F')}" + (" (generic)" if m.get("assumed") else "")
        if d.kind == "cap":
            return f"{fmt_si(m.get('value', 0), 'F')}, ESL {fmt_si(m.get('esl', 0), 'H')}, ESR {fmt_si(m.get('esr', 0), 'Ω')}"
        return f"{fmt_si(m.get('r', 0), 'Ω')}"

    def _from_ui(self, quiet=False):
        s = self.setup
        try:
            for key, (t, kind) in self.fields.items():
                txt = t.GetValue()
                setattr(s, key, parse_si(txt) if kind == "si" else kind(float(txt)) if kind is int else kind(txt))
        except ValueError as e:
            if not quiet:
                wx.MessageBox(f"Options: {e}", "Value", parent=self)
            return False
        s.topology = list(M.TOPOLOGIES)[self.topo.GetSelection()]
        s.adaptive = self.adaptive.GetValue()
        s.transient = self.tran.GetValue()
        s.simulator = ["auto", "builtin", "ngspice"][max(self.simc.GetSelection(), 0)]
        s.i_dir = DIRS[max(self.dirc.GetSelection(), 0)][0]
        return True

    def _selected(self):
        i = self.plist.GetFirstSelected()
        return self._rows[i] if i >= 0 else None

    def _selected_all(self):
        """Device indices of every selected row (Shift / Ctrl-click select several)."""
        out, i = [], self.plist.GetFirstSelected()
        while i >= 0:
            out.append(self._rows[i])
            i = self.plist.GetNextSelected(i)
        return out

    def _reselect(self, refs):
        for i, k in enumerate(self._rows):
            if self.setup.devices[k].ref in refs:
                self.plist.Select(i)

    # ------------------------------------------------------------- actions
    def _detect(self):
        k = self.sw.GetSelection()
        net = self.sw_names[k] if k >= 0 else None
        try:
            new = M.detect(self.board, net)
        except ValueError as e:
            return wx.MessageBox(str(e), "Detect", parent=self)
        self._from_ui(quiet=True)
        for key in ("v_rail", "margin", "cells", "f_min", "f_max", "adaptive", "i_switch", "name", "i_ripple",
                    "f_sw", "i_dir", "transient", "simulator"):
            setattr(new, key, getattr(self.setup, key))
        self.setup = new
        note = self._apply_saved(new)
        self.results.pop(self.ci, None)
        self._to_ui()
        self._show_or_clear()
        self._status(f"{len(new.devices)} parts found on {new.sw_net}.{note} Check the models in the Checks box.")

    def _cell_nets(self):
        """Rail / return changed by hand: move the switches' pins, search the caps again."""
        kr, kg = self.railc.GetSelection(), self.retc.GetSelection()
        rail = self.net_names[kr] if kr >= 0 else None
        ret = self.net_names[kg] if kg >= 0 else None
        if rail == ret:
            return wx.MessageBox("The rail and the return must be different nets.", "Cell", parent=self)
        M.set_cell_nets(self.board, self.setup, rail, ret)
        self.results.pop(self.ci, None)
        self._to_ui()
        self._show_or_clear()
        self._status(f"Rail {self.setup.rail_net}, return {self.setup.ret_net}: "
                     f"{len(self.setup.by_role('cap'))} hot-loop capacitor(s) on.")

    def _switch_cell(self, k):
        if k < 0 or k == self.ci:
            return
        self._from_ui(quiet=True)
        self.ci = k
        self._to_ui()
        self._show_or_clear()

    def _new_cell(self):
        used = {c.sw_net for c in self.cells}
        cands = [n for n in M.switch_nets(self.board) if n not in used]
        names = cands + sorted(n for n in self.board.nets if n not in cands and n not in used)
        with wx.SingleChoiceDialog(self, "Switch node of the new cell (nets joining an inductor and a "
                                         "switch first):", "New cell", names) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            net = names[dlg.GetSelection()]
        try:
            new = M.detect(self.board, net)
        except ValueError as e:
            wx.MessageBox(f"{e}\n\nThe cell is created empty: add its parts by hand.", "New cell", parent=self)
            new = M.Setup(sw_net=net)
        self._from_ui(quiet=True)
        for key in ("v_rail", "margin", "cells", "f_min", "f_max", "adaptive"):
            setattr(new, key, getattr(self.setup, key))
        self._apply_saved(new)
        self.cells.append(new)
        self.ci = len(self.cells) - 1
        self._to_ui()
        self._show_or_clear()

    def _del_cell(self):
        if len(self.cells) <= 1:
            return wx.MessageBox("This is the only cell.", "Remove cell", parent=self)
        if wx.MessageBox(f"Remove the cell {self._cell_name(self.setup)}?", "Remove cell",
                         wx.YES_NO, self) != wx.YES:
            return
        del self.cells[self.ci]
        self.results = {(k if k < self.ci else k - 1): v for k, v in self.results.items() if k != self.ci}
        self.ci = max(0, self.ci - 1)
        self._to_ui()
        self._show_or_clear()

    def _show_or_clear(self):
        """Show this cell's result, or empty the result tabs if it has none (yet)."""
        if self.res is not None:
            return self._show()
        self.clist.DeleteAllItems()
        self.gres.DeleteAllItems()
        self.info.SetValue("Not analysed yet: press Analyse.")
        self.verdict.SetValue("Not analysed yet: press Analyse. Check the Checks box on the left first.")
        self.spice.SetValue("")
        self.elist.DeleteAllItems()
        self.tsel.Set([])
        self.tinfo.SetLabel("")
        for fig, canvas in ((self.bfig, self.bcanvas), (self.zfig, self.zcanvas), (self.mfig, self.mcanvas),
                            (self.tfig, self.tcanvas)):
            fig.clf()
            canvas.draw()

    def _from_kicad(self):
        try:
            ids = self.session.kicad.selected_pad_ids(self.board)
        except Exception as e:      # noqa: BLE001
            return wx.MessageBox(f"Could not read the selection: {e}", "KiCad", parent=self)
        nets = {p.net for p in self.board.pads if p.id in ids and p.net}
        if not nets:
            return wx.MessageBox("Select a pad on the switch node in the PCB editor.", "KiCad selection", parent=self)
        net = sorted(nets)[0]
        self.sw.SetSelection(self.sw_names.index(net))
        self._detect()

    def _edit(self, d=None):
        if d is None:
            k = self._selected()
            if k is None:
                return
            d = self.setup.devices[k]
        with ModelDialog(self, d, self.session.library, self.board, self._store()) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                d.model, d.nets, d.kelvin = dlg.result
                self._to_ui()

    def _toggle(self):
        """On/off for every selected part: all off if they were all on, else all on."""
        ks = self._selected_all()
        if not ks:
            return
        on = not all(self.setup.devices[k].enabled for k in ks)
        refs = {self.setup.devices[k].ref for k in ks}
        for k in ks:
            self.setup.devices[k].enabled = on
        self._to_ui()
        self._reselect(refs)

    def _add(self, role=None):
        with AddPartDialog(self, self.board, role) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            ref, role = dlg.value()
        if not ref:
            return
        try:
            d = M.make_device(self.board, self.setup, ref, role)
        except ValueError as e:
            return wx.MessageBox(str(e), "Add part", parent=self)
        self.setup.devices = [x for x in self.setup.devices if x.key != d.key] + [d]
        if role == "link":           # the rail now reaches further: its caps are searched again
            caps = [x for x in self.setup.devices if x.kind == "cap"]
            self.setup.devices = [x for x in self.setup.devices if x.kind != "cap"] + \
                M.find_caps(self.board, self.setup, old=caps)
        self._to_ui()

    def _remove(self):
        ks = set(self._selected_all())
        if ks:
            self.setup.devices = [d for k, d in enumerate(self.setup.devices) if k not in ks]
            self._to_ui()

    def _drive_dialog(self, gd, index=None):
        with DriveDialog(self, self.board, self.setup, gd, self.session.library) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            drive, parts = dlg.result
        s = self.setup
        old = set(s.drives[index].chain) if index is not None else set()
        for p in parts:                                 # add / replace the series parts
            s.devices = [d for d in s.devices if d.key != p.key] + [p]
        if index is None:
            s.drives = [g for g in s.drives if g.fet != drive.fet] + [drive]
        else:
            s.drives[index] = drive
        in_use = {k for g in s.drives for k in g.chain}
        s.devices = [d for d in s.devices if not (d.role == "gate_r" and d.key in old and d.key not in in_use)]
        self.results.pop(self.ci, None)
        self._to_ui()

    def _edit_drive(self):
        i = self.glist.GetFirstSelected()
        if i >= 0:
            self._drive_dialog(self.setup.drives[i], i)

    def _add_drive(self):
        fets = [d for d in self.setup.devices if d.kind == "fet" and not d.internal and d.nets.get("G")]
        free = [d for d in fets if not any(self.setup.device(g.fet) is d for g in self.setup.drives)]
        if not fets:
            return wx.MessageBox("No FET with a gate pin in this cell.", "Add drive", parent=self)
        d = (free or fets)[0]
        self._drive_dialog(M.GateDrive(d.key, ["", ""], ["", ""], [], 2.0))

    def _del_drive(self):
        i = self.glist.GetFirstSelected()
        if i < 0:
            return
        s = self.setup
        gone = set(s.drives[i].chain)
        del s.drives[i]
        in_use = {k for g in s.drives for k in g.chain}
        s.devices = [d for d in s.devices if not (d.role == "gate_r" and d.key in gone and d.key not in in_use)]
        self._to_ui()

    def _run(self):
        if self._running or not self._from_ui():
            return
        self.save()
        board, st, setup = self.board, self.session.stackup, M.Setup.from_dict(self.setup.to_dict())
        self._running, self._run_ci = True, self.ci
        self.run_b.Disable()

        def work():
            try:
                r = A.run(board, st, setup, progress=lambda m: wx.CallAfter(self._status, m))
                if setup.transient and setup.i_switch > 0:
                    try:
                        r.edges, r.tran = TR.run(board, setup, r, progress=lambda m: wx.CallAfter(self._status, m))
                    except (ValueError, RuntimeError) as e:
                        r.tran_error = str(e)
                wx.CallAfter(self._done, r, None)
            except ValueError as e:
                wx.CallAfter(self._done, None, str(e))
            except Exception:      # noqa: BLE001
                wx.CallAfter(self._done, None, traceback.format_exc())
        threading.Thread(target=work, daemon=True).start()

    def _done(self, r, err):
        self._running = False
        self.run_b.Enable()
        if err:
            self._status("Analysis failed.")
            return wx.MessageBox(err, "Switching Loop", parent=self)
        self.results[self._run_ci] = r
        if self._run_ci != self.ci:          # the user moved to another cell meanwhile
            return self._status(f"{self._cell_name(self.cells[self._run_ci])}: done in {r.seconds:.1f} s.")
        self._show()
        self._status(f"Done in {r.seconds:.1f} s: {r.cells} cells at {r.pitch:g} mm, {r.bars} current bars.")

    # ------------------------------------------------------------- results
    def _show(self):
        r = self.res
        self.clist.DeleteAllItems()
        for c in r.commutations:
            short = (f"{', '.join(x.key for x in M.Setup.from_dict(r.setup).by_role(c['off']))} off → "
                     f"{', '.join(x.key for x in M.Setup.from_dict(r.setup).by_role(c['on']))} on")
            self.clist.Append([short, "yes" if c["active"] else "", f"{c['L_total_nH']:.2f}",
                               f"{c['L_board_nH']:.2f}", f"{c['L_parts_nH']:.2f}", f"{c['R_mOhm']:.0f}",
                               fmt_si(c["f_ring"], "Hz"), f"{c['Q']:.1f}", f"{c['C_eff_pF']:.0f}",
                               f"{c['didt_A_ns']:.2f}" if math.isfinite(c.get("didt_A_ns", math.nan)) else "-",
                               f"{c['V_over']:.2f}" if math.isfinite(c.get("V_over", math.nan)) else "-"])
        self.gres.DeleteAllItems()
        for g in r.gates:
            self.gres.Append([g["fet"], g["driver"], f"{g['L_gate_nH']:.2f}", f"{g['L_cs_nH']:.3f}",
                              f"{g['R_gate']:.2f}", fmt_si(g["f_gate"], "Hz"), f"{g['Q_gate']:.2f}"])
        self._fit_columns(self.clist)
        self._fit_columns(self.gres)
        s = M.Setup.from_dict(r.setup)
        lines = [f"{s.label()}: switch node {s.sw_net}, rail {s.rail_net}, return {s.ret_net}; {s.v_rail:g} V across the switch.",
                 f"Mesh: {r.cells} copper cells at {r.pitch:g} mm, {r.bars} current bars, {r.meshes} mesh currents, "
                 f"window {r.window[2] - r.window[0]:.0f} x {r.window[3] - r.window[1]:.0f} mm.",
                 "",
                 "L loop: the inductance seen across the switch that turns off, through the copper, the other switch"
                 " (on) and the capacitors (ESL); 'board' is the copper alone, 'parts' the packages and ESL.",
                 "Ringing: that loop with the switch's own capacitance (Coss or Cj) and the board's; Q from the "
                 "half-power width.  Overshoot ≈ L di/dt; a snubber across the switch starts from R ≈ √(L/C).",
                 "L common-source: the mutual between the gate loop and the power loop through the FET (shared "
                 "copper, package source L, magnetic coupling): V_gs,induced = -L_cs di/dt."]
        for c in r.commutations:
            os_ = (f"{c['V_over']:.2f} V at {c['didt_A_ns']:.2f} A/ns ({s.i_switch:g} A over {c['didt_from']})"
                   if math.isfinite(c.get("V_over", math.nan)) else
                   f"{c['L_total_nH']:.1f} V per A/ns (give the switched current and the rise / fall times for volts)")
            lines.append(f"  {c['title']}: overshoot ≈ {os_}; snubber R ≈ "
                         f"{math.sqrt(c['L_total_nH'] / max(c['C_eff_pF'], 1e-9) * 1e3):.2f} Ω, C ≈ "
                         f"{3 * c['C_eff_pF']:.0f} pF.")
        cn, cm = r.cap_nets, np.array(r.cap_pF)
        pairs = sorted(((cm[i, j], cn[i], cn[j]) for i in range(len(cn)) for j in range(i + 1, len(cn))
                        if cm[i, j] > 0.05), reverse=True)
        if pairs:
            lines += ["", "Board capacitance between nets (in the window): "
                      + ", ".join(f"{a} - {b} {v:.2f} pF" for v, a, b in pairs[:10])]
        if r.warnings:
            lines += ["", "Warnings:"] + ["• " + w for w in dict.fromkeys(r.warnings)]
        self.info.SetValue("\n".join(lines))
        self.verdict.SetValue("\n".join(ADV.verdict(r)).strip())
        titles = [c["title"] for c in r.commutations]
        self.bsel.Set(titles)
        self.bsel.SetSelection(0)
        self.msel.Set(titles)
        self.msel.SetSelection(0)
        self.mlay.Set(r.layers)
        self.mlay.SetSelection(0)
        self.spice.SetValue(r.spice)
        self._show_edges()
        self._plot_breakdown()
        self._plot_z()
        self._plot_map()

    @staticmethod
    def _edge_title(e):
        return f"{e.switch} turns {e.kind}" + (f" ({e.other} {'takes' if e.kind == 'off' else 'gives up'} the current)"
                                               if e.other else "")

    def _show_edges(self):
        r = self.res
        self.elist.DeleteAllItems()
        if not r.edges:
            self.tsel.Set([])
            msg = (f"The switching simulation failed: {r.tran_error}" if r.tran_error else
                   "Not simulated: tick 'Simulate the switching edges' and give the inductor current "
                   "under Operating point.")
            self.tinfo.SetLabel(msg)
            self.tinfo.Wrap(700)
            self.tinfo.GetParent().Layout()
            self.tfig.clf()
            self.tcanvas.draw()
            return
        f = lambda v, fmt="{:.2f}": "-" if v is None or not math.isfinite(v) else fmt.format(v)      # noqa: E731
        for e in r.edges:
            m = e.metrics
            gp = (f"{m['vgs_other_peak']:.2f} (V_th {m['vth_other']:g})" if "vgs_other_peak" in m else "-")
            self.elist.Append([self._edge_title(e), f(e.current, "{:.3g}"), f(m["V_peak"], "{:.1f}"),
                               f(m["overshoot"], "{:.1f}"), f(m["dvdt_V_ns"], "{:.1f}"), f(m["didt_A_ns"]),
                               fmt_si(m["f_ring"], "Hz") if math.isfinite(m["f_ring"]) else "-",
                               f(m["Q_ring"], "{:.0f}"), f(m["E_uJ"], "{:.3g}"), gp,
                               f(m.get("i_rr_peak"), "{:.2f}") if e.kind == "on" else "-"])
        self._fit_columns(self.elist)
        t = r.tran
        d, why = t["direction"]
        self.tinfo.SetLabel(f"Switching loss ≈ {t['P_sw_W']:.3g} W at {fmt_si(t['f_sw'], 'Hz')} "
                            f"(E_on {t['E_on_uJ']:.3g} µJ + E_off {t['E_off_uJ']:.3g} µJ per cycle).\nInductor "
                            f"current {'out of' if d == 'out' else 'into'} the switch node ({why}); simulated "
                            f"{'in ngspice' if t.get('simulator') == 'ngspice' else 'with the built-in solver'}.")
        self.tinfo.GetParent().Layout()
        self.tsel.Set([self._edge_title(e) for e in r.edges])
        self.tsel.SetSelection(0)
        self._plot_edge(0)

    def _plot_edge(self, k):
        r = self.res
        if not r or not r.edges or k < 0:
            return
        if self.tsel.GetSelection() != k:
            self.tsel.SetSelection(k)
        e = r.edges[k]
        s = M.Setup.from_dict(r.setup)
        m = e.metrics
        t = e.t * 1e9
        self.tfig.clf()
        gate = any(n.startswith("V_GS") for n in e.waves)
        rows = 3 if gate else 2
        axv = self.tfig.add_subplot(rows, 1, 1)
        axi = self.tfig.add_subplot(rows, 1, 2, sharex=axv)
        axg = self.tfig.add_subplot(rows, 1, 3, sharex=axv) if gate else None
        for name, y in e.waves.items():
            ax = axg if name.startswith("V_GS") else axv if name.startswith("V_") else axi
            if ax is None:
                continue
            ax.plot(t, y, lw=1.1, label=name)
        axv.axhline(s.v_rail, color="0.5", ls="--", lw=0.8, label=f"supply {s.v_rail:g} V")
        pk = int(np.argmax(e.v_sw))
        axv.annotate(f"peak {m['V_peak']:.1f} V\n(+{m['overshoot']:.1f} V)", (t[pk], e.v_sw[pk]),
                     textcoords="offset points", xytext=(12, -6), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.6))
        if gate and "vth_other" in m and math.isfinite(m["vth_other"]):
            axg.axhline(m["vth_other"], color="C3", ls="--", lw=0.8, label=f"V_th of {e.other}")
        # start zoomed on the edge and its first rings (the toolbar's pan / zoom shows the rest)
        f_r = m["f_ring"] if math.isfinite(m["f_ring"]) else 1e8
        axv.set_xlim(0, min(t[-1], TR.T0 * 1e9 + 3 * max(m.get("t_rise_ns", 0) or 0, 2) + 12e9 / f_r))
        axv.set_ylabel("voltage [V]")
        axi.set_ylabel("current [A]")
        if gate:
            axg.set_ylabel("gate [V]")
        (axg or axi).set_xlabel("time [ns]")
        for ax in [a for a in (axv, axi, axg) if a is not None]:
            ax.grid(alpha=0.3)
            if ax.lines:
                ax.legend(fontsize=7, loc="upper right")
        axv.set_title(f"{self._edge_title(e)} at {e.current:.3g} A", fontsize=9)
        self.tcanvas.draw()

    def _export_edges(self):
        r = self.res
        if not r or not r.edges:
            return
        with wx.FileDialog(self, "Save waveforms", wildcard="CSV (*.csv)|*.csv", defaultFile="switching_edges.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            for e in r.edges:
                names = list(e.waves)
                w.writerow([f"# {self._edge_title(e)} at {e.current:g} A"])
                w.writerow(["t [ns]"] + names)
                for i in range(len(e.t)):
                    w.writerow([f"{e.t[i] * 1e9:.4f}"] + [f"{e.waves[n][i]:.5g}" for n in names])
                w.writerow([])
        self._status(f"Saved {path}")

    def _plot_breakdown(self):
        r = self.res
        if not r:
            return
        c = r.commutations[max(self.bsel.GetSelection(), 0)]
        self.bfig.clf()
        ax = self.bfig.add_subplot(111)
        items = c["breakdown"][:18][::-1]
        labels = [lab for lab, _ in items]
        vals = [v for _, v in items]
        cols = ["C0" if not any(w in lab for w in ("package", "ESL", "driver")) else "C1" for lab in labels]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals)), labels)
        ax.set_xlabel("share of the loop inductance [nH]")
        ax.set_title(f"{c['title']}: {c['L_total_nH']:.2f} nH at {fmt_si(c['f_eval'], 'Hz')} "
                     f"(blue: board copper, orange: parts)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        self.bcanvas.draw()

    def _plot_z(self):
        r = self.res
        self.zfig.clf()
        ax1 = self.zfig.add_subplot(211)
        ax2 = self.zfig.add_subplot(212, sharex=ax1)
        f = r.freqs
        w = 2 * np.pi * f
        for k, c in enumerate(r.commutations):
            ax1.loglog(f, np.abs(c["Z_ring"]), color=f"C{k}", label=c["title"])
            ax2.semilogx(f, c["Z_loop"].imag / w * 1e9, color=f"C{k}", label=f"{c['title']}: with parts")
            ax2.semilogx(f, c["Z_board"].imag / w * 1e9, color=f"C{k}", ls="--", label="board copper only")
            if math.isfinite(c["f_ring"]):
                ax1.axvline(c["f_ring"], color=f"C{k}", lw=0.8, alpha=0.6)
        ax1.set_ylabel("|Z| across the switch [Ω]")
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)
        ax2.set_ylabel("loop L [nH]")
        ax2.set_xlabel("frequency [Hz]")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
        self.zcanvas.draw()

    def _plot_map(self):
        r = self.res
        if not r or not r.display:
            return
        k = max(self.msel.GetSelection(), 0)
        L = r.layers[max(self.mlay.GetSelection(), 0)]
        q = "J" if self.mq.GetSelection() == 0 else "E"
        dsp = r.display
        a = dsp["data"][L][:, :, dsp["keys"].index((k, q))]
        self.mfig.clf()
        ax = self.mfig.add_subplot(111)
        v = a[np.isfinite(a)]
        vmax = float(np.percentile(v, 99.5)) if v.size else 1.0
        cmap = matplotlib.colormaps["inferno"].copy()
        cmap.set_bad("0.3")
        im = ax.imshow(a, extent=dsp["grid"].extent, cmap=cmap, vmin=0, vmax=vmax or 1.0, interpolation="bilinear")
        ax.set_aspect("equal")
        self.mfig.colorbar(im, ax=ax, label="A/mm per A of loop current" if q == "J" else "nH/mm²", shrink=0.85)
        # the parts: pad outlines on this layer and their names
        s = M.Setup.from_dict(r.setup)
        refs = {d.ref for d in s.devices if d.enabled} | {g.out[0] for g in s.drives}
        loop_nets = {s.sw_net, s.rail_net, s.ret_net}
        for ref in refs:
            ps = [p for p in self.board.pads if p.ref == ref and p.geom is not None]
            for p in ps:
                if L in p.layers:
                    x0, y0, x1, y1 = p.geom.bbox
                    ax.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                              edgecolor="cyan", lw=0.6, alpha=0.8,
                                                              hatch="///" if p.net in loop_nets else None))
            if ps:
                x = sum(p.pos[0] for p in ps) / len(ps)
                y = sum(p.pos[1] for p in ps) / len(ps)
                ax.text(x, y, ref, color="cyan", fontsize=8, ha="center", va="center", fontweight="bold")
        x0, y0, x1, y1 = r.window
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{r.commutations[k]['title']} - {L} - at {fmt_si(r.commutations[k]['f_eval'], 'Hz')} "
                     f"(mesh {r.pitch_min:g} mm finest)", fontsize=9)
        self.mcanvas.draw()

    def _save_spice(self):
        if not self.res:
            return
        with wx.FileDialog(self, "Save subcircuit", wildcard="SPICE (*.lib;*.cir)|*.lib;*.cir",
                           defaultFile="loop_copper.lib", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                with open(dlg.GetPath(), "w") as f:
                    f.write(self.res.spice)

    def _copy_spice(self):
        if self.res and wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(self.res.spice))
            wx.TheClipboard.Close()
