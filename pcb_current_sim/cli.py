"""Command-line front end.

    python3 -m pcb_current_sim board.kicad_pcb --row "J1.2>J2.2:10" --row "J2.1>J1.1:10" --out results

A row is SRC>SNK:AMPS[:NAME]; several pads per side are joined with '+'
(e.g. "U1.3+U1.4>J1.1:5").  All rows are solved simultaneously.

Voltage sources and loads use the same syntax: --vsource "U1.5>U1.2:3.3:3V3"
(+ pads > - pads : volts) and --load "U2.1>U2.4:0.5:MCU" (+ pads > - pads : amps).

Resistor footprints reached from the rows are included automatically.  Add
any other two-pad resistance with --element PAD_A-PAD_B=OHMS[:NAME[:WATTS]][:ext]
(e.g. "Q1.2-Q1.3=0.004:Q1" for a FET's R_on, "L1.1-L1.2=12m:L1"; ":ext" = an off-board load
or wire whose dissipation does not heat the board, e.g. "J2.1-J2.2=2.4:LOAD:ext").
"""
import argparse
import os
import re
import sys

from . import board as B
from . import solver as S

UNITS = {"um": 1e-6, "µm": 1e-6, "mm": 1e-3, "oz": 34.79e-6, "mil": 25.4e-6}


def parse_row(text, k, kind="I"):
    m = re.fullmatch(r"\s*([^>]+)>([^:]+):([-+0-9.eE]+)(?::(.+))?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"bad row '{text}' (expected SRC>SNK:AMPS[:NAME])")
    src = [p.strip() for p in m.group(1).split("+")]
    snk = [p.strip() for p in m.group(2).split("+")]
    return S.Row(m.group(4) or f"row{k + 1}", src, snk, float(m.group(3)), kind)


def parse_element(text):
    from .elements import Element, parse_ohms
    ext = text.rstrip().endswith(":ext")
    text = text.rstrip()[:-4] if ext else text
    m = re.fullmatch(r"\s*([^-=\s]+)-([^=\s]+)=([^:]+)(?::([^:]*))?(?::([0-9.eE+-]+))?\s*", text)
    R = parse_ohms(m.group(3)) if m else None
    if R is None:
        raise argparse.ArgumentTypeError(f"bad element '{text}' (expected PAD_A-PAD_B=OHMS[:NAME[:WATTS]])")
    name = m.group(4) or m.group(1).split(".")[0]
    return Element(name, m.group(1), m.group(2), R, float(m.group(5)) if m.group(5) else None, heats=not ext)


def parse_thickness(text):
    """'F.Cu=35um' -> ('F.Cu', 3.5e-5)"""
    L, v = text.split("=")
    m = re.fullmatch(r"([0-9.]+)\s*([a-zµ]*)", v.strip())
    return L.strip(), float(m.group(1)) * UNITS.get(m.group(2) or "um", 1e-6)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pcb_current_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board")
    ap.add_argument("--row", action="append", default=[], help="SRC>SNK:AMPS[:NAME]")
    ap.add_argument("--vsource", action="append", default=[], help="PLUS>MINUS:VOLTS[:NAME]")
    ap.add_argument("--load", action="append", default=[], help="PLUS>MINUS:AMPS[:NAME]")
    ap.add_argument("--pulse", help="SCALE:MS[:steady] - transient pulse of all rows x SCALE for MS ms "
                                    "(from ambient, or from the steady state)")
    ap.add_argument("--no-magnetic", action="store_true", help="skip B maps and loop inductance")
    ap.add_argument("--b-height", type=float, default=1.0, help="B-map distance from the board [mm]")
    ap.add_argument("--element", action="append", default=[], help="PAD_A-PAD_B=OHMS[:NAME[:WATTS]]")
    ap.add_argument("--no-auto-resistors", action="store_true", help="ignore resistor footprints")
    ap.add_argument("--no-crop", action="store_true", help="grid over the whole board")
    ap.add_argument("--scenario", help="JSON scenario file saved by the GUI")
    ap.add_argument("--pitch", default="auto", help="grid pitch in mm, or 'auto'")
    ap.add_argument("--thickness", action="append", default=[], help="LAYER=VALUE[um|mm|oz]")
    ap.add_argument("--plating", type=float, default=25, help="via/barrel plating [µm]")
    ap.add_argument("--no-thermal", action="store_true")
    ap.add_argument("--ambient", type=float, default=25.0)
    ap.add_argument("--h-conv", type=float, help="fixed h [W/m²K] per face (default: h from temperature)")
    ap.add_argument("--emissivity", type=float, help="surface emissivity for the automatic h (default 0.9)")
    ap.add_argument("--no-error-estimate", action="store_true")
    ap.add_argument("--out", default="current_sim_out")
    ap.add_argument("--formats", default="png,pdf", help="png,svg,pdf,html,html3d,vtk,csv,npz")
    a = ap.parse_args(argv)

    board = B.load(a.board)
    if a.scenario:
        from .scenario import load_scenario
        rows, s, els = load_scenario(a.scenario)
    else:
        rows, s, els = [], S.Settings(), []
    rows += [parse_row(r, k) for k, r in enumerate(a.row)]
    rows += [parse_row(r, len(rows) + k, "V") for k, r in enumerate(a.vsource)]
    rows += [parse_row(r, len(rows) + k, "load") for k, r in enumerate(a.load)]
    if a.pulse:
        p = a.pulse.split(":")
        s.pulse, s.pulse_scale, s.pulse_ms = True, float(p[0]), float(p[1])
        s.pulse_from_steady = len(p) > 2 and p[2].startswith("s")
    if a.no_magnetic:
        s.magnetic = False
    s.b_height = a.b_height
    els += [parse_element(e) for e in a.element]
    if a.no_auto_resistors:
        s.auto_resistors = False
    if a.no_crop:
        s.crop = False
    if a.pitch != "auto":
        s.pitch = float(a.pitch)
    s.thickness.update(dict(parse_thickness(t) for t in a.thickness))
    s.t_plate = a.plating * 1e-6
    s.thermal = not a.no_thermal and s.thermal
    s.T_amb = a.ambient
    if a.h_conv is not None:
        s.h_conv, s.h_auto = a.h_conv, False
    if a.emissivity is not None:
        s.emissivity = a.emissivity
    s.error_estimate = not a.no_error_estimate
    try:
        from . import elements as EL
        auto, warn = EL.detect(board) if s.auto_resistors else ([], [])
        board.warnings += warn
        res = S.simulate(board, rows, s, progress=lambda m: print(m, file=sys.stderr),
                         elements=EL.merge(auto, els))
    except S.SimError as e:
        sys.exit(f"error: {e}")
    print(res.summary_text())
    from . import export
    os.makedirs(a.out, exist_ok=True)
    fm = set(a.formats.split(","))
    for f in ("png", "svg"):
        if f in fm:
            export.export_images(res, a.out, f)
    if "pdf" in fm:
        export.export_report_pdf(res, os.path.join(a.out, "report.pdf"))
    if "html" in fm:
        export.export_html(res, os.path.join(a.out, "current_density.html"))
    if "html3d" in fm:
        export.export_html_3d(res, os.path.join(a.out, "current_density_3d.html"), js="inline")
    if "vtk" in fm:
        from .vtk_export import export_vtk
        export_vtk(res, os.path.join(a.out, "paraview"))
    if "csv" in fm:
        res.write_csv(a.out)
    if "npz" in fm:
        res.save(os.path.join(a.out, "result.npz"))
    print(f"\nOutputs written to {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
