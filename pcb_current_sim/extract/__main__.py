"""python3 -m pcb_current_sim.extract BOARD PAD PAD [PAD ...] [--return NET ...] [-o FILE.lib]

Prints the per-path totals and the segment table, and writes (or prints) the SPICE subcircuit."""
import argparse
import os
import sys

from .. import board as B
from ..suite import stackup as ST
from ..suite.library import parse_si
from . import rlc as X
from . import spice as SP


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m pcb_current_sim.extract",
                                 description="R, L, C of the copper between pads as a SPICE RLC subcircuit.")
    ap.add_argument("board")
    ap.add_argument("pads", nargs="+", help="REF.PAD, e.g. U1.3 J2.1 (P1, P2, ...)")
    ap.add_argument("--return", dest="ret", action="append", default=None,
                    help="return net (repeat for several; default: GND-like nets)")
    ap.add_argument("--fmax", default="1G", help="highest frequency, e.g. 500M, 2.5G (default 1G)")
    ap.add_argument("--sample", type=float, default=0.5, help="cross-section spacing [mm]")
    ap.add_argument("--window", type=float, default=5.0, help="return search half-width [mm]")
    ap.add_argument("--max-section", type=float, default=0.0, help="section length limit [mm]")
    ap.add_argument("--no-stubs", action="store_true")
    ap.add_argument("--no-dc", action="store_true", help="skip the DC field solution for R")
    ap.add_argument("--g", action="store_true", help="add dielectric loss G at f max")
    ap.add_argument("-o", "--output", help="write the subcircuit here (default: print it)")
    a = ap.parse_args(argv)
    board = B.load(a.board)
    if len(a.pads) < 2:
        ap.error("give at least two pads")
    net = next((p.net for p in board.pads if p.id == a.pads[0]), None)
    ret = a.ret if a.ret is not None else X.guess_return_nets(board, net)
    o = X.Options(f_max=parse_si(a.fmax), sample=a.sample, window=a.window, max_section=a.max_section,
                  stubs=not a.no_stubs, dc_solver=not a.no_dc, dielectric_g=a.g)
    try:
        e = X.Extractor(board, ST.from_board(board), ret, o, progress=lambda m: print(m, file=sys.stderr)).run(a.pads)
    except ValueError as ex:
        sys.exit(f"error: {ex}")
    log = sys.stderr if not a.output else sys.stdout
    print(f"net {e.net}, return {', '.join(ret) or '(none)'}", file=log)
    for pr in e.pairs:
        where = (f"through {pr['pours']} pour(s)" if pr.get("pours") else
                 f"{pr['length_mm']:.2f} mm, Z0 {pr['Z0_avg']:.1f} Ω ({pr['Z0_min']:.0f}–{pr['Z0_max']:.0f}), "
                 f"delay {pr['delay_ps']:.1f} ps")
        print(f"{pr['pair']} ({pr['a']} → {pr['b']}): {pr['vias']} via(s), R {pr['R_mOhm']:.3f} mΩ, "
              f"L {pr['L_nH']:.3f} nH, copper C {pr['C_pF']:.3f} pF"
              + (f" (+ {pr['C_el_pF'] * 1e-12:.4g} F of elements)" if pr.get("C_el_pF") else "") + f", {where}",
              file=log)
    for r in e.segments:
        print(f"  #{r['chain']} {r['layer']} w {r['width_mm']:g} l {r['length_mm']:.2f} mm: Z0 {r['Z0']:.1f} Ω, "
              f"εeff {r['eeff']:.2f}, L' {r['L_nH_mm']:.3f} nH/mm, C' {r['C_pF_mm']:.4f} pF/mm  [{r['reference']}]",
              file=log)
    for w in e.warnings:
        print("warning: " + w, file=log)
    text = SP.write(e, board_name=os.path.basename(a.board))
    if a.output:
        with open(a.output, "w") as f:
            f.write(text)
        print(f"wrote {a.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
