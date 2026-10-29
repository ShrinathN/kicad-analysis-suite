"""python3 -m pcb_current_sim.wxgui [board.kicad_pcb] [--kicad] [--home]

Opens the DC & Thermal tool, or with --home the suite's Home window."""
import sys

from .suite import main

args = [a for a in sys.argv[1:] if not a.startswith("--")]
main(args[0] if args else None, use_kicad="--kicad" in sys.argv, tool="home" if "--home" in sys.argv else "dc")
