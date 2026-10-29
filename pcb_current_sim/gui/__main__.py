"""python3 -m pcb_current_sim.gui [board.kicad_pcb] [--kicad]"""
import sys

from .app import main

args = [a for a in sys.argv[1:] if not a.startswith("--")]
main(args[0] if args else None, use_kicad="--kicad" in sys.argv)
