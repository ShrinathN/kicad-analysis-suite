# Development

Notes for working on the code: environment, tests, smoke tests, screenshots and how the code is organised.
The running design log, kept up to date during development, is [CLAUDE.md](../CLAUDE.md); the plan is in [ROADMAP.md](ROADMAP.md).

## Environment

- **Host Python** (3.9 or newer) with numpy, scipy, matplotlib and plotly runs the core, the CLI, the tests and the Tk GUI.
  wxPython is optional on the host.
- **KiCad 10** runs the plugin. The tool was developed against the Flathub Flatpak (`org.kicad.KiCad`), whose Python has wxPython but no Tk.
  - There is no host `kicad-cli` with the Flatpak; use `flatpak run --command=kicad-cli org.kicad.KiCad`.
  - Run commands in the sandbox with `flatpak run --command=sh org.kicad.KiCad -c '...'`. The sandbox sees your home directory.
  - KiCad builds the plugin's venv from `kicad_plugin/requirements.txt` on first load, at
    `~/.var/app/org.kicad.KiCad/cache/kicad/10.0/python-environments/io.github.shrinathn.kicad-analysis-suite/`.
    wxPython from the sandbox stays visible inside that venv.
  - *Preferences → Plugins → Enable KiCad API* must be on; the interpreter is set in
    `~/.var/app/org.kicad.KiCad/config/kicad/10.0/kicad_common.json`.
- **Installing the plugin**: `kicad_plugin/install.sh` copies the plugin into KiCad's plugin folder, so re-run it after code changes.
  `install.sh --link` symlinks instead, for development.

**If the toolbar buttons don't appear:**
- The plugin must be in `~/.var/app/org.kicad.KiCad/data/kicad/10.0/plugins/` (where `install.sh` puts it), not in `scripting/plugins/`, which is only for the old SWIG plugins.
- KiCad shows the buttons only after it has installed `requirements.txt` into the plugin venv. If KiCad was closed during that install, or it failed, the venv contains only pip. Install the requirements by hand (with `VENV` as below) and restart KiCad:
  ```sh
  flatpak run --command=sh org.kicad.KiCad -c "$VENV -m pip install -r ~/.var/app/org.kicad.KiCad/data/kicad/10.0/plugins/kicad-analysis-suite/requirements.txt"
  ```

In the commands below, `VENV` is the plugin venv's `bin/python3` inside the Flatpak:

```sh
VENV=~/.var/app/org.kicad.KiCad/cache/kicad/10.0/python-environments/io.github.shrinathn.kicad-analysis-suite/bin/python3
```

## Tests

```sh
python3 -m pytest tests
```

About 60 tests in about 20 s:
- `tests/test_solver.py`: the DC, thermal, magnetic and pulse solvers against analytic results.
- `tests/test_suite.py`: stackup parsing and the εr(f) model, scenario migration, the session, the component library and external tool detection.
- `tests/test_matching.py`: the matching network maths (L-match, transformers, stubs, microstrip, Touchstone, S2P) and the tolerance analyses.

## GUI smoke tests and screenshots

These open real windows, so run them inside the Flatpak with the plugin venv, and leave the desktop alone while they run.

| Script | What it does | Output |
|---|---|---|
| `sim/demo/smoke.py` | DC window, element dialog, a run, the result viewer | PNGs next to the script |
| `sim/demo/smoke3d.py` | The viewer's 3D tab | `sim/demo/viewer_3d.png` |
| `sim/demo/smoke_suite.py` | Home and the shared panels; a stackup edit reaching the DC window | `sim/demo/smoke_out/` (cleared first) |
| `sim/demo/smoke_matching.py` | Builds a network in the matching editor, edits properties, runs every analysis | `sim/demo/smoke_out/` |
| `sim/demo/screenshots.py` | Regenerates the documentation screenshots | `docs/screenshots/` |

```sh
flatpak run --command=sh org.kicad.KiCad -c "$VENV sim/demo/screenshots.py"
```

The screenshot script works on a temporary copy of the demo board and a temporary component library.
It grabs windows from the screen, so anything on top of them ends up in the picture.

## Code layout

```
pcb_current_sim/
  sexpr.py, board.py        .kicad_pcb parser (tracks, arcs, vias, zones, pads, stackup with plies)
  raster.py, solver.py      cut-cell finite-volume DC solver (rows, elements, voltage sources)
  thermal.py, magnetic.py   electro-thermal model, B-field maps and loop inductance
  elements.py               resistance elements (auto-detected resistors, user elements)
  results.py, export.py     result container, plots, PDF / HTML / CSV, 3D (plotly)
  vtk_export.py             ParaView (.vtm / .vti / .vtp / .pvd) writer, numpy only
  scenario.py               <board>.current_sim.json, one section per tool (version 2)
  kicad_link.py             IPC API link (kicad-python): board path, pad selection, save, refill
  cli.py                    command line
  suite/                    toolkit-neutral suite core
    session.py              shared board, stackup, library and KiCad link, with change events
    stackup.py              copper and dielectrics, Djordjevic-Sarkar εr(f) / tanδ(f)
    library.py              real-part models (L, C, R), SI parsing, user library file
    tools.py                tool registry for Home and the Suite menu
    external.py             openEMS / FastHenry / FastCap / ParaView detection (incl. Flatpak apps)
    results_index.py        finds saved results
  matching/                 Matching & Tolerance core
    model.py                elements, ports, bands, ABCD cascade
    analysis.py             corners, Monte Carlo, worst case, sensitivity, BOM
    microstrip.py           Hammerstad-Jensen microstrip with loss
    touchstone.py           .s1p / .s2p reader, S -> ABCD
    plots.py                matplotlib figures (Smith, S11, S21, VSWR, histograms, tornado)
  wxgui/                    wx GUI (used inside KiCad)
    suite.py                Home, Suite menu, stackup / library / results / external windows
    app.py, viewer.py       DC & Thermal window and result viewer (2D, 3D, tables)
    matching.py             Matching & Tolerance window
    schematic.py            symbol canvas for the matching editor
  gui/                      Tk GUI (DC & Thermal only, for hosts without wx)
  tables.py                 helpers shared by both GUIs
kicad_plugin/               plugin.json (two toolbar actions), entry points, icons, install.sh
sim/demo/                   demo board, smoke tests, screenshot script
tests/                      pytest suite
docs/                       user guide, development notes, roadmap, screenshots
```

### Conventions
- Core modules never import a GUI toolkit; the wx and Tk layers only call them.
- Every window gets one shared `Session`. Changes go through it (`set_stackup`, `library_changed`, `load_board`) so the other windows are notified.
- Each tool saves only its own section of the scenario file through `scenario.save_section`, which keeps the other sections.
- New tools are built in wx only; the Tk GUI stays DC-only.
