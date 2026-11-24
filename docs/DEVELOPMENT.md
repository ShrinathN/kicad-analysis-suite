# Development

Notes for working on the code: environment, tests, smoke tests, screenshots and how the code is organised.
The running design log, kept up to date during development, is [CLAUDE.md](../CLAUDE.md); the plan is in [ROADMAP.md](ROADMAP.md).

## Environment

- **Host Python** (3.11 or newer: the tests file reader uses `tomllib`) with numpy, scipy, matplotlib and plotly runs the core, the CLI, the tests and the Tk GUI.
  wxPython is optional on the host. On a system whose own Python has no numpy / scipy (or is too new for their wheels), use a separate interpreter, e.g. pyenv: `pyenv install 3.11.10`, then `~/.pyenv/versions/3.11.10/bin/python -m pip install numpy scipy matplotlib plotly pytest sympy`.
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
- **Zip package**: `python3 kicad_plugin/build_pcm.py` builds `dist/kicad-analysis-suite-<version>.zip` for *Plugin and Content Manager → Install from File…*.
  It contains `metadata.json` (from `kicad_plugin/metadata.json`, PCM schema v1, `runtime: ipc`), `plugins/` (the files `install.sh` copies plus `pcb_current_sim/`, without caches) and a 64 × 64 `resources/icon.png` drawn by the script.
  The build stops if the identifiers in `metadata.json` and `plugin.json` differ. Bump `versions[0].version` for each release.
  Remove an `install.sh` copy before installing the zip (and the reverse): both would register the same identifier.

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

About 200 tests in about 135 s (ngspice and FastHenry cross-checks are skipped when those programs are missing):
- `tests/test_solver.py`: the DC, thermal, magnetic and pulse solvers against analytic results; regulators (LDO, dropout, switching, junction estimate), heat at parts, row values with SI prefixes.
- `tests/test_loop.py`: Switching Loop: PEEC against analytic inductances, detection (buck, boost ICs, rail links, Kelvin sources, gate chains, 4-switch buck-boost), topologies and the operating point from the converter, transients against analytic RLC and ngspice, snubber, EMI spectrum, losses.
- `tests/test_suite.py`: stackup parsing and the εr(f) model, scenario migration, the session, the component library and external tool detection.
- `tests/test_linsolve.py`: the solver backends, copper coverage as a union, the setup cache and the thermal cell.
- `tests/test_extract.py`: the cross-section solver against Hammerstad–Jensen and Wheeler, path tracing (vias, stubs, coplanar, missing return), pours (C, L and R against analytic values and the DC solver), elements across nets and to REF, R calibration, SPICE output, and runs of the subcircuit and a diode element in ngspice (skipped without ngspice).
- `tests/test_components.py`: the shared component list: migration of the old per-tool lists, per-tool switches, saving, the DC file API, Switching Loop capacitor models, the Components window's operations.
- `tests/test_pdn.py`: PDN: detection of the rail's parts, a plane pair's C and first cavity resonance, a decap's series resonance, low-f R against the DC solver, the target check, decap table, what-if and maps, Touchstone export (read back; the bare-board file with the parts connected gives the analysed Z), layout checks (a slot in the return plane crossed by the current, copper with no return, a far decap, mounting L), saved regulator fits by part number.
- `tests/test_regulator.py`: the regulator loop model (droop, crossover), fits recovering known loops from noisy synthetic plots, the boost RHP zero, digitising a synthetic scope image, a fit from a few dozen hand-clicked points (`digitise.from_points`), a ringing recovery found by the auto type, a fitted test capacitance, the fit window, hue colour matching and RDP simplification.
- `tests/test_batch.py`: simulation tests: TOML round trip, tests-file errors, check evaluation (worst item, `at`, missing values), a run of a matching and a DC test (exit codes, files, report, results.json, comparison with the previous run, `--test` patterns), tests that cannot run, `init` / `list` / `check` / `metrics`.
- `tests/test_pdn_transient.py`: vector fitting, ramp responses against a time simulation, PULSE edges, superposition of several loads.
- `tests/test_em3d.py`: 3D extraction's filament solver: round-wire skin effect against Bessel, bar and loop inductance against Grover and Maxwell, the reduced model against direct solves, FastHenry cross-checks (skipped without FastHenry).
- `tests/test_em3d_pcb.py`: board copper in 3D extraction: a track's R and L, a KiCad spiral against the analytic geometry, a track over a pour against microstrip, coupled coils, SPICE in ngspice, coarse cells, the window and filament budget, a coil under a ferrite sheet.
- `tests/test_em3d_magnetic.py`: the magnetic core solver: charge integrals against brute force, a cube's polarisability, the weak-core (Born) limit, a plate against the image method, a gapped ring against the gap reluctance, lossy (Debye) cores as modes against direct complex solves, the closed ring's loss tangent.
- `tests/test_em3d_cores.py`: material library and Debye fit, staircase areas, E core boxes / rotation / placement, plates on either side and by footprint, slot snapping (board and footprint Edge.Cuts, outer legs past the edge), a planar E core around a board spiral.
- `tests/test_matching.py`: the matching network maths (L-match, transformers, stubs, microstrip, Touchstone, S2P) and the tolerance analyses.

## GUI smoke tests and screenshots

These open real windows, so run them inside the Flatpak with the plugin venv, and leave the desktop alone while they run.

| Script | What it does | Output |
|---|---|---|
| `sim/demo/smoke.py` | DC window, element dialog, a run, the result viewer | PNGs next to the script |
| `sim/demo/smoke3d.py` | The viewer's 3D tab | `sim/demo/viewer_3d.png` |
| `sim/demo/smoke_suite.py` | Home and the shared panels; a stackup edit reaching the DC window | `sim/demo/smoke_out/` (cleared first) |
| `sim/demo/smoke_extract.py` | Opens Trace R/L/C on the demo board, extracts a trace, shows every tab | `sim/demo/smoke_out/` |
| `sim/demo/smoke_loop.py [BOARD]` | Switching Loop: detection, one analysis, every tab (default board: the synthetic buck cell `sim/demo/buck_demo.kicad_pcb`, made by `tests/test_loop.py:buck_cell`) | `sim/demo/smoke_out/` |
| `sim/demo/smoke_matching.py` | Builds a network in the matching editor, edits properties, runs every analysis | `sim/demo/smoke_out/` |
| `sim/demo/screenshots.py` | Regenerates the documentation screenshots | `docs/screenshots/` |
| `sim/demo/screenshots_extract.py BOARD` | Trace R/L/C screenshots from a board's saved `extract` setup (the board used for the docs is not in the repository) | `docs/screenshots/extract_*.png` |
| `sim/demo/screenshots_pdn.py BOARD` | PDN screenshots from a board's saved `pdn` setup (the board used for the docs is not in the repository). The regulator-fit window's shots in the docs (`pdn_regfit_source / image / fit / in_pdn.png`) were taken by hand on a datasheet plot; the script writes its own as `check_pdn_regfit_*.png` (git-ignored) so a rerun does not replace them | `docs/screenshots/pdn_*.png` |
| `sim/demo/screenshots_loop.py` | Switching Loop on `buck_demo.kicad_pcb`: an analysis with the edges at 10 A, the baseline, the suggested snubber, every tab | `docs/screenshots/loop_*.png` |
| `sim/demo/screenshots_em3d.py` | 3D extraction on `coil_demo.kicad_pcb` (coil over a pour, then with a ferrite sheet) and `planar_demo.kicad_pcb` (an E core snapped to slots), the material dialogs | `docs/screenshots/em3d_*.png` |

```sh
flatpak run --command=sh org.kicad.KiCad -c "$VENV sim/demo/screenshots.py"
```

The screenshot scripts work on temporary copies of the boards and a temporary component library.
They grab windows from the screen, so anything on top of them ends up in the picture: leave the PC alone while they run (about 15 minutes for all of them).
A wx smoke or screenshot script must open a window before `app.MainLoop()`: with no window open the loop returns at once and the script ends silently.

Demo boards in `sim/demo/`: `demo.kicad_pcb` (DC & Thermal: a supply, a FET, a load, resistors), `buck_demo.kicad_pcb` (a synchronous buck cell for Switching Loop), `coil_demo.kicad_pcb` (a 4-turn coil over a ground pour), `planar_demo.kicad_pcb` (a 3-turn spiral with slots for an E18 core in footprint T1). The last two are made from the helpers in `tests/test_loop.py` and `tests/test_em3d_pcb.py`.

## Code layout

```
pcb_current_sim/
  sexpr.py, board.py        .kicad_pcb parser (tracks, arcs, vias, zones, pads, stackup with plies)
  raster.py, solver.py      cut-cell finite-volume DC solver (rows, elements, voltage sources); setup cache
  linsolve.py               sparse factorisation for every tool (SuperLU, optional PARDISO, CG fallback)
  thermal.py, magnetic.py   electro-thermal model, B-field maps and loop inductance
  elements.py               resistance elements (auto-detected resistors, user elements)
  components.py             the board's shared component list: elements defined once, switched per tool
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
  extract/                  Trace R/L/C → SPICE core
    path.py                 traces a net's copper between pads as a graph (tracks, vias, pads, zones)
    xsection.py             2D finite-volume field solver: C', L', Z0, εeff of a cross-section
    rlc.py                  cross-sections along the path, via / pad models, ladder, DC calibration
    spice.py                .subckt writer (plain R, L, C)
    __main__.py             command line
  em3d/                     3D extraction core
    kernels.py              partial inductance of filament pairs (exact, GMD, Neumann), threaded
    geometry.py             3D nodes / segments / ports, filament grading to the skin depth
    system.py               filament network (mesh basis), direct solve, PRIMA reduced model, cores as modes
    magnetic.py             magnetic cores: facet volume-integral method, Debye materials, modes
    materials.py            magnetic material grades (built-in classes + materials.json), Debye fit
    cores.py                parametric cores (plate, E, rod, toroid, box), placement, slots and snapping
    pcb.py                  board copper -> geometry (window, adaptive cells, filament budget), core meshing
    analysis.py             run: wideband Z, R / L / Q / k, current maps
    spice.py, shapes.py     SPICE (Foster wideband, fixed R-L-K); test shapes (spirals, loops, wires)
    fasthenry.py            FastHenry input writer / runner (cross-checks)
  pdn/                      PDN core: model.py (setup, detection), network.py (mesh over an ideal return, Z(f) sweep), analysis.py (target, decaps, what-if, maps), regulator.py (loop model and fit), digitise.py (traces from a plot image), transient.py (load steps: vector fitting, ramp responses, PULSE loads), layout.py (layout checks), regstore.py (saved regulator fits)
  batch/                    simulation tests (tests file, runner, report, command line)
    tomlio.py               TOML writer (reader: tomllib); None <-> "null"
    spec.py                 the tests file: TestFile ([shared] stackup + components), Test, Check
    make.py                 tests from saved scenario sections or a window's setup
    runner.py               select, run, evaluate checks, simulations/<timestamp>/, compare with the previous run
    report.py               report.html / report.md
    cli.py                  run / init / list / check / metrics (dispatched from pcb_current_sim/cli.py)
    tools/                  one adapter per tool: METRICS, OUTPUTS, from_scenario, run(ctx)
  loop/                     Switching Loop core
    peec.py                 mesh (cells, bars, wires, barrels), partial L, mesh-current solve, capacitance
    circuit.py              small complex nodal solver: copper block + parts, port Z matrices, resonance
    model.py                cell setup (parts in roles, gate drives), FET / diode models, auto-detection
    analysis.py             commutations, energy breakdown, gate loops, maps, SPICE
  matching/                 Matching & Tolerance core
    model.py                elements, ports, bands, ABCD cascade
    analysis.py             corners, Monte Carlo, worst case, sensitivity, BOM
    microstrip.py           Hammerstad-Jensen microstrip with loss
    touchstone.py           .sNp reader and writer, S -> ABCD
    plots.py                matplotlib figures (Smith, S11, S21, VSWR, histograms, tornado)
  wxgui/                    wx GUI (used inside KiCad)
    suite.py                Home, Suite menu, stackup / library / results / external windows
    components.py           Components window (the board's shared parts between pads)
    app.py, viewer.py       DC & Thermal window and result viewer (2D, 3D, tables)
    matching.py             Matching & Tolerance window
    extract.py              Trace R/L/C → SPICE window
    loop.py                 Switching Loop window
    em3d.py, cores.py       3D extraction window; core, material and grade dialogs
    pdn.py                  PDN window
    regfit.py               regulator model from a datasheet plot (digitiser + fit)
    tests.py                Tests window, "Save as test…" (add_button / save_as_test), check editor
    schematic.py            symbol canvas for the matching editor
    layout.py               scrolled() input panels and refit() after their texts change
  gui/                      Tk GUI (DC & Thermal only, for hosts without wx)
  tables.py                 helpers shared by both GUIs
kicad_plugin/               plugin.json (two toolbar actions), entry points, icons, install.sh, metadata.json + build_pcm.py (PCM zip)
sim/demo/                   demo boards, smoke tests, screenshot scripts
tests/                      pytest suite
docs/                       user guide, development notes, roadmap, screenshots
```

### Conventions
- Core modules never import a GUI toolkit; the wx and Tk layers only call them.
- Every window gets one shared `Session`. Changes go through it (`set_stackup`, `library_changed`, `load_board`) so the other windows are notified.
- Each tool saves only its own section of the scenario file through `scenario.save_section`, which keeps the other sections.
- New tools are built in wx only; the Tk GUI stays DC-only.
