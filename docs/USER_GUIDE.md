# User guide

The full reference for the PCB Analysis Suite: installation, every tool and option, the physical models, outputs and tests. For a short overview see the [README](../README.md); for development notes see [DEVELOPMENT.md](DEVELOPMENT.md).

## Overview

The suite currently has two tools: **DC & Thermal** and **Matching & Tolerance** (see [Suite](#suite)). The DC & Thermal tool runs a DC simulation of copper nets in a `.kicad_pcb`. For each simulated net it produces:
- current density |J|
- voltage (IR drop), with current paths, loads and voltage sources
- Joule power density
- temperature (2D electro-thermal model), steady and for a current pulse (fusing)
- magnetic field maps above and below the board, and loop inductance

It also builds per-pad, per-via and per-track tables, with IPC-2221 and Onderdonk checks.

## Use

**From KiCad 10 (IPC API plugin, native or Flatpak)**
1. Run `kicad_plugin/install.sh`. Use `--link` to symlink instead of copy.
   It finds the Flatpak's plugin folder (`~/.var/app/org.kicad.KiCad/data/kicad/10.0/plugins`) or the native one (`~/.local/share/kicad/10.0/plugins`).
2. In KiCad, open *Preferences → Plugins* and tick **Enable KiCad API**. Then restart the PCB editor.
3. On first load, KiCad creates the plugin's Python venv from `requirements.txt` (kicad-python, numpy, scipy, matplotlib, pyamg, plotly).
   This needs network access, and the toolbar button appears once it finishes.
   The GUI uses wxPython, which ships with KiCad (Flatpak included). On native installs without wx it falls back to Tk.
4. The plugin adds two toolbar buttons: **PCB Analysis Suite** (the tiles icon) opens the Home window, and **DC & Thermal** opens that tool directly.
   **Save the board and refill zones first**: the simulator reads the saved file.
   The window's "Refill zones + save" button does both through the API.
5. To add a row from KiCad, select pads in the editor (source first, then sink) and click *Add from KiCad selection*.
   Use the selection filter to pick pads rather than whole footprints.

**Standalone GUI:** `python3 -m pcb_current_sim.wxgui board.kicad_pcb [--home]` (wx, DC & Thermal or the Home window) or `python3 -m pcb_current_sim.gui board.kicad_pcb` (Tk, DC & Thermal only)

**CLI**
```
python3 -m pcb_current_sim board.kicad_pcb --row "J1.2>J2.2:3:signal" --row "J2.1>J1.1:3:return" \
        --out results --formats png,svg,pdf,html,csv,npz [--no-thermal] [--pitch 0.05] \
        [--thickness F.Cu=48um] [--ambient 25] [--h-conv 12 | --emissivity 0.9] [--scenario board.current_sim.json]
```
A row is written `SRC>SNK:AMPS[:NAME]`. Join several pads on one side with `+`, for example `U1.3+U1.4>J1.1:5`.
Voltage sources and loads: `--vsource "PLUS>MINUS:VOLTS[:NAME]"`, `--load "PLUS>MINUS:AMPS[:NAME]"`.
Add a resistance element with `--element PAD_A-PAD_B=OHMS[:NAME[:WATTS]][:ext]`, for example `Q1.3-Q1.2=20m:Q1:0.35`.
`:ext` marks an off-board load or wire whose dissipation does not heat the board.
Pulse: `--pulse SCALE:MS[:steady]` (all rows × SCALE for MS ms, from ambient or from the steady state).
Other options: `--no-auto-resistors`, `--no-crop`, `--no-magnetic`, `--b-height MM`.

Example board with a shunt, a FET element and pull resistors: `sim/demo/demo.kicad_pcb`
(`python3 -m pcb_current_sim sim/demo/demo.kicad_pcb --vsource "J1.1>J1.2:12:supply" --element "J2.1-J2.2=2.4:LOAD:ext" --element "Q1.3-Q1.2=20m:Q1:0.35" --pulse 4:20:steady`).

## Suite
All windows run in one process and share the loaded board, the stackup and the component library. Every window has a **Suite** menu that leads to Home, the tools and the shared panels.
- **Home** shows the board, its stackup and what the scenario file holds, with tiles for each tool:
  - POWER: DC & Thermal, PDN (AC)
  - SIGNAL & FIELDS: SI & Fields, 3D extraction
  - RF / CIRCUIT: Matching & Tolerance, Full-wave (openEMS)

  DC & Thermal and Matching & Tolerance are built; the other tiles are marked *planned*.
- **Stackup & materials** shows copper thickness (with overrides) and dielectric thickness, εr and tanδ.
  - εr and tanδ come from the board's stackup, default to 4.3 / 0.02, and can be edited at a reference frequency.
  - Plies inside one dielectric are combined in series. KiCad's `addsublayer` plies are read.
  - The frequency dependence uses the wideband Djordjevic–Sarkar model, which is causal: εr falls slowly with f and tanδ stays nearly flat. A constant model is the alternative.
  - The DC tool's copper-thickness list edits the same overrides.
- **Component library** holds real-part models used by the matching and PDN tools. Each part has a tolerance (± % or absolute; capacitor codes A/B/C/D) and parasitics:
  - L: DCR, Q at f, SRF.
  - C: ESR or Q, ESL or SRF, dielectric with a temperature and DC-bias curve.
  - R: series L, parallel C.

  The panel plots |Z| and Q against f, at nominal and at the tolerance limits.
  User parts are stored in `~/.config/pcb_current_sim/library.json`, which the Flatpak and host installs share. Built-in parts hold generic typical values; duplicate one to edit it.
- **Results browser** lists saved results (`.npz` plus its `.json` sidecar) under the board folder, and the last unsaved run.
- **External tools** looks for openEMS, FastHenry, FastCap and ParaView.
  Inside the Flatpak, host programs can only be run after `flatpak override --user --talk-name=org.freedesktop.Flatpak org.kicad.KiCad`; without that, the 3D tools will export their input files instead.

**Matching & Tolerance** checks whether a matching network still works across real part tolerances.
- **Visual editor.** Symbols appear on the schematic as you add parts from the palette: series or shunt L, C and R, transmission lines, open and shorted stubs, and a two-port from an `.s2p` file.
  - New parts go in at the ▼ marker. Drag a part to move it; right-click it or use the buttons to flip it between series and shunt, duplicate or delete it.
  - Click a part, the source or the load to edit it in the property grid. Every edit redraws the plots.
- **Parts.** Pick a part from the component library or type your own values. Values accept SI prefixes (`4n7`, `2.2p`); tolerances take `5 %`, an absolute value (`0.1 pF`) or a capacitor code (A/B/C/D). "Save part to library…" stores a custom part.
- **Lines.** An ideal line (Z₀, εeff, loss) or a quasi-static microstrip (Hammerstad–Jensen, with conductor and dielectric loss). "Use board stackup" fills in h, εr and tanδ; "Width for Z0…" solves for the width. Tolerances on Z₀, width (etching) and εr are optional.
- **Source and load.** R + jX, series or parallel RLC, or an `.s1p` file such as an antenna's measured S11.
- **Specs.** One or more bands, each with a minimum return loss, a maximum insertion loss and a maximum VSWR.
- **Analyses.** Responses use an ABCD cascade computed for many samples at once (10k samples in about 1.4 s).
  - Corners: every variable at both limits.
  - Monte Carlo: uniform, or Gaussian with 3σ = tolerance (truncated at the tolerance).
  - Worst-case search: bounded Powell, started from the nominal point and the worst corners and samples. Resonant responses can be worst inside the tolerance box, not at a corner.
  - Sensitivity: a tornado chart (one variable at a time) and each variable's share of the Monte Carlo variance. A low total share means the response is non-linear in the tolerances, for example a match sitting at its optimum.
  - Optional temperature sweep, which drifts capacitors by dielectric.
- **Outputs.** Smith chart (nominal, Monte Carlo curves in band, load alone, RL spec circle), S11, S21 with the loss in the parts, VSWR, histograms with yield per band, sensitivity, a worst-case table and a BOM (CSV). The response can also be exported as CSV.
- **Saving.** The network goes into the scenario's `matching` section when a board is open, or into a `.match.json` file. It saves on every run and when the window closes.
- Not modelled: microstrip dispersion and thickness correction, coupling between parts, and pad or via parasitics (add them as elements).

**Scenario file.** `<board>.current_sim.json` (version 2) has one section per tool: `stackup`, `dc`, and later `matching`, `pdn` and so on.
Each tool rewrites only its own section. Version-1 files load as the `dc` section, and their copper-thickness overrides move to `stackup`.

## Model
- **Excitations.** Every row of the table is applied **at the same time**. A row is one of:
  - *Current path*: +I at the source pads and −I at the sink pads.
  - *Load*: draws I out of the copper at its + pads and returns it at its − pads (an IC, a motor). The table shows the voltage the load actually sees.
  - *Voltage source*: holds V(+) − V(−) (a regulator or supply). It is a Norton equivalent with 10⁶ S, so the error is 10 µV at 10 A. Its current is set by the loads and resistance elements on it.
  - A shared return path therefore carries the sum of the row currents.
  - The pads on one side of a row are tied together as a single equipotential.
  - For a THT pad, the electrode is its plated barrel on every layer. For an SMD pad, it is the pad area.
  - Each row reports ΔV and R_eff = ΔV/I. When rows share copper, ΔV includes the drop caused by the other rows.
- **Resistance elements.** A lumped resistance joins two pad electrodes, which may be on different nets.
  - Resistor footprints (reference `R…`/`RS…` or a `Resistor_*` library, exactly 2 pads, not DNP) are found automatically.
    The value is read from the Value field (`10k`, `4k7`, `0R1`, `R005`, `5m` = milli, `1M` = mega), and the power rating is guessed from the package (0603 = 0.1 W … 2512 = 1 W, DIN0207 = 0.25 W).
    Parts that can't be read (4-pad Kelvin shunts, odd values) are listed as warnings.
  - Any other part can be added by picking two of its pads and entering R and an optional rating: a FET's R_on, an inductor's or ferrite's DCR, a fuse, or a load or wire between two connectors.
  - Only elements that can carry current are used: starting from the rows' nets, enabled elements pull in further nets, and dead-end nets (reached through one element only and not excited) are dropped.
  - Each element's I²R heats its two pads in the thermal model. The Elements table shows R, I, V, P and P against the rating.
  - With current excitation, an element that bridges two separately excited loops (for example a bleeder across the supply) carries ≈0 A, because nothing sets the voltage between the loops.
  - 0 Ω parts are ideal links (1 µΩ). TCR and Kelvin sensing are not modelled.
- **Speed.** By default the electrical grid covers only the bounding box of the simulated copper, snapped to the whole-board lattice, so the result is identical.
  The thermal grid adds 20 mm of board around it (`thermal_margin`), and its cut edges are adiabatic.
  Sparse factorisations use a symmetric minimum-degree ordering, which is about 3× faster than the default.
- **Geometry.** The tool reads:
  - tracks and arcs on all copper layers
  - vias, with barrel resistance R = ρL / (π·t_plate·(d + t_plate))
  - filled zones
  - pads: rect, roundrect, oval, circle, trapezoid, and custom (anchor plus polygons), including footprint rotation
  - drills, which are removed from the copper
- **Copper thickness and dielectric gaps** come from `(setup (stackup))`. If the file has no stackup, the tool uses 35 µm copper and the board thickness split evenly.
  Override the thickness per layer if your fab's finished outer copper is thicker (often 45–60 µm after plating).
- **Discretisation.** The tool uses a cut-cell finite-volume grid: each link conducts through the copper fraction of the shared cell face, sampled 4×4.
  - Axis-aligned and 45° tracks both converge to within 1–2% at 8 cells per track width.
  - With the error estimate enabled, a second solve at twice the pitch gives a ± error for every row.
- **Thermal model.** Each copper layer is a thermal sheet: copper of all nets plus a share of the dielectric.
  - Sheets are coupled through the dielectric and through the copper barrels of all vias and plated holes.
  - Both outer faces lose heat to ambient. By default h comes from the local surface temperature: natural convection of a vertical plate, 1.42·(ΔT/L)^0.25, plus radiation εσ(T⁴ − Ta⁴)/ΔT with ε = 0.9. That is about 11 W/m²K at a 10 K rise and 18 W/m²K at a 100 K rise. Untick "h from temperature" (CLI `--h-conv 12`) to use a fixed h.
  - Element I²R is spread over the area of its two pads, not put into a single cell.
  - The solver iterates ρ(T) = ρ20·(1 + 0.00393·(T − 20)) until T stops changing (Picard iteration).
  - If the copper would exceed its melting point, it stops and flags **thermal runaway**.
  - It does not model component self-heating, enclosures or airflow.
- **Current density J** comes from the face-normal fields (E/ρ), averaged per cell. The viewer can draw streamlines.
- **Magnetic field.** Biot–Savart from the solved currents: sheet currents per layer (FFT convolution), vertical barrel currents and resistance-element bodies.
  It is shown on a plane `b_height` (default 1 mm) above the top surface (top layer view) and below the bottom surface (bottom layer view), as |B| and the normal component Bz (up), with Bx, By and Bz in the cursor readout.
  Only the board's own currents are included; external wiring is not.
  Checked against an infinite strip (within 1 %).
- **Loop inductance.** Each row is driven alone with 1 A (other current sources open, voltage sources shorted).
  The partial inductance matrix L_mn = μ0/4π ΣΣ m_m·m_n/R is computed over the current moments, with the exact cell self-term and Gauss quadrature for near cells. Barrels use Grover's parallel-filament formula.
  Each loop is closed by a straight line from the pads where the current leaves the copper back to where it enters (the source or component). Define rows so that this closure is short: a capacitor's two pads, a connector's pins, or a voltage source with its load as an element.
  It uses the DC current distribution; at MHz the return current hugs its track and L drops a little. Checked against Grover for a hairpin (within 4 %).
- **Pulse (transient).** All rows × `pulse_scale` for `pulse_ms`, starting from ambient or from the steady state, using implicit Euler.
  Heat capacity is copper plus the FR4 the heat can reach (0.886·√(αt), re-lumped with energy conservation as the pulse goes on).
  ρ(T) is coupled; the electrical problem is re-solved every 20 K.
  The pulse uses its own thermal grid: the copper plus 3·√(α_cu·t) of board.
  The FR4 capacity growth is applied every step as an energy-conserving sink C′(t)·φ, so T(t) is smooth. The default is about 1.6 % conservative against the converged result.
  The tool reports the hottest point over time, the time to melt if the copper melts, the T map at the end, and the peak T per track.
  Validation: with FR4 removed it reproduces Onderdonk (10.03 ms against 10 ms). With FR4, a 1 mm 1 oz trace lasts about 1.8× longer (18.2 ms converged, 17.9 ms at the default steps).
  Resistance elements heat their pads; their body heat capacity is ignored, which errs on the hot side.

## Outputs
- **Viewer.**
  - Pick the layer and the quantity: |J|, V, T, power density, |B|, B normal, or T at the end of the pulse.
  - The cursor readout shows each quantity at the pointer.
  - Click the map to set a ΔV reference point.
  - Selecting a table row marks its location on the map.
  - Resistance elements are drawn as dashed yellow links.
  - **3D tab** (wx viewer): an interactive Plotly scene in an embedded web view (WebKitGTK).
    - Each copper layer is a sheet at its stackup height (z exaggerated), coloured by |J|, V, T, power density or end-of-pulse T. Vias and barrels are vertical bars coloured by current.
    - It also shows the resistance elements, the board outline, and the B-normal planes above and below.
    - Drag to rotate, scroll to zoom; hover shows values. *Open in browser* and *Save 3D HTML…* give a self-contained page that works offline.
    - The Tk viewer opens the same scene in the browser. On the CLI it is `--formats html3d`.
    - It is a visualisation of the 2.5D results; the solver itself is not 3D.
- **ParaView.** The *ParaView…* button or `--formats vtk` writes a folder you can open in ParaView. It uses VTK XML (zlib, base64), written with numpy only.
  - `board.vtm` is a multiblock with the copper sheets at their true heights (|J|, J vector, V, power density, copper fraction, net id), the thermal sheets (T and end-of-pulse T), the B planes (B vector, |B|, Bz), and line sets for the vias (current), elements (I, P, R) and outline.
  - `pulse.pvd` is the pulse temperature as a time series of 20 frames that ParaView animates.
  - Axes: x right, y up (KiCad y flipped), z up with the top copper surface at 0, true scale. Use *Transform* to scale z.
  - `README.txt` in the folder lists useful filters: Threshold, Glyph on J_vector, Contour on V, Plot Over Line.
  - Install ParaView with `flatpak install flathub org.paraview.ParaView`.
  - Checked by reading every file back with VTK 9.7's XML readers.
- **Exports.** You can export:
  - the current view as PNG, SVG or PDF
  - every layer as images
  - a multi-page PDF report (summary, tables, maps)
  - the tables as CSV
  - an interactive Plotly HTML page
  - the result as `.npz` + `.json`, which can be reopened later
- **Tables.**
  - Excitations.
  - Pads: role, current, V, T.
  - Vias and barrels: current, IPC capacity, parallel-via groups and each via's share.
  - Elements: R, I, V, P, rating use and pad temperature.
  - Inductance: self L per row, mutual M and coupling k between rows, closure length.
  - Pulse: hottest temperature over time (also plotted in the PDF report).
  - Loss ranking per copper item: P, share of the net's loss, ΔV, I, crowding factor, IPC-2221 use at 10 °C and 20 °C, and Onderdonk fusing time.

## Tests
`python3 -m pytest tests`. The tests cover:
- analytic straight, arc and via resistances
- rotated footprints
- parallel vias
- a shared return path
- error cases
- 45° tracks against axis-aligned ones
- the analytic thermal sheet, the automatic h against the heat balance, via barrels coupling the layers, element heat spread over pads
- scanline zone fill against matplotlib's point-in-polygon test
- IPC and Onderdonk formulas
- the reference board (27.2 mΩ and 20.2 mΩ)
- voltage source with a resistive load, load voltage against IR drop
- B above and below a strip, hairpin loop inductance against Grover, Grover filaments
- adiabatic pulse against Onderdonk, and a pulse from steady state that leaves T unchanged
- resistor values and ratings, a series resistor across nets, parallel resistors sharing by conductance, a user element on a 3-pad part with dead-end pruning, cropped against full-board grids, and element heating
- matching (`tests/test_matching.py`): an L-match is matched and lossless; Q loss in the inductor; a quarter-wave transformer; quarter-wave open and shorted stubs; a 50 Ω microstrip on FR4; Touchstone formats and S → ABCD; an S2P element equal to a series inductor; corners, Monte Carlo yield, Gaussian against uniform, the worst case at least as bad as corners and samples; the temperature variable on X7R; setup round trip and the BOM
- suite (`tests/test_suite.py`): stackup plies and `addsublayer`, the Djordjevic–Sarkar fit and its causality, scenario v1 → v2 migration with sections kept, stackup overrides persisting across sessions, SI value parsing, L/C part models (SRF, Q fit, DC bias, Monte Carlo broadcasting), and the library round trip

## Layout
`pcb_current_sim/` holds the core: `sexpr`, `board`, `elements`, `raster`, `solver`, `thermal`, `magnetic`, `results`, `export`, `vtk_export`, `cli` and `scenario`.
It also holds `kicad_link` (the kipy wrapper), `tables` (helpers shared by both GUIs), `wxgui/` (the wx app and viewer, used inside KiCad) and `gui/` (the Tk app and viewer).
`pcb_current_sim/matching/` holds the matching tool's core: `model` (elements, ports, bands, ABCD cascade), `analysis` (corners, Monte Carlo, worst case, sensitivity, BOM), `touchstone`, `microstrip` and `plots`. `wxgui/matching.py` is its window and `wxgui/schematic.py` the symbol canvas.
`pcb_current_sim/suite/` holds the toolkit-neutral suite core: `session` (shared state and change events), `stackup`, `library`, `tools` (the tool registry), `external` (solver detection) and `results_index`. `wxgui/suite.py` holds Home, the Suite menu and the shared panels.
`kicad_plugin/` holds the KiCad plugin files: `plugin.json` (two actions), `_bootstrap.py`, `home.py`, `action.py`, icons and `install.sh`.
`sim/current_density.py` is the original single-board prototype, kept for reference.
`sim/demo/` holds the demo board and four wx smoke tests: `smoke.py` (main window and viewer), `smoke3d.py` (3D tab), `smoke_suite.py` (Home, the shared panels and the DC window) and `smoke_matching.py` (builds a network in the editor, runs every analysis and screenshots every tab). The last two work on a copy in `smoke_out/`, and `smoke_suite.py` clears that folder first. Run them with the plugin venv's python inside the Flatpak; they write screenshots next to themselves.
