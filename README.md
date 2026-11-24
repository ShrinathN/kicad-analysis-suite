# kicad-analysis-suite

# tl;dr (no AIsloppa)

 This is the software suite for KiCad which lets kycad run a variety of simulations in it. I just wanted to have access to the same features as many paid and professional EDA packages out there in KiCad, which is why this was made. There will be bugs and there will be quirks. This was made by me using Claude Opus 5.5 over a few days and is being developed right now. It is mostly for my personal use but you are free to do as you see fit.

**PCB Analysis Suite for KiCad**: a KiCad 10 plugin, also usable standalone, for analysing printed circuit boards. It has six tools so far:

- **DC & Thermal**: current density, IR drop and temperature on a real board, including resistance elements, linear and switching regulators (with their losses and junction temperature), current pulses (fusing) and magnetic field maps.
- **Switching Loop**: the hot loop and gate loops of a switching power stage (buck, boost, half-bridge, 4-switch buck-boost; synchronous or with a diode): loop inductance and where it sits, ringing frequency, gate-loop and common-source inductance, from a partial-inductance (PEEC) solve of the layout; switching edges in time, a snubber designer and an EMI estimate.
- **PDN (AC)**: the impedance of a supply rail seen by each chip, from the board's copper, the decoupling capacitors and the regulator, against the target impedance; which capacitors matter, plane resonances, and the rail's voltage over time when the loads step, with the regulator's control loop fitted to its datasheet's load-transient plot.
- **Trace R/L/C → SPICE**: R, L, C, Z₀ and delay of the copper between any pads, from a 2D field solve of each cross-section, exported as a SPICE RLC subcircuit.
- **3D extraction**: R(f), L(f), Q and coupling of any copper between pads, with skin and proximity effect, eddy currents in nearby copper and **magnetic cores** (ferrite sheets, planar E cores in the board's slots, rods, toroids; lossy materials from a datasheet fit), and a wideband SPICE model. FastHenry's method, built in.
- **Matching & Tolerance**: draw an RF matching network with real parts, then see whether it still meets its specs across part tolerances.

**Simulation tests**: keep any tool's setup as a test with pass / fail checks (T_max < 85 °C, the rail within its target, loop L < 3 nH, …) and rerun them whenever the board changes, from the command line, a Makefile or a script. Each run writes a report to `simulations/<timestamp>/` and compares it with the previous run:

```sh
python3 -m pcb_current_sim init board.kicad_pcb                       # tests from the setups you made in the windows
python3 -m pcb_current_sim run board.simtests.toml --test "3V3 rail"   # or --all; exit code 1 when a check fails
```

More tools (full-wave export, signal integrity) are planned. See the [roadmap](docs/ROADMAP.md).

![Home window](docs/screenshots/home.png)

## Typical flows

Save the board in KiCad (refill zones first), click **PCB Analysis Suite** on the toolbar and open a tool from Home. The [user guide's workflows](docs/USER_GUIDE.md#workflows) give every step; in short:

| Question | Tool | Steps |
|---|---|---|
| How much voltage does this path lose, and how hot does it get? | DC & Thermal | select source then load pads in KiCad → *Add from KiCad selection* → check the elements → *Run simulation* → V, \|J\|, T maps |
| Do my regulators stay cool and out of dropout? | DC & Thermal | *Add regulator…* per regulator → a voltage source at the input, loads on the outputs → run → Regulators tab |
| How big is my converter's hot loop, and does it ring? | Switching Loop | pick the switch node → *Detect parts* → fix the Checks → operating point → *Analyse* → *What it means*, Snubber, EMI, Compare |
| Is the rail decoupled well enough? | PDN | pick rail and returns → target → regulator (fit its datasheet plot) → *Analyse* → impedance against the target, decaps, what-if, load steps |
| What SPICE model is this trace or power path? | Trace R/L/C → SPICE | ports (pads in order) → return nets → *Extract* → *Save subcircuit…* |
| What are this coil's L and Q, with the pour under it and a ferrite sheet? | 3D extraction | port across the coil → tick the pour's net → *Magnetic cores → Add…* → *Analyse* → R, L, Q (f), SPICE |
| What does my planar inductor give with this E core and gap? | 3D extraction | slots in Edge.Cuts → core *E*, dimensions, gap → *Snap to the board's slots* → *Analyse* |
| Does the matching network survive tolerances? | Matching & Tolerance | source, load, bands → parts from the library → *Run analysis* → yield, Smith cloud, sensitivity |
| Did the last layout change break anything? | Simulation tests | *Save as test…* in each tool → checks → `python3 -m pcb_current_sim run board.simtests.toml --all` |

## DC & Thermal

Pick the pads where current enters and leaves, or add voltage sources and loads, and solve all of them together on the board's copper.

| Current density | Temperature (electro-thermal) |
|---|---|
| ![Current density](docs/screenshots/dc_viewer_current_density.png) | ![Temperature](docs/screenshots/dc_viewer_temperature.png) |
| **Magnetic field above the board** | **3D view** |
| ![B field](docs/screenshots/dc_viewer_bfield.png) | ![3D view](docs/screenshots/dc_viewer_3d.png) |

- Reads tracks, arcs, vias, filled zones, pads and the stackup directly from the `.kicad_pcb` file.
- Current paths, loads and voltage sources. Resistors on the board are found automatically, and any two pads can be joined by a resistance (FET R_on, inductor DCR, external load).
- Electro-thermal coupling with temperature-dependent copper resistance and natural convection plus radiation.
- Pulse heating and fusing, magnetic field maps and loop inductance.
- Tables per pad, via, track and element; export to PDF, HTML, CSV, PNG/SVG and ParaView.

![DC & Thermal window](docs/screenshots/dc_main.png)

## Switching Loop

Pick the switch node (or let the tool find it): it finds the switching parts, the hot-loop capacitors and the gate drives (driver pin, gate resistor, gate, return pin) and meshes the copper of all their nets, ground included, as conductors. A partial-inductance (PEEC) solve gives the copper's impedance between the parts' pins; the parts add their datasheet models (package inductance, capacitances, ESL).

- For each switching edge: the loop inductance across the switch that turns off, split into board copper and parts, and a bar chart of where it sits (each net and layer, vias, each package, each capacitor's ESL).
- The ringing frequency and Q with the switch's own capacitance, the expected overshoot per A/ns, and a starting snubber.
- Per FET: the gate-loop inductance, gate ringing, and the common-source inductance (the mutual between the gate loop and the power loop), which slows switching.
- Current-density and inductance-density maps at the ringing frequency, board capacitance between the nets, and a SPICE subcircuit of the copper (coupled inductors).

- **Switching edges** (simulated in time): one turn-on and one turn-off of the hard-switched switch at the inductor current you give, with the copper and the parts as one circuit: switch-node overshoot and ringing, dv/dt, di/dt, switching loss (E_on + E_off) × f_sw, reverse recovery, and how far the other FET's gate is pushed toward its threshold (shoot-through risk). A built-in solver uses the datasheet values; a FET can instead use its manufacturer's SPICE model, simulated in ngspice (bundled with KiCad).
- A cell diagram, readiness checks with one-click fixes, and a plain-language "What it means" summary.

| | |
|---|---|
| ![Summary](docs/screenshots/loop_summary.png) | ![Where the L is](docs/screenshots/loop_breakdown.png) |
| ![Switching edges](docs/screenshots/loop_switching.png) | ![Snubber](docs/screenshots/loop_snubber.png) |
| ![EMI estimate](docs/screenshots/loop_emi.png) | ![Compare with the baseline](docs/screenshots/loop_compare.png) |
| ![Current map](docs/screenshots/loop_map.png) | ![Impedance](docs/screenshots/loop_impedance.png) |

## PDN (AC)

Pick a rail and its return nets: the tool finds the decoupling capacitors, the regulator and the chips on the rail, meshes the rail's copper over its return planes, and sweeps the impedance at every chip from 10 Hz to the GHz range.

- Z(f) at each load against Z_target = V × deviation / ΔI, for the bare board, with the capacitors, and complete; the anti-resonance peaks and plane resonances.
- A decap table: C at the rail voltage (DC bias), ESR, ESL, mounting inductance, resonance, distance to the nearest load and the band where each one carries the most current; "what if" each capacitor were left off.
- Voltage maps of the rail at a chosen frequency (where a plane resonance sits).
- The regulator as R + L or as its control loop, fitted to the load-transient plot of its datasheet: a built-in digitiser reads a screenshot (traces by colour, or by clicked points for black-and-white plots), and the fit transfers to your board's capacitance.
- Load steps: each chip's current as a SPICE-style PULSE, and the rail's voltage at each chip over time, split by which load causes it.
- Layout findings with their place on the board: voids and slots in the return plane the rail's current crosses, copper with no return under it, decaps with long return paths, chips with no decap nearby.
- Touchstone export (`.sNp`) of the complete rail between the loads, or of the bare board with a port at every load, decap and the regulator; regulator fits saved by part number for other boards.

| | |
|---|---|
| ![Impedance](docs/screenshots/pdn_impedance.png) | ![Load steps](docs/screenshots/pdn_load_steps.png) |
| ![Plane resonance map](docs/screenshots/pdn_map.png) | ![Regulator fit](docs/screenshots/pdn_regfit_fit.png) |

## Trace R/L/C → SPICE

Pick two or more pads on one net and the return nets. The tool follows the copper between the pads (tracks, arcs, vias, stubs), solves the field of every cross-section against the return copper, and builds an RLC ladder.

- Any geometry: microstrip, stripline, coplanar lines, and a GND pour on the other side of a 2-layer board, all from one 2D field solver (within 2 % of Hammerstad–Jensen and Wheeler).
- DC R from the board's real copper (the DC solver), L and C per segment, via L and C, pad C, open stubs.
- Copper pours as real conductors: meshed over their return plane and reduced exactly to R, L and C between everything that touches them.
- Resistors, diodes, capacitors and inductors between any two pads, also across nets (trace → part → trace), with return-net pads as the reference.
- Outputs: per-path totals, a segment table with Z₀ and the return found, the path coloured by Z₀ with impedance steps marked, a drawn schematic of the ladder, and a `.subckt` of plain R, L and C that runs in any SPICE.

The example below is a DC-DC converter's input path on a 2-layer board: from the input connector through a supply pour, the inductor (added as an element) and the switch-node pour to the switch, over a ground plane. The input capacitors are capacitor elements to ground.

| | |
|---|---|
| ![Schematic tab](docs/screenshots/extract_schematic.png) | ![Board tab](docs/screenshots/extract_board.png) |
| ![Summary tab](docs/screenshots/extract_summary.png) | ![SPICE tab](docs/screenshots/extract_spice.png) |

## 3D extraction

Pick pads (a port: current in at the + pads, out at the − pads); tick nearby nets to include their eddy currents; add magnetic cores. The copper is cut into filaments graded to the skin depth (FastHenry's method, checked against FastHenry), cores into magnetisation cells, and one solve gives R(f) and L(f) from DC up, with the coupling between several ports.

| R, L, Q against frequency | Current: eddy currents in the pour under a coil |
|---|---|
| ![R, L, Q](docs/screenshots/em3d_rlq.png) | ![Current map](docs/screenshots/em3d_current.png) |
| **A ferrite sheet over the coil** | **A planar E core snapped to the board's slots** |
| ![Core dialog](docs/screenshots/em3d_core_dialog.png) | ![E core](docs/screenshots/em3d_core_snap.png) |

- Uses: PCB coils (L, R_ac, Q, with the pour under them and a ferrite sheet), AC resistance and loss of power copper, magnetic pickup between a switching loop and a sense loop, shunt Kelvin routing, current sharing in via arrays, planar inductors and transformers.
- Magnetic cores: plate / ferrite sheet, E-E or E-I (rectangular or round centre leg, centre and outer gaps), rod, toroid, or any boxes; placed by board coordinates, relative to a footprint or snapped to Edge.Cuts slots. Materials from a library of generic grade classes or your own grade fitted to its datasheet's μ', μ'' points; their loss appears as R.
- Outputs: per port R and L at DC, mid band and f max, Q, M and k between ports, current maps per layer with track cross-sections, and a wideband SPICE subcircuit (plain R-L branches, right from DC to f max, core loss included).

## Matching & Tolerance

Build a ladder network in a visual editor: the symbols appear as you add parts, and the plots follow every edit.

![Matching editor](docs/screenshots/matching_editor.png)

| Monte Carlo S11 against the spec | Yield histograms |
|---|---|
| ![S11](docs/screenshots/matching_s11.png) | ![Histograms](docs/screenshots/matching_histograms.png) |
| **Smith chart** | **Sensitivity** |
| ![Smith chart](docs/screenshots/matching_smith.png) | ![Sensitivity](docs/screenshots/matching_sensitivity.png) |

- Series and shunt L, C and R, transmission lines (ideal, or microstrip from the board stackup), open and shorted stubs, and S2P two-ports.
- Real parts from a shared component library: tolerance, Q, self-resonance, ESR, ESL, and capacitor drift with temperature and DC bias.
- Source and load as R + jX, an RLC model or a measured `.s1p` file.
- Corners, Monte Carlo (uniform or Gaussian), a worst-case search, sensitivity, yield per band and a BOM.

The screenshot example matches a small 2.4 GHz antenna (20 Ω + 1.5 nH + 3 pF) to 50 Ω. It uses a 1.3 pF shunt capacitor and a 1.5 nH series inductor, both 0402 parts with ±0.1 pF / ±0.1 nH tolerance.

## Shared panels

| Stackup & materials | Component library |
|---|---|
| ![Stackup](docs/screenshots/stackup.png) | ![Component library](docs/screenshots/component_library.png) |

**Components**: the board's parts between pads (detected resistors and anything you add: a FET's R_on, an inductor's DCR, a diode, a capacitor, a regulator), defined once and switched on or off per tool.

![Components](docs/screenshots/components.png)

## Install

1. Install KiCad 10 and enable the API: *Preferences → Plugins → Enable KiCad API*.
2. Either install the zip package: run `python3 kicad_plugin/build_pcm.py` (writes `dist/kicad-analysis-suite-<version>.zip`), then in KiCad open *Plugin and Content Manager → Install from File…* and pick the zip.
   Or run `kicad_plugin/install.sh`, which copies the plugin into the plugin folder of a native or Flatpak KiCad. Use one method, not both.
3. Restart the PCB editor. On first load KiCad installs the Python dependencies (numpy, scipy, matplotlib, plotly, kicad-python), which needs network access.
4. Two toolbar buttons appear: **PCB Analysis Suite** opens the Home window, and **DC & Thermal** opens that tool directly.

Save the board (and refill zones) before running: the tools read the saved `.kicad_pcb` file.

Standalone, without KiCad running:

```sh
python3 -m pcb_current_sim.wxgui board.kicad_pcb --home                        # wx GUI, Home window
python3 -m pcb_current_sim board.kicad_pcb --row "J1.2>J2.2:3" --out results   # command line
```

## Documentation

- [Documentation (HTML)](docs/index.html): workflows, what each tool does, how it works, its limits, interactive calculators and realistic use cases for every tool. Open the file in any browser; it needs no network.
- [User guide](docs/USER_GUIDE.md): step-by-step [workflows](docs/USER_GUIDE.md#workflows), every tool and option, the physical models, outputs and the CLI.
- [Development](docs/DEVELOPMENT.md): environment, tests, smoke tests, screenshots and code layout.
- [Roadmap](docs/ROADMAP.md): the plan for the remaining tools.
- [CLAUDE.md](CLAUDE.md): the detailed design log kept during development.

## Status

- All six tools work and are covered by tests (`python3 -m pytest tests`: about 200 tests, 2 minutes).
- Tested in a running KiCad 10 (Flatpak): both toolbar buttons, the KiCad link (pad selection, save, refill, highlighting) and the Plugin and Content Manager install.
- 3D extraction: magnetic cores are linear for now; DC bias (L against current), core loss against flux density and temperature are next.
- Developed and tested on Linux with the KiCad 10 Flatpak.

## License

[GPL-3.0](LICENSE), the same licence as KiCad. You may use, modify and share this software. If you distribute it, or a modified version, you must do so under the same licence and include the source.
