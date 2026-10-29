# Roadmap

The suite is being built in this order. Steps 1 and 2 are done; the rest are planned. The plan below was worked out with the commercial tools as a reference for how simulations are grouped.

| Step | Tool | Status |
|---|---|---|
| 1 | Suite framework: Home, stackup & materials, component library, results browser, external tools | done |
| 2 | Matching & Tolerance | done (synthesis and design centering still to come) |
| 3 | PDN (AC power integrity) | next |
| 4 | 3D extraction (FastHenry / FastCap) | planned |
| 5 | Full-wave export (openEMS) | planned |
| 6 | SI & Fields | planned |

## The plan (agreed 2026-10-29)
The user looked at how the commercial tools group their simulations and agreed this structure:
- Cadence splits PowerDC (DC IR plus electro-thermal) from PowerSI and OptimizePI (AC PDN, decaps).
  Aurora covers impedance, crosstalk and return path; Celsius is thermal; Clarity is full-wave.
- Ansys SIwave holds DC, SYZ, resonance and EMI in one tool. Icepak does thermal, fed by SIwave's loss maps.
- Keysight PIPro holds DC IR, electro-thermal, AC Z, resonance and decap optimisation, all with one shared setup.

The decisions:

1. **DC & Thermal**: the current window.
   - Contents: DC current density, IR drop, resistance elements, electro-thermal, pulse/fusing and DC B-field maps.
   - Later: electromigration lifetime (Black), via barrel fatigue, tolerance Monte Carlo, component self-heating and airflow.
2. **PDN (AC power integrity)**: the first new window to build.
   - Contents: Z(f) against target impedance (Z_t = V·ripple / I_step), a decap table (auto-detect C footprints; ESR, ESL and mounting L), a VRM model (R + L), plane resonances, L(f) and EMI from resonances.
   - Later: decap optimisation.
   - It shares the VRM and load definitions with window 1, as PIPro does: the "V" and "load" rows can be imported.
   - **Loop inductance moves here** as L(f), with the current DC-pattern value as its low-frequency point. The DC B maps stay in window 1.
   - Solver plan:
     - A frequency-domain mesh of the power-net copper on a coarser AC grid (about 0.25 to 0.5 mm).
     - Each link: (R_s,pow(f) + R_s,ret(f) + jω·μ0·h) / g, where g = w/l comes from the existing link conductance (G = σ·t·g).
     - Skin effect: R_s = ρ / (δ·(1 − e^(−t/δ))).
     - h is the dielectric distance to the nearest return-net copper. Treat return planes on both sides as two in parallel.
     - Each cell gets a shunt jωC + ω·C·tanδ to the return, with C = ε0·εr·A / h. Defaults are εr = 4.3 and tanδ = 0.02.
     - Power-net vias: L ≈ μ0·Δz/(2π)·ln(s/r), with s the distance to the nearest return via.
     - Ports are super-nodes against the return as reference: decaps as series RLC, VRM as R + L, load = the measured port.
     - Warn where power copper has no return copper under it.
     - Why not reuse the DC current pattern for L: at PDN frequencies the return current runs under the power copper, while the DC return spreads out, so the DC-pattern L is too large.
3. **Signal integrity & fields**:
   - Trace Z₀, effective permittivity and crosstalk from a 2D cross-section Laplace solve.
   - Return-current path at frequency (moved here from PDN).
   - HV E-field and clearance/creepage.
4. **Full-wave export**: openEMS, for antennas and S-parameters.

Framework, to build first:
- Launching: superseded by the update below: a Suite (Home) button plus a direct DC & Thermal button, as two actions in the IPC `plugin.json`.
- A shared board, stackup and materials panel (εr and tanδ get added).
- One scenario file, `board.current_sim.json`, with one section per window. Old files must keep loading.
- The shared result viewer (2D, 3D, tables, ParaView), extended with frequency plots for PDN.

### Update, later on 2026-10-29: Home window, Matching & Tolerance tool, 3D tools, build order
**Home window and launching.** This replaces "one button per window"; the user left the details to me.
- KiCad buttons:
  - "PCB Analysis Suite" opens the **Home** window.
  - A direct "DC & Thermal" button stays as a shortcut.
- All tools run in one process and share the loaded board, stackup and component library. Every window has a "Suite" menu (Home, the other tools).
- Home layout:
  - A header with the board, layers and stackup summary, and its saved state.
  - Tiles in three groups:
    - POWER: DC & Thermal, PDN (AC).
    - SIGNAL & FIELDS: SI & Fields, 3D extraction.
    - RF / CIRCUIT: Matching & Tolerance, Full-wave (openEMS).
  - A shared row:
    - Stackup & materials (εr and tanδ against f).
    - **Component library**: real-part models shared by the PDN decaps and the matching tool.
    - Results browser.
    - External tools: detect openEMS and FastHenry/FastCap, and handle the Flatpak sandbox (`flatpak-spawn --host` needs `--talk-name=org.freedesktop.Flatpak`, or export only).

**Matching & Tolerance tool.** Circuit-level, pure numpy, no external solver.
- Purpose, in the user's words: play with *real* component values and see if the matching network still works across tolerances.
- **Ladder builder only.** Series and shunt slots, each holding L, C, R, a transmission-line section (Z₀ and εeff from the stackup) or an open/short stub. A netlist editor and KiCad import are deferred.
- Input and output ports against ground. Z_S and Z_L can be:
  - a constant R + jX;
  - an RLC model;
  - optionally a `.s1p` file, for example the antenna S11 from openEMS later.
- Real models:
  - **L:** tolerance (±% or ±nH), DCR, Q at f_Q, SRF. Model: L in series with R(f) = DCR + k√f (k fitted to Q), in parallel with C_p = 1/((2π·SRF)²·L).
  - **C:** tolerance (±% or pF codes A/B/C/D = 0.05/0.1/0.25/0.5), Q or ESR, ESL or SRF, C0G against X7R (tempco, DC bias).
  - **R:** tolerance, L_s, C_p.
- **Vendor S2P:** only if cheap. Plan: our own small Touchstone parser, with an S2P part used as a fixed element (no tolerance). Drop it if it costs more than that.
- Analysis:
  - corners (2^N);
  - Monte Carlo (10k–100k samples), with **uniform or Gaussian (3σ = tolerance) selectable in the UI**;
  - a bounded worst-case optimiser, because resonant responses are non-monotonic and corners can miss the worst case;
  - sensitivity / variance contribution;
  - optional temperature sweep.
  - Batched: ABCD cascade over samples × frequencies.
- Specs are multi-band: RL / VSWR / IL limits per band.
- Outputs:
  - a Smith chart with the nominal curve and the MC cloud or envelope;
  - S11, S21 and VSWR against f, with min/max and 5–95% bands;
  - IL and efficiency;
  - yield, overall and per band;
  - histograms and a sensitivity chart;
  - a BOM.
- Step 2: synthesise L / π / T from Z_S and Z_L, snap to E-series or library values, then design centering (maximise yield).

**3D tools (discussed):**
- **(1) openEMS** (full-wave): antennas (S11, input Z, patterns, efficiency, ground and enclosure effects), interconnect S-parameters and TDR (Touchstone export), near- and far-field EMI, resonances. Palace is optional.
- **(2) FastHenry / FastCap** (3D quasi-static): commutation-loop L, via arrays, shunt ESL and sense coupling, R_ac(f), PCB coils (L, Q, k), switch-node C to ground (common-mode EMI), capacitive crosstalk and touch pads. They auto-generate SPICE subcircuits and feed the PDN and SI windows.
- 3D electro-thermal (Elmer) comes later. CFD (OpenFOAM) and thermo-mechanical are skipped.

**Build order (my call; the user left it to me):**
1. Suite framework: Home, shared stackup and materials, component library, scenario sections, "Suite" menus, plugin.json actions.
2. **Matching & Tolerance**: self-contained and quick to finish, and it establishes the component library.
3. PDN.
4. FastHenry / FastCap extraction.
5. openEMS export.
6. SI & Fields.

## Other ideas
- Canvas overlay in KiCad: |J| contours as user-layer graphics through a kipy commit.
- Per-pad contact resistance; finished-copper and plating options; thermal reliefs.
