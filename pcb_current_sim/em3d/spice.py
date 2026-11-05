"""SPICE subcircuits from a 3D extraction.

* `foster`: the wideband model.  One port: plain parallel R-L branches between the two pins
  (Y = Σ 1 / (R_k + s L_k): exact at DC, R and L right at every frequency, any SPICE).  Several
  ports: one R-L branch per mode, driven by the ports through voltage-controlled voltage sources
  and feeding them back through current-controlled current sources (E, F, V, R, L: SPICE3,
  ngspice, LTspice).
* `fixed`: R and coupled L at one frequency (R, L, K only), for any number of ports.

Pins: one pair per port, NAME_p and NAME_m (the + and - pads).
"""
import math

import numpy as np

TWO_PI = 2 * math.pi


def _n(x):
    """SPICE number (Meg for 1e6, plain exponent otherwise)."""
    return f"{x:.6e}"


def _pins(names):
    return [f"{n}_{s}" for n in names for s in ("p", "m")]


def _safe(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def foster(res, subckt="EM3D", drop=1e-9, header=()):
    model, names = res.model, [_safe(n) for n in res.names]
    C, tau = model.C, model.tau
    g = (C ** 2).sum(axis=1)
    keep = np.flatnonzero(g > drop * g.sum())
    out = [f"* {line}" for line in header]
    out += [f"* Wideband model ({len(keep)} modes), valid {res.f[0]:.3g} Hz - {res.f[-1]:.3g} Hz and at DC",
            f".subckt {subckt} " + " ".join(_pins(names))]
    if len(names) == 1:
        p, m = f"{names[0]}_p", f"{names[0]}_m"
        for j, k in enumerate(keep):
            R = 1.0 / g[k]
            if tau[k] * R > 0:
                out.append(f"R{j} {p} n{j} {_n(R)}")
                out.append(f"L{j} n{j} {m} {_n(tau[k] * R)}")
            else:
                out.append(f"R{j} {p} {m} {_n(R)}")
    else:
        for j, k in enumerate(keep):
            # mode voltage v_k = Σ c_kp V_p: a chain of E sources from node 0 of the mode
            prev = "0"
            for q, nm in enumerate(names):
                if abs(C[k, q]) < 1e-15:
                    continue
                node = f"e{j}_{q}"
                out.append(f"E{j}_{q} {node} {prev} {nm}_p {nm}_m {_n(C[k, q])}")
                prev = node
            out.append(f"R{j} {prev} r{j} 1")
            if tau[k] > 0:
                out.append(f"L{j} r{j} s{j} {_n(tau[k])}")
                out.append(f"V{j} s{j} 0 0")
            else:
                out.append(f"V{j} r{j} 0 0")
            for q, nm in enumerate(names):
                if abs(C[k, q]) >= 1e-15:
                    out.append(f"F{j}_{q} {nm}_p {nm}_m V{j} {_n(C[k, q])}")
    out += [f".ends {subckt}", ""]
    return "\n".join(out)


def fixed(res, f, subckt="EM3D_F", header=()):
    names = [_safe(n) for n in res.names]
    Z = res.at(f)
    L = np.imag(Z) / (TWO_PI * f)
    out = [f"* {line}" for line in header]
    out += [f"* R and L at {f:.4g} Hz (coupled inductors); off-diagonal R left out",
            f".subckt {subckt} " + " ".join(_pins(names))]
    for q, nm in enumerate(names):
        out.append(f"R{q} {nm}_p x{q} {_n(max(Z[q, q].real, 1e-9))}")
        out.append(f"L{q} x{q} {nm}_m {_n(max(L[q, q], 1e-15))}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k = L[i, j] / math.sqrt(max(L[i, i] * L[j, j], 1e-300))
            if abs(k) > 1e-6:
                out.append(f"K{i}_{j} L{i} L{j} {max(-0.999999, min(0.999999, k)):.6f}")
    out += [f".ends {subckt}", ""]
    return "\n".join(out)
