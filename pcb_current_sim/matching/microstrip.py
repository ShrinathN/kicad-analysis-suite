"""Quasi-static microstrip: Hammerstad-Jensen Z0 and εeff, plus conductor and dielectric loss.

Accurate to about 1 % for 0.01 < w/h < 100 at frequencies where h << λ; no dispersion or
thickness correction. εr may vary with f (pass arrays from the stackup's εr(f) model).
"""
import numpy as np

ETA0 = 376.730313668
C0 = 299792458.0
MU0 = 4e-7 * np.pi
RHO_CU = 1.72e-8


def z0_eeff(w, h, er):
    u = np.asarray(w, float) / np.asarray(h, float)
    er = np.asarray(er, float)
    a = 1 + np.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)) / 49 + np.log(1 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    fu = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / u) ** 0.7528)
    z01 = ETA0 / (2 * np.pi) * np.log(fu / u + np.sqrt(1 + (2 / u) ** 2))
    return z01 / np.sqrt(eeff), eeff


def gamma(f, w, h, er, tand, rho=RHO_CU):
    """(Z0, γ = α + jβ per metre) for a microstrip at f."""
    f = np.asarray(f, float)
    z0, eeff = z0_eeff(w, h, er)
    beta = 2 * np.pi * f * np.sqrt(eeff) / C0
    rs = np.sqrt(np.pi * f * MU0 * rho)
    a_c = rs / (z0 * np.asarray(w, float))
    a_d = np.pi * f / C0 * er / np.sqrt(eeff) * (eeff - 1) / np.maximum(er - 1, 1e-9) * tand
    return z0, a_c + a_d + 1j * beta


def width_for(z0, h, er):
    """Width (same unit as h) giving Z0, by bisection on the monotonic Z0(w)."""
    lo, hi = 1e-3 * h, 50 * h
    for _ in range(80):
        m = np.sqrt(lo * hi)
        if z0_eeff(m, h, er)[0] > z0:
            lo = m
        else:
            hi = m
    return float(np.sqrt(lo * hi))
