"""Shared stackup and materials: copper thickness and dielectric εr / tanδ against frequency.

The board file gives the geometry (and εr / tanδ when the stackup has them).
The user's overrides live in the scenario's "stackup" section, so every tool
sees the same values.

Frequency dependence follows the wideband Djordjevic-Sarkar model:
    ε(ω) = ε∞ + Δε/(m2 − m1) · log10((ω2 + jω) / (ω1 + jω)),   ω1 = 10^m1, ω2 = 10^m2 rad/s
fitted to εr and tanδ at f_ref. This keeps εr(f) and tanδ(f) causal (Kramers-Kronig consistent).
"""
import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_ER = 4.3
DEFAULT_TAND = 0.02
EPS0 = 8.8541878128e-12


@dataclass
class Dielectric:
    """One dielectric gap between copper layers k and k+1 (plies merged)."""
    name: str
    material: str
    t: float                     # m
    er: float                    # at f_ref
    tand: float                  # at f_ref
    f_ref: float = 1e9           # Hz
    from_board: bool = True      # False once the user overrides εr / tanδ

    def eps(self, f, model="ds", m1=4.0, m2=12.0):
        """Complex relative permittivity ε' − jε'' at f (Hz, scalar or array)."""
        f = np.asarray(f, float)
        if model != "ds":
            return np.full(f.shape, self.er * (1 - 1j * self.tand))
        w1, w2 = 10.0 ** m1, 10.0 ** m2

        def lg(fr):
            w = 2 * np.pi * fr
            return np.log10((w2 + 1j * w) / (w1 + 1j * w)) / (m2 - m1)
        L0 = lg(self.f_ref)
        de = -(self.er * self.tand) / L0.imag
        einf = self.er - de * L0.real
        return einf + de * lg(f)          # imag part is negative: ε'' = −imag

    def er_tand(self, f, **kw):
        e = self.eps(f, **kw)
        return e.real, -e.imag / e.real


@dataclass
class Stackup:
    """Board stackup with user overrides applied."""
    copper_layers: list
    cu_board: dict                                  # layer -> m, from the board file
    cu_override: dict = field(default_factory=dict)  # layer -> m
    dielectrics: list = field(default_factory=list)  # Dielectric per gap
    model: str = "ds"                               # "ds" (Djordjevic-Sarkar) or "const"
    board_thickness: float = 1.6                    # mm
    stackup_found: bool = True

    def cu(self, layer):
        return self.cu_override.get(layer, self.cu_board[layer])

    @property
    def cu_thickness(self):
        return {L: self.cu(L) for L in self.copper_layers}

    def layer_z(self):
        """Top-of-copper z (m) of each layer, top layer at 0, increasing downwards."""
        z, out = 0.0, {}
        for k, L in enumerate(self.copper_layers):
            out[L] = z
            z += self.cu(L) + (self.dielectrics[k].t if k < len(self.dielectrics) else 0.0)
        return out

    def gap(self, a, b):
        """Dielectric thickness (m) between copper layers a and b."""
        i, j = sorted((self.copper_layers.index(a), self.copper_layers.index(b)))
        return sum(d.t for d in self.dielectrics[i:j]) + sum(self.cu(L) for L in self.copper_layers[i + 1:j])

    def er_between(self, a, b, f=None):
        """Effective εr between two layers: series plates, t / Σ(t_i / εr_i)."""
        i, j = sorted((self.copper_layers.index(a), self.copper_layers.index(b)))
        ds = self.dielectrics[i:j]
        ers = [d.er if f is None else d.er_tand(f, model=self.model)[0] for d in ds]
        return sum(d.t for d in ds) / sum(d.t / e for d, e in zip(ds, ers))

    # ---- persistence: only the overrides go into the scenario --------------
    def overrides(self):
        return dict(cu_thickness=dict(self.cu_override), model=self.model,
                    dielectrics=[dict(er=d.er, tand=d.tand, f_ref=d.f_ref) if not d.from_board else None
                                 for d in self.dielectrics])

    def apply(self, ov):
        if not ov:
            return self
        self.cu_override = {L: float(t) for L, t in (ov.get("cu_thickness") or {}).items()
                            if L in self.cu_board}
        self.model = ov.get("model", self.model)
        for d, o in zip(self.dielectrics, ov.get("dielectrics") or []):
            if o:
                d.er, d.tand, d.f_ref, d.from_board = float(o["er"]), float(o["tand"]), \
                    float(o.get("f_ref", d.f_ref)), False
        return self

    def summary(self):
        n = len(self.copper_layers)
        cu = ", ".join(f"{self.cu(L) * 1e6:.0f}" for L in self.copper_layers)
        ers = sorted({f"{d.er:.2f}" for d in self.dielectrics})
        return (f"{n} layers, {self.board_thickness:.2f} mm, Cu {cu} µm, εr {'/'.join(ers) or '-'}"
                + ("" if self.stackup_found else " (no stackup in file: defaults)"))


def from_board(board, overrides=None):
    diel = []
    for k, plies in enumerate(board.dielectrics):
        t = sum(p["t"] for p in plies) or board.gaps[k]
        known = [p for p in plies if p.get("er")]
        if known:   # series combination of plies that state εr; tanδ weighted by t/εr (their share of the field)
            w = [p["t"] / p["er"] for p in known]
            er = sum(p["t"] for p in known) / sum(w)
            tand = sum(wi * (p.get("tand") or DEFAULT_TAND) for wi, p in zip(w, known)) / sum(w)
        else:
            er, tand = DEFAULT_ER, DEFAULT_TAND
        mats = sorted({p["material"] for p in plies if p.get("material")})
        diel.append(Dielectric(plies[0]["name"] if plies else f"dielectric {k + 1}", "/".join(mats) or "FR4",
                               t, er, tand))
    s = Stackup(list(board.copper_layers), dict(board.cu_thickness), dielectrics=diel,
                board_thickness=board.thickness, stackup_found=board.stackup_found)
    return s.apply(overrides)


def plate_capacitance(area_m2, h_m, er):
    return EPS0 * er * area_m2 / h_m


def wavelength(f, er):
    return 299792458.0 / (f * math.sqrt(er))
