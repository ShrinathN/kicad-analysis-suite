"""Magnetic material library for the 3D extraction's cores.

A grade is an initial permeability μi and a first-order (Debye) relaxation frequency f_r:

    μ(f) = 1 + (μi - 1) / (1 + j f / f_r)

so μ' falls to about half at f_r, where μ'' peaks (μ'' = μ' there): a first fit to a datasheet's
complex permeability plot.  A single relaxation has a narrower μ'' peak than most real ferrites:
fit it (`fit_debye`) to the datasheet points around the frequencies that matter.  Ferrite permeability roughly follows Snoek's limit, μi · f_r ≈ constant
(MnZn some GHz, NiZn tens of GHz), which the built-in classes respect.

The built-in grades are generic material classes with typical values, not any maker's product:
for a real design enter the grade from its datasheet (Add / Fit).  User grades live in
materials.json next to the component library ({name: grade dict}).

DC bias, core loss (Steinmetz) and temperature come in a later step (fields reserved: notes).
"""
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from . import magnetic as Mg

FILE = "materials.json"
KINDS = ("MnZn ferrite", "NiZn ferrite", "Ferrite sheet", "Iron powder", "Alloy powder", "Nanocrystalline",
         "Ideal", "Custom")


@dataclass
class Grade:
    name: str
    mu_r: float                       # initial (low-frequency) relative permeability μi
    f_r: float = None                 # Debye relaxation [Hz]; None: lossless, constant μ
    kind: str = "Custom"
    notes: str = ""
    builtin: bool = False

    def material(self):
        return Mg.Material(self.name, float(self.mu_r), None if not self.f_r else float(self.f_r))

    def mu(self, f):
        """Complex μ(f) = μ' - j μ''."""
        return 1.0 + self.material().chi(f)

    def to_dict(self):
        return dict(name=self.name, mu_r=float(self.mu_r), f_r=None if not self.f_r else float(self.f_r),
                    kind=self.kind, notes=self.notes)

    @classmethod
    def from_dict(cls, d, builtin=False):
        return cls(str(d.get("name", "material")), float(d.get("mu_r", 1000.0)),
                   None if not d.get("f_r") else float(d["f_r"]), str(d.get("kind", "Custom")),
                   str(d.get("notes", "")), builtin)


TYPICAL = "Typical values of the class, not a product: use your grade's datasheet for a design."
BUILTIN = [Grade(*g, notes=f"{n} {TYPICAL}", builtin=True) for *g, n in (
    ("MnZn power ferrite, μi 2000", 2000.0, 3e6, "MnZn ferrite",
     "Power transformers and inductors, 25-500 kHz."),
    ("MnZn ferrite, μi 5000", 5000.0, 1.2e6, "MnZn ferrite",
     "Common-mode chokes, broadband transformers."),
    ("MnZn high-μ ferrite, μi 10000", 10000.0, 4e5, "MnZn ferrite",
     "Common-mode chokes and signal transformers below ~200 kHz."),
    ("NiZn ferrite, μi 800", 800.0, 8e6, "NiZn ferrite",
     "EMI suppression (beads, cable cores): lossy from a few MHz on purpose."),
    ("NiZn ferrite, μi 125", 125.0, 4e7, "NiZn ferrite",
     "Inductors and transformers 1-30 MHz."),
    ("NiZn ferrite, μi 40", 40.0, 1.5e8, "NiZn ferrite",
     "RF inductors and transformers up to ~100 MHz."),
    ("Ferrite sheet (NFC), μ' 150", 150.0, 4e8, "Ferrite sheet",
     "Shielding sheet behind 13.56 MHz antennas: μ' ~150, μ'' a few there."),
    ("Ferrite sheet (wireless power), μ' 700", 700.0, 5e6, "Ferrite sheet",
     "Shielding sheet behind 100-200 kHz charging coils."),
    ("Iron powder, μ 10", 10.0, 1e8, "Iron powder",
     "RF and high-Q inductors, 1-50 MHz (distributed gap)."),
    ("Iron powder, μ 75", 75.0, 3e6, "Iron powder",
     "DC chokes and output filters below ~500 kHz (distributed gap)."),
    ("FeSiAl alloy powder, μ 60", 60.0, 3e6, "Alloy powder",
     "Power inductors with DC bias, 50-500 kHz (distributed gap)."),
    ("Nanocrystalline, μ 20000", 20000.0, 1.5e5, "Nanocrystalline",
     "Common-mode chokes, current transformers (tape-wound: the μ(f) shown is the effective one)."),
    ("Ideal, μ 10000 (no loss)", 10000.0, None, "Ideal",
     "A lossless reference: what the geometry gives at most."),
)]


def path_for(library_path):
    return os.path.join(os.path.dirname(library_path) or ".", FILE)


class Library:
    """Built-in grades plus the user's (materials.json); a user grade with a built-in's name
    replaces it."""

    def __init__(self, path=None):
        self.path = path
        self.user = []
        if path:
            try:
                with open(path) as fh:
                    d = json.load(fh)
                self.user = [Grade.from_dict({**v, "name": k}) for k, v in d.items() if isinstance(v, dict)]
            except (OSError, ValueError):
                self.user = []

    @property
    def grades(self):
        names = {g.name.lower() for g in self.user}
        return [g for g in BUILTIN if g.name.lower() not in names] + self.user

    def names(self):
        return [g.name for g in self.grades]

    def get(self, name):
        k = (name or "").strip().lower()
        return next((g for g in self.grades if g.name.lower() == k), None)

    def save(self, grade):
        if not grade.name.strip():
            raise ValueError("A material needs a name.")
        self.user = [g for g in self.user if g.name.lower() != grade.name.lower()]
        self.user.append(Grade.from_dict(grade.to_dict()))
        self._write()

    def remove(self, name):
        n = len(self.user)
        self.user = [g for g in self.user if g.name.lower() != name.lower()]
        if len(self.user) != n:
            self._write()

    def _write(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        d = {g.name: {k: v for k, v in g.to_dict().items() if k != "name"} for g in self.user}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


def fit_debye(f, mu1, mu2=None):
    """(μi, f_r, rms relative error) of the Debye model fitted to datasheet points: f [Hz], μ'
    and optionally μ'' (same length; None entries skipped).  Least squares on log(1 + |error|)
    relative to |μ|, so the points count alike across decades."""
    from scipy.optimize import least_squares
    f = np.asarray(f, float)
    m1 = np.asarray(mu1, float)
    m2 = None if mu2 is None else np.array([np.nan if v is None else v for v in mu2], float)
    ok = np.isfinite(f) & np.isfinite(m1) & (f > 0)
    if ok.sum() < 2:
        raise ValueError("At least two points (f, μ') are needed.")
    data = m1 - 1j * (np.nan_to_num(m2) if m2 is not None else 0.0)
    has2 = m2 is not None and np.isfinite(m2).any()

    def model(p):
        return 1.0 + (math.exp(p[0]) - 1.0) / (1.0 + 1j * f / math.exp(p[1]))

    def res(p):
        mu = model(p)
        r = [(mu.real - data.real)[ok] / np.abs(data)[ok]]
        if has2:
            k = ok & np.isfinite(m2)
            r.append((mu.imag - data.imag)[k] / np.abs(data)[k])
        return np.concatenate(r)

    mu0 = max(float(m1[ok][np.argmin(f[ok])]), 1.5)
    # f_r start: where μ' first drops below half of the first point, else the top frequency
    below = f[ok][m1[ok] < 0.5 * mu0]
    fr0 = float(below[0]) if len(below) else float(f[ok].max()) * 3
    best = None
    for s in (0.3, 1.0, 3.0):
        r = least_squares(res, [math.log(mu0), math.log(fr0 * s)])
        if best is None or r.cost < best.cost:
            best = r
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    return math.exp(best.x[0]), math.exp(best.x[1]), rms
