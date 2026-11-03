"""Small Touchstone v1 reader and writer (.sNp): '# <unit> S <MA|DB|RI> R <z0>' plus data lines."""
import os
import re

import numpy as np

UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
_cache = {}


class Touchstone:
    def __init__(self, f, S, z0, path=""):
        self.f, self.S, self.z0, self.path = f, S, z0, path    # S: (F, n, n)

    @property
    def nports(self):
        return self.S.shape[1]

    def at(self, f):
        """S interpolated (linear in re/im) to f, shape (len(f), n, n); held constant outside the data."""
        f = np.asarray(f, float)
        out = np.empty((len(f),) + self.S.shape[1:], complex)
        for i in range(self.nports):
            for j in range(self.nports):
                s = self.S[:, i, j]
                out[:, i, j] = np.interp(f, self.f, s.real) + 1j * np.interp(f, self.f, s.imag)
        return out

    def covers(self, f1, f2):
        return self.f[0] <= f1 * (1 + 1e-9) and f2 <= self.f[-1] * (1 + 1e-9)


def parse(text, nports, path=""):
    unit, fmt, z0 = 1e9, "MA", 50.0
    nums = []
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            tok = line[1:].upper().split()
            for k, t in enumerate(tok):
                if t in UNITS:
                    unit = UNITS[t]
                elif t in ("MA", "DB", "RI"):
                    fmt = t
                elif t == "R" and k + 1 < len(tok):
                    z0 = float(tok[k + 1])
                elif t in ("Y", "Z", "H", "G"):
                    raise ValueError(f"only S-parameter files are supported (found {t})")
            continue
        if line.startswith("["):
            raise ValueError("Touchstone v2 keywords are not supported")
        nums += [float(x) for x in line.split()]
    per = 1 + 2 * nports * nports
    if not nums or len(nums) % per:
        raise ValueError(f"expected {per} numbers per frequency for a {nports}-port file")
    a = np.array(nums).reshape(-1, per)
    f = a[:, 0] * unit
    p = a[:, 1::2], a[:, 2::2]
    if fmt == "RI":
        s = p[0] + 1j * p[1]
    elif fmt == "MA":
        s = p[0] * np.exp(1j * np.radians(p[1]))
    else:
        s = 10 ** (p[0] / 20) * np.exp(1j * np.radians(p[1]))
    S = s.reshape(-1, nports, nports)
    if nports == 2:                       # v1 two-port order is S11 S21 S12 S22
        S = S.transpose(0, 2, 1)
    if np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must increase")
    return Touchstone(f, S, z0, path)


def load(path):
    """Cached by path and modification time."""
    m = re.fullmatch(r"\.s(\d+)p", os.path.splitext(path)[1].lower())
    if not m or int(m.group(1)) < 1:
        raise ValueError("expected a .sNp file (.s1p, .s2p, …)")
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _cache:
        with open(path) as fh:
            _cache[key] = parse(fh.read(), int(m.group(1)), path)
    return _cache[key]


def s2abcd(S, z0):
    """(F, 2, 2) S referenced to real z0 -> A, B, C, D arrays of shape (F,)."""
    s11, s12, s21, s22 = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]
    d = 2 * s21
    A = ((1 + s11) * (1 - s22) + s12 * s21) / d
    B = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / d
    C = ((1 - s11) * (1 - s22) - s12 * s21) / (d * z0)
    D = ((1 - s11) * (1 + s22) + s12 * s21) / d
    return A, B, C, D


def z2s(Z, z0):
    """(F, n, n) impedance matrix -> S referenced to real z0 in every port: (Z - z0)(Z + z0)^-1."""
    eye = np.eye(Z.shape[1])
    return np.linalg.solve((Z + z0 * eye).transpose(0, 2, 1), (Z - z0 * eye).transpose(0, 2, 1)).transpose(0, 2, 1)


def write(path, f, Z, z0=50.0, names=(), comments=()):
    """Write the impedance matrices Z (F, n, n) at f [Hz] as a Touchstone v1 S-parameter file
    (RI, reference z0 in every port). n ports: the file's extension should be .s<n>p. Port names
    and comments go into '!' lines. Rows of 3+ port files are broken after 4 pairs, as v1 asks."""
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    n = Z.shape[1]
    S = z2s(Z, z0)
    if n == 2:
        S = S.transpose(0, 2, 1)              # v1 two-port order S11 S21 S12 S22
    num = lambda c: f"{c.real: .12e} {c.imag: .12e}"
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        for c in comments:
            fh.write(f"! {c}\n")
        for k, name in enumerate(names):
            fh.write(f"! Port {k + 1}: {name}\n")
        fh.write(f"# HZ S RI R {z0:g}\n")
        for i in range(len(f)):
            head = f"{f[i]:.12e}"
            if n <= 2:
                fh.write(head + " " + " ".join(num(c) for c in S[i].ravel()) + "\n")
                continue
            for r in range(n):
                row = S[i, r]
                for q in range(0, n, 4):
                    lead = head if (r == 0 and q == 0) else " " * len(head)
                    fh.write(lead + " " + " ".join(num(c) for c in row[q:q + 4]) + "\n")
