"""Small Touchstone v1 reader (.s1p / .s2p): '# <unit> S <MA|DB|RI> R <z0>' plus data lines."""
import os

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
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".s1p", ".s2p"):
        raise ValueError("expected a .s1p or .s2p file")
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _cache:
        with open(path) as fh:
            _cache[key] = parse(fh.read(), int(ext[2]), path)
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
