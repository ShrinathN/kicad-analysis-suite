"""FastHenry input files from a Geometry, running FastHenry and reading its Zc.mat.

Used to cross-check the built-in solver (same segments, same filament counts and ratios) and,
later, as an export.  FastHenry: https://github.com/ediloren/FastHenry2 (MIT licence).
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np


def find():
    """Path of the fasthenry program, or None (env FASTHENRY, PATH, ~/.local/src build)."""
    for p in (os.environ.get("FASTHENRY"), shutil.which("fasthenry"),
              os.path.expanduser("~/.local/src/FastHenry2/bin/fasthenry")):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def write(geom, path, f_min, f_max, ndec=1, title="pcb_current_sim em3d"):
    """FastHenry input (units m).  Frequencies: f_min .. f_max, ndec points per decade."""
    out = [f"* {title}", ".units m"]
    for i, (x, y, z) in enumerate(geom.xyz):
        out.append(f"N{i} x={x:.12g} y={y:.12g} z={z:.12g}")
    for k, (s, (pa, u, l, wd, hd)) in enumerate(zip(geom.segs, geom.frames())):
        out.append(f"E{k} N{s['a']} N{s['b']} w={s['w']:.12g} h={s['h']:.12g} sigma={s['sigma']:.12g} "
                   f"nwinc={s['nw']} nhinc={s['nh']} rw={s['rw']:.6g} rh={s['rh']:.6g} "
                   f"wx={wd[0]:.12g} wy={wd[1]:.12g} wz={wd[2]:.12g}")
    groups = {}
    for i in range(len(geom.xyz)):
        groups.setdefault(geom._find(i), []).append(i)
    for nodes in groups.values():
        if len(nodes) > 1:
            out.append(".equiv " + " ".join(f"N{i}" for i in nodes))
    for name, a, b in geom.ports:
        out.append(f".external N{a} N{b} {name}")
    out.append(f".freq fmin={f_min:.12g} fmax={f_max:.12g} ndec={ndec}")
    out.append(".end")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def parse(path):
    """(f [Hz], Z (F x p x p complex)) from a FastHenry Zc.mat."""
    fs, Zs = [], []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    num = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([-+]\s*\d*\.?\d+(?:[eE][-+]?\d+)?)j")
    while i < len(lines):
        m = re.match(r"Impedance matrix for frequency = (\S+) (\d+) x (\d+)", lines[i])
        if m:
            f, p = float(m.group(1)), int(m.group(2))
            Z = np.zeros((p, p), complex)
            for r in range(p):
                vals = num.findall(lines[i + 1 + r])
                Z[r] = [complex(float(a), float(b.replace(" ", ""))) for a, b in vals]
            fs.append(f)
            Zs.append(Z)
            i += p
        i += 1
    return np.array(fs), np.array(Zs)


def run(geom, f_min, f_max, ndec=1, exe=None, timeout=3600):
    """Run FastHenry on the geometry; (f, Z) with ports in geom.ports order (FastHenry lists them
    last first in Zc.mat; reordered here)."""
    exe = exe or find()
    if not exe:
        raise FileNotFoundError("FastHenry not found (set FASTHENRY to the program).")
    with tempfile.TemporaryDirectory() as d:
        write(geom, os.path.join(d, "g.inp"), f_min, f_max, ndec)
        subprocess.run([exe, "g.inp"], cwd=d, check=True, capture_output=True, timeout=timeout)
        f, Z = parse(os.path.join(d, "Zc.mat"))
        with open(os.path.join(d, "Zc.mat")) as fh:
            head = fh.read().split("Impedance matrix", 1)[0]
    rows = re.findall(r"Row (\d+):.*port name: (\S+)", head)
    if rows:
        order = {name.lower(): int(r) - 1 for r, name in rows}      # FastHenry lowercases names
        idx = [order[name.lower()] for name, _, _ in geom.ports]
        Z = Z[:, idx][:, :, idx]
    return f, Z
