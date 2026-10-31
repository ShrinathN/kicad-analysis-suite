"""The switching-transient circuit as an ngspice deck, and running it.

transient.Circuit is translated element by element: R, C, inductors (the copper block as coupled
inductors with K lines; its off-diagonal resistance is dropped), sources as PWL, and the built-in
device equations as behavioural sources (the MOSFET channel as a B current, voltage-dependent
capacitors as ddt(q(V)), junctions as SPICE diodes with IS, N and TT), so the two simulators run the
same models.  A FET with a vendor model is a subcircuit instance instead (Circuit.subs).

Back-ends: KiCad's bundled libngspice (ctypes, shared mode) or an ngspice program on the PATH.
"""
import ctypes
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

FC = 0.5               # forward-bias coefficient of the capacitance law (transient.qc_fun uses the same)
LIB_PATHS = ("/app/lib/libngspice.so.0", "/app/lib/libngspice.so", "libngspice.so.0", "libngspice.so")


# ---------------------------------------------------------------- vendor model files
def subckts(path):
    """{name: [pins]} of the .subckt definitions in a SPICE library file."""
    out = {}
    try:
        with open(path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        return out
    text = re.sub(r"\n\+", " ", text)            # continuation lines
    for line in text.splitlines():
        t = line.strip().split()
        if len(t) >= 3 and t[0].lower() == ".subckt":
            pins = []
            for x in t[2:]:
                if "=" in x or x.lower() in ("params:", "param:"):
                    break
                pins.append(x)
            out[t[1]] = pins
    return out


def guess_pins(pins):
    """A pin order ('D G S', 'D G S T', ...) for a subcircuit's pin names: by name where they say
    drain / gate / source / temperature, else the usual vendor order D G S (then T)."""
    role = []
    for p in pins:
        q = p.lower()
        if q in ("d", "drain") or q.startswith("drain"):
            role.append("D")
        elif q in ("g", "gate") or q.startswith("gate"):
            role.append("G")
        elif q in ("s", "source") or q.startswith("source"):
            role.append("S")
        elif q in ("t", "tj", "tc", "tcase", "temp") or q.startswith(("tj", "tcase")):
            role.append("T")
        else:
            role.append("")
    if sorted(r for r in role if r in "DGS" and r) == ["D", "G", "S"]:
        return " ".join(r or "NC" for r in role)
    base = ["D", "G", "S", "T", "T"]
    return " ".join(base[k] if k < len(base) else "NC" for k in range(len(pins)))


# ---------------------------------------------------------------- the deck
def _fmt(v):
    return f"{v:.6g}"


def _pwl(ts, ys):
    """PWL points of samples ys at times ts, without the points a straight line already passes
    through (breakpoints closer than ngspice's minimum step stop it)."""
    ys = np.asarray(ys, float)
    keep = [0]
    tol = 1e-9 * max(np.max(np.abs(ys)), 1e-30)
    for i in range(1, len(ts) - 1):
        a = keep[-1]
        y_lin = ys[a] + (ys[i + 1] - ys[a]) * (ts[i] - ts[a]) / (ts[i + 1] - ts[a])
        if abs(ys[i] - y_lin) > tol:
            keep.append(i)
    keep.append(len(ts) - 1)
    return " ".join(f"{_fmt(ts[i])} {_fmt(ys[i])}" for i in keep)


def deck(c, t_end, h, title="switching edge"):
    """(deck text, vector names {('v', node name) | ('i', branch index): spice vector}) of Circuit c."""
    c._setup()
    nm = {-1: "0"}
    for name, k in c.names.items():
        if k >= 0:
            nm[k] = f"n{k}"
    L = [f"* {title} (written by the PCB analysis suite)", ".options rshunt=1e9 reltol=1e-3 abstol=1e-10 "
         "vntol=1e-6 method=gear itl4=100"]
    incs = sorted({p for _, p, _ in c.subs})
    for p in incs:
        L.append(f'.include "{p}"')
    for k, (a, b, r) in enumerate(c.res):
        L.append(f"R{k} {nm[a]} {nm[b]} {_fmt(r)}")
    for k, (a, b, cc) in enumerate(c.caps):
        L.append(f"C{k} {nm[a]} {nm[b]} {_fmt(cc)}")
    for k, q in enumerate(c.qs):
        sp = q.f.spec
        if sp[0] == "lin":
            L.append(f"CQ{k} {nm[q.a]} {nm[q.b]} {_fmt(sp[1])}")
            continue
        # the built-in charge q(x) = A (sqrt(max(x, 0) + phi) - sqrt(phi)) + c0 min(x, 0), x = p V(a, b):
        # c = c0 / sqrt(1 + x / phi) above 0 V, c0 below (a junction model would grow when forward biased)
        _, A, phi, c0 = sp
        x = f"({_fmt(q.p)}*V({nm[q.a]},{nm[q.b]}))"
        # c0 / sqrt(1 + x / phi) as a junction's depletion capacitance (CJO = c0, VJ = phi, M = 0.5, the
        # same forward-bias law as the built-in model) that never conducts; cathode where x > 0
        ka, an = (q.a, q.b) if q.p > 0 else (q.b, q.a)
        L.append(f"DQ{k} {nm[an]} {nm[ka]} DQM{k}")
        L.append(f".model DQM{k} D(IS=1e-30 N=100 CJO={_fmt(c0)} VJ={_fmt(phi)} M=0.5 FC={FC} RS=0)")
    for k, d in enumerate(c.ds):
        i_s, n = d.f.spec
        L.append(f"D{k} {nm[d.a]} {nm[d.b]} DM{k}")
        L.append(f".model DM{k} D(IS={_fmt(i_s)} N={_fmt(n)} TT={_fmt(d.tau)} RS=0 CJO=0)")
    for k, ch in enumerate(c.chs):
        K, vth, s, kl = ch.f.spec
        p = _fmt(ch.p)
        vgs = f"({p}*V({nm[ch.g]},{nm[ch.s]}))"
        vds = f"({p}*V({nm[ch.d]},{nm[ch.s]}))"
        x = f"({vgs}-{_fmt(vth)})"
        vov = f"(max({x},0)+{_fmt(s)}*ln(1+exp(-abs({x})/{_fmt(s)})))"
        isat = f"({_fmt(K)}*{vov}*{vov})"
        L.append(f"BCH{k} {nm[ch.d]} {nm[ch.s]} I={p}*{isat}*tanh({_fmt(kl)}*{vov}*{vds}/({isat}+1e-12))")
    ts = np.arange(0.0, t_end * (1 + 1e-9), max(h, t_end / 3000))
    for k, g in enumerate(c.gs):
        pts = _pwl(ts, [g.g(t) for t in ts])
        L.append(f"VG{k} g{k} 0 PWL({pts})")
        L.append(f"BG{k} {nm[g.a]} {nm[g.b]} I=V(g{k})*V({nm[g.a]},{nm[g.b]})")
    vec = {}
    for k, (a, b, r, l, _) in enumerate(c.lbr):
        rr, ll = c.Rm[k, k], c.Lm[k, k]
        if rr > 0:
            L.append(f"RL{k} {nm[a]} m{k} {_fmt(rr)}")
            L.append(f"L{k} m{k} {nm[b]} {_fmt(ll)}")
        else:
            L.append(f"L{k} {nm[a]} {nm[b]} {_fmt(ll)}")
        vec[("i", k)] = f"l{k}#branch"
    for k0, Lb, _ in c.blocks:
        n = len(Lb)
        for i in range(n):
            for j in range(i + 1, n):
                kk = Lb[i, j] / np.sqrt(Lb[i, i] * Lb[j, j])
                if abs(kk) > 1e-4:
                    L.append(f"K{k0 + i}_{k0 + j} L{k0 + i} L{k0 + j} {_fmt(min(max(kk, -0.9999), 0.9999))}")
    for k, (a, b, e, r, _) in enumerate(c.vsrc):
        pts = _pwl(ts, [e(t) for t in ts])
        L.append(f"V{k} {nm[a]} v{k} PWL({pts})")
        L.append(f"RV{k} v{k} {nm[b]} {_fmt(r)}")
    for k, (a, b, i) in enumerate(c.isrc):
        L.append(f"I{k} {nm[a]} {nm[b]} DC {_fmt(i(0.0))}")
    for k, (name, _, nodes) in enumerate(c.subs):
        L.append(f"X{k} " + " ".join(nm[x] for x in nodes) + f" {name}")
    L.append(f".tran {_fmt(h)} {_fmt(t_end)} 0 {_fmt(h)}")
    for name, k in c.names.items():
        if k >= 0:
            vec[("v", name)] = nm[k]
    return "\n".join(L) + "\n", vec


# ---------------------------------------------------------------- running it
_lib = None
_out = []


def _load():
    global _lib
    if _lib is not None:
        return _lib
    for p in LIB_PATHS:
        try:
            lib = ctypes.CDLL(p)
        except OSError:
            continue
        P = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        E = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)
        B = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)
        cbs = (P(lambda s, i, u: (_out.append(s.decode(errors="replace")), 0)[1]),
               P(lambda s, i, u: 0), E(lambda st, unl, q, i, u: 0), B(lambda r, i, u: 0))
        lib.ngSpice_Init(cbs[0], cbs[1], cbs[2], None, None, cbs[3], None)
        lib._cbs = cbs                            # keep the callbacks alive
        lib.ngGet_Vec_Info.restype = ctypes.POINTER(_VecInfo)
        _lib = lib
        return lib
    return None


class _VecInfo(ctypes.Structure):
    _fields_ = [("v_name", ctypes.c_char_p), ("v_type", ctypes.c_int), ("v_flags", ctypes.c_short),
                ("v_realdata", ctypes.POINTER(ctypes.c_double)), ("v_compdata", ctypes.c_void_p),
                ("v_length", ctypes.c_int)]


def available():
    """'library' (KiCad's libngspice), 'program' (ngspice on the PATH) or None."""
    if _load() is not None:
        return "library"
    if shutil.which("ngspice"):
        return "program"
    return None


def run(text, vectors):
    """Simulate deck `text`; {vector: array} for the names in `vectors` plus 'time'."""
    how = available()
    if how == "library":
        return _run_lib(text, vectors)
    if how == "program":
        return _run_prog(text, vectors)
    raise RuntimeError("ngspice was not found (KiCad's libngspice, or an ngspice program).")


def _run_lib(text, vectors):
    lib = _lib
    _out.clear()
    lib.ngSpice_Command(b"destroy all")
    lib.ngSpice_Command(b"remcirc")
    lines = text.splitlines() + [".end"]
    arr = (ctypes.c_char_p * (len(lines) + 1))(*[x.encode() for x in lines], None)
    if lib.ngSpice_Circ(arr) != 0:
        raise RuntimeError("ngspice did not accept the circuit:\n" + "".join(_out[-20:]))
    lib.ngSpice_Command(b"run")
    res = {}
    for name in ["time"] + list(vectors):
        vi = lib.ngGet_Vec_Info(name.encode())
        if not vi or not vi.contents.v_realdata:
            raise RuntimeError(f"ngspice gave no vector {name}:\n" + "\n".join(_out[-20:]))
        n = vi.contents.v_length
        res[name] = np.ctypeslib.as_array(vi.contents.v_realdata, shape=(n,)).copy()
    return res


def _run_prog(text, vectors):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.txt")
        names = list(vectors)
        ctl = ("\n.control\nset wr_singlescale\nset wr_vecnames\nrun\nwrdata " + out + " "
               + " ".join(names) + "\nquit\n.endc\n.end\n")
        path = os.path.join(d, "deck.cir")
        with open(path, "w") as fh:
            fh.write(text + ctl)
        p = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True, timeout=600)
        if not os.path.exists(out):
            raise RuntimeError("ngspice failed:\n" + (p.stdout + p.stderr)[-2000:])
        data = np.loadtxt(out, skiprows=1, ndmin=2)
    res = {"time": data[:, 0]}
    for k, n in enumerate(names):
        res[n] = data[:, k + 1]
    return res
