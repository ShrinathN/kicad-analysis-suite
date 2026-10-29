"""Find the external solvers the 3D tools can hand work to.

A program counts as found when it is on PATH, or installed as a Flatpak app
(`flatpak info <app id>`; Flatpak apps are never on PATH under their own name).

Inside the KiCad Flatpak the host's programs and Flatpak apps are invisible. They can be
seen and run with `flatpak-spawn --host` only if the sandbox may talk to org.freedesktop.Flatpak
(`flatpak override --user --talk-name=org.freedesktop.Flatpak org.kicad.KiCad`).
Without that, host programs are reported as unknown and the tools export their input files
for the user to run on the host.
"""
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field

PROGRAMS = (  # (label, executables tried in order, Flatpak app ids, used by)
    ("openEMS", ("openEMS",), (), "Full-wave"),
    ("AppCSXCAD", ("AppCSXCAD",), (), "Full-wave (geometry viewer)"),
    ("FastHenry", ("fasthenry", "FastHenry2"), (), "3D extraction (L, R)"),
    ("FastCap", ("fastcap", "FasterCap", "fastercap"), (), "3D extraction (C)"),
    ("ParaView", ("paraview",), ("org.paraview.ParaView",), "Result viewing"),
)
PY_MODULES = (("openEMS", "openEMS python"), ("CSXCAD", "CSXCAD python"))
ENABLE_HOST = "flatpak override --user --talk-name=org.freedesktop.Flatpak org.kicad.KiCad"


@dataclass
class Found:
    label: str
    used_by: str
    where: str = ""       # "sandbox", "host", "host (flatpak)", "unknown" or "" (not found)
    path: str = ""        # executable path, or the Flatpak ref and version
    argv: list = field(default_factory=list)   # command that starts it from here (flatpak-spawn included)

    @property
    def ok(self):
        return bool(self.argv)


def in_flatpak():
    return os.path.exists("/.flatpak-info")


def host_reachable(timeout=3):
    """True when `flatpak-spawn --host` works from inside the sandbox."""
    if not in_flatpak() or not shutil.which("flatpak-spawn"):
        return False
    try:
        return subprocess.run(["flatpak-spawn", "--host", "true"], timeout=timeout,
                              capture_output=True).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run(argv, timeout=5):
    """stdout of argv, or None when it fails."""
    try:
        r = subprocess.run(argv, timeout=timeout, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _host_which(exe):
    out = _run(["flatpak-spawn", "--host", "sh", "-c", f'command -v "{exe}"'])
    return (out or "").strip()


def flatpak_app(app_id, host_prefix=()):
    """'ref (version, installation)' if the Flatpak app is installed, else ''."""
    out = _run(list(host_prefix) + ["flatpak", "info", app_id])
    if not out:
        return ""
    info = dict(line.strip().split(": ", 1) for line in out.splitlines() if ": " in line)
    ver = info.get("Version", "")
    return f"{info.get('Ref', app_id)} ({', '.join(x for x in (ver, info.get('Installation', '')) if x)})"


def detect():
    """(list of Found, notes). Probes the host too when running in the Flatpak and allowed to."""
    sandbox = in_flatpak()
    host = host_reachable()
    spawn = ["flatpak-spawn", "--host"] if sandbox else []
    can_flatpak = host if sandbox else bool(shutil.which("flatpak"))
    out, notes = [], []
    for label, exes, apps, used in PROGRAMS:
        f = Found(label, used)
        for exe in exes:
            p = shutil.which(exe)
            if p:
                f.where, f.path, f.argv = "sandbox" if sandbox else "host", p, [p]
                break
            if host:
                p = _host_which(exe)
                if p:
                    f.where, f.path, f.argv = "host", p, spawn + [p]
                    break
        if not f.ok and can_flatpak:
            for app in apps:
                ref = flatpak_app(app, spawn)
                if ref:
                    f.where, f.path, f.argv = "host (flatpak)", ref, spawn + ["flatpak", "run", app]
                    break
        if not f.ok and sandbox and not host:
            f.where = "unknown"     # the host is invisible from the sandbox
        out.append(f)
    for mod, label in PY_MODULES:
        spec = importlib.util.find_spec(mod)
        out.append(Found(label, "Full-wave scripting", "sandbox" if sandbox else "host", spec.origin or "",
                         ["python", "-m", mod]) if spec else Found(label, "Full-wave scripting"))
    if sandbox:
        notes.append("Running inside the KiCad Flatpak.")
        notes.append("Host programs and Flatpak apps are reachable through flatpak-spawn." if host else
                     "Host programs and Flatpak apps (e.g. ParaView from Flathub) can't be seen from the sandbox, "
                     "so they show as unknown; tools will export input files instead. To let the suite find and "
                     f"run them, run this once on the host and restart KiCad:  {ENABLE_HOST}")
    return out, notes
