"""Find saved results (.npz + .json sidecar) for the results browser, reading only the small sidecar."""
import json
import os
import time

SKIP_DIRS = {".git", "__pycache__", "node_modules", "test-backups"}


def scan(root, max_depth=3, limit=200):
    out = []
    root = os.path.abspath(root)
    base = root.count(os.sep)
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for f in files:
            if not f.endswith(".npz"):
                continue
            meta = os.path.join(d, f[:-4] + ".json")
            if os.path.exists(meta):
                info = describe(os.path.join(d, f), meta)
                if info:
                    out.append(info)
            if len(out) >= limit:
                return sorted(out, key=lambda r: -r["mtime"])
    return sorted(out, key=lambda r: -r["mtime"])


def describe(npz, meta):
    try:
        with open(meta) as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return None
    if "summary" not in m or "grid" not in m:
        return None
    s = m.get("summary") or {}
    st = os.stat(npz)
    feats = [k for k, on in (("thermal", m.get("tgrid")), ("B", m.get("bgrid")), ("pulse", m.get("pulse")))
             if on]
    return dict(tool="DC & Thermal", path=npz, mtime=st.st_mtime,
                date=time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                board=os.path.basename(m.get("board_path") or ""),
                rows=", ".join(r.get("name", "") for r in m.get("rows") or []),
                nets=len(m.get("nets") or []), T_max=s.get("T_max"),
                features=", ".join(feats), size_kb=st.st_size / 1024)
