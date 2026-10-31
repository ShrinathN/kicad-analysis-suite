#!/usr/bin/env python3
"""Build a KiCad Plugin and Content Manager (PCM) package of the plugin.

    python3 kicad_plugin/build_pcm.py [-o OUT_DIR] [--version X.Y.Z]

Writes OUT_DIR/kicad-analysis-suite-<version>.zip (default OUT_DIR: dist/ at the repository root).
In KiCad: Plugin and Content Manager > Install from File... and pick the zip.

Archive layout (what the PCM expects):
    metadata.json          package metadata (from kicad_plugin/metadata.json)
    plugins/               plugin.json, entry points, icons, requirements.txt, pcb_current_sim/
    resources/icon.png     64 x 64 icon shown in the PCM

The PCM unpacks plugins/ to <KiCad data>/<version>/3rdparty/plugins/<identifier with . -> _>/.
Uses only the standard library (the icon is written as a plain PNG), so any Python 3 can run it.
"""
import argparse
import json
import os
import struct
import sys
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLUGIN_FILES = ["plugin.json", "requirements.txt", "_bootstrap.py", "action.py", "home.py",
                "icon-24.png", "icon-48.png", "icon-home-24.png", "icon-home-48.png"]
PACKAGE = "pcb_current_sim"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_EXT = {".pyc", ".pyo"}
# The Home tile icon's four colours, drawn at the PCM's 64 x 64.
TILES = [(245, 208, 0), (232, 89, 12), (123, 44, 191), (58, 12, 163)]


def icon_png(size=64):
    """The Home icon (2 x 2 coloured tiles on transparent) as PNG bytes."""
    pad, gap = size // 12, size // 8
    tile = (size - 2 * pad - gap) // 2
    rows = []
    for y in range(size):
        row = bytearray([0])                       # filter type 0
        for x in range(size):
            px = (0, 0, 0, 0)
            for i, (ox, oy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                x0, y0 = pad + ox * (tile + gap), pad + oy * (tile + gap)
                if x0 <= x < x0 + tile and y0 <= y < y0 + tile:
                    px = TILES[i] + (255,)
            row += bytes(px)
        rows.append(bytes(row))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
            + chunk(b"IEND", b""))


def package_files():
    """(path on disk, path in the archive) for every plugin file."""
    for f in PLUGIN_FILES:
        yield os.path.join(HERE, f), "plugins/" + f
    src = os.path.join(ROOT, PACKAGE)
    for d, dirs, files in os.walk(src):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            if os.path.splitext(f)[1] in SKIP_EXT:
                continue
            p = os.path.join(d, f)
            yield p, "plugins/" + os.path.relpath(p, ROOT).replace(os.sep, "/")


def build(out_dir, version=None):
    with open(os.path.join(HERE, "metadata.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(HERE, "plugin.json")) as fh:
        plugin = json.load(fh)
    if meta["identifier"] != plugin["identifier"]:
        sys.exit(f"metadata.json identifier {meta['identifier']!r} differs from "
                 f"plugin.json {plugin['identifier']!r}")
    ver = meta["versions"][0]
    if version:
        ver["version"] = version
    # A package installed from a file must not carry download fields (they describe a repository entry).
    for k in ("download_url", "download_sha256", "download_size", "install_size"):
        ver.pop(k, None)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"kicad-analysis-suite-{ver['version']}.zip")
    tmp = out + ".part"
    size = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.json", json.dumps(meta, indent=2) + "\n")
        icon = icon_png()
        z.writestr("resources/icon.png", icon)
        for disk, arc in package_files():
            z.write(disk, arc)
            size += os.path.getsize(disk)
    os.replace(tmp, out)
    print(f"Wrote {out} (version {ver['version']}, {size / 1e6:.1f} MB unpacked)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "dist"), help="output folder")
    ap.add_argument("--version", help="override the version in metadata.json")
    a = ap.parse_args()
    build(a.out, a.version)


if __name__ == "__main__":
    main()
