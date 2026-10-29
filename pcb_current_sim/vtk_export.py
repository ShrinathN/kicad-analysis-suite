"""ParaView / VTK export (XML formats, base64 binary, numpy only - no vtk package needed).

Writes a folder with
  board.vtm     multiblock: one ImageData sheet per copper layer at its stackup height
                (|J|, J vector, V, power density, copper fraction, net id), thermal sheets
                (T, end-of-pulse T), B planes above / below (B vector), and PolyData for
                vias / barrels (current), resistance elements and the board outline
  pulse.pvd     time series of the pulse temperature sheets (ParaView animates it)

Coordinates are mm in a right-handed frame: x right, y up (= -y of KiCad), z up with the top
copper surface at 0, so ParaView's +Z view matches the KiCad top view.  z is true scale; use
a Transform filter (Scale z) to exaggerate the stackup.
"""
import base64
import os
import zlib

import numpy as np

HEAD = 'byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor"'


# ---------------------------------------------------------------------------- low level
def _b64(a):
    """zlib-compressed binary DataArray payload: base64(header) + base64(data), one block."""
    raw = np.ascontiguousarray(a).tobytes()
    comp = zlib.compress(raw, 6)
    head = np.array([1, len(raw), len(raw), len(comp)], np.uint32).tobytes()
    return base64.b64encode(head).decode("ascii") + base64.b64encode(comp).decode("ascii")


def _array(name, a, ncomp=1):
    a = np.asarray(a)
    t = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.int32): "Int32",
         np.dtype(np.uint8): "UInt8", np.dtype(np.int64): "Int64"}[a.dtype]
    return (f'<DataArray type="{t}" Name="{name}" NumberOfComponents="{ncomp}" format="binary">'
            f"{_b64(a)}</DataArray>")


def _rows_up(A):
    """KiCad row order (y down) -> VTK order (y up)."""
    return np.asarray(A)[::-1]


def write_image(path, grid, z, cell_arrays):
    """2D ImageData at height z [mm] on `grid` (KiCad coords); cell_arrays: name -> (ny, nx[, 3])."""
    g = grid
    ox, oy = g.x0, -(g.y0 + g.ny * g.h)
    body = []
    for name, a in cell_arrays.items():
        a = np.asarray(a)
        if a.ndim == 3:
            flat = _rows_up(a).reshape(-1, a.shape[2]).astype(np.float32)
            body.append(_array(name, flat, a.shape[2]))
        else:
            dt = np.float32 if a.dtype.kind == "f" or a.dtype == bool else np.int32
            body.append(_array(name, _rows_up(a).astype(dt).ravel()))
    ext = f"0 {g.nx} 0 {g.ny} 0 0"
    xml = (f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" {HEAD}>\n<ImageData WholeExtent="{ext}" Origin="{ox} {oy} {z}" '
           f'Spacing="{g.h} {g.h} 1">\n<Piece Extent="{ext}">\n<CellData>\n' + "\n".join(body) +
           "\n</CellData>\n</Piece>\n</ImageData>\n</VTKFile>\n")
    with open(path, "w") as f:
        f.write(xml)


def write_lines(path, segments, cell_arrays=None):
    """PolyData of line segments [(p0, p1), ...] with KiCad (x, y, z) points; one cell per segment."""
    pts = np.array([[p[0], -p[1], p[2]] for seg in segments for p in seg], np.float32).reshape(-1, 3)
    n = len(segments)
    conn = np.arange(2 * n, dtype=np.int32)
    offs = np.arange(2, 2 * n + 1, 2, dtype=np.int32)
    cd = "".join(_array(k, np.asarray(v, np.float32)) for k, v in (cell_arrays or {}).items())
    xml = (f'<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" {HEAD}>\n<PolyData>\n<Piece NumberOfPoints="{2 * n}" NumberOfLines="{n}">\n'
           f"<Points>{_array('Points', pts.ravel(), 3)}</Points>\n"
           f"<Lines>{_array('connectivity', conn)}{_array('offsets', offs)}</Lines>\n"
           f"<CellData>{cd}</CellData>\n</Piece>\n</PolyData>\n</VTKFile>\n")
    with open(path, "w") as f:
        f.write(xml)


def _multiblock(path, blocks):
    """blocks: [(group name, [(dataset name, relative file)])]"""
    out = ['<?xml version="1.0"?>', '<VTKFile type="vtkMultiBlockDataSet" version="1.0">', "<vtkMultiBlockDataSet>"]
    for gi, (gname, items) in enumerate(blocks):
        out.append(f'<Block index="{gi}" name="{gname}">')
        for di, (dname, rel) in enumerate(items):
            out.append(f'<DataSet index="{di}" name="{dname}" file="{rel}"/>')
        out.append("</Block>")
    out += ["</vtkMultiBlockDataSet>", "</VTKFile>"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------- result export
def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def export_vtk(res, outdir):
    """Write board.vtm (+ pulse.pvd) for ParaView into outdir; returns the list of top-level files."""
    s = res.summary
    zL = s.get("layer_z_mm")
    if not zL:
        raise ValueError("this result has no layer heights; re-run the simulation")
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    blocks, written = [], []

    def rel(name):
        return os.path.join("data", name)

    copper = []
    for L in res.layers:
        F = res.fields[L]
        J = np.stack([F["Jx"], -np.asarray(F["Jy"]), np.zeros_like(F["Jx"])], axis=-1)   # y up
        arrs = {"J_A_per_mm2": F["J"], "J_vector": J, "V_mV": F["V"], "p_W_per_mm2": F["p"],
                "copper_fraction": F["f"], "net_id": np.asarray(F["net"], np.int32)}
        fn = rel(f"copper_{_safe(L)}.vti")
        write_image(os.path.join(outdir, fn), res.grid, zL[L], arrs)
        copper.append((L, fn))
    blocks.append(("copper", copper))

    thermal = []
    for key, maps, grid in (("T_C", res.T, res.tgrid), ("T_pulse_end_C", res.Tp, res.pgrid)):
        if maps is None or grid is None:
            continue
        for L in res.layers:
            fn = rel(f"{key}_{_safe(L)}.vti")
            write_image(os.path.join(outdir, fn), grid, zL[L], {key: maps[L]})
            thermal.append((f"{key} {L}", fn))
    if thermal:
        blocks.append(("thermal", thermal))

    if res.B is not None:
        planes = []
        bh = s.get("b_height_mm", 1.0)
        for side, b in res.B.items():
            z = bh if side == "above" else s["z_bottom_mm"] - bh
            Bv = np.stack([b["Bx"], -np.asarray(b["By"]), b["Bz"]], axis=-1)
            mag = np.sqrt(b["Bx"] ** 2 + b["By"] ** 2 + b["Bz"] ** 2)
            fn = rel(f"B_{side}.vti")
            write_image(os.path.join(outdir, fn), res.bgrid, z, {"B_uT": Bv, "B_mag_uT": mag, "Bz_uT": b["Bz"]})
            planes.append((f"B {side} ({bh:g} mm)", fn))
        blocks.append(("magnetic", planes))

    geo = []
    vias = [v for v in res.vias if v.get("L_top") in zL and v.get("L_bot") in zL]
    if vias:
        fn = rel("vias.vtp")
        write_lines(os.path.join(outdir, fn),
                    [((v["x"], v["y"], zL[v["L_top"]]), (v["x"], v["y"], zL[v["L_bot"]])) for v in vias],
                    {"I_A": [v["I_A"] for v in vias], "drill_mm": [v["drill_mm"] for v in vias],
                     "P_W": [v["P_W"] for v in vias]})
        geo.append(("vias and barrels", fn))
    els = []
    for e in res.elements:
        for kind, data in res.outlines.get(res.layers[0], []):
            if kind == "elem" and data[2] == e["name"]:
                (xa, ya), (xb, yb), _ = data
                els.append(((xa, ya, 0.15), (xb, yb, 0.15), e))
    if els:
        fn = rel("elements.vtp")
        write_lines(os.path.join(outdir, fn), [(a, b) for a, b, _ in els],
                    {"I_A": [e["I_A"] for *_, e in els], "P_W": [e["P_W"] for *_, e in els],
                     "R_ohm": [e["R_ohm"] for *_, e in els]})
        geo.append(("resistance elements", fn))
    segs = []
    for loop in s.get("board_outline", []):
        a = np.asarray(loop)
        for z in (0.0, s["z_bottom_mm"]):
            segs += [((p[0], p[1], z), (q[0], q[1], z)) for p, q in zip(a[:-1], a[1:])]
    if segs:
        fn = rel("outline.vtp")
        write_lines(os.path.join(outdir, fn), segs)
        geo.append(("board outline", fn))
    if geo:
        blocks.append(("geometry", geo))

    _multiblock(os.path.join(outdir, "board.vtm"), blocks)
    written.append(os.path.join(outdir, "board.vtm"))

    if res.pulse_frames and res.pgrid is not None:
        lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "<Collection>"]
        for n, (t, maps) in enumerate(res.pulse_frames):
            for part, L in enumerate(res.layers):
                fn = rel(f"pulse_{_safe(L)}_{n:03d}.vti")
                write_image(os.path.join(outdir, fn), res.pgrid, zL[L], {"T_C": maps[L]})
                lines.append(f'<DataSet timestep="{t / 1e3:.6g}" part="{part}" name="{L}" file="{fn}"/>')
        lines += ["</Collection>", "</VTKFile>"]
        with open(os.path.join(outdir, "pulse.pvd"), "w") as f:
            f.write("\n".join(lines) + "\n")
        written.append(os.path.join(outdir, "pulse.pvd"))

    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write("Open board.vtm in ParaView (File > Open), then Apply.\n"
                "Colour by J_A_per_mm2, V_mV, p_W_per_mm2 (copper), T_C (thermal) or B_mag_uT (magnetic).\n"
                "Hide copper outside the traces: Threshold on copper_fraction > 0.\n"
                "Current arrows: Glyph on J_vector; equipotentials: Contour on V_mV (after Cell Data to Point Data).\n"
                "The stack is true scale: Transform > Scale z (e.g. 5) to separate the layers.\n"
                + ("pulse.pvd is the pulse temperature over time (time in seconds): press Play.\n"
                   if res.pulse_frames else ""))
    return written
