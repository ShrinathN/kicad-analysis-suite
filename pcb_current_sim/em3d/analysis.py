"""Run a 3D extraction of board copper: wideband port impedances, current maps, SPICE.

`run(board, stackup, setup)` -> Result:
* f, Z (F x p x p): from the reduced (Foster) model, which is exact at DC and passive;
* per port: R_dc, L_dc (DC current pattern), R and L at every f, Q = ωL / R;
* L and coupling k between ports at every f (Im Z / ω; k = L_ij / √(L_ii L_jj));
* per mesh bar and display frequency: the bar's current for 1 A into each port, and per
  filament for the cross-section view (`maps`).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import pcb, system

TWO_PI = 2 * math.pi
SHOW = 3                              # display frequencies: DC-ish, middle, top


@dataclass
class Result:
    names: list
    f: np.ndarray
    Z: np.ndarray                     # F x p x p
    model: object                     # system.Reduced
    window: list
    n_fils: int
    seconds: float
    f_show: list = field(default_factory=list)
    bar_I: dict = field(default_factory=dict)   # (port, k_show) -> complex current per mesh bar for 1 A
    fil_I: dict = field(default_factory=dict)   # (port, k_show) -> complex current per filament
    bars: dict = field(default_factory=dict)    # drawing data per mesh bar (see _bars)
    fils: dict = field(default_factory=dict)    # per filament: seg, bar, w, h, offsets in the bar
    warnings: list = field(default_factory=list)

    @property
    def R(self):
        return np.real(np.diagonal(self.Z, axis1=1, axis2=2))

    @property
    def L(self):
        """Inductance matrices (F x p x p) [H]."""
        return np.imag(self.Z) / (TWO_PI * self.f[:, None, None])

    @property
    def Q(self):
        return TWO_PI * self.f[:, None] * np.diagonal(self.L, axis1=1, axis2=2) / np.maximum(self.R, 1e-300)

    @property
    def k(self):
        L = self.L
        d = np.sqrt(np.abs(np.diagonal(L, axis1=1, axis2=2)))
        return L / (d[:, :, None] * d[:, None, :])

    def dc(self):
        """(R_dc vector, L_dc matrix) of the ports (DC current pattern)."""
        Y0 = np.einsum("kp,kq->pq", self.model.C, self.model.C)
        Z0 = np.linalg.inv(Y0)
        # dZ/ds at s = 0: Z Y' Z with Y'(0) = -Σ c cᵀ τ
        Y1 = -np.einsum("kp,k,kq->pq", self.model.C, self.model.tau, self.model.C)
        L0 = -Z0 @ Y1 @ Z0
        return np.real(np.diag(Z0)), (L0 + L0.T) / 2

    def at(self, f):
        """Z (p x p) at one frequency from the model."""
        return self.model.Z(f)[0]


def run(board, stackup, s, progress=lambda m: None):
    t0 = time.time()
    built = pcb.build(board, stackup, s, progress)
    progress(f"{built.n_fils} filaments: building the network...")
    sysm = system.System(built.geom, progress)
    progress("Wideband model...")
    model = sysm.reduce(s.f_min, s.f_max, progress=progress)
    f = np.geomspace(s.f_min, s.f_max, max(2, int(s.points)))
    Z = model.Z(f)
    res = Result(list(sysm.names), f, Z, model, built.window, built.n_fils, 0.0, warnings=list(built.warnings))
    res.f_show = [float(x) for x in (s.f_min, math.sqrt(s.f_min * s.f_max), s.f_max)]
    nseg = len(built.geom.segs)
    seg_bar = built.bar_of_seg
    for p in range(len(res.names)):
        u = np.zeros(len(res.names))
        u[p] = 1.0
        for k, fs in enumerate(res.f_show):
            I = sysm.currents_reduced(model, fs, u)
            res.fil_I[(p, k)] = I.astype(np.complex64)
            Iseg = np.zeros(nseg, complex)
            np.add.at(Iseg, sysm.seg, I)
            Ibar = np.zeros(built.mesh.n_branches, complex)
            np.add.at(Ibar, seg_bar, Iseg)          # barrel rings: the ring's total
            res.bar_I[(p, k)] = Ibar
    res.bars = _bars(built.mesh)
    F = sysm.fils
    res.fils = dict(seg=sysm.seg, bar=seg_bar[sysm.seg], w=F.w, h=F.h,
                    off_w=np.einsum("ik,ik->i", F.c0 - _seg_start(built.geom)[sysm.seg], F.wd),
                    off_h=np.einsum("ik,ik->i", F.c0 - _seg_start(built.geom)[sysm.seg], F.hd))
    res.seconds = time.time() - t0
    progress(f"Done in {res.seconds:.1f} s.")
    return res


def _seg_start(geom):
    return np.array([geom.xyz[sg["a"]] for sg in geom.segs])


def _bars(mesh):
    """Per mesh bar: kind, layer, centre (mm), direction, length (mm), width (mm), net, end nodes;
    the cell grid (x0, y0, pitch, shape) and each layer's cell -> node map, for drawing."""
    g = mesh.grid
    return dict(kind=mesh.br_kind.copy(), layer=mesh.br_layer.copy(), x=mesh.br_x.copy(), y=mesh.br_y.copy(),
                ux=mesh.br_ux.copy(), uy=mesh.br_uy.copy(), len=mesh.br_len * 1e3, w=mesh.br_w * 1e3,
                t=mesh.br_t * 1e3, net=mesh.br_net.copy(), a=mesh.br_a.copy(), b=mesh.br_b.copy(),
                n_nodes=mesh.n_nodes, layers=list(mesh.layers), nets=list(mesh.nets),
                grid=(g.x0, g.y0, g.h, g.ny, g.nx),
                cell_node={L: mesh.cell_node[L].astype(np.int32) for L in mesh.layers})


def sheet_current(res, port, k_show):
    """(per node: |K| [A/mm] in the cell grid, per bar: |I| / w [A/mm]) for 1 A into a port at
    display frequency k_show.  A node's K combines its x and y bars (mean magnitude of each)."""
    b = res.bars
    I = np.abs(res.bar_I[(port, k_show)])
    Kb = I / np.maximum(b["w"], 1e-9)
    n = b["n_nodes"]
    K2 = np.zeros(n)
    for kind in (0, 1):
        m = b["kind"] == kind
        s = np.zeros(n)
        c = np.zeros(n)
        for end in ("a", "b"):
            np.add.at(s, b[end][m], Kb[m])
            np.add.at(c, b[end][m], 1.0)
        K2 += np.where(c > 0, s / np.maximum(c, 1), 0.0) ** 2
    return np.sqrt(K2), Kb
