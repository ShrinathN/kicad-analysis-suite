"""The filament network: port impedances at any frequency, and a reduced model for whole bands.

Branches are filaments (resistance R, partial inductances L, dense).  Kirchhoff's voltage law is
written in a mesh-current basis Q = [M | P]: M the fundamental cycles of a spanning forest (two
filaments of one segment make a two-branch cycle, so most columns are tiny), P one column per
port, the tree path from the port's first node to its second (closed outside the conductors by
the port's source).  With Z_q(s) = Qᵀ (R + sL) Q, a port voltage V drives mesh currents
x = Z_q⁻¹ E V (E selects the port columns), so the port admittance matrix is

    Y(s) = Eᵀ (R_q + s L_q)⁻¹ E,     Z = Y⁻¹.

`System.direct(f)` solves this as is (dense complex, one solve per frequency).  `System.reduce`
builds a passive reduced model by PRIMA (block Krylov subspaces of (R_q + s₀ L_q)⁻¹ L_q at a
few real expansion points s₀ including DC, congruence projection): R_r, L_r stay symmetric
positive definite, so the model is passive and has a Foster form, Y(s) = Σ c_k c_kᵀ / (1 + s τ_k)
(one R-L branch per pole for a single port).  Its Z(f) costs nothing per frequency.
"""
import math

import numpy as np
import scipy.sparse as sp
from scipy import linalg

from ..loop.peec import _forest, _path, available_ram, MEM_SHARE
from .kernels import inductance_matrix

TWO_PI = 2 * math.pi


def dense_bytes(B, n):
    """Peak memory of System for B filaments and n = meshes + ports: L, Qᵀ L, L_q and one factor."""
    return int(8 * (B * B + B * n + 2 * n * n))


class System:
    def __init__(self, geom, progress=lambda m: None):
        self.geom = geom
        F, R, a, b, seg = geom.filaments()
        self.fils, self.R, self.a, self.b, self.seg = F, R, a, b, seg
        _, N = geom.electrical_nodes()
        enode, _ = geom.electrical_nodes()
        B = len(R)
        parent, pbr, depth, comp = _forest(N, a, b)
        self.comp = comp
        tree = np.zeros(B, bool)
        tree[pbr[pbr >= 0]] = True
        rows, cols, vals = [], [], []
        cot = np.flatnonzero(~tree)
        for col, e in enumerate(cot):
            rows.append(e)
            cols.append(col)
            vals.append(1.0)
            for k, s in _path(b[e], a[e], parent, pbr, depth, a):
                rows.append(k)
                cols.append(col)
                vals.append(float(s))
        m = len(cot)
        self.names = []
        for name, na, nb in geom.ports:
            u, v = enode[na], enode[nb]
            if comp[u] != comp[v]:
                raise ValueError(f"Port {name}: its two nodes are not joined by conductors.")
            col = m + len(self.names)
            for k, s in _path(u, v, parent, pbr, depth, a):
                rows.append(k)
                cols.append(col)
                vals.append(float(s))
            self.names.append(name)
        p = len(self.names)
        if not p:
            raise ValueError("No ports.")
        n = m + p
        self.Q = sp.csc_matrix((vals, (rows, cols)), shape=(B, n))
        self.m, self.p = m, p
        need, have = dense_bytes(B, n), available_ram()
        if need > MEM_SHARE * have:
            raise ValueError(f"Too many filaments for this computer: {B} filaments need about "
                             f"{need / 2 ** 30:.1f} GB, {have / 2 ** 30:.1f} GB is free. Use fewer "
                             f"filaments per segment or a smaller region.")
        progress(f"Partial inductances of {B} filaments...")
        L = inductance_matrix(F)
        progress(f"Mesh matrices ({m} meshes, {p} ports)...")
        QL = np.asarray(self.Q.T @ L)                       # n x B
        del L
        self.Lq = np.asarray((self.Q.T @ QL.T).T)           # n x n
        self.Lq = (self.Lq + self.Lq.T) / 2
        del QL
        self.Rq = np.asarray((self.Q.T @ sp.diags(R) @ self.Q).todense())
        self.E = np.zeros((n, p))
        self.E[m + np.arange(p), np.arange(p)] = 1.0

    @property
    def n_filaments(self):
        return len(self.R)

    # -- direct solve -----------------------------------------------------------------
    def direct(self, f):
        """(Z, X): port impedance matrix at f [Hz] (p x p complex) and the mesh currents per unit
        port current (n x p): branch currents I = Q X i_port."""
        A = self.Rq + 1j * TWO_PI * f * self.Lq
        S = linalg.solve(A, self.E.astype(complex), assume_a="sym")
        Y = self.E.T @ S
        Z = linalg.inv(Y)
        return (Z + Z.T) / 2, S @ Z

    def currents(self, X, i_port):
        """Filament currents [A] for port currents i_port (X from direct)."""
        return self.Q @ (X @ np.asarray(i_port, complex))

    def currents_reduced(self, model, f, i_port):
        """Filament currents [A] at f for port currents i_port, from the reduced model (its
        projection basis V spans the mesh currents)."""
        s = 1j * TWO_PI * f
        Zp = model.Z(f)[0]
        v = Zp @ np.asarray(i_port, complex)
        xr = linalg.solve(model.Rr + s * model.Lr, model.Er @ v)
        return self.Q @ (model.V @ xr)

    # -- reduced model ----------------------------------------------------------------
    def reduce(self, f_min, f_max, tol=1e-5, max_blocks=60, progress=lambda m: None):
        """Passive reduced model valid from DC to f_max (checked from f_min).  Expansion points:
        DC and 3 frequencies spread logarithmically up to f_max; blocks are added round-robin until
        Z at 40 test frequencies changes by less than tol (relative to the largest |Z| entry at
        each frequency)."""
        pts = [0.0] + list(TWO_PI * np.geomspace(max(f_min, f_max / 1e4), f_max, 3))
        facs = []
        for s0 in pts:
            progress(f"Factorising at {s0 / TWO_PI:.3g} Hz...")
            facs.append(linalg.cho_factor(self.Rq + s0 * self.Lq, lower=False, check_finite=False))
        test = np.geomspace(max(f_min, 1e-3), f_max, 40)
        V = np.zeros((self.E.shape[0], 0))
        heads = [None] * len(pts)
        Zold = None
        model = None
        for it in range(max_blocks):
            for k, fac in enumerate(facs):
                if heads[k] is not None and heads[k].shape[1] == 0:
                    continue                                 # this point's Krylov space is exhausted
                W = linalg.cho_solve(fac, self.E if heads[k] is None else self.Lq @ heads[k], check_finite=False)
                W = _orth(W, V)
                if W.shape[1] == 0:
                    heads[k] = np.zeros((V.shape[0], 0))
                    continue
                V = np.hstack([V, W])
                heads[k] = W
            if all(h is not None and h.shape[1] == 0 for h in heads):
                break
            model = Reduced(V.T @ self.Rq @ V, V.T @ self.Lq @ V, V.T @ self.E, self.names, V)
            Z = model.Z(test)
            if Zold is not None:
                scale = np.abs(Z).reshape(len(test), -1).max(axis=1)
                err = (np.abs(Z - Zold).reshape(len(test), -1).max(axis=1) / scale).max()
                progress(f"Reduced model order {V.shape[1]}: change {err:.1e}")
                if err < tol:
                    break
            Zold = Z
        return model


def _orth(W, V, drop=1e-10):
    """W orthonormalised against V and itself (two Gram-Schmidt passes, QR), columns that fall
    below drop x their original norm removed (deflation)."""
    if W.shape[1] == 0:
        return W
    n0 = np.linalg.norm(W, axis=0)
    for _ in range(2):
        if V.shape[1]:
            W = W - V @ (V.T @ W)
    Qw, Rw = np.linalg.qr(W)
    keep = np.abs(np.diag(Rw)) > drop * max(n0.max(), 1e-300)
    Qw = Qw[:, keep]
    if V.shape[1]:
        Qw = Qw - V @ (V.T @ Qw)
        Qw, _ = np.linalg.qr(Qw)
    return Qw


class Reduced:
    """Reduced model: Y(s) = E_rᵀ (R_r + s L_r)⁻¹ E_r, as a Foster sum."""

    def __init__(self, Rr, Lr, Er, names, V=None):
        self.Rr, self.Lr, self.Er, self.names, self.V = (Rr + Rr.T) / 2, (Lr + Lr.T) / 2, Er, list(names), V
        tau, W = linalg.eigh(self.Lr, self.Rr)              # Wᵀ R_r W = I, Wᵀ L_r W = diag(tau)
        self.tau = np.maximum(tau, 0.0)                      # time constants L/R of the modes [s]
        self.C = W.T @ Er                                    # (order x p)

    @property
    def order(self):
        return len(self.tau)

    def Y(self, f):
        f = np.atleast_1d(np.asarray(f, float))
        s = 1j * TWO_PI * f
        g = 1.0 / (1.0 + s[:, None] * self.tau[None, :])    # F x k
        return np.einsum("kp,fk,kq->fpq", self.C, g, self.C)

    def Z(self, f):
        """Port impedance matrices (F x p x p) at frequencies f [Hz]."""
        Z = np.linalg.inv(self.Y(f))
        return (Z + np.swapaxes(Z, 1, 2)) / 2
