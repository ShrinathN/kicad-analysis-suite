"""Sparse linear solves for every tool: one place to choose and improve the backend.

    f = factorize(A)          # A: scipy sparse, square; SPD by default
    x = f.solve(b)            # b: (n,) or (n, k) columns; x has b's shape
    x = solve(A, b)           # one-off

Backends, best available first:
  "pardiso"  Intel MKL PARDISO through pypardiso (optional, multithreaded, real matrices).
             Used only when importable and PCB_SIM_SOLVER is not "superlu".
  "superlu"  scipy's SuperLU with a symmetric minimum-degree ordering (always there; real
             or complex).  It releases the GIL, so independent solves can run in threads.
On MemoryError the factorisation falls back to Jacobi-preconditioned CG (slow but small).

Iterative multigrid (pyamg) is deliberately not used: on copper-next-to-FR4 matrices the
conductance contrast makes it slow and inaccurate (measured: 50x slower than SuperLU and
8 % off at 0.7 M nodes).  A GPU or C++ backend would plug in here as another Factor class.
"""
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ENV = "PCB_SIM_SOLVER"          # "superlu" forces SuperLU, "pardiso" requires PARDISO


def _pardiso():
    if os.environ.get(ENV, "").lower() == "superlu":
        return None
    try:
        import pypardiso
        return pypardiso
    except Exception:          # noqa: BLE001 - missing or broken MKL means SuperLU
        if os.environ.get(ENV, "").lower() == "pardiso":
            raise
        return None


def backend(complex_=False):
    """Name of the backend factorize() would use."""
    return "pardiso" if (not complex_ and _pardiso() is not None) else "superlu"


class Factor:
    """A factorised matrix; solve() takes a vector or (n, k) columns."""
    name = "?"

    def __init__(self, A):
        self.n = A.shape[0]

    def solve(self, b):
        b = np.asarray(b)
        if self.n == 0:
            return np.zeros(b.shape, b.dtype)
        return self._solve(b)


class SuperLUFactor(Factor):
    name = "superlu"

    def __init__(self, A):
        super().__init__(A)
        if self.n:
            self.lu = spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True))

    def _solve(self, b):
        return self.lu.solve(b)


class PardisoFactor(Factor):
    name = "pardiso"

    def __init__(self, A, spd=True):
        super().__init__(A)
        pp = _pardiso()
        self.ps = pp.PyPardisoSolver()
        self.ps.set_matrix_type(2 if spd else 1)       # real SPD / real structurally symmetric
        self.A = sp.triu(A, format="csr") if spd else sp.csr_matrix(A)
        if self.n:
            self.A.sort_indices()
            self.ps.factorize(self.A)

    def _solve(self, b):
        x = self.ps.solve(self.A, np.ascontiguousarray(b, dtype=float))
        return x.reshape(b.shape)


class CGFactor(Factor):
    """Fallback when a direct factorisation does not fit in memory."""
    name = "cg"

    def __init__(self, A):
        super().__init__(A)
        self.A = sp.csr_matrix(A)
        self.M = sp.diags(1 / self.A.diagonal())

    def _solve(self, b):
        cols = b.reshape(self.n, -1)
        x = np.column_stack([spla.cg(self.A, c, M=self.M, rtol=1e-10, maxiter=20 * self.n)[0] for c in cols.T])
        return x.reshape(b.shape)


def factorize(A, spd=True):
    """Factorise a square sparse matrix.  spd: symmetric positive definite (conductance and
    thermal matrices); False for general (e.g. complex symmetric AC) systems."""
    complex_ = np.iscomplexobj(A.data) if sp.issparse(A) else np.iscomplexobj(A)
    try:
        if backend(complex_) == "pardiso" and spd:
            return PardisoFactor(A)
        return SuperLUFactor(A)
    except MemoryError:
        return CGFactor(A)


def solve(A, b, spd=True):
    return factorize(A, spd).solve(b)
