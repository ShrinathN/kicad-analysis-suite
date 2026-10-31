"""PCB analysis suite (KiCad plugin): DC & thermal, switching loops, trace extraction, matching."""
import os

# CPU threads for the heavy numerics (dense solves, threaded matrix builds).  Twice the PC powered
# off with all 12 cores busy, so the default is a third of the cores; PCB_SIM_THREADS overrides.
# The BLAS limits must be set before numpy is first imported, which importing this package does.
THREADS = max(1, int(os.environ.get("PCB_SIM_THREADS") or max(2, (os.cpu_count() or 4) // 3)))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))
