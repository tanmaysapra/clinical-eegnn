"""OpenMP isolation between xgboost and torch on macOS/arm64.

The xgboost wheel links `@rpath/libomp.dylib` but bundles no copy; torch ships
its own. Loading both into one process is broken here in *both* orders:

    torch then xgboost  -> segfault (exit 139) on the first fit
    xgboost then torch  -> torch/PyG deadlocks at 0% CPU

Neither `KMP_DUPLICATE_LIB_OK=TRUE` (documented as unsafe -- duplicate OpenMP
runtimes can silently corrupt numerics) nor conda-forge llvm-openmp fixes it,
since torch still prefers its bundled copy. `OMP_NUM_THREADS=1` does work but
costs ~2.2x on torch training.

So xgboost never runs in the torch process at all: `run_xgb_worker` shells out
to scripts/_xgb_worker.py, which imports xgboost/shap and never imports torch.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_WORKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "_xgb_worker.py",
)


def run_xgb_worker(X, y, masks, seed=42, want_shap=False):
    """Fit XGBoost out-of-process. Returns a dict of arrays."""
    train, val, test = masks
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.npz")
        out_path = os.path.join(tmp, "out.npz")
        np.savez(in_path, X=X, y=y, train=train, val=val, test=test,
                 seed=seed, want_shap=want_shap)
        proc = subprocess.run(
            [sys.executable, _WORKER, in_path, out_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"xgboost worker failed ({proc.returncode}):\n{proc.stderr}"
            )
        d = np.load(out_path)
        return {k: d[k] for k in d.files}
