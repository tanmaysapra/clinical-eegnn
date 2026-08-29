# Shared preamble. Every notebook in the series starts from the same state:
# repo on the path, warnings quietened, and one config factory so a parameter
# changed in one notebook cannot silently differ in the next.
import os, sys, time, json, warnings

REPO = "/kaggle/working/clinical-eegnn"
RAW  = "/kaggle/input/datasets/akshaybe/updated-mimic-iv"
CACHE = "/kaggle/working/processed"

os.chdir(REPO); sys.path.insert(0, REPO)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np, pandas as pd, yaml

def ensure_pyg():
    """Install torch_geometric if the session does not already have it.

    Not in the Kaggle base image, and a session that has restarted has lost it
    along with everything else in /kaggle/working. Notebook 00 installs it, but
    anything run without going through 00 -- a standalone script, a resumed
    session -- needs this or it fails at `import graph_build`.
    """
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"],
                       check=True)
        import torch_geometric  # noqa: F401
    return True

def mimic_cfg(blocks=("demographics", "comorbidity", "labs", "vitals", "treatment"),
              **overrides):
    """The one place the MIMIC config is assembled."""
    cfg = yaml.safe_load(open(f"{REPO}/configs/default.yaml"))
    cfg["data"].update(source="mimic", raw_dir=RAW, cache_dir=CACHE,
                       blocks=list(blocks), **overrides)
    return cfg
