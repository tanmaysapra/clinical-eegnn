"""Fetch the Kaggle `akshaybe/updated-mimic-iv` tables into data/raw/.

The Kaggle mirror ships MIMIC-IV v3.1 verbatim: `mimic-iv-3.1/hosp/*.csv` and
`mimic-iv-3.1/icu/*.csv`, roughly 100 GB uncompressed. Almost none of that is
needed for a first-24h ICU mortality cohort, so downloads are tiered:

    core   ~0.9 GB  patients, admissions, icustays, diagnoses, dictionaries
    events ~3.5 GB  inputevents / outputevents / procedureevents (ICU treatment)
    labs    ~18 GB  labevents          (opt-in, hours on a normal link)
    charts  ~42 GB  chartevents        (opt-in, needs ~45 GB free)

`src/mimic_iv.py` degrades gracefully: whichever tiers are present on disk are
used, and the feature blocks sourced from missing tiers are simply absent.

Usage:
    python scripts/fetch_mimic.py --tier core
    python scripts/fetch_mimic.py --tier core --tier events
    python scripts/fetch_mimic.py --tier labs        # long
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

DATASET = "akshaybe/updated-mimic-iv"
ROOT = "mimic-iv-3.1"

TIERS = {
    "core": [
        "hosp/patients.csv",
        "hosp/admissions.csv",
        "icu/icustays.csv",
        "hosp/diagnoses_icd.csv",
        "hosp/d_icd_diagnoses.csv",
        "hosp/d_labitems.csv",
        "icu/d_items.csv",
        "hosp/services.csv",
        "hosp/transfers.csv",
        "hosp/procedures_icd.csv",
        "hosp/omr.csv",
    ],
    "events": [
        "icu/outputevents.csv",
        "icu/procedureevents.csv",
        "icu/inputevents.csv",
    ],
    "labs": ["hosp/labevents.csv"],
    "charts": ["icu/chartevents.csv"],
}


def fetch(rel_path, dest_root):
    """Download one table unless it is already on disk. Returns True if present."""
    out_path = os.path.join(dest_root, ROOT, rel_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  [skip] {rel_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)")
        return True

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download", "-d", DATASET,
        "-f", f"{ROOT}/{rel_path}", "-p", out_dir, "--force",
    ]
    print(f"  [get ] {rel_path}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr.strip()[-500:], file=sys.stderr)
        return False

    # Kaggle sometimes serves a .zip wrapper for larger files.
    zpath = out_path + ".zip"
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
        os.remove(zpath)
    return os.path.exists(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", action="append", default=None,
                    choices=sorted(TIERS), help="repeatable; default: core")
    ap.add_argument("--dest", default="data/raw")
    args = ap.parse_args()
    tiers = args.tier or ["core"]

    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found: pip install kaggle, then place your "
                 "API token at ~/.kaggle/kaggle.json")

    failed = []
    for tier in tiers:
        print(f"[{tier}]")
        for rel in TIERS[tier]:
            if not fetch(rel, args.dest):
                failed.append(rel)
    if failed:
        print("\nFAILED: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)
    print("\nok")


if __name__ == "__main__":
    main()
