#%% md
# 00 · Environment and data audit

**Series:** Early-Exit GNN for clinical risk stratification on MIMIC-IV.

This notebook establishes what we are running on and what data is actually
present, before any modelling claim depends on it. It is deliberately the
cheapest notebook in the series: it reads no event tables.

It answers three questions:

1. **What compute do we have?** Determines whether the 42 GB `chartevents`
   scan and the 47k-node graph are feasible in one session.
2. **Which MIMIC-IV tables are on disk?** The ETL degrades gracefully — a
   missing table means a missing feature block, not a crash — so "which blocks
   are even possible" is a property of the mount, and needs recording.
3. **Do the schemas match what the ETL expects?** Column names in MIMIC-IV
   v3.1 differ from v2.x in places; catching that here costs seconds, catching
   it inside a 40-minute scan does not.

#%% code
exec(open("/kaggle/working/clinical-eegnn/notebooks/src/_bootstrap.py").read())

import platform, subprocess, importlib

print("python  ", sys.version.split()[0], "|", platform.platform())
print("cwd     ", os.getcwd())

for m in ["numpy", "pandas", "duckdb", "torch", "torch_geometric", "sklearn",
          "xgboost", "shap", "pyarrow", "matplotlib"]:
    try:
        print("  %-18s %s" % (m, getattr(importlib.import_module(m), "__version__", "?")))
    except ImportError:
        print("  %-18s MISSING" % m)

#%% md
## Compute

`duckdb` is the load-bearing dependency here. The event aggregation in
`src/mimic_iv.py` scans `labevents` (18 GB) and `chartevents` (42 GB) with
predicate pushdown and constant memory. Without it the code falls back to a
chunked pandas reader that works but is far slower.

A GPU matters less than it looks: the model is small and the cohort is a single
graph. Where it does pay is kNN construction, which is one big matmul.

#%% code
import torch
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.1f} GB")
print("cpus:", os.cpu_count())
print("ram :", round(int(open("/proc/meminfo").readline().split()[1]) / 1e6, 1), "GB")
print(subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True).stdout)

#%% md
## Install `torch_geometric`

Not in the Kaggle base image. The model uses only `MessagePassing`, `GCNConv`
and `GATConv`, so the plain wheel is enough — none of the compiled
`torch-scatter` / `torch-sparse` extras are required.

#%% code
try:
    import torch_geometric
    print("already present:", torch_geometric.__version__)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"],
                   check=True)
    import torch_geometric
    print("installed:", torch_geometric.__version__)

#%% md
## Dataset inventory

The Kaggle mirror ships PhysioNet's layout verbatim. `find_root` locates the
tree by looking for a directory containing `hosp/`, so it tolerates the extra
nesting level Kaggle adds.

#%% code
from src import mimic_iv

root = mimic_iv.find_root(RAW)
print("root:", root)
assert root is not None, f"no MIMIC-IV tree under {RAW}"

rows = []
for sub in ("hosp", "icu"):
    d = os.path.join(root, sub)
    for f in sorted(os.listdir(d)):
        rows.append({"table": f"{sub}/{f}", "GB": os.path.getsize(os.path.join(d, f)) / 1e9})
inv = pd.DataFrame(rows).sort_values("GB", ascending=False)
print(f"{len(inv)} tables, {inv['GB'].sum():.1f} GB total\n")
inv.head(12).to_string(index=False)

#%% md
## Which tables the pipeline actually reads

Only a small fraction of that 100 GB is touched. Each feature block names its
source; a block whose source is absent is skipped and simply contributes no
columns.

#%% code
NEEDED = {
    "hosp/patients.csv":       "cohort: age, sex, date of death",
    "hosp/admissions.csv":     "cohort: admission/discharge times, outcome flag",
    "icu/icustays.csv":        "cohort: ICU window, care unit",
    "hosp/diagnoses_icd.csv":  "block: comorbidity (prior admissions only)",
    "hosp/labevents.csv":      "block: labs",
    "icu/chartevents.csv":     "block: vitals",
    "icu/inputevents.csv":     "block: treatment (drugs, fluids)",
    "icu/procedureevents.csv": "block: treatment (lines, ventilation)",
    "icu/outputevents.csv":    "block: treatment (urine output)",
}
missing = []
for rel, why in NEEDED.items():
    ok = mimic_iv.has_table(root, rel)
    missing += [] if ok else [rel]
    print(f"  {'OK ' if ok else 'GONE'}  {rel:26s} {why}")
print("\nall required tables present" if not missing else f"\nMISSING: {missing}")

#%% md
## Schema check

The ETL reads specific columns by name. This reads only the header plus a
handful of rows from each table — cheap even for the 42 GB files, because
`nrows` stops the reader immediately.

#%% code
EXPECTED = {
    "hosp/patients.csv":   ["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    "hosp/admissions.csv": ["subject_id", "hadm_id", "admittime", "dischtime",
                            "deathtime", "admission_type", "admission_location",
                            "insurance", "marital_status", "race",
                            "hospital_expire_flag"],
    "icu/icustays.csv":    ["subject_id", "hadm_id", "stay_id", "first_careunit",
                            "intime", "outtime", "los"],
    "hosp/labevents.csv":  ["hadm_id", "itemid", "charttime", "valuenum"],
    "icu/chartevents.csv": ["stay_id", "itemid", "charttime", "valuenum"],
    "icu/inputevents.csv": ["stay_id", "itemid", "starttime"],
    "icu/outputevents.csv":["stay_id", "itemid", "charttime", "value"],
}
for rel, cols in EXPECTED.items():
    head = pd.read_csv(mimic_iv.table_path(root, rel), nrows=3)
    absent = [c for c in cols if c not in head.columns]
    print(f"  {rel:24s} {len(head.columns):3d} cols  " +
          ("all expected present" if not absent else f"!! MISSING {absent}"))

#%% md
## Concept dictionaries

Features are defined by `itemid` sets in `src/mimic_concepts.py`. A concept
whose itemids never appear would silently produce an all-missing column, so
confirm the ids resolve against the dictionaries.

#%% code
from src import mimic_concepts as mc

d_labs = pd.read_csv(mimic_iv.table_path(root, "hosp/d_labitems.csv"))
d_items = pd.read_csv(mimic_iv.table_path(root, "icu/d_items.csv"))

for label, items, dic in [("labs", mc.LAB_ITEMS, d_labs),
                          ("chart", mc.CHART_ITEMS, d_items),
                          ("input", mc.INPUT_ITEMS, d_items),
                          ("proc", mc.PROC_ITEMS, d_items)]:
    known = set(dic["itemid"])
    unknown = [i for i in items if i not in known]
    concepts = sorted({v[0] for v in items.values()})
    print(f"  {label:6s} {len(items):3d} itemids -> {len(concepts):2d} concepts"
          f"   unresolved: {unknown if unknown else 'none'}")

urine = set(mc.OUTPUT_URINE_ITEMS) | set(mc.OUTPUT_URINE_NEGATIVE_ITEMS)
print("\nurine itemids:")
print(d_items[d_items["itemid"].isin(urine)][["itemid", "label"]].to_string(index=False))

#%% md
### On the urine itemids

`227488` ("GU Irrigant Volume In") is fluid *instilled* for bladder irrigation,
not urine produced, and `227489` ("GU Irrigant/Urine Volume Out") returns it
mixed with urine. MIMIC's own `urine_output` concept subtracts the former from
the latter, and `src/mimic_iv._urine_totals` now does the same. Summing both as
output overstates urine for anyone on irrigation — mostly post-urology
patients, i.e. a systematic bias in a specific subgroup rather than noise.

#%% md
## Audit result

Record the outcome so later notebooks can assert they are running against the
same mount.

#%% code
audit = {
    "root": root,
    "tables_present": sorted(NEEDED),
    "tables_missing": missing,
    "blocks_possible": ["demographics", "comorbidity"]
                       + (["labs"] if mimic_iv.has_table(root, "hosp/labevents.csv") else [])
                       + (["vitals"] if mimic_iv.has_table(root, "icu/chartevents.csv") else [])
                       + (["treatment"] if mimic_iv.has_table(root, "icu/inputevents.csv") else []),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cpus": os.cpu_count(),
}
os.makedirs(CACHE, exist_ok=True)
json.dump(audit, open(f"{CACHE}/audit.json", "w"), indent=2)
print(json.dumps(audit, indent=2))

#%% md
**Next:** `01_cohort_extraction` turns these tables into one patient-level
design matrix, and asserts the leakage controls that make the label honest.
