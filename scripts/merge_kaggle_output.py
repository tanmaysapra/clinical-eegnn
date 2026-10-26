#!/usr/bin/env python
"""Mirror a Kaggle run's log back into the local notebook as cell outputs.

    python scripts/merge_kaggle_output.py            # notebooks 12-18, results/kaggle_runs/kNN_v1
    python scripts/merge_kaggle_output.py 17         # one notebook
    python scripts/merge_kaggle_output.py 17 path/to/run.log

`kaggle kernels output` returns a flat stdout/stderr stream (no cell
boundaries), so placement is heuristic: `Writing <path>` lines go to the
`%%writefile <path>` cell they came from; everything else goes to the last
code cell. Re-run after `nbmake.py` (which regenerates the notebook with empty
outputs). The pushed copy under notebooks/kaggle/ is left untouched.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE = re.compile(r"Debugger warning|frozen modules|PYDEVD_DISABLE_FILE_VALIDATION|"
                   r"Debugging will proceed|^0\.00s - ")


def _stream(name, lines):
    return {"output_type": "stream", "name": name, "text": lines}


def merge(nb_path, log_path):
    recs = json.load(open(log_path, encoding="utf-8"))
    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    writefile = {}
    for c in cells:
        c["outputs"], c["execution_count"] = [], None
        m = re.match(r"%%writefile\s+(\S+)", "".join(c["source"]))
        if m:
            writefile[m.group(1)] = c

    last = []
    for r in recs:
        text = r["data"]
        if r["stream_name"] == "stderr" and all(NOISE.search(l) for l in text.splitlines() or [text]):
            continue
        m = re.match(r"Writing (\S+)\n?$", text)
        if r["stream_name"] == "stdout" and m and m.group(1) in writefile:
            writefile[m.group(1)]["outputs"].append(_stream("stdout", [text]))
        else:
            last.append((r["stream_name"], text))

    merged = []
    for name, text in last:
        if merged and merged[-1][0] == name:
            merged[-1][1].append(text)
        else:
            merged.append((name, [text]))
    cells[-1]["outputs"] = [_stream(n, t) for n, t in merged]

    for i, c in enumerate(cells, 1):
        if c["outputs"]:
            c["execution_count"] = i
    nb["metadata"]["kaggle_run"] = {
        "log": os.path.relpath(log_path, ROOT).replace(os.sep, "/"),
        "note": "outputs mirrored from the Kaggle log by scripts/merge_kaggle_output.py; "
                "non-writefile output is attached to the last code cell",
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return sum(len(c["outputs"]) for c in cells)


def main(argv):
    nums = [argv[0]] if argv else [f"{n}" for n in range(12, 19)]
    for num in nums:
        nbs = glob.glob(os.path.join(ROOT, "notebooks", f"{num}_*.ipynb"))
        logs = [argv[1]] if len(argv) > 1 else glob.glob(
            os.path.join(ROOT, "results", "kaggle_runs", f"k{num}_v*", "*.log"))
        if len(nbs) != 1 or not logs:
            print(f"[skip] {num}: notebook={nbs} logs={logs}")
            continue
        log = sorted(logs)[-1]
        n = merge(nbs[0], log)
        print(f"[ok] {os.path.basename(nbs[0])} <- {os.path.relpath(log, ROOT)} ({n} output blocks)")


if __name__ == "__main__":
    main(sys.argv[1:])
