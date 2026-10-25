#!/usr/bin/env python
"""Build a `.ipynb` from this repo's marker-split `.py` notebook source
convention (`notebooks/README.md`): `#%% md` / `#%% code` on their own
line, one cell per marker, everything after it (until the next marker)
is that cell's content.

    python scripts/nbmake.py notebooks/src/09_arch4r_phase0.py notebooks/09_arch4r_phase0.ipynb

The `.ipynb` is a build artifact -- always regenerate it from the `.py`
source after editing the source, never hand-edit the `.ipynb` directly
(see notebooks/README.md).
"""
import json
import sys


def _embed_cell(rel_path, repo_root):
    """`#%% embed src/foo.py` -> one code cell that `%%writefile`s that repo
    file into /kaggle/working/<rel_path>, so a notebook can carry the modules
    it needs without cloning the (private) repo -- no GITHUB_TOKEN secret."""
    import os
    body = open(os.path.join(repo_root, rel_path), encoding="utf-8").read()
    lines = [f"%%writefile /kaggle/working/{rel_path}\n"] + [l + "\n" for l in body.splitlines()]
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def convert(src_path, out_path):
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(src_path, encoding="utf-8").read()
    section, cells = None, []
    for line in text.splitlines():
        if line.strip().startswith("#%% embed "):
            cells.append(_embed_cell(line.strip()[len("#%% embed "):].strip(), repo_root))
            section = None
            continue
        if line.strip() == "#%% md":
            section = "markdown"
            cells.append({"cell_type": "markdown", "metadata": {}, "source": []})
            continue
        if line.strip() == "#%% code":
            section = "code"
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                         "outputs": [], "source": []})
            continue
        if section is not None:
            cells[-1]["source"].append(line + "\n")

    for c in cells:
        src = "".join(c["source"])
        c["source"] = src.rstrip("\n").splitlines(keepends=True) if src.strip() else []

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"{src_path} -> {out_path}  ({len(cells)} cells)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
