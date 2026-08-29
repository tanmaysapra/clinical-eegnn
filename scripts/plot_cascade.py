#!/usr/bin/env python
"""The accuracy/compute frontier, which HANDOVER 5h said did not exist.

    python scripts/plot_cascade.py results/headline.json

Left panel is the frontier itself: the certified gate against the two trivial
policies the depth policy was dominated by. Right panel is the safety
objective -- what share of each severity stratum the gate is willing to release.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical slots 1-3 (all-pairs, light surface). Slot 3's contrast
# obliges direct labels, which every mark using it carries.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def main(path, out):
    runs = json.load(open(path))["runs"]
    mean = lambda f: float(np.nanmean([f(r) for r in runs]))
    std = lambda f: float(np.nanstd([f(r) for r in runs]))

    keys = [k for k in runs[0]["gates"]
            if k.startswith("global|") and k.endswith("twosided|conformal")]
    alphas = sorted(float(k.split("|")[1].split("=")[1]) for k in keys)
    keys = [f"global|alpha={a}|twosided|conformal" for a in alphas]

    g = lambda field: [mean(lambda r, k=k: r["gates"][k][field]) for k in keys]
    gs = lambda field: [std(lambda r, k=k: r["gates"][k][field]) for k in keys]
    saved, auroc, aerr = g("compute_saved_pct"), g("auroc"), gs("auroc")
    nolab = g("resolved_before_labs_pct")
    hi = [100 * v for v in g("high_risk_release_rate")]
    lo = [100 * v for v in g("low_risk_release_rate")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)

    # -------------------------------------------------------- the frontier
    ax.errorbar(saved, auroc, yerr=aerr, marker="o", ms=8, color=BLUE, capsize=3,
                lw=2, ecolor=BLUE, mec=SURFACE, mew=2, zorder=3,
                label="certified two-sided gate")
    # Hand-placed so labels clear the line, the error bars and each other; the
    # two tightest budgets sit almost on top of one another.
    default = [(11, 6), (11, -26), (12, 6), (11, -26)]
    offsets = (default[:len(alphas)] if len(alphas) <= len(default)
               else default + [(11, -26)] * (len(alphas) - len(default)))
    for a, s, u, nl, off in zip(alphas, saved, auroc, nolab, offsets):
        ax.annotate(f"α={a}\n{nl:.0f}% no labs", (s, u), textcoords="offset points",
                    xytext=off, fontsize=8, color=MUTED, linespacing=1.35)
    for nm, colour, mark, label in (
            ("always_stage0", ORANGE, "s", "always stage 0"),
            ("always_full", AQUA, "^", "always full cascade")):
        x = mean(lambda r, n=nm: r["baselines"][n]["compute_saved_pct"])
        yv = mean(lambda r, n=nm: r["baselines"][n]["auroc"])
        ax.errorbar(x, yv, yerr=std(lambda r, n=nm: r["baselines"][n]["auroc"]),
                    marker=mark, color=colour, capsize=3, ms=11, ls="none",
                    mec=SURFACE, mew=2, ecolor=colour, zorder=4, label=label)
        xy, ha = ((4, -50), "left") if nm == "always_full" else ((-15, -3), "right")
        ax.annotate(label, (x, yv), textcoords="offset points", xytext=xy,
                    ha=ha, fontsize=8.5, color=MUTED)
    ax.set_xlabel("compute saved (%)", color=MUTED, fontsize=10)
    ax.set_ylabel("test AUROC", color=MUTED, fontsize=10)
    ax.set_title("An accuracy/compute trade-off exists\n"
                 "and the gate is not dominated by either constant policy",
                 fontsize=11, color=INK, loc="left")
    ax.margins(x=0.22, y=0.20)
    _style(ax)

    # ----------------------------------------------------- who gets released
    w, x = 0.36, np.arange(len(alphas))
    for offs, vals, colour, label in ((-w / 2 - 0.01, hi, BLUE, "top severity decile"),
                                      (+w / 2 + 0.01, lo, ORANGE, "everyone else")):
        bx.bar(x + offs, vals, w, color=colour, label=label, zorder=3,
               edgecolor=SURFACE, lw=2)
        for xi, v in zip(x + offs, vals):
            bx.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8.5, color=MUTED)
    for i, (h, l) in enumerate(zip(hi, lo)):
        if h > 0:
            bx.annotate(f"{l / h:.0f}× separation", (i, max(h, l) + 7),
                        ha="center", fontsize=9, color=INK)
    bx.set_xticks(x)
    bx.set_xticklabels([f"α = {a}" for a in alphas])
    bx.set_ylabel("released early (%)", color=MUTED, fontsize=10)
    bx.set_xlabel("safety budget: share of released patients allowed to be deaths",
                  color=MUTED, fontsize=9.5)
    bx.set_title("Only verified low-risk patients exit early\n"
                 "α = 0.005 releases nobody — the bound declines to certify it",
                 fontsize=11, color=INK, loc="left")
    bx.legend(fontsize=8.5, frameon=False, labelcolor=MUTED, loc="upper left")
    bx.margins(y=0.22)
    _style(bx)
    bx.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/headline.json",
         sys.argv[2] if len(sys.argv) > 2 else "results/figures/cascade_frontier.png")
