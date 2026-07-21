"""Figures: reliability diagrams, accuracy-efficiency trade-off, exit analysis."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluate import reliability_curve  # noqa: E402


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def reliability_diagram(y, p, out_dir, name="reliability.png", n_bins=15, title=""):
    conf, acc, cnt = reliability_curve(y, p, n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    ok = ~np.isnan(conf)
    ax.plot(conf[ok], acc[ok], "o-", color="#2b6cb0", label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title or "Reliability")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


def tradeoff_curve(points, out_dir, name="tradeoff.png"):
    """points: list of dicts with avg_depth, auroc, and the swept label."""
    depth = [q["avg_depth"] for q in points]
    auroc = [q["auroc"] for q in points]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(depth, auroc, "o-", color="#2b6cb0")
    for q in points:
        ax.annotate(q["label"], (q["avg_depth"], q["auroc"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("average exit depth (layers per patient)")
    ax.set_ylabel("test AUROC")
    ax.set_title("Accuracy vs efficiency")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, name)


def exit_distribution(dist, out_dir, name="exit_distribution.png"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    layers = sorted(dist)
    ax.bar([str(l) for l in layers], [dist[l] for l in layers], color="#2b6cb0")
    ax.set_xlabel("exit layer")
    ax.set_ylabel("patients")
    ax.set_title("Where patients exit")
    return _save(fig, out_dir, name)


def depth_vs_difficulty(analysis, out_dir, name="depth_vs_difficulty.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(analysis["margin"], analysis["depth"], s=6, alpha=0.4, color="#2b6cb0")
    ax.set_xlabel("|p - 0.5|  (larger = more clear-cut)")
    ax.set_ylabel("exit depth")
    ax.set_title(f"Depth vs case difficulty (r = {analysis['corr_depth_vs_margin']:.3f})")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, name)
