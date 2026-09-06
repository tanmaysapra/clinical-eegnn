#!/usr/bin/env python
"""Aggregate `run_architectures.py` into the controlled comparison.

The question the spec set and the project never answered: **does ACT actually
beat discrete exit stages?** Everything here is paired by seed, because the
three architectures share splits and graph, and every headline in this project
that was not paired-and-multi-seeded has since failed to reproduce.

    python scripts/analyse_architectures.py results/architectures.json
"""
import json
import sys

import numpy as np

CRIT = 2.776          # df=4, two-sided 0.05 -- the project's standard


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float("nan")), v.size


def paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), d
    s = d.std(ddof=1)
    t = d.mean() / (s / np.sqrt(d.size)) if s > 0 else float("inf")
    return float(d.mean()), float(s), float(t), d


def col(runs, arch, *path, default=None):
    out = []
    for r in runs:
        cur = r["archs"].get(arch)
        for k in path:
            if cur is None:
                break
            cur = cur.get(k) if isinstance(cur, dict) else None
        out.append(default if cur is None else cur)
    return out


def main(path):
    d = json.load(open(path))
    runs = d["runs"]
    args = d["args"]
    archs = [a for a in ("1_graph_act", "2_ppr_push", "3_fixed_block")
             if a in runs[0]["archs"]]
    seeds = [r["seed"] for r in runs]
    L = args["n_blocks"]

    print(f"== {path}")
    print(f"seeds {seeds}   n={runs[0]['n']}   prevalence={runs[0]['prevalence']:.5f}")
    print(f"depth budget shared by all architectures: {L}")
    print("all three share: cohort, splits, graph, calibration split, "
          "certified gate\n")

    # ------------------------------------------------------------- accuracy
    print("## Full-depth quality (no gating), test set")
    print(f"{'architecture':>16} {'AUROC':>18} {'AUPRC':>18} {'params':>9} "
          f"{'train s':>8}")
    for a in archs:
        au = col(runs, a, "full_depth", "auroc")
        ap = col(runs, a, "full_depth", "auprc")
        pr = col(runs, a, "n_params")
        ts = col(runs, a, "train_s")
        m, s, _ = ms(au); m2, s2, _ = ms(ap)
        print(f"{a:>16} {m:>9.4f} ±{s:.4f} {m2:>9.4f} ±{s2:.4f} "
              f"{int(np.mean(pr)):>9} {np.mean(ts):>8.1f}")

    # ------------------------------------------- THE controlled comparison
    print("\n## The spec's question: does ACT beat discrete exit stages?")
    print("   (Architecture 1 vs Architecture 3, same operator family, same")
    print("    budget, same graph, same gate. Paired over seeds, |t|crit=2.776)")
    if "1_graph_act" in archs and "3_fixed_block" in archs:
        a1 = col(runs, "1_graph_act", "full_depth", "auroc")
        a3 = col(runs, "3_fixed_block", "full_depth", "auroc")
        m, s, t, per = paired_t(a1, a3)
        verdict = ("ACT significantly better" if t > CRIT else
                   "discrete stages significantly better" if t < -CRIT else
                   "INDISTINGUISHABLE")
        print(f"   AUROC(ACT) - AUROC(fixed-block) = {m:+.4f} ±{s:.4f}  "
              f"t={t:+.2f}  ->  {verdict}")
        print(f"   per-seed {np.round(per, 4).tolist()}")

    # ------------------------------------------------------ stage structure
    print("\n## Are the exits differentiated? (last stage AUROC - first)")
    print("   This is the SS5n diagnostic. A flat profile means the exit")
    print("   mechanism is choosing between options that are all the same,")
    print("   which is what made the depth cascade fail.")
    for a in archs:
        gap = col(runs, a, "stage_gap_last_minus_first")
        m, s, t, _ = paired_t(gap, np.zeros(len(gap)))
        sa = np.array([col(runs, a, "stage_auroc")[i] for i in range(len(runs))],
                      dtype=object)
        prof = np.mean([np.asarray(x, float) for x in sa], axis=0)
        print(f"   {a:>16} gap {m:+.4f} ±{s:.4f}  t={t:+.2f}  "
              f"{'DIFFERENTIATED' if abs(t) > CRIT else 'flat'}")
        print(f"   {'':>16} stage profile {np.round(prof, 4).tolist()}")

    # ------------------------------------------------------------- compute
    print("\n## Compute saving: measured vs analytical")
    print("   Architecture 1 has no measured figure -- `model.py` propagates")
    print("   every node at every step and masks afterwards, so its saving is")
    print("   exit-depth arithmetic. 2 and 3 count the work they actually did.")
    for a in archs:
        meas = col(runs, a, "compute_is_measured")
        tag = "measured" if all(meas) else "ANALYTICAL ONLY"
        if a == "1_graph_act":
            an = col(runs, a, "analytical_depth_saved_pct")
            dep = col(runs, a, "mean_realised_depth")
            m, s, _ = ms(an); m2, s2, _ = ms(dep)
            print(f"   {a:>16} [{tag}] depth {m2:.2f}/{L} ±{s2:.2f}  "
                  f"analytical saving {m:.1f}% ±{s:.1f}")
        else:
            for al in args["alphas"]:
                k = f"alpha={al}"
                ge = col(runs, a, "gates", k, "measured_edge_work_saved_pct")
                gs = col(runs, a, "gates", k, "compute_saved_pct")
                m, s, n = ms(ge); m2, s2, _ = ms(gs)
                if n:
                    print(f"   {a:>16} [{tag}] {k}: measured edge work saved "
                          f"{m:.1f}% ±{s:.1f}   (gate's stage arithmetic said "
                          f"{m2:.1f}% ±{s2:.1f})")

    # ------------------------------------------------ Architecture 2's curve
    if "2_ppr_push" in archs:
        print("\n## Architecture 2: the work/accuracy curve (its whole claim)")
        floor = col(runs, "2_ppr_push", "encoder_only_auroc")
        fm, fs, _ = ms(floor)
        print(f"   graph-free encoder floor: AUROC {fm:.4f} ±{fs:.4f}")
        sweeps = col(runs, "2_ppr_push", "eps_sweep")
        if sweeps and sweeps[0]:
            eps = [r["eps_push"] for r in sweeps[0]]
            print(f"   {'eps_push':>10} {'edges saved %':>16} {'exit round':>12} "
                  f"{'AUROC':>16} {'vs floor':>10}")
            for i, e in enumerate(eps):
                sv = ms([s[i]["edge_work_saved_pct"] for s in sweeps])
                au = ms([s[i]["test_auroc"] for s in sweeps])
                er = ms([s[i]["mean_exit_round"] for s in sweeps])
                print(f"   {e:>10g} {sv[0]:>9.1f} ±{sv[1]:<5.1f} "
                      f"{er[0]:>7.2f} ±{er[1]:<4.2f} {au[0]:>9.4f} ±{au[1]:.4f} "
                      f"{au[0]-fm:>+10.4f}")
            best = max(range(len(eps)),
                       key=lambda i: ms([s[i]["test_auroc"] for s in sweeps])[0])
            print(f"   best AUROC at eps={eps[best]:g}; diffusion beats the "
                  f"encoder floor by "
                  f"{ms([s[best]['test_auroc'] for s in sweeps])[0]-fm:+.4f}")

    # --------------------------------------------------------------- safety
    print("\n## Under the shared certified gate")
    for al in args["alphas"]:
        k = f"alpha={al}"
        print(f"  alpha={al}")
        for a in archs:
            au = col(runs, a, "gates", k, "auroc")
            mr = col(runs, a, "gates", k, "miss_rate_among_released")
            rf = col(runs, a, "gates", k, "released_frac")
            cs = col(runs, a, "gates", k, "compute_saved_pct")
            m, s, _ = ms(au); mm, msd, mn = ms(mr)
            rm, rs, _ = ms(rf); cm, csd, _ = ms(cs)
            fires = sum(1 for x in rf if x and x > 0)
            breach = sum(1 for x in mr
                         if x is not None and np.isfinite(x) and x > al)
            miss = "— never fires" if mn == 0 else f"{mm:.4f} ±{msd:.4f}"
            print(f"    {a:>16} AUROC {m:.4f} ±{s:.4f}  released "
                  f"{rm:.3f} ±{rs:.3f}  miss {miss}  "
                  f"fires {fires}/{len(rf)}  breaches {breach}/{len(rf)}  "
                  f"saved {cm:.1f}%")

    print("\n## What this licenses saying")
    if "1_graph_act" in archs and "3_fixed_block" in archs:
        a1 = col(runs, "1_graph_act", "full_depth", "auroc")
        a3 = col(runs, "3_fixed_block", "full_depth", "auroc")
        _, _, t, _ = paired_t(a1, a3)
        if abs(t) <= CRIT:
            print("  Architecture 1 and Architecture 3 are indistinguishable at")
            print("  matched budget. The negative result is therefore about")
            print("  early exit on this data, NOT about the ACT halting")
            print("  mechanism specifically -- both exit designs fail the same")
            print("  way. That is the stronger claim, and it is now supported.")
        else:
            print("  The two exit mechanisms differ significantly. The earlier")
            print("  negative result was mechanism-specific and should be")
            print("  restated accordingly.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/architectures.json")
