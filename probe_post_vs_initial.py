"""
Does the post-assert residual stream predict flipping beyond what the initial-turn residual
stream already predicts, layer by layer?

For initially right questions, per layer, on the same seeded 5-fold stratified splits as
probe_initial.py / probe_post_assert.py:
    assert_flipped ~ z(initial-probe OOF logit) + z(post-assert-probe OOF logit)
Reports the post-assert term (coef, 95% CI, p), plus the same fit with a dataset covariate
and the initial term for reference. Expectation: post-assert term near zero through layer 17,
significant from layer 18.

Usage:
    python probe_post_vs_initial.py --n 200 --out-dir results/probe_post_vs_initial/subset_n200   # quick test
    python probe_post_vs_initial.py                                                         # full run
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from probe_initial import BLUE, INK, MUTED, probe_cv
from probe_baselines import FOCUS, HEADLINE, ORANGE, Tee, beyond, ranges, style


def plot(df, path):
    x = df["layer_1idx"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.axvspan(FOCUS[0] - 0.5, FOCUS[-1] + 0.5, color="#f0efec", zorder=0)
    for pre, color, label in [("post", BLUE, "post-assert term"), ("init", ORANGE, "initial term")]:
        ax.fill_between(x, df[f"{pre}_ci_lo"], df[f"{pre}_ci_hi"], color=color, alpha=0.18, linewidth=0)
        ax.plot(x, df[f"{pre}_coef"], color=color, linewidth=2, marker="o", markersize=4,
                label=f"{label} (95% CI)")
    sig = df[(df.post_p < 0.05) & (df.post_coef > 0)]
    ax.scatter(sig["layer_1idx"], sig["post_coef"], s=60, facecolors="none", edgecolors=INK, zorder=5,
               label="post-assert term p < 0.05")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(HEADLINE, color=MUTED, linewidth=0.8, alpha=0.7)
    ax.text(FOCUS[0] - 0.3, ax.get_ylim()[0], "layers 8–17", color=MUTED, fontsize=8, va="bottom")
    ax.text(HEADLINE + 0.2, ax.get_ylim()[0], f"L{HEADLINE}", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlim(0.5, x[-1] + 0.5)
    ax.set_title(f"flipped ~ z(initial-probe logit) + z(post-assert-probe logit), initially right "
                 f"(n={df.n.iloc[0]})", color=INK, fontsize=9, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    style(ax, "coefficient (per SD of probe logit)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--n", type=int, default=None, help="random subset of rows (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/probe_post_vs_initial")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import sys
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    recs = [json.loads(line) for line in open(args.records)]
    acts_init = np.load("data/acts_initial.npy", mmap_mode="r")
    acts_assert = np.load("data/acts_assert.npy", mmap_mode="r")
    assert len(recs) == acts_init.shape[0] == acts_assert.shape[0], "records and activations disagree"
    n_layers = acts_init.shape[1]
    rows = np.arange(len(recs))
    if args.n:  # same subset as the earlier scripts with the same --n/--seed
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.n, replace=False))
    right = np.array([recs[i]["correct"] for i in rows], dtype=bool)
    r = rows[right]
    y = np.array([recs[i]["assert"]["flipped"] for i in r]).astype(int)
    ds = np.array([recs[i]["dataset"] == "ARC-Challenge" for i in r], dtype=float)
    two_ds = 0 < ds.sum() < len(ds)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(r, y))
    print(f"Initially right: n={len(y)}, flipped={y.sum()} ({y.mean():.1%}); layers 1-indexed")

    out = []
    for layer in range(n_layers):
        t0 = time.time()
        i_oof, i_aucs, _ = probe_cv(np.asarray(acts_init[:, layer, :], dtype=np.float32)[r], y, folds, args.seed)
        p_oof, p_aucs, _ = probe_cv(np.asarray(acts_assert[:, layer, :], dtype=np.float32)[r], y, folds, args.seed)
        post = beyond(y, i_oof, p_oof)   # post-assert term, controlling for initial
        init = beyond(y, p_oof, i_oof)   # initial term, controlling for post-assert (same fit)
        post_ds = beyond(y, i_oof, p_oof, ds) if two_ds else (np.nan,) * 4
        row = {"layer": layer, "layer_1idx": layer + 1, "n": len(y), "n_pos": int(y.sum()),
               "initial_auroc_mean": i_aucs.mean(), "post_assert_auroc_mean": p_aucs.mean()}
        for pre, vals in [("post", post), ("init", init), ("post_ds", post_ds)]:
            row.update(dict(zip([f"{pre}_coef", f"{pre}_ci_lo", f"{pre}_ci_hi", f"{pre}_p"], vals)))
        out.append(row)
        print(f"  L{layer + 1:2d}: post-assert term {post[0]:+.2f} [{post[1]:+.2f}, {post[2]:+.2f}] "
              f"p={post[3]:.2g}   initial term {init[0]:+.2f} p={init[3]:.2g}   "
              f"(AUROC initial {i_aucs.mean():.3f}, post {p_aucs.mean():.3f}; {time.time() - t0:.0f}s)")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(args.out_dir, "post_vs_initial.csv"), index=False)
    plot(df, os.path.join(args.out_dir, "post_vs_initial.png"))

    if not args.n:
        old = pd.read_csv("results/probe_baselines/diff_probe.csv")
        d = max(np.abs(df.initial_auroc_mean - old.initial_auroc_mean).max(),
                np.abs(df.post_assert_auroc_mean - old.post_assert_auroc_mean).max())
        print(f"\nReproduction check vs probe_baselines (same folds): max |ΔAUROC| {d:.2g}")
    print(f"Wrote CSV, plot and run_log.txt to {args.out_dir}/")

    sig = df[(df.post_p < 0.05) & (df.post_coef > 0)]["layer_1idx"].tolist()
    sig_ds = df[(df.post_ds_p < 0.05) & (df.post_ds_coef > 0)]["layer_1idx"].tolist()
    early, late = df[df.layer_1idx < HEADLINE], df[df.layer_1idx >= HEADLINE]
    h = df.iloc[HEADLINE - 1]
    print("\n=== SUMMARY ===")
    print(f"Post-assert term significant (p<0.05, positive) at: {ranges(sig)}"
          + (f"; with dataset covariate: {ranges(sig_ds)}" if two_ds else ""))
    print(f"  L1–17: coefficients {early.post_coef.min():+.2f} to {early.post_coef.max():+.2f}, "
          f"significant at {int(((early.post_p < 0.05) & (early.post_coef > 0)).sum())}/17 layers")
    print(f"  L18: {h.post_coef:+.2f} [{h.post_ci_lo:+.2f}, {h.post_ci_hi:+.2f}] p={h.post_p:.2g}; "
          f"L18–26 coefficients {late.post_coef.min():+.2f} to {late.post_coef.max():+.2f}")
    near_zero = not ((early.post_p < 0.05) & (early.post_coef > 0)).any()
    from_18 = ((late.post_p < 0.05) & (late.post_coef > 0)).all()
    print(f"  → Expectation (near zero through L17, significant from L18): "
          f"{'confirmed' if near_zero and from_18 else 'NOT confirmed'} — "
          f"{'no' if near_zero else 'some'} early layers add information beyond the initial turn; "
          f"{'every' if from_18 else 'not every'} layer from L18 does.")


if __name__ == "__main__":
    main()
