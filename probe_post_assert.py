"""
Linear probes on the post-assert residual stream (data/acts_assert.npy), per layer:
is the flip decision linearly readable before the logit lens shows it (~layers 17–19)?

Target: record["assert"]["flipped"], in two versions: all questions, and initially right only.
Probe setup is imported from probe_initial.py (scaled L2 logistic, inner-CV C, seeded
5-fold stratified outer CV). Per-layer baselines on the same folds:
  - logit-lens score: lens[layer, initial] − max of the other three letters (from assert lens)
  - initial margin (as in probe_initial.py)
Layers of interest: 8–17 (1-indexed).

Usage:
    python probe_post_assert.py --n 200 --out-dir results/probe_post_assert/subset_n200   # quick test
    python probe_post_assert.py                                                           # full run
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

from runner import LETTERS
from probe_initial import BLUE, INK, MUTED, margin_cv, probe_cv

ORANGE = "#eb6834"
FOCUS = range(8, 18)  # 1-indexed layers of interest
SUSTAIN = 3  # consecutive clear wins needed for the verdict


def lens_score(rec, layer):
    """Logit-lens lead of the initial answer at this layer, after the assert follow-up."""
    lens = np.array(rec["assert"]["lens"])[layer]
    init = LETTERS.index(rec["initial"])
    return lens[init] - np.delete(lens, init).max()


def run_version(name, acts, recs, rows, n_layers, seed):
    y = np.array([recs[i]["assert"]["flipped"] for i in rows]).astype(int)
    margin = np.array([recs[i]["initial_margin"] for i in rows])
    print(f"\n=== {name}: n={len(y)}, flipped={y.sum()} ({y.mean():.1%}) ===")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rows, y))
    _, m_aucs = margin_cv(margin, y, folds)  # margin_cv is a generic one-feature classifier
    print(f"  Initial-margin baseline AUROC: {m_aucs.mean():.3f} ± {m_aucs.std():.3f}")

    out = []
    for layer in range(n_layers):
        t0 = time.time()
        X = np.asarray(acts[:, layer, :], dtype=np.float32)[rows]
        _, p_aucs, cs = probe_cv(X, y, folds, seed)
        lens = np.array([lens_score(recs[i], layer) for i in rows])
        _, l_aucs = margin_cv(lens, y, folds)
        clear = p_aucs.mean() - p_aucs.std() > l_aucs.mean() + l_aucs.std()
        out.append({
            "version": name, "layer": layer, "layer_1idx": layer + 1, "n": len(y), "n_pos": int(y.sum()),
            "probe_auroc_mean": p_aucs.mean(), "probe_auroc_std": p_aucs.std(),
            "lens_auroc_mean": l_aucs.mean(), "lens_auroc_std": l_aucs.std(),
            "margin_auroc_mean": m_aucs.mean(), "margin_auroc_std": m_aucs.std(),
            "probe_minus_lens": p_aucs.mean() - l_aucs.mean(),
            "probe_clearly_beats_lens": bool(clear),
            "folds_probe_beats_lens": int((p_aucs > l_aucs).sum()),
            "median_C": float(np.median(cs)),
        })
        print(f"  layer {layer + 1:2d}: probe {p_aucs.mean():.3f} ± {p_aucs.std():.3f}   "
              f"lens {l_aucs.mean():.3f} ± {l_aucs.std():.3f}   "
              f"{'CLEAR ' if clear else '      '}C={np.median(cs):g}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(out)


def plot(df, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    x = df["layer_1idx"].to_numpy()
    ax.axvspan(FOCUS[0] - 0.5, FOCUS[-1] + 0.5, color="#f0efec", zorder=0)
    ax.text(FOCUS[0] - 0.3, 0.985, "layers of interest", color=MUTED, fontsize=8, va="top")
    for col, color, label in [("probe", BLUE, "linear probe"), ("lens", ORANGE, "logit-lens score")]:
        mean, std = df[f"{col}_auroc_mean"].to_numpy(), df[f"{col}_auroc_std"].to_numpy()
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=4, label=label + " (± fold std)")
    m = df["margin_auroc_mean"].iloc[0]
    ax.axhline(m, color=MUTED, linestyle="--", linewidth=1.5, label=f"initial margin ({m:.3f})")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(x[0] + 0.3, 0.495, "chance", color=MUTED, fontsize=8, ha="left", va="top")
    lo = min(0.45, (df["probe_auroc_mean"] - df["probe_auroc_std"]).min() - 0.02,
             (df["lens_auroc_mean"] - df["lens_auroc_std"]).min() - 0.02)
    ax.set_ylim(lo, 1.0)
    ax.set_xlim(0.5, x[-1] + 0.5)
    ax.set_xlabel("Layer", color=INK)
    ax.set_ylabel("AUROC for flip under assert (out-of-fold)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(df, name):
    clear = df[df["probe_clearly_beats_lens"]]
    focus = df[df["layer_1idx"].isin(FOCUS)]
    print(f"\n{name} (n={df.n.iloc[0]}, flipped {df.n_pos.iloc[0]}):")
    if len(clear):
        print(f"  Earliest layer where the probe clearly beats the logit lens (non-overlapping "
              f"fold std): layer {int(clear['layer_1idx'].iloc[0])}.")
    else:
        print("  The probe never clearly beats the logit lens (fold std bands always overlap).")
    # A lone early "win" can be noise (e.g. a below-chance lens AUROC), so the verdict uses the
    # first layer that starts a run of SUSTAIN consecutive clear wins.
    flags = df["probe_clearly_beats_lens"].to_numpy()
    starts = [i for i in range(len(flags) - SUSTAIN + 1) if flags[i:i + SUSTAIN].all()]
    first = int(df["layer_1idx"].iloc[starts[0]]) if starts else None
    if len(clear) and first != int(clear["layer_1idx"].iloc[0]):
        print(f"  (That is an isolated win; first sustained win, ≥{SUSTAIN} consecutive layers: "
              f"{'layer ' + str(first) if first else 'none'}.)")
    print(f"  Layers {FOCUS[0]}–{FOCUS[-1]}: probe clearly ahead at "
          f"{int(focus['probe_clearly_beats_lens'].sum())}/{len(focus)} layers; "
          + "; ".join(f"L{int(r.layer_1idx)} {r.probe_auroc_mean:.2f} vs {r.lens_auroc_mean:.2f}"
                      for r in focus.itertuples()))
    best_lens = df.loc[df["lens_auroc_mean"].idxmax()]
    m = df["margin_auroc_mean"].iloc[0]
    above = df[df["probe_auroc_mean"] > m]
    print(f"  Logit lens is best at layer {int(best_lens.layer_1idx)} ({best_lens.lens_auroc_mean:.3f}); "
          f"probe first exceeds the initial margin ({m:.3f}) at "
          f"{'layer ' + str(int(above.layer_1idx.iloc[0])) if len(above) else 'no layer'}.")
    if first is not None and first < 17:
        print(f"  → The flip decision is linearly readable from layer {first}, before the logit lens "
              "shows it around layers 17–19.")
        if not len(above) or int(above.layer_1idx.iloc[0]) > first:
            print(f"    But at layer {first} the probe is still below the initial-margin baseline, so the "
                  "early signal is weaker than what confidence alone predicts.")
    elif first is not None:
        print(f"  → The probe only pulls clearly ahead at layer {first}, i.e. not before the logit-lens zone.")
    else:
        print("  → No evidence the flip decision is readable earlier than the logit lens shows it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--acts", default="data/acts_assert.npy")
    ap.add_argument("--n", type=int, default=None, help="random subset of rows (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/probe_post_assert")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.records)]
    acts = np.load(args.acts, mmap_mode="r")
    assert len(recs) == acts.shape[0], "records and activations disagree"
    n_layers = acts.shape[1]
    rows = np.arange(len(recs))
    if args.n:
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.n, replace=False))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(rows)} questions, {n_layers} layers, d_model {acts.shape[2]} (layers printed 1-indexed)")

    right = np.array([recs[i]["correct"] for i in rows], dtype=bool)
    df_all = run_version("all questions", acts, recs, rows, n_layers, args.seed)
    df_right = run_version("initially right only", acts, recs, rows[right], n_layers, args.seed)

    pd.concat([df_all, df_right]).to_csv(os.path.join(args.out_dir, "probe_post_assert.csv"), index=False)
    plot(df_all, f"Flip under assert from post-assert residual: all questions (n={df_all.n.iloc[0]})",
         os.path.join(args.out_dir, "probe_post_assert_all.png"))
    plot(df_right, f"Flip under assert from post-assert residual: initially right (n={df_right.n.iloc[0]})",
         os.path.join(args.out_dir, "probe_post_assert_right.png"))
    print(f"\nWrote CSV and plots to {args.out_dir}/")

    print("\n=== SUMMARY ===")
    summarize(df_all, "All questions")
    summarize(df_right, "Initially right only")


if __name__ == "__main__":
    main()
