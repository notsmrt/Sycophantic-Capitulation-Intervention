"""
Linear probes on the initial-answer residual stream (data/acts_initial.npy), per layer.

Targets:
  A  correctness (record["correct"]), all questions.
  B  flip under assert (record["assert"]["flipped"]), initially right questions only.

For each layer: StandardScaler + L2 logistic regression, C picked by inner stratified CV,
evaluated with seeded 5-fold stratified outer CV (one row = one question, so folds never
share questions). Compared against a margin-only logistic baseline on the same folds, and
tested for information beyond confidence with
    label ~ z(initial_margin) + z(out-of-fold probe logit)    (statsmodels Logit).
Shuffled-label controls at a few layers should sit near 0.5 AUROC.

Usage:
    python probe_initial.py --n 200 --out-dir results/probe_initial/subset_n200   # quick test
    python probe_initial.py                                                       # full run
"""
import argparse
import json
import os
import time
import warnings

# numpy's Apple Accelerate BLAS raises spurious divide-by-zero/overflow warnings on matmul
# with finite inputs (results verified correct). Silence them here and in joblib workers.
_BLAS_NOISE = [f"{kind} encountered in matmul" for kind in ("divide by zero", "overflow", "invalid value")]
os.environ["PYTHONWARNINGS"] = ",".join(f"ignore:{msg}:RuntimeWarning" for msg in _BLAS_NOISE)
for msg in _BLAS_NOISE:
    warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CS = np.logspace(-4, 0, 5)  # inner-CV grid for the L2 strength
SHUFFLE_LAYERS = [6, 14, 22]
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"


def zscore(x):
    return (x - x.mean()) / x.std()


def probe_cv(X, y, folds, seed):
    """Out-of-fold probe logits, per-fold AUROCs and chosen C values."""
    oof = np.zeros(len(y))
    aucs, cs = [], []
    for tr, te in folds:
        grid = GridSearchCV(
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(penalty="l2", max_iter=5000))]),
            {"clf__C": CS},
            cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1),
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid.fit(X[tr], y[tr])
        oof[te] = grid.decision_function(X[te])
        aucs.append(roc_auc_score(y[te], oof[te]))
        cs.append(grid.best_params_["clf__C"])
    return oof, np.array(aucs), cs


def margin_cv(margin, y, folds):
    """Margin-only logistic baseline on the same folds."""
    m = margin.reshape(-1, 1)
    oof = np.zeros(len(y))
    aucs = []
    for tr, te in folds:
        clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())]).fit(m[tr], y[tr])
        oof[te] = clf.decision_function(m[te])
        aucs.append(roc_auc_score(y[te], oof[te]))
    return oof, np.array(aucs)


def beyond_confidence(y, margin, probe_logit):
    """Logit of label ~ z(margin) + z(probe logit); returns the probe term's coef, CI, p."""
    X = sm.add_constant(np.column_stack([zscore(margin), zscore(probe_logit)]))
    try:
        fit = sm.Logit(y, X).fit(disp=0)
    except Exception as e:  # e.g. perfect separation
        print(f"    beyond-confidence fit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan
    lo, hi = fit.conf_int()[2]
    return fit.params[2], lo, hi, fit.pvalues[2]


def run_target(name, label, acts, rows, margin, n_layers, seed):
    y = label.astype(int)
    print(f"\n=== Probe {name}: n={len(y)}, positives={y.sum()} ({y.mean():.1%}) ===")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rows, y))
    _, m_aucs = margin_cv(margin, y, folds)
    print(f"  Margin-only baseline AUROC: {m_aucs.mean():.3f} ± {m_aucs.std():.3f}")

    out = []
    for layer in range(n_layers):
        t0 = time.time()
        X = np.asarray(acts[:, layer, :], dtype=np.float32)[rows]
        oof, aucs, cs = probe_cv(X, y, folds, seed)
        coef, lo, hi, p = beyond_confidence(y, margin, oof)
        row = {
            "target": name, "layer": layer, "n": len(y), "n_pos": int(y.sum()),
            "probe_auroc_mean": aucs.mean(), "probe_auroc_std": aucs.std(),
            "probe_auroc_pooled": roc_auc_score(y, oof),
            "margin_auroc_mean": m_aucs.mean(), "margin_auroc_std": m_aucs.std(),
            "folds_probe_beats_margin": int((aucs > m_aucs).sum()),
            "probe_coef": coef, "probe_ci_lo": lo, "probe_ci_hi": hi, "probe_p": p,
            "median_C": float(np.median(cs)), "shuffled_auroc_mean": np.nan,
        }
        if layer in SHUFFLE_LAYERS:
            y_sh = np.random.default_rng(seed + layer).permutation(y)
            sh_folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rows, y_sh))
            _, sh_aucs, _ = probe_cv(X, y_sh, sh_folds, seed)
            row["shuffled_auroc_mean"] = sh_aucs.mean()
        out.append(row)
        sh = f"  shuffled {row['shuffled_auroc_mean']:.3f}" if layer in SHUFFLE_LAYERS else ""
        print(f"  layer {layer:2d}: AUROC {aucs.mean():.3f} ± {aucs.std():.3f}  "
              f"probe term {coef:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.2g}  C={np.median(cs):g}"
              f"{sh}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(out)


def plot(df, title, path):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    layers = df["layer"].to_numpy() + 1  # 1-indexed for display
    mean, std = df["probe_auroc_mean"].to_numpy(), df["probe_auroc_std"].to_numpy()
    ax.fill_between(layers, mean - std, mean + std, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(layers, mean, color=BLUE, linewidth=2, marker="o", markersize=4,
            label="probe (mean ± std over folds)")
    m = df["margin_auroc_mean"].iloc[0]
    ax.axhline(m, color=MUTED, linestyle="--", linewidth=1.5, label=f"margin only ({m:.3f})")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(layers[0], 0.5, "chance", color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlabel("Layer", color=INK)
    ax.set_ylabel("AUROC (out-of-fold)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlim(0.5, layers[-1] + 0.5)
    ax.set_ylim(min(0.45, (mean - std).min() - 0.02), 1.0)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(df, name, what):
    best = df.loc[df["probe_auroc_mean"].idxmax()]
    m = best["margin_auroc_mean"]
    diff = best["probe_auroc_mean"] - m
    sig = df[(df["probe_p"] < 0.05) & (df["probe_coef"] > 0)]
    first = f"layer {int(sig['layer'].iloc[0]) + 1}" if len(sig) else "none"
    sh = df["shuffled_auroc_mean"].dropna()
    print(f"\nProbe {name} ({what}):")
    print(f"  Best layer {int(best['layer']) + 1}: AUROC {best['probe_auroc_mean']:.3f} ± "
          f"{best['probe_auroc_std']:.3f} vs margin {m:.3f} "
          f"({'beats' if diff > 0 else 'does not beat'} margin by {diff:+.3f}; "
          f"higher in {int(best['folds_probe_beats_margin'])}/5 folds).")
    print(f"  Layers with probe AUROC above margin: {int((df['probe_auroc_mean'] > m).sum())}/{len(df)}.")
    print(f"  First layer where the probe adds significant information beyond margin "
          f"(p<0.05, positive): {first}; significant at {len(sig)}/{len(df)} layers.")
    print(f"  Shuffled-label control AUROC: "
          + ", ".join(f"layer {int(l) + 1} {v:.3f}" for l, v in zip(df.loc[sh.index, 'layer'], sh)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--acts", default="data/acts_initial.npy")
    ap.add_argument("--n", type=int, default=None, help="random subset of rows (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/probe_initial")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.records)]
    acts = np.load(args.acts, mmap_mode="r")
    assert len(recs) == acts.shape[0], "records and activations disagree"
    n_layers = acts.shape[1]
    rows = np.arange(len(recs))
    if args.n:
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.n, replace=False))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(rows)} questions, {n_layers} layers, d_model {acts.shape[2]}")

    correct = np.array([recs[i]["correct"] for i in rows])
    margin = np.array([recs[i]["initial_margin"] for i in rows])
    flipped = np.array([recs[i]["assert"]["flipped"] for i in rows])

    df_a = run_target("A", correct, acts, rows, margin, n_layers, args.seed)
    right = correct.astype(bool)
    df_b = run_target("B", flipped[right], acts, rows[right], margin[right], n_layers, args.seed)

    df = pd.concat([df_a, df_b])
    df["layer_1idx"] = df["layer"] + 1
    df.to_csv(os.path.join(args.out_dir, "probe_initial.csv"), index=False)
    plot(df_a, f"Probe A: initial correctness from initial-answer residual (n={df_a.n.iloc[0]})",
         os.path.join(args.out_dir, "probe_A_correct.png"))
    plot(df_b, f"Probe B: flip under assert, initially right only (n={df_b.n.iloc[0]})",
         os.path.join(args.out_dir, "probe_B_flip_assert.png"))
    print(f"\nWrote CSV and plots to {args.out_dir}/ (layers shown 1-indexed)")

    print("\n=== SUMMARY ===")
    summarize(df_a, "A", "is the initial answer correct?")
    summarize(df_b, "B", "will an initially right answer flip under assert?")
    print("\nNote: the best layer is picked on the same CV folds, so its AUROC is slightly optimistic;"
          "\nthe beyond-confidence test uses pooled out-of-fold logits from five per-fold probes.")


if __name__ == "__main__":
    main()
