"""
Robustness checks for the probes in probe_initial.py and probe_post_assert.py.

  1. Stronger baselines for Probe A (correctness, all questions) and Probe B (assert flip,
     initially right only): margin, output (sorted letter log-softmax + letter mass) and
     lens_all (full initial logit-lens table). Per layer, beyond-baseline Logit:
         label ~ z(baseline OOF logit) + z(probe OOF logit) [+ ARC-Challenge]
  2. Dataset control: dataset covariate in every beyond-baseline regression, and Probe B
     rerun on ARC-Challenge only (own seeded stratified folds).
  3. Fixed headline layer 18 (1-indexed), reported alongside the best layer.
  4. Difference probe: acts_assert − acts_initial, initially right, target assert flipped,
     with shuffled-label controls at layers 8, 15, 22.
  5. Where wrong answers go: share of wrong→flipped answers that land on the key vs 1/3.

The real initial letter logits are not stored, and the last logit-lens row does not
reproduce them exactly (Gemma 2 soft-caps final logits and the two paths round differently
in bf16), so they are recomputed by rerunning the initial prompts and cached in
results/probe_baselines/initial_letter_logits.npy; every row is checked against the
recorded initial answer, margin and confidence.

Usage:
    python probe_baselines.py --n 200 --out-dir results/probe_baselines/subset_n200   # quick test
    python probe_baselines.py                                                          # full run
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold

from runner import LETTERS
from probe_initial import BLUE, INK, MUTED, margin_cv, probe_cv, zscore

ORANGE, AQUA, VIOLET = "#eb6834", "#1baf7a", "#6250d6"
HEADLINE = 18            # 1-indexed, fixed in advance
FOCUS = range(8, 18)     # 1-indexed layers of interest
DIFF_SHUFFLE = [8, 15, 22]
BASELINES = ["margin", "output", "lens_all"]
BASE_COLORS = {"margin": MUTED, "output": ORANGE, "lens_all": AQUA}


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def L(layer):
    """0-indexed layer → 1-indexed label."""
    return layer + 1


def ranges(layers):
    """[8, 9, 10, 14] → 'L8–10, L14' (1-indexed input)."""
    if not len(layers):
        return "none"
    layers, out, start = sorted(layers), [], None
    for i, l in enumerate(layers):
        if start is None:
            start = l
        if i + 1 == len(layers) or layers[i + 1] != l + 1:
            out.append(f"L{start}" if start == l else f"L{start}–{l}")
            start = None
    return ", ".join(out)


# ---------- inputs ----------

def initial_letter_logits(recs, rows, cache):
    """Real initial letter logits (after soft-capping), recomputed from the model and cached."""
    full = np.load(cache) if os.path.exists(cache) else np.full((len(recs), 4), np.nan, np.float32)
    todo = [i for i in rows if np.isnan(full[i]).any()]
    if todo:
        import torch
        from datasets import load_dataset
        from runner import FIRST_PREFILL, Phase1Runner, format_question

        items = {}
        for name in sorted({recs[i]["dataset"] for i in todo}):
            for x in load_dataset("allenai/ai2_arc", name, split="test"):
                items[(name, x["id"])] = x
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        runner = Phase1Runner(torch.bfloat16, device)
        print(f"Recomputing initial letter logits for {len(todo)} questions on {device}...")
        t0 = time.time()
        for k, i in enumerate(todo):
            item = items[(recs[i]["dataset"], recs[i]["id"])]
            logits, _, _ = runner.run(runner.prompt([("user", format_question(item))], FIRST_PREFILL))
            full[i] = logits.float().numpy()
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
                np.save(cache, full)
        np.save(cache, full)
        del runner

    lg = full[rows].astype(np.float64)
    top2 = -np.sort(-lg, 1)[:, :2]
    p = np.exp(lg - logsumexp(lg, 1, keepdims=True))
    init_ok = np.mean(np.array(LETTERS)[lg.argmax(1)] == [recs[i]["initial"] for i in rows])
    d_margin = np.abs(top2[:, 0] - top2[:, 1] - [recs[i]["initial_margin"] for i in rows])
    d_conf = np.abs(p.max(1) - [recs[i]["initial_conf"] for i in rows])
    print(f"Initial letter logits check vs records: argmax = initial {init_ok:.1%}; margin exact "
          f"{np.mean(d_margin == 0):.1%} (max |Δ| {d_margin.max():.2g}); max |Δconf| {d_conf.max():.2g}")
    assert init_ok == 1.0, "recomputed logits disagree with recorded initial answers"
    return lg


def output_features(logits, mass):
    """Sorted letter log-softmax (top → bottom) + letter mass. Sorting makes the features
    independent of which letter was chosen, so a linear model can use them directly."""
    ls = logits - logsumexp(logits, 1, keepdims=True)
    return np.column_stack([-np.sort(-ls, 1), mass])


def lens_all_features(recs, rows, aligned=True):
    """Initial logit-lens table flattened (26 × 4). Aligned: per layer, the initial answer's
    column first, then the other three letters sorted high → low (same numbers, re-ordered so
    the answer's lead is linearly available). Not aligned: raw A–D columns."""
    out = []
    for i in rows:
        lens = np.array(recs[i]["initial_lens"])
        if aligned:
            init = LETTERS.index(recs[i]["initial"])
            others = -np.sort(-np.delete(lens, init, 1), 1)
            lens = np.column_stack([lens[:, init], others])
        out.append(lens.ravel())
    return np.array(out)


# ---------- analyses ----------

def beyond(y, base_logit, probe_logit, dataset=None):
    """Probe term of Logit(label ~ z(baseline) + z(probe) [+ dataset]): coef, CI lo, CI hi, p."""
    cols = [zscore(base_logit), zscore(probe_logit)]
    if dataset is not None:
        cols.append(dataset)
    try:
        fit = sm.Logit(y, sm.add_constant(np.column_stack(cols))).fit(disp=0)
    except Exception as e:  # e.g. perfect separation
        print(f"    beyond-baseline fit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan
    lo, hi = fit.conf_int()[2]
    return fit.params[2], lo, hi, fit.pvalues[2]


def baselines(feats, y, folds, seed):
    """OOF logits and fold AUROCs for each baseline, on the given folds."""
    out = {}
    oof, aucs = margin_cv(feats["margin"], y, folds)
    out["margin"] = (oof, aucs)
    for name in ("output", "lens_all", "lens_all_raw"):
        oof, aucs, _ = probe_cv(feats[name], y, folds, seed)
        out[name] = (oof, aucs)
    return out


def layer_probes(acts_fn, n_layers, y, folds, seed, label):
    """Per-layer OOF probe logits and fold AUROCs; acts_fn(layer) gives the (n, d) features."""
    res = []
    for layer in range(n_layers):
        t0 = time.time()
        oof, aucs, cs = probe_cv(acts_fn(layer), y, folds, seed)
        res.append((oof, aucs))
        print(f"  {label} L{L(layer):2d}: AUROC {aucs.mean():.3f} ± {aucs.std():.3f}  "
              f"C={np.median(cs):g}  ({time.time() - t0:.0f}s)")
    return res


def run_target(name, y, rows, acts_fn, feats, dataset, n_layers, seed):
    """Probe + baselines + beyond-baseline regressions for one target. Returns (baseline rows,
    per-layer long rows)."""
    print(f"\n=== {name}: n={len(y)}, positives={y.sum()} ({y.mean():.1%}) ===")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rows, y))
    base = baselines(feats, y, folds, seed)
    base_rows = []
    for b, (_, aucs) in base.items():
        base_rows.append({"target": name, "baseline": b, "n": len(y), "n_pos": int(y.sum()),
                          "auroc_mean": aucs.mean(), "auroc_std": aucs.std()})
        print(f"  baseline {b:12s} AUROC {aucs.mean():.3f} ± {aucs.std():.3f}")
    two_ds = dataset is not None and 0 < dataset.sum() < len(dataset)

    probes = layer_probes(acts_fn, n_layers, y, folds, seed, "probe")
    rows_out = []
    for layer, (p_oof, p_aucs) in enumerate(probes):
        for b in BASELINES:
            b_oof, b_aucs = base[b]
            coef, lo, hi, p = beyond(y, b_oof, p_oof)
            row = {"target": name, "layer": layer, "layer_1idx": L(layer), "baseline": b,
                   "n": len(y), "probe_auroc_mean": p_aucs.mean(), "probe_auroc_std": p_aucs.std(),
                   "baseline_auroc_mean": b_aucs.mean(), "baseline_auroc_std": b_aucs.std(),
                   "folds_probe_beats_baseline": int((p_aucs > b_aucs).sum()),
                   "coef": coef, "ci_lo": lo, "ci_hi": hi, "p": p}
            if two_ds:
                coef, lo, hi, p = beyond(y, b_oof, p_oof, dataset)
            else:  # single dataset: the covariate is constant, so the controlled fit equals the plain one
                coef, lo, hi, p = (np.nan,) * 4
            row.update({"coef_ds": coef, "ci_lo_ds": lo, "ci_hi_ds": hi, "p_ds": p})
            rows_out.append(row)
    return base_rows, rows_out


def shuffled_auroc(X, y, rows, seed, layer):
    y_sh = np.random.default_rng(seed + layer).permutation(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rows, y_sh))
    return probe_cv(X, y_sh, folds, seed)[1].mean()


def wrong_flip_destinations(recs, rows, logits):
    out = []
    pos = {i: k for k, i in enumerate(rows)}
    sel = [i for i in rows if not recs[i]["correct"] and recs[i]["assert"]["flipped"]]
    for ds in ["all", "ARC-Easy", "ARC-Challenge"]:
        g = [i for i in sel if ds == "all" or recs[i]["dataset"] == ds]
        if not g:
            continue
        k = sum(recs[i]["assert"]["to_correct"] for i in g)
        bt = binomtest(k, len(g), 1 / 3, alternative="two-sided")
        ci = bt.proportion_ci(0.95, method="exact")
        runner_up = [LETTERS[int(np.argsort(-logits[pos[i]])[1])] for i in g]
        to_ru = np.mean([recs[i]["assert"]["final"] == r for i, r in zip(g, runner_up)])
        ru_key = np.mean([recs[i]["answer_key"] == r for i, r in zip(g, runner_up)])
        out.append({"dataset": ds, "n_wrong_flipped": len(g), "n_to_correct": k,
                    "frac_to_correct": k / len(g), "ci_lo": ci.low, "ci_hi": ci.high,
                    "p_vs_one_third": bt.pvalue, "frac_to_initial_runner_up": to_ru,
                    "frac_runner_up_is_key": ru_key})
    return pd.DataFrame(out)


# ---------- plots ----------

def style(ax, ylabel):
    ax.set_xlabel("Layer", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_baselines(layers_df, base_df, target, title, path):
    d = layers_df[(layers_df.target == target) & (layers_df.baseline == "margin")]
    x = d["layer_1idx"].to_numpy()
    mean, std = d["probe_auroc_mean"].to_numpy(), d["probe_auroc_std"].to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), dpi=150, sharex=True)

    a1.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.18, linewidth=0)
    a1.plot(x, mean, color=BLUE, linewidth=2, marker="o", markersize=4, label="probe (± fold std)")
    for b in BASELINES:
        v = base_df[(base_df.target == target) & (base_df.baseline == b)]["auroc_mean"].iloc[0]
        a1.axhline(v, color=BASE_COLORS[b], linestyle="--", linewidth=1.5, label=f"{b} baseline ({v:.3f})")
    a1.axvline(HEADLINE, color=MUTED, linewidth=0.8, alpha=0.7)
    a1.set_ylim(min(0.45, (mean - std).min() - 0.02), 1.0)
    a1.text(HEADLINE + 0.2, a1.get_ylim()[0] + 0.01, f"L{HEADLINE} (fixed)", color=MUTED, fontsize=8, va="bottom")
    a1.set_title(title, color=INK, fontsize=10, loc="left")
    a1.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    style(a1, "AUROC (out-of-fold)")

    has_ds = d["coef_ds"].notna().any()
    sfx = "_ds" if has_ds else ""
    for b in BASELINES:
        e = layers_df[(layers_df.target == target) & (layers_df.baseline == b)]
        a2.fill_between(x, e[f"ci_lo{sfx}"], e[f"ci_hi{sfx}"], color=BASE_COLORS[b], alpha=0.15, linewidth=0)
        a2.plot(x, e[f"coef{sfx}"], color=BASE_COLORS[b], linewidth=2, marker="o", markersize=3,
                label=f"beyond {b}")
    a2.axhline(0, color=MUTED, linewidth=1)
    a2.axvline(HEADLINE, color=MUTED, linewidth=0.8, alpha=0.7)
    a2.set_title("Probe term in label ~ z(baseline) + z(probe)"
                 + (" + dataset" if has_ds else "") + "  (95% CI)", color=INK, fontsize=9, loc="left")
    a2.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    a2.set_xlim(0.5, x[-1] + 0.5)
    style(a2, "probe coefficient (per SD)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_diff(diff_df, path):
    x = diff_df["layer_1idx"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.axvspan(FOCUS[0] - 0.5, FOCUS[-1] + 0.5, color="#f0efec", zorder=0)
    for col, color, label in [("initial", VIOLET, "initial-turn probe B"),
                              ("post_assert", ORANGE, "post-assert probe"),
                              ("diff", BLUE, "difference probe (assert − initial)")]:
        m, s = diff_df[f"{col}_auroc_mean"].to_numpy(), diff_df[f"{col}_auroc_std"].to_numpy()
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=4, label=label)
    sh = diff_df.dropna(subset=["diff_shuffled_auroc"])
    ax.scatter(sh["layer_1idx"], sh["diff_shuffled_auroc"], marker="x", s=40, color=INK, zorder=5,
               label="difference probe, shuffled labels")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylim(min(0.4, min((diff_df[f"{c}_auroc_mean"] - diff_df[f"{c}_auroc_std"]).min()
                             for c in ("initial", "post_assert", "diff")) - 0.02), 1.0)
    ax.text(FOCUS[0] - 0.3, ax.get_ylim()[0] + 0.01, "layers 8–17", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlim(0.5, x[-1] + 0.5)
    ax.set_title(f"Flip under assert, initially right (n={diff_df.n.iloc[0]}): AUROC by layer",
                 color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    style(ax, "AUROC (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------- summary ----------

def summarize_target(layers_df, base_df, target, desc):
    d = layers_df[layers_df.target == target]
    b = base_df[base_df.target == target].set_index("baseline")["auroc_mean"]
    pm = d[d.baseline == "margin"].set_index("layer_1idx")
    best = int(pm["probe_auroc_mean"].idxmax())
    has_ds = d["coef_ds"].notna().any()
    print(f"\n{desc} — n={int(pm.n.iloc[0])}")
    print(f"  Baselines: " + ", ".join(f"{k} {b[k]:.3f}" for k in BASELINES)
          + f" (raw A–D lens_all, for reference: {b['lens_all_raw']:.3f})")
    for layer, tag in [(best, "best layer"), (HEADLINE, "fixed layer")]:
        r = pm.loc[layer]
        print(f"  {tag} L{layer}: probe AUROC {r.probe_auroc_mean:.3f} ± {r.probe_auroc_std:.3f}")
    for base in ("output", "lens_all", "margin"):
        e = d[d.baseline == base].set_index("layer_1idx")
        above = e.index[e.probe_auroc_mean > e.baseline_auroc_mean].tolist()
        sig = e.index[(e.p < 0.05) & (e.coef > 0)].tolist()
        line = (f"  vs {base}: higher AUROC at {ranges(above)}; adds significant info at {ranges(sig)}")
        if has_ds:
            sig_ds = e.index[(e.p_ds < 0.05) & (e.coef_ds > 0)].tolist()
            line += f"; with dataset covariate: {ranges(sig_ds)}"
        print(line)
        r = e.loc[HEADLINE]
        c, lo, hi, p = (r.coef_ds, r.ci_lo_ds, r.ci_hi_ds, r.p_ds) if has_ds else (r.coef, r.ci_lo, r.ci_hi, r.p)
        print(f"      at L{HEADLINE}: AUROC {r.probe_auroc_mean:.3f} vs {r.baseline_auroc_mean:.3f} "
              f"({int(r.folds_probe_beats_baseline)}/5 folds); probe term {c:+.2f} [{lo:+.2f}, {hi:+.2f}] "
              f"p={p:.2g}{' (dataset-controlled)' if has_ds else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--n", type=int, default=None, help="random subset of rows (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/probe_baselines")
    ap.add_argument("--logits-cache", default="results/probe_baselines/initial_letter_logits.npy")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.logits_cache), exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    recs = [json.loads(line) for line in open(args.records)]
    acts_init = np.load("data/acts_initial.npy", mmap_mode="r")
    acts_assert = np.load("data/acts_assert.npy", mmap_mode="r")
    assert len(recs) == acts_init.shape[0] == acts_assert.shape[0], "records and activations disagree"
    n_layers = acts_init.shape[1]
    rows = np.arange(len(recs))
    if args.n:  # same subset as probe_initial.py / probe_post_assert.py with the same --n/--seed
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.n, replace=False))
    print(f"{len(rows)} questions, {n_layers} layers (all layer numbers below are 1-indexed)")

    logits = initial_letter_logits(recs, rows, args.logits_cache)
    feats_all = {
        "margin": np.array([recs[i]["initial_margin"] for i in rows]),
        "output": output_features(logits, np.array([recs[i]["initial_letter_mass"] for i in rows])),
        "lens_all": lens_all_features(recs, rows),
        "lens_all_raw": lens_all_features(recs, rows, aligned=False),
    }
    correct = np.array([recs[i]["correct"] for i in rows], dtype=bool)
    flipped = np.array([recs[i]["assert"]["flipped"] for i in rows]).astype(int)
    challenge = np.array([recs[i]["dataset"] == "ARC-Challenge" for i in rows])

    def subset(mask):
        return rows[mask], {k: v[mask] for k, v in feats_all.items()}

    def init_layer(r):
        return lambda layer: np.asarray(acts_init[:, layer, :], dtype=np.float32)[r]

    base_rows, layer_rows = [], []
    # Probe A: correctness, all questions
    b, l = run_target("A", correct.astype(int), rows, init_layer(rows), feats_all,
                      challenge.astype(float), n_layers, args.seed)
    base_rows += b
    layer_rows += l
    # Probe B: assert flip, initially right
    r_right, f_right = subset(correct)
    b, l = run_target("B", flipped[correct], r_right, init_layer(r_right), f_right,
                      challenge[correct].astype(float), n_layers, args.seed)
    base_rows += b
    layer_rows += l
    # Probe B, ARC-Challenge only (own folds)
    m = correct & challenge
    r_ch, f_ch = subset(m)
    b, l = run_target("B_challenge", flipped[m], r_ch, init_layer(r_ch), f_ch, None, n_layers, args.seed)
    base_rows += b
    layer_rows += l

    base_df, layers_df = pd.DataFrame(base_rows), pd.DataFrame(layer_rows)
    base_df.to_csv(os.path.join(args.out_dir, "baselines.csv"), index=False)
    layers_df.to_csv(os.path.join(args.out_dir, "beyond_baseline.csv"), index=False)
    for t, title in [("A", f"Probe A: initial correctness (n={int(correct.size)})"),
                     ("B", f"Probe B: flip under assert, initially right (n={int(correct.sum())})"),
                     ("B_challenge", f"Probe B, ARC-Challenge only (n={int(m.sum())})")]:
        plot_baselines(layers_df, base_df, t, title, os.path.join(args.out_dir, f"baselines_{t}.png"))

    # 4. Difference probe (same rows and folds as Probe B)
    print(f"\n=== Difference probe (assert − initial), initially right: n={len(r_right)} ===")
    y = flipped[correct]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(r_right, y))

    def diff_layer(layer):
        return (np.asarray(acts_assert[:, layer, :], dtype=np.float32)[r_right]
                - np.asarray(acts_init[:, layer, :], dtype=np.float32)[r_right])

    def assert_layer(layer):
        return np.asarray(acts_assert[:, layer, :], dtype=np.float32)[r_right]

    diff = layer_probes(diff_layer, n_layers, y, folds, args.seed, "diff")
    post = layer_probes(assert_layer, n_layers, y, folds, args.seed, "post-assert")
    init_b = layers_df[(layers_df.target == "B") & (layers_df.baseline == "margin")].reset_index(drop=True)
    diff_rows = []
    for layer in range(n_layers):
        sh = np.nan
        if L(layer) in DIFF_SHUFFLE:
            sh = shuffled_auroc(diff_layer(layer), y, r_right, args.seed, layer)
            print(f"  diff L{L(layer)} shuffled-label AUROC {sh:.3f}")
        diff_rows.append({
            "layer": layer, "layer_1idx": L(layer), "n": len(y), "n_pos": int(y.sum()),
            "diff_auroc_mean": diff[layer][1].mean(), "diff_auroc_std": diff[layer][1].std(),
            "post_assert_auroc_mean": post[layer][1].mean(), "post_assert_auroc_std": post[layer][1].std(),
            "initial_auroc_mean": init_b.probe_auroc_mean[layer], "initial_auroc_std": init_b.probe_auroc_std[layer],
            "diff_shuffled_auroc": sh,
        })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(args.out_dir, "diff_probe.csv"), index=False)
    plot_diff(diff_df, os.path.join(args.out_dir, "diff_probe.png"))

    # 5. Where wrong answers go
    dest = wrong_flip_destinations(recs, rows, logits)
    dest.to_csv(os.path.join(args.out_dir, "wrong_flip_destinations.csv"), index=False)

    # Consistency with earlier runs (full data only: same folds should give the same AUROCs)
    if not args.n:
        old_i = pd.read_csv("results/probe_initial/probe_initial.csv")
        old_p = pd.read_csv("results/probe_post_assert/probe_post_assert.csv")
        da = np.abs(layers_df[(layers_df.target == "A") & (layers_df.baseline == "margin")].probe_auroc_mean.to_numpy()
                    - old_i[old_i.target == "A"].probe_auroc_mean.to_numpy()).max()
        db = np.abs(init_b.probe_auroc_mean.to_numpy() - old_i[old_i.target == "B"].probe_auroc_mean.to_numpy()).max()
        dp = np.abs(diff_df.post_assert_auroc_mean.to_numpy()
                    - old_p[old_p.version == "initially right only"].probe_auroc_mean.to_numpy()).max()
        print(f"\nReproduction check, max |ΔAUROC| vs earlier runs: Probe A {da:.2g}, Probe B {db:.2g}, "
              f"post-assert {dp:.2g}")
    print(f"\nWrote CSVs, plots and run_log.txt to {args.out_dir}/")

    # ---------- summary ----------
    print("\n=== SUMMARY (layers 1-indexed; L18 fixed in advance) ===")
    summarize_target(layers_df, base_df, "B", "Probe B (will an initially right answer flip under assert?)")
    summarize_target(layers_df, base_df, "B_challenge", "Probe B, ARC-Challenge only")
    summarize_target(layers_df, base_df, "A", "Probe A (is the initial answer correct?)")

    early = diff_df[diff_df.layer_1idx < HEADLINE]
    late = diff_df[diff_df.layer_1idx >= HEADLINE]
    print(f"\nDifference probe (assert − initial, initially right):")
    print(f"  L1–17: AUROC {early.diff_auroc_mean.min():.3f}–{early.diff_auroc_mean.max():.3f} "
          f"(max at L{int(early.loc[early.diff_auroc_mean.idxmax(), 'layer_1idx'])}); "
          f"L18: {diff_df.diff_auroc_mean[HEADLINE - 1]:.3f} ± {diff_df.diff_auroc_std[HEADLINE - 1]:.3f}; "
          f"L18–26: {late.diff_auroc_mean.min():.3f}–{late.diff_auroc_mean.max():.3f}")
    print("  Shuffled labels: " + ", ".join(f"L{int(r.layer_1idx)} {r.diff_shuffled_auroc:.3f}"
                                           for r in diff_df.dropna(subset=['diff_shuffled_auroc']).itertuples()))
    near = early.diff_auroc_mean.max() < 0.6
    print(f"  → {'Near chance' if near else 'NOT near chance'} through L17 "
          f"(threshold: every layer < 0.60); jump at L18: "
          f"{diff_df.diff_auroc_mean[HEADLINE - 1] - diff_df.diff_auroc_mean[HEADLINE - 2]:+.3f}.")

    print("\nWhere initially wrong answers go when they flip under assert:")
    for r in dest.itertuples():
        print(f"  {r.dataset}: {r.n_to_correct}/{r.n_wrong_flipped} = {r.frac_to_correct:.1%} to the key "
              f"[95% CI {r.ci_lo:.1%}, {r.ci_hi:.1%}], binomial p={r.p_vs_one_third:.2g} vs 1/3; "
              f"{r.frac_to_initial_runner_up:.1%} went to the initial runner-up, which was the key "
              f"{r.frac_runner_up_is_key:.1%} of the time.")


if __name__ == "__main__":
    main()
