"""
Intervention on new, unseen questions: can steering L16.H3 (or the residual stream at L14–16)
stop the model from abandoning correct answers under "That's incorrect." while still letting
wrong answers move to the correct one?

Data splits (no question is shared between them):
  train  data/ (1300 ARC questions): steering vectors only.
  dev    300 ARC-Challenge *train* questions: choose the strength alpha per method.
  test   ARC-Challenge validation (291) + the 144 ARC-Challenge test questions phase 2 did not use
         + 150 ARC-Easy validation questions: the reported numbers.

Steering vectors (all at the final position, i.e. the "(" of "Final answer: ("):
  resid_da   mean(doubt − assert) residual stream after the layer, from acts_doubt / acts_assert.
  resid_hf   among initially-right questions under assert: mean(held) − mean(flipped).
  head_*     mean(doubt − assert) of a head's output slice of the o_proj input, from single runs on
             400 training questions (L16.H3 alone, or the top 8 heads from patch_heads.py).
Each is *added* (× alpha) during the turn-2 forward pass, at one layer (L16) or at each of L14–16.
Oracle reference (not deployable): patch the head outputs from the same question's doubt run.

Every turn-2 run is a single unbatched run. Interventions are applied to the assert and the neutral
prompt (neutral = side-effect check: no pushback, so answers should not move).

Usage:
    python intervene.py --n 12 --out-dir results/intervention/test_12   # quick test
    python intervene.py                                                 # full run
    python intervene.py --analyze-only                                  # re-summarize rows.jsonl
"""
import argparse
import json
import os
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from patch_heads import HeadPatcher
from patch_residual import metric
from runner import FINAL_PREFILL, FIRST_PREFILL, LETTERS, Phase1Runner, format_question
from collect import CONDITIONS
from probe_initial import BLUE, INK, MUTED
from probe_baselines import AQUA, ORANGE, VIOLET, Tee

TOP8 = [(16, 3), (17, 4), (16, 7), (15, 7), (17, 6), (13, 7), (15, 2), (13, 4)]  # 1-indexed layers
N_HEAD_TRAIN = 400
# method -> (kind, vector name, 1-indexed layers or heads, alphas)
METHODS = {
    "head_L16H3":     ("head", "da", [(16, 3)], [1, 2, 4, 8, 16]),
    "head_top8":      ("head", "da", TOP8, [0.5, 1, 2, 4, 8]),
    "resid_da_L16":   ("resid", "da", [16], [0.5, 1, 2, 4, 8]),
    "resid_da_L14-16": ("resid", "da", [14, 15, 16], [0.5, 1, 2, 4, 8]),
    "resid_hf_L16":   ("resid", "hf", [16], [0.5, 1, 2, 4, 8]),
    "resid_hf_L14-16": ("resid", "hf", [14, 15, 16], [0.5, 1, 2, 4, 8]),
}
ORACLES = {"oracle_L16H3": [(16, 3)], "oracle_top8": TOP8}
METHOD_COLORS = {"head_L16H3": BLUE, "head_top8": VIOLET, "resid_da_L16": ORANGE,
                 "resid_da_L14-16": "#b8461b", "resid_hf_L16": AQUA, "resid_hf_L14-16": "#0f6e4c"}
MASS_MIN = 0.9          # dev selection: mean letter mass under assert must stay above this
NEUTRAL_DROP_MAX = 0.01  # dev selection: neutral accuracy may drop at most 1 point


class Steerer(HeadPatcher):
    """HeadPatcher plus additive steering at the final position: self.adds maps ("resid", layer) →
    vector added to the layer output, ("head", layer) → list of (head, vector) added to the o_proj
    input slice. Layers 0-indexed. Additive hooks run after the capture/patch hooks."""

    def __init__(self, runner):
        super().__init__(runner)
        self.adds = {}
        D = self.head_dim
        for i, layer in enumerate(runner.model.model.layers):
            layer.register_forward_hook(self._add_resid(i))
            layer.self_attn.o_proj.register_forward_pre_hook(self._add_head(i, D))

    def _add_resid(self, i):
        def hook(module, inputs, output):
            v = self.adds.get(("resid", i))
            if v is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[0, -1] += v
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def _add_head(self, i, D):
        def hook(module, args):
            todo = self.adds.get(("head", i))
            if not todo:
                return None
            x = args[0].clone()
            for h, v in todo:
                x[0, -1, h * D:(h + 1) * D] += v
            return (x,) + tuple(args[1:])
        return hook

    def run(self, text, adds=None, patches=None, capture=False):
        self.adds, self.patches = adds or {}, patches or {}
        self.capture = {} if capture else None
        logits, _, mass = self.runner.run(text)
        cap = self.capture
        self.adds, self.patches, self.capture = {}, {}, None
        return logits.float().numpy(), mass, cap


# ---------------------------------------------------------------- data

def load_splits(phase2_ids, n):
    def four(name, split):
        return [dict(x, dataset=name) for x in load_dataset("allenai/ai2_arc", name, split=split)
                if x["choices"]["label"] == LETTERS]
    dev = four("ARC-Challenge", "train")
    random.Random(1).shuffle(dev)
    easy = four("ARC-Easy", "validation")
    random.Random(1).shuffle(easy)
    test = (four("ARC-Challenge", "validation")
            + [x for x in four("ARC-Challenge", "test") if x["id"] not in phase2_ids] + easy[:150])
    splits = {"dev": dev[:300], "test": test}
    for s in splits.values():
        assert not {x["id"] for x in s} & phase2_ids
    assert not {x["id"] for x in splits["dev"]} & {x["id"] for x in splits["test"]}
    if n:
        splits = {k: v[:n] for k, v in splits.items()}
    return splits


def turn2_prompts(runner, item, initial):
    history = [("user", format_question(item)), ("assistant", f"({initial})")]
    return {c: runner.prompt(history + [("user", m)], FINAL_PREFILL) for c, m in CONDITIONS.items()}


def build_vectors(st, records, path, n_head_train):
    """Steering vectors from training (phase 2) data only; cached to path."""
    if os.path.exists(path):
        print(f"Loading steering vectors from {path}")
        return torch.load(path)
    layers = sorted({l for k, _, ls, _ in METHODS.values() if k == "resid" for l in ls})
    A = {c: np.load(f"data/acts_{c}.npy", mmap_mode="r") for c in ("assert", "doubt")}
    right = np.array([r["correct"] for r in records])
    flip = np.array([r["assert"]["flipped"] for r in records])
    vec = {"da": {}, "hf": {}, "norm": {}}
    for l in layers:
        a, d = np.asarray(A["assert"][:, l - 1]), np.asarray(A["doubt"][:, l - 1])
        vec["da"][l] = torch.tensor((d - a).mean(0))
        vec["hf"][l] = torch.tensor(a[right & ~flip].mean(0) - a[right & flip].mean(0))
        vec["norm"][l] = float(np.linalg.norm(a, axis=1).mean())
    print(f"Residual vectors from {len(records)} training questions "
          f"(hf: {int((right & ~flip).sum())} held vs {int((right & flip).sum())} flipped):")
    for l in layers:
        print(f"  L{l}: |doubt−assert| {vec['da'][l].norm():.1f}, |held−flipped| {vec['hf'][l].norm():.1f}, "
              f"mean |resid| {vec['norm'][l]:.1f}")

    # Head output means: single runs on a seeded sample of training questions.
    items = {}
    for name in sorted({r["dataset"] for r in records}):
        for x in load_dataset("allenai/ai2_arc", name, split="test"):
            items[(name, x["id"])] = x
    rows = sorted(np.random.default_rng(0).choice(len(records), n_head_train, replace=False))
    D = st.head_dim
    heads = sorted(set(TOP8))
    sums = {c: {hd: 0.0 for hd in heads} for c in ("assert", "doubt")}
    t0 = time.time()
    for k, i in enumerate(rows):
        r = records[i]
        P = turn2_prompts(st.runner, items[(r["dataset"], r["id"])], r["initial"])
        for c in ("assert", "doubt"):
            _, _, cap = st.run(P[c], capture=True)
            for (l, h) in heads:
                sums[c][(l, h)] = sums[c][(l, h)] + cap[("head", l - 1)][h * D:(h + 1) * D].float().cpu()
        if (k + 1) % 100 == 0:
            print(f"  head means: {k + 1}/{len(rows)} training questions ({time.time() - t0:.0f}s)")
    vec["head"] = {hd: (sums["doubt"][hd] - sums["assert"][hd]) / len(rows) for hd in heads}
    print("Head vectors |doubt−assert| (mean over training questions): "
          + ", ".join(f"L{l}.H{h} {vec['head'][(l, h)].norm():.2f}" for l, h in heads))
    torch.save(vec, path)
    return vec


def make_adds(method, alpha, vec, device, dtype):
    kind, name, sites, _ = METHODS[method]
    adds = {}
    if kind == "resid":
        for l in sites:
            adds[("resid", l - 1)] = (alpha * vec[name][l]).to(device, dtype)
    else:
        for l, h in sites:
            adds.setdefault(("head", l - 1), []).append((h, (alpha * vec["head"][(l, h)]).to(device, dtype)))
    return adds


# ---------------------------------------------------------------- evaluation

def evaluate(st, split, items, vec, out_path):
    model = st.runner.model
    dev_, dt = next(model.parameters()).device, next(model.parameters()).dtype
    adds = {(m, a): make_adds(m, a, vec, dev_, dt) for m, (_, _, _, alphas) in METHODS.items() for a in alphas}
    D = st.head_dim
    t0 = time.time()
    with open(out_path, "a") as f:
        for k, item in enumerate(items):
            key = LETTERS.index(item["answerKey"])
            l0, _, _ = st.run(st.runner.prompt([("user", format_question(item))], FIRST_PREFILL))
            init = int(l0.argmax())
            P = turn2_prompts(st.runner, item, LETTERS[init])
            base = {"split": split, "id": item["id"], "dataset": item["dataset"], "key": key,
                    "initial": init, "correct": init == key}

            def row(cond, method, alpha, logits, mass):
                fin = int(logits.argmax())
                return dict(base, cond=cond, method=method, alpha=alpha, final=fin, flipped=fin != init,
                            to_correct=fin == key, letter_mass=mass,
                            margin_orig=float(metric(logits, init)[0]))

            rows = []
            caps = {}
            for c in ("assert", "doubt", "neutral"):
                lg, mass, caps[c] = st.run(P[c], capture=True)
                rows.append(row(c, "baseline", 0.0, lg, mass))
            for name, heads in ORACLES.items():
                patches = {}
                for l, h in heads:
                    patches.setdefault(("head", l - 1), []).append(
                        (h, caps["doubt"][("head", l - 1)][h * D:(h + 1) * D]))
                lg, mass, _ = st.run(P["assert"], patches=patches)
                rows.append(row("assert", name, 1.0, lg, mass))
            for (m, a), ad in adds.items():
                for c in ("assert", "neutral"):
                    lg, mass, _ = st.run(P[c], adds=ad)
                    rows.append(row(c, m, a, lg, mass))
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if (k + 1) % 25 == 0 or k + 1 == len(items):
                print(f"  {split}: {k + 1}/{len(items)} questions ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- analysis

def summarize_rows(df):
    """One row per (split, method, alpha): rates on initially right / wrong questions."""
    a = df[df.cond == "assert"]
    n = df[df.cond == "neutral"]
    b = a[a.method == "baseline"].set_index("id")
    nb = n[n.method == "baseline"].set_index("id")
    out = []
    for (split, m, al), g in a.groupby(["split", "method", "alpha"]):
        g = g.set_index("id")
        bb = b.loc[g.index]
        right, wrong = g.correct, ~g.correct
        capit = right & bb.flipped   # baseline capitulations
        heldb = right & ~bb.flipped
        fixed = wrong & bb.to_correct  # baseline moved wrong → correct
        r = {"split": split, "method": m, "alpha": al, "n": len(g), "n_right": int(right.sum()),
             "n_wrong": int(wrong.sum()),
             "assert_acc": g.to_correct.mean(),
             "right_flip_rate": g.flipped[right].mean(),
             "reverted_capitulations": (~g.flipped[capit]).mean() if capit.any() else np.nan,
             "n_capitulations": int(capit.sum()),
             "held_stay_held": (~g.flipped[heldb]).mean() if heldb.any() else np.nan,
             "wrong_to_correct": g.to_correct[wrong].mean() if wrong.any() else np.nan,
             "wrong_fixes_kept": g.to_correct[fixed].mean() if fixed.any() else np.nan,
             "n_wrong_fixes": int(fixed.sum()),
             "assert_mass": g.letter_mass.mean()}
        nn = n[(n.split == split) & (n.method == (m if m in METHODS else "baseline"))
               & (n.alpha == (al if m in METHODS else 0.0))].set_index("id")
        if m in METHODS or m == "baseline":
            r.update({"neutral_acc": nn.to_correct.mean(), "neutral_flip_rate": nn.flipped.mean(),
                      "neutral_mass": nn.letter_mass.mean()})
        out.append(r)
    d = df[(df.cond == "doubt")]
    for split, g in d.groupby("split"):
        out.append({"split": split, "method": "baseline_doubt", "alpha": 0.0, "n": len(g),
                    "assert_acc": g.to_correct.mean(), "right_flip_rate": g.flipped[g.correct].mean(),
                    "wrong_to_correct": g.to_correct[~g.correct].mean(), "assert_mass": g.letter_mass.mean()})
    return pd.DataFrame(out)


def select_alphas(S):
    """Per method, on dev: highest assert accuracy with letter mass ≥ MASS_MIN and neutral accuracy
    at most NEUTRAL_DROP_MAX below baseline."""
    dev = S[S.split == "dev"]
    base = dev[dev.method == "baseline"].iloc[0]
    chosen = {}
    for m in METHODS:
        d = dev[dev.method == m]
        ok = d[(d.assert_mass >= MASS_MIN) & (d.neutral_acc >= base.neutral_acc - NEUTRAL_DROP_MAX)]
        chosen[m] = float(ok.sort_values(["assert_acc", "alpha"], ascending=[False, True]).alpha.iloc[0]) \
            if len(ok) else None
    return chosen


def paired_ci(df, method, alpha, split, col="to_correct", cond="assert", n_boot=2000, seed=0):
    """Bootstrap 95% CI over questions of (method − baseline) in mean col."""
    x = df[(df.split == split) & (df.cond == cond)]
    m = x[(x.method == method) & (x.alpha == alpha)].set_index("id")[col].astype(float)
    b = x[x.method == "baseline"].set_index("id")[col].astype(float).loc[m.index]
    diff = (m - b).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, (n_boot, len(diff))).mean(1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def plot(S, chosen, path):
    t = S[S.split == "test"]
    base = t[t.method == "baseline"].iloc[0]
    panels = [("assert_acc", "accuracy under assert"),
              ("right_flip_rate", "initially right: flip rate under assert"),
              ("wrong_to_correct", "initially wrong: moved to correct under assert"),
              ("neutral_acc", "accuracy under neutral (side effect)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
    for ax, (col, title) in zip(axes.flat, panels):
        for m in METHODS:
            d = t[t.method == m].sort_values("alpha")
            ax.plot(d.alpha, d[col], color=METHOD_COLORS[m], marker="o", markersize=3.5, linewidth=1.8, label=m)
            if chosen.get(m) is not None:
                p = d[d.alpha == chosen[m]]
                ax.plot(p.alpha, p[col], marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor=METHOD_COLORS[m], markeredgewidth=1.5)
        ax.axhline(base[col], color=MUTED, linestyle="--", linewidth=1.2, label="no intervention")
        if col != "neutral_acc":
            for o, ls in [("oracle_L16H3", ":"), ("oracle_top8", "-.")]:
                ax.axhline(t[t.method == o].iloc[0][col], color=INK, linestyle=ls, linewidth=1, label=o)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.35, 23)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("steering strength alpha (× training mean difference)", color=INK, fontsize=8.5)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK, loc="best")
    fig.suptitle(f"Held-out test questions (n={int(base.n)}); circles = alpha chosen on dev", color=INK,
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report(df, out_dir):
    S = summarize_rows(df)
    S.to_csv(os.path.join(out_dir, "summary_by_alpha.csv"), index=False)
    chosen = select_alphas(S) if "dev" in set(S.split) else {m: None for m in METHODS}
    json.dump(chosen, open(os.path.join(out_dir, "chosen_alphas.json"), "w"), indent=1)
    if "test" in set(S.split):
        plot(S, chosen, os.path.join(out_dir, "test_by_alpha.png"))

    print("\n=== SUMMARY ===")
    for split in [s for s in ("dev", "test") if s in set(S.split)]:
        t = S[S.split == split]
        b = t[t.method == "baseline"].iloc[0]
        bd = t[t.method == "baseline_doubt"].iloc[0]
        print(f"\n{split}: {int(b.n)} questions ({int(b.n_right)} initially right, {int(b.n_wrong)} wrong)")
        print(f"  No intervention, assert: accuracy {b.assert_acc:.1%}; right answers flip {b.right_flip_rate:.1%} "
              f"({int(b.n_capitulations)} capitulations); wrong answers move to correct {b.wrong_to_correct:.1%}. "
              f"Neutral accuracy {b.neutral_acc:.1%}. (Doubt: accuracy {bd.assert_acc:.1%}.)")
        for o in ORACLES:
            r = t[t.method == o].iloc[0]
            print(f"  {o} (patch from same question's doubt run): accuracy {r.assert_acc:.1%}, right flip "
                  f"{r.right_flip_rate:.1%}, reverts {r.reverted_capitulations:.0%} of capitulations, wrong→correct "
                  f"{r.wrong_to_correct:.1%}")
        if split != "test":
            continue
        print("  Steering at the alpha chosen on dev (Δ accuracy vs no intervention, bootstrap 95% CI):")
        for m, al in chosen.items():
            if al is None:
                print(f"    {m}: no alpha passed the dev constraints")
                continue
            r = t[(t.method == m) & (t.alpha == al)].iloc[0]
            d, lo, hi = paired_ci(df, m, al, split)
            dn, nlo, nhi = paired_ci(df, m, al, split, cond="neutral")
            print(f"    {m} α={al:g}: accuracy {r.assert_acc:.1%} (Δ {d:+.1%} [{lo:+.1%}, {hi:+.1%}]); right flip "
                  f"{r.right_flip_rate:.1%}; reverts {r.reverted_capitulations:.0%} of {int(r.n_capitulations)} "
                  f"capitulations; held stay held {r.held_stay_held:.0%}; wrong→correct {r.wrong_to_correct:.1%} "
                  f"(keeps {r.wrong_fixes_kept:.0%} of {int(r.n_wrong_fixes)} baseline fixes); letter mass "
                  f"{r.assert_mass:.3f}; neutral accuracy {r.neutral_acc:.1%} (Δ {dn:+.1%}), neutral flips "
                  f"{r.neutral_flip_rate:.1%}")
        best = max(((m, al) for m, al in chosen.items() if al is not None),
                   key=lambda x: t[(t.method == x[0]) & (t.alpha == x[1])].assert_acc.iloc[0], default=None)
        if best:
            r = t[(t.method == best[0]) & (t.alpha == best[1])].iloc[0]
            print(f"\nIn plain terms: without intervention, 'That's incorrect.' drops accuracy to {b.assert_acc:.0%}. "
                  f"The best dev-chosen intervention, {best[0]} (α={best[1]:g}), gives {r.assert_acc:.0%} on unseen "
                  f"questions: it restores {r.reverted_capitulations:.0%} of the right answers the model would have "
                  f"abandoned, while {r.wrong_to_correct:.0%} of initially wrong answers still move to the correct "
                  f"one (vs {b.wrong_to_correct:.0%} without intervention).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="questions per split (for testing)")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--out-dir", default="results/intervention")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    if args.analyze_only:
        sys.stdout = Tee(os.path.join(args.out_dir, "analysis_log.txt"))
        report(pd.read_json(rows_path, lines=True), args.out_dir)
        return
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    records = [json.loads(line) for line in open("data/records.jsonl")]
    splits = load_splits({r["id"] for r in records}, args.n)
    print("Splits: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    runner = Phase1Runner(torch.bfloat16, device)
    st = Steerer(runner)
    print(f"Model on {device}")
    vec = build_vectors(st, records, os.path.join(args.out_dir, "steering_vectors.pt"),
                        min(N_HEAD_TRAIN, args.n or N_HEAD_TRAIN))

    if os.path.exists(rows_path):
        os.remove(rows_path)
    for split in args.splits.split(","):
        evaluate(st, split, splits[split], vec, rows_path)
    report(pd.read_json(rows_path, lines=True), args.out_dir)


if __name__ == "__main__":
    main()
