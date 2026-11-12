"""
Robustness of the patching findings to the turn-2 prefill.

Everything in patch_residual.py / patch_heads.py used the turn-2 prefill "Final answer: (". Here the same
tests run with other prefills (default: bare "(" and "The answer is ("). Turn 1 is unchanged, so the
recorded initial answers stay valid.

Held-out questions only: every question in patch_heads.py's screening sample (where the top heads
were chosen) is excluded.

Step 1  behaviour: assert / doubt / neutral with the new prefill on every held-out initially-right
        question; flip rates vs the recorded "Final answer: (" runs. New minimal pairs = flipped under
        assert and held under doubt / neutral, with the new prefill.
Step 2  patching on a seeded sample of the new pairs (--max-pairs per type), all at the final position:
        a. residual stream, layers 1–26, forward (source → assert) and reverse (assert → source)
        b. re-screen of all 40 heads in L13–17, forward
        c. the original top-8 heads per type (results/patch_heads/stage2_cumulative.csv), cumulative
           k = 1, 2, 3, 5, 8, forward and reverse
Step 3  comparison with the original results (plots + summary).

All runs are single unbatched runs (as in patch_residual.py). Control: --prefill "Final answer: (" must
reproduce the original per-pair numbers; the script checks this itself.

Usage:
    python patch_prefill_robustness.py --prefill "Final answer: (" --max-pairs 5 --out-dir results/patch_prefill_robustness/test_control
    python patch_prefill_robustness.py --max-pairs 10 --out-dir results/patch_prefill_robustness/test_10
    python patch_prefill_robustness.py                                                  # full run
"""
import argparse
import json
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import spearmanr

from patch_heads import KS, N_SCREEN, SCREEN_LAYERS, HeadPatcher, hname, plot_heatmaps
from patch_residual import PAIR_TYPES, agg, metric
from runner import FINAL_PREFILL, LETTERS, Phase1Runner, format_question
from collect import CONDITIONS
from probe_initial import BLUE, INK, MUTED
from probe_baselines import ORANGE, Tee, style

DEFAULT_PREFILLS = ["(", "The answer is ("]
PREFILL_COLORS = [BLUE, ORANGE, "#1baf7a", "#6250d6"]
COLS = ["recovery", "reverted", "reverse_effect", "reverse_flipped"]


def slug(prefill):
    s = re.sub(r"[^a-z0-9]+", "_", prefill.lower()).strip("_")
    return s or "paren"


def parse_head(name):
    l, h = re.fullmatch(r"L(\d+)\.H(\d+)", name).groups()
    return int(l) - 1, int(h)


def build_prompts_pf(runner, item, initial, prefill):
    history = [("user", format_question(item)), ("assistant", f"({initial})")]
    return {c: runner.prompt(history + [("user", msg)], prefill) for c, msg in CONDITIONS.items()}


def screened_rows(pdf, seed=0):
    """Exactly patch_heads.main's screening sample (rows of all screened questions)."""
    rows = set()
    rng = np.random.default_rng(seed)
    for c in PAIR_TYPES:
        p = list(pdf[pdf.pair == c].row)
        rows |= {p[i] for i in rng.choice(len(p), min(N_SCREEN, len(p)), replace=False)}
    return rows


# ---------------------------------------------------------------- steps

def behaviour(runner, recs, items, prefill):
    rows = []
    t0 = time.time()
    for k, rec in enumerate(recs):
        prompts = build_prompts_pf(runner, items[(rec["dataset"], rec["id"])], rec["initial"], prefill)
        row = {"row": rec["row"], "id": rec["id"], "dataset": rec["dataset"], "initial": rec["initial"]}
        for c, text in prompts.items():
            logits, _, mass = runner.run(text)
            final = LETTERS[int(logits.argmax())]
            row.update({f"{c}_final": final, f"{c}_flipped": final != rec["initial"], f"{c}_mass": mass,
                        f"{c}_orig_flipped": rec[c]["flipped"]})
        rows.append(row)
        if (k + 1) % 200 == 0 or k + 1 == len(recs):
            print(f"  behaviour: {k + 1}/{len(recs)} questions ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def patch_pair(patcher, rec, item, cond, prefill, top, n_layers, n_heads):
    prompts = build_prompts_pf(patcher.runner, item, rec["initial"], prefill)
    orig = LETTERS.index(rec["initial"])
    src_logits, src_cap = patcher.clean(prompts[cond])
    asr_logits, asr_cap = patcher.clean(prompts["assert"])
    assert src_logits.argmax() == orig and asr_logits.argmax() != orig, f"row {rec['row']}: clean run changed"
    m_src, m_asr = float(metric(src_logits, orig)[0]), float(metric(asr_logits, orig)[0])
    gap = m_src - m_asr
    base = {"row": rec["row"], "pair": cond}
    pair_row = {**base, "original": rec["initial"], "assert_final": LETTERS[int(asr_logits.argmax())],
                "source_clean": m_src, "assert_clean": m_asr}

    sites = [("resid", i) for i in range(n_layers)]
    fwd, rev = patcher.patched(prompts["assert"], sites, src_cap), patcher.patched(prompts[cond], sites, asr_cap)
    m_fwd, m_rev = metric(fwd, orig), metric(rev, orig)
    resid = [{**base, "layer": i, "layer_1idx": i + 1, "recovery": (m_fwd[i] - m_asr) / gap,
              "reverted": float(fwd[i].argmax() == orig), "reverse_effect": (m_src - m_rev[i]) / gap,
              "reverse_flipped": float(rev[i].argmax() != orig)} for i in range(n_layers)]

    heads = [(l - 1, h) for l in SCREEN_LAYERS for h in range(n_heads)]
    hl = patcher.patched_heads(prompts["assert"], [[s] for s in heads], src_cap)
    m_h = metric(hl, orig)
    screen = [{**base, "layer": l, "layer_1idx": l + 1, "head": h, "head_name": hname(l, h),
               "recovery": (m_h[j] - m_asr) / gap, "reverted": float(hl[j].argmax() == orig)}
              for j, (l, h) in enumerate(heads)]

    cfg = [top[cond][:k] for k in KS]
    cf, cr = patcher.patched_heads(prompts["assert"], cfg, src_cap), patcher.patched_heads(prompts[cond], cfg, asr_cap)
    m_cf, m_cr = metric(cf, orig), metric(cr, orig)
    cum = [{**base, "k": k, "heads": " ".join(hname(*h) for h in cfg[j]), "recovery": (m_cf[j] - m_asr) / gap,
            "reverted": float(cf[j].argmax() == orig), "reverse_effect": (m_src - m_cr[j]) / gap,
            "reverse_flipped": float(cr[j].argmax() != orig)} for j, k in enumerate(KS)]
    return pair_row, resid, screen, cum


def control_check(out):
    """With the original prefill, per-pair numbers must match patch_residual.py / patch_heads.py."""
    print("\n=== Control check against the original per-pair results ===")
    worst = 0.0
    for name, new, old_paths, keys in [
        ("residual", out["resid"], [f"results/patch_residual/stage1_resid_{c}_per_pair.csv" for c in PAIR_TYPES],
         ["pair", "row", "layer_1idx"]),
        ("cumulative heads", out["cum"], ["results/patch_heads/stage2_cumulative_per_pair.csv"], ["pair", "row", "k"])]:
        old = pd.concat([pd.read_csv(p) for p in old_paths])
        m = new.merge(old, on=keys, suffixes=("", "_old"))
        d = max((m[c] - m[f"{c}_old"]).abs().max() for c in COLS)
        worst = max(worst, d)
        print(f"  {name}: {len(m)}/{len(new)} rows matched, max |diff| {d:.2g}")
    print(f"  {'OK' if worst < 1e-3 else 'MISMATCH'}: control reproduces the original results (max |diff| {worst:.2g})")


# ---------------------------------------------------------------- plots

def plot_resid(results, path):
    fig, axes = plt.subplots(1, len(PAIR_TYPES), figsize=(13, 4.4), dpi=150, squeeze=False)
    for ax, (cond, label) in zip(axes[0], PAIR_TYPES.items()):
        o = pd.read_csv(f"results/patch_residual/stage1_resid_{cond}.csv")
        ax.plot(o.layer_1idx, o.recovery, color=MUTED, linewidth=2, linestyle="--",
                label=f'"{FINAL_PREFILL}" original (n={int(o.n_pairs.iloc[0])})')
        for (pf, out), color in zip(results.items(), PREFILL_COLORS):
            d = out["resid_agg"][out["resid_agg"].pair == cond]
            if d.empty:
                continue
            ax.fill_between(d.layer_1idx, d.recovery - d.recovery_se, d.recovery + d.recovery_se,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(d.layer_1idx, d.recovery, color=color, linewidth=2, marker="o", markersize=3,
                    label=f'"{pf}" (n={int(d.n_pairs.iloc[0])})')
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(f"{label}: residual patching recovery by prefill (mean ± SE)", color=INK, fontsize=9, loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK)
        style(ax, "recovery")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative(results, orig_cum, path):
    fig, axes = plt.subplots(len(PAIR_TYPES), 2, figsize=(12, 8), dpi=150, squeeze=False)
    for row, (cond, label) in zip(axes, PAIR_TYPES.items()):
        series = [(f'"{FINAL_PREFILL}" original', orig_cum[orig_cum.pair == cond], MUTED)]
        series += [(f'"{pf}"', out["cum_agg"][out["cum_agg"].pair == cond], c)
                   for (pf, out), c in zip(results.items(), PREFILL_COLORS)]
        for name, d, color in series:
            if d.empty:
                continue
            d = d.sort_values("k")
            for ax, col, ls, what in [(row[0], "recovery", "-", "recovery"), (row[0], "reverse_effect", "--", "reverse effect"),
                                      (row[1], "reverted", "-", "reverts"), (row[1], "reverse_flipped", "--", "reverse flips")]:
                ax.plot(d.k, d[col], color=color, linestyle=ls, linewidth=2, marker="o", markersize=4,
                        label=f"{name}: {what}")
        for ax, yl in [(row[0], "normalized effect"), (row[1], "fraction of pairs")]:
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.set_xticks(KS)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{label}: original top-k heads patched together", color=INK, fontsize=9, loc="left")
            style(ax, yl)
            ax.set_xlabel("k (original top heads)", color=INK)
            ax.legend(loc="lower right", frameon=False, fontsize=6.5, labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--pairs", default="results/patch_residual/pairs.csv")
    ap.add_argument("--prefill", action="append", default=None, help="repeatable; default: '(' and 'The answer is ('")
    ap.add_argument("--max-pairs", type=int, default=150, help="patched pairs per type and prefill")
    ap.add_argument("--max-questions", type=int, default=None, help="behaviour step subset (testing)")
    ap.add_argument("--skip-patching", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/patch_prefill_robustness")
    args = ap.parse_args()
    prefills = args.prefill or DEFAULT_PREFILLS
    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    recs = [json.loads(line) for line in open(args.records)]
    pdf = pd.read_csv(args.pairs)
    pdf = pdf[pdf.src_ok & pdf.assert_ok]
    excluded = screened_rows(pdf)
    pool = [r for r in recs if r["correct"] and r["row"] not in excluded]
    print(f"Initially right: {sum(r['correct'] for r in recs)}; excluding {len(excluded)} questions from "
          f"patch_heads's screening sample → {len(pool)} held-out questions")
    if args.max_questions:
        pool = [pool[i] for i in sorted(np.random.default_rng(args.seed).choice(len(pool), args.max_questions, replace=False))]
        print(f"Test run: behaviour on {len(pool)} questions")
    by_row = {r["row"]: r for r in pool}
    orig_pairs = {c: set(pdf[pdf.pair == c].row) & set(by_row) for c in PAIR_TYPES}

    cum_orig = pd.read_csv("results/patch_heads/stage2_cumulative.csv")
    top = {c: [parse_head(h) for h in cum_orig[(cum_orig.pair == c) & (cum_orig.k == KS[-1])].heads.iloc[0].split()]
           for c in PAIR_TYPES}
    print("Original top-8 heads: " + "; ".join(f"{c}: {' '.join(hname(*h) for h in hs)}" for c, hs in top.items()))
    screen_orig = pd.read_csv("results/patch_heads/stage1_screen.csv")

    items = {}
    for name in sorted({r["dataset"] for r in pool}):
        for x in load_dataset("allenai/ai2_arc", name, split="test"):
            items[(name, x["id"])] = x
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    runner = Phase1Runner(torch.bfloat16, device)
    patcher = HeadPatcher(runner)
    n_layers, n_heads = runner.n_layers, runner.model.config.num_attention_heads
    print(f"Model on {device}, {n_layers} layers, {n_heads} heads")

    results = {}
    for pf in prefills:
        sub = os.path.join(args.out_dir, slug(pf))
        os.makedirs(sub, exist_ok=True)
        print(f'\n################ prefill "{pf}" → {sub}/')
        print(f"  example ending: {build_prompts_pf(runner, items[(pool[0]['dataset'], pool[0]['id'])], pool[0]['initial'], pf)['assert'][-60:]!r}")

        # ---- step 1
        print("\n=== Step 1: behaviour ===")
        beh = behaviour(runner, pool, items, pf)
        beh.to_csv(os.path.join(sub, "behaviour.csv"), index=False)
        out = {"beh": beh, "pairs": {}}
        for c in CONDITIONS:
            print(f"  {c:8s} flips {beh[f'{c}_flipped'].mean():6.1%} (\"{FINAL_PREFILL}\": {beh[f'{c}_orig_flipped'].mean():6.1%}); "
                  f"same flip outcome as original on {(beh[f'{c}_flipped'] == beh[f'{c}_orig_flipped']).mean():.0%}; "
                  f"letter prob mass {beh[f'{c}_mass'].mean():.3f}")
            if beh[f"{c}_mass"].mean() < 0.5:
                print(f"  WARNING: under {c} the prefill puts little mass on the four letters")
        for c in PAIR_TYPES:
            rows = set(beh[beh.assert_flipped & ~beh[f"{c}_flipped"]].row)
            out["pairs"][c] = rows
            both = rows & orig_pairs[c]
            print(f"  {PAIR_TYPES[c]} pairs: {len(rows)} (original {len(orig_pairs[c])}; {len(both)} in both, "
                  f"{len(both) / max(len(orig_pairs[c]), 1):.0%} of the original pairs survive)")

        if args.skip_patching:
            results[pf] = out
            continue

        # ---- step 2
        print(f"\n=== Step 2: patching (up to {args.max_pairs} pairs per type) ===")
        rng = np.random.default_rng(args.seed)
        pair_rows, resid, screen, cum = [], [], [], []
        for c in PAIR_TYPES:
            rows = sorted(out["pairs"][c])
            rows = [rows[i] for i in sorted(rng.choice(len(rows), min(args.max_pairs, len(rows)), replace=False))]
            t0 = time.time()
            for k, r in enumerate(rows):
                rec = by_row[r]
                p, a, b, d = patch_pair(patcher, rec, items[(rec["dataset"], rec["id"])], c, pf, top, n_layers, n_heads)
                pair_rows.append(p)
                resid += a
                screen += b
                cum += d
                if (k + 1) % 25 == 0 or k + 1 == len(rows):
                    print(f"  {PAIR_TYPES[c]}: {k + 1}/{len(rows)} pairs ({time.time() - t0:.0f}s)")
        out["pairs_df"] = pd.DataFrame(pair_rows)
        out["resid"], out["screen"], out["cum"] = pd.DataFrame(resid), pd.DataFrame(screen), pd.DataFrame(cum)
        out["resid_agg"] = agg(out["resid"], ["pair", "layer", "layer_1idx"], COLS)
        out["screen_agg"] = agg(out["screen"], ["pair", "layer", "layer_1idx", "head", "head_name"], ["recovery", "reverted"])
        out["cum_agg"] = agg(out["cum"], ["pair", "k", "heads"], COLS)
        for name in ("pairs_df", "resid", "screen", "cum"):
            out[name].to_csv(os.path.join(sub, f"{name.replace('_df', '')}_per_pair.csv" if name != "pairs_df" else "pairs.csv"), index=False)
        for name in ("resid_agg", "screen_agg", "cum_agg"):
            out[name].to_csv(os.path.join(sub, f"{name.replace('_agg', '')}.csv"), index=False)
        last = out["resid"][out["resid"].layer_1idx == n_layers]
        print(f"  Sanity check, L{n_layers} residual: max |recovery − 1| {np.abs(last.recovery - 1).max():.2g}")
        plot_heatmaps(out["screen_agg"], os.path.join(sub, "head_screen_heatmap.png"))
        if pf == FINAL_PREFILL:
            control_check(out)
        results[pf] = out

    if args.skip_patching:
        return
    plot_resid(results, os.path.join(args.out_dir, "resid_recovery_by_prefill.png"))
    plot_cumulative(results, cum_orig, os.path.join(args.out_dir, "top_heads_by_prefill.png"))
    print(f"\nWrote CSVs, plots and run_log.txt to {args.out_dir}/")
    summarize(results, screen_orig, cum_orig)


def resid_shape(d):
    rec = d.set_index("layer_1idx").recovery
    jumps = rec.diff()
    big = int(jumps.idxmax())
    above = rec.index[rec > 0.5]
    return rec, big, jumps[big], (int(above[0]) if len(above) else None)


def summarize(results, screen_orig, cum_orig):
    print("\n=== SUMMARY (layers 1-indexed, heads 0–7; held-out questions only) ===")
    print(f'Reference: "{FINAL_PREFILL}" (original). Original numbers are over all original pairs.')
    for pf, out in results.items():
        beh = out["beh"]
        print(f'\n--- prefill "{pf}" ---')
        print("1. Behaviour: " + "; ".join(f"{c} flips {beh[f'{c}_flipped'].mean():.0%} (was {beh[f'{c}_orig_flipped'].mean():.0%})"
                                           for c in CONDITIONS)
              + "; pairs " + ", ".join(f"{PAIR_TYPES[c]} {len(out['pairs'][c])}" for c in PAIR_TYPES))
        for cond, label in PAIR_TYPES.items():
            n = out["pairs_df"][out["pairs_df"].pair == cond].shape[0]
            if not n:
                print(f"  {label}: no pairs")
                continue
            print(f"  {label} (n={n} patched):")
            rec, big, jump, half = resid_shape(out["resid_agg"][out["resid_agg"].pair == cond])
            o_rec, o_big, o_jump, o_half = resid_shape(pd.read_csv(f"results/patch_residual/stage1_resid_{cond}.csv"))
            print(f"    2. Residual: max through L12 {rec.loc[:12].max():.2f} (was {o_rec.loc[:12].max():.2f}); first > 0.5 at "
                  f"L{half} (was L{o_half}); largest jump L{big - 1}→L{big} {jump:+.2f} (was L{o_big - 1}→L{o_big} {o_jump:+.2f}); "
                  f"L16 {rec[16]:.2f} (was {o_rec[16]:.2f})")
            s = out["screen_agg"][out["screen_agg"].pair == cond].sort_values("recovery", ascending=False)
            so = screen_orig[screen_orig.pair == cond].set_index("head_name").recovery
            new = s.set_index("head_name").recovery.loc[so.index]
            top_new, top_old = list(s.head_name[:8]), list(so.sort_values(ascending=False).index[:8])
            rank_h3 = list(s.head_name).index("L16.H3") + 1
            print(f"    3. Heads: top 5 " + ", ".join(f"{r.head_name} {r.recovery:+.2f}" for r in s.head(5).itertuples())
                  + f"; L16.H3 rank {rank_h3}; Spearman vs original screen {spearmanr(new, so)[0]:.2f}; "
                  f"top-8 overlap {len(set(top_new) & set(top_old))}/8; most negative {s.iloc[-1].head_name} {s.iloc[-1].recovery:+.2f}")
            c = out["cum_agg"][out["cum_agg"].pair == cond].set_index("k")
            co = cum_orig[cum_orig.pair == cond].set_index("k")
            print(f"    4. Original top-k heads: " + "; ".join(
                f"k={k} recovery {c.recovery[k]:+.2f} (was {co.recovery[k]:+.2f}), reverts {c.reverted[k]:.0%} "
                f"(was {co.reverted[k]:.0%}), reverse flips {c.reverse_flipped[k]:.0%} (was {co.reverse_flipped[k]:.0%})"
                for k in (1, KS[-1])))
    print("\nIn plain words: compare each line to its 'was' value. The findings are robust to the prefill if the "
          "residual rise stays in L13–17, L16.H3 stays at or near the top with a high Spearman correlation, and "
          "the top-8 heads still revert most assert flips (and, for doubt pairs, still induce most flips in reverse).")


if __name__ == "__main__":
    main()
