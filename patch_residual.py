"""
Activation patching at the final position: where does the assert run's flip get decided?

Pairs (initially right questions):
  doubt→assert    flipped under assert, held under doubt
  neutral→assert  flipped under assert (neutral never flips)
Prompts are rebuilt exactly as in collect.py (history uses the recorded initial answer).
Clean runs must reproduce the recorded final answers; pairs that don't are skipped.

Metric: logit(original answer) − max logit of the other three letters (four letter logits).
  recovery = (patched − assert_clean) / (source_clean − assert_clean)

Stage 1  residual stream output of each layer (1–26), final position only:
         source → assert (recovery, fraction reverting to the original answer) and
         assert → source (normalized drop, fraction induced to flip).
Stage 2  for layers 12–22, only the attention contribution (post_attention_layernorm output)
         or only the MLP contribution (post_feedforward_layernorm output) at the final position,
         source → assert. These are the two terms Gemma2DecoderLayer adds to the residual stream.

Every run is a single unbatched Phase1Runner.run call, so patched runs have exactly the numerics
of the recorded runs (batching changes bf16 results on MPS by up to ~0.25 logits). Sanity check:
patching the last layer's residual reproduces the source run, so recovery at L26 must be 1.

Usage:
    python patch_residual.py --max-pairs 10 --out-dir results/patch_residual/test_10   # quick test
    python patch_residual.py                                                     # full run
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
import torch
from datasets import load_dataset

from runner import FINAL_PREFILL, LETTERS, Phase1Runner, format_question
from collect import CONDITIONS
from probe_initial import BLUE, INK, MUTED
from probe_baselines import ORANGE, Tee, style

STAGE2_LAYERS = range(12, 23)  # 1-indexed
PAIR_TYPES = {"doubt": "doubt→assert", "neutral": "neutral→assert"}
COLORS = {"doubt": BLUE, "neutral": ORANGE}


class Patcher:
    """Capture or overwrite final-position outputs of layers ("resid"), attention contributions
    ("attn", post_attention_layernorm) and MLP contributions ("mlp", post_feedforward_layernorm)."""

    def __init__(self, runner):
        self.runner = runner
        self.capture = None   # dict to fill on the next forward pass, or None
        self.patches = {}     # (kind, layer) -> list of (batch index, vector)
        for i, layer in enumerate(runner.model.model.layers):
            for kind, mod in [("resid", layer), ("attn", layer.post_attention_layernorm),
                              ("mlp", layer.post_feedforward_layernorm)]:
                mod.register_forward_hook(self._hook(kind, i))

    def _hook(self, kind, i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.capture is not None:
                self.capture[(kind, i)] = h[0, -1].detach().clone()
            todo = self.patches.get((kind, i))
            if not todo:
                return None
            h = h.clone()
            for b, v in todo:
                h[b, -1] = v
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def clean(self, text):
        """Single unbatched run, exactly as in collect.py; returns letter logits and captures."""
        self.capture, self.patches = {}, {}
        logits, _, _ = self.runner.run(text)
        cap, self.capture = self.capture, None
        return logits.float().numpy(), cap

    def patched(self, text, sites, source):
        """One single run per site = (kind, layer), each patched with source[site]; (len(sites), 4)."""
        out = []
        for site in sites:
            self.patches = {site: [(0, source[site])]}
            logits, _, _ = self.runner.run(text)
            out.append(logits.float().numpy())
        self.patches = {}
        return np.array(out)


def metric(logits, orig):
    """logit(orig) − max of the other three; works on (4,) or (B, 4)."""
    logits = np.atleast_2d(logits)
    return logits[:, orig] - np.delete(logits, orig, axis=1).max(axis=1)


def build_prompts(runner, item, initial, cond):
    q = format_question(item)
    history = [("user", q), ("assistant", f"({initial})")]
    return {c: runner.prompt(history + [("user", CONDITIONS[c])], FINAL_PREFILL) for c in (cond, "assert")}


def run_pair(patcher, rec, item, cond, n_layers):
    orig = LETTERS.index(rec["initial"])
    prompts = build_prompts(patcher.runner, item, rec["initial"], cond)
    src_logits, src_cap = patcher.clean(prompts[cond])
    asr_logits, asr_cap = patcher.clean(prompts["assert"])
    src_ans, asr_ans = LETTERS[int(src_logits.argmax())], LETTERS[int(asr_logits.argmax())]
    ok = src_ans == rec[cond]["final"] and asr_ans == rec["assert"]["final"]
    base = {"row": rec["row"], "id": rec["id"], "dataset": rec["dataset"], "pair": cond,
            "original": rec["initial"], "assert_final": rec["assert"]["final"],
            "src_ok": src_ans == rec[cond]["final"], "assert_ok": asr_ans == rec["assert"]["final"]}
    if not ok:
        return base, None, None
    m_src, m_asr = float(metric(src_logits, orig)[0]), float(metric(asr_logits, orig)[0])
    gap = m_src - m_asr
    base.update({"source_clean": m_src, "assert_clean": m_asr})

    sites = [("resid", i) for i in range(n_layers)]
    fwd = patcher.patched(prompts["assert"], sites, src_cap)    # source → assert
    rev = patcher.patched(prompts[cond], sites, asr_cap)        # assert → source
    m_fwd, m_rev = metric(fwd, orig), metric(rev, orig)
    stage1 = []
    for i in range(n_layers):
        stage1.append({"row": rec["row"], "pair": cond, "layer": i, "layer_1idx": i + 1,
                       "recovery": (m_fwd[i] - m_asr) / gap,
                       "reverted": bool(fwd[i].argmax() == orig),
                       "reverse_effect": (m_src - m_rev[i]) / gap,
                       "reverse_flipped": bool(rev[i].argmax() != orig)})

    sites2 = [(k, l - 1) for l in STAGE2_LAYERS for k in ("attn", "mlp")]
    comp = patcher.patched(prompts["assert"], sites2, src_cap)
    m_comp = metric(comp, orig)
    stage2 = [{"row": rec["row"], "pair": cond, "layer": l, "layer_1idx": l + 1, "component": k,
               "recovery": (m_comp[j] - m_asr) / gap, "reverted": bool(comp[j].argmax() == orig)}
              for j, (k, l) in enumerate(sites2)]
    return base, stage1, stage2


def agg(df, by, cols):
    g = df.groupby(by)
    out = g[cols].mean()
    for c in cols:
        out[f"{c}_se"] = g[c].std() / np.sqrt(g[c].count())
    out["n_pairs"] = g.size()
    return out.reset_index()


def plot_stage1(s1, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), dpi=150, sharex=True)
    for cond, label in PAIR_TYPES.items():
        d = s1[s1.pair == cond]
        if d.empty:
            continue
        x, n = d.layer_1idx, int(d.n_pairs.iloc[0])
        for col, ls, what in [("recovery", "-", "source → assert recovery"),
                              ("reverse_effect", "--", "assert → source effect")]:
            a1.fill_between(x, d[col] - d[f"{col}_se"], d[col] + d[f"{col}_se"], color=COLORS[cond],
                            alpha=0.15, linewidth=0)
            a1.plot(x, d[col], color=COLORS[cond], linestyle=ls, linewidth=2, marker="o", markersize=3,
                    label=f"{label} (n={n}): {what}")
        a2.plot(x, d["reverted"], color=COLORS[cond], linewidth=2, marker="o", markersize=3,
                label=f"{label}: answer reverts to original (source → assert)")
        a2.plot(x, d["reverse_flipped"], color=COLORS[cond], linestyle="--", linewidth=2, marker="o",
                markersize=3, label=f"{label}: flip induced (assert → source)")
    for a in (a1, a2):
        a.axhline(0, color=MUTED, linewidth=1)
        a.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
        a.legend(loc="center left", frameon=False, fontsize=7.5, labelcolor=INK)
    a1.set_title("Residual-stream patching at the final position (mean ± SE)", color=INK, fontsize=10, loc="left")
    style(a1, "normalized effect")
    a2.set_title("Answer changes", color=INK, fontsize=10, loc="left")
    style(a2, "fraction of pairs")
    a2.set_xlim(0.5, s1.layer_1idx.max() + 0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stage2(s2, s1, path):
    conds = [c for c in PAIR_TYPES if c in set(s2.pair)]
    fig, axes = plt.subplots(1, len(conds), figsize=(6.5 * len(conds), 4.2), dpi=150, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        d = s2[s2.pair == cond]
        r = s1[(s1.pair == cond) & s1.layer_1idx.isin(STAGE2_LAYERS)]
        ax.plot(r.layer_1idx, r.recovery, color=MUTED, linewidth=1.5, linestyle="--", label="full residual stream")
        for comp, color, label in [("attn", BLUE, "attention contribution only"), ("mlp", ORANGE, "MLP contribution only")]:
            e = d[d.component == comp]
            ax.fill_between(e.layer_1idx, e.recovery - e.recovery_se, e.recovery + e.recovery_se,
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(e.layer_1idx, e.recovery, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(f"{PAIR_TYPES[cond]} (n={int(d.n_pairs.iloc[0])}): recovery by component (mean ± SE)",
                     color=INK, fontsize=9, loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
        ax.set_xticks(list(STAGE2_LAYERS))
        style(ax, "recovery")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(s1, s2):
    print("\n=== SUMMARY (layers 1-indexed) ===")
    for cond, label in PAIR_TYPES.items():
        d = s1[s1.pair == cond].set_index("layer_1idx")
        if d.empty:
            continue
        rec = d.recovery
        jumps = rec.diff()
        big = int(jumps.idxmax())
        half = rec.index[rec > 0.5]
        first_half = int(half[0]) if len(half) else None
        print(f"\n{label} (n={int(d.n_pairs.iloc[0])} pairs)")
        print(f"  Recovery: max {rec.loc[:16].max():.2f} through L16; "
              + ", ".join(f"L{l} {rec[l]:.2f} ± {d.recovery_se[l]:.2f}" for l in (16, 17, 18, 19, 20, 22, 26)))
        print(f"  Largest one-layer rise: L{big - 1}→L{big} ({jumps[big]:+.2f}); first layer above 0.5: "
              f"{'L' + str(first_half) if first_half else 'none'}; reverts to original at L26: {d.reverted[26]:.0%}")
        early_ok = rec.loc[:16].max() < 0.2
        sharp_ok = big in (17, 18, 19)
        late_ok = rec.loc[24:].min() > 0.8
        print(f"  Prediction (near 0 through ~L16, sharp rise ~L17–18, near 1 late): "
              f"near zero early {'yes' if early_ok else 'NO'}; sharp rise at 17–18 {'yes' if sharp_ok else 'NO'}; "
              f"near 1 by final layers {'yes' if late_ok else 'NO'}.")
        print(f"  Reverse (assert → source): effect L16 {d.reverse_effect[16]:.2f}, L18 {d.reverse_effect[18]:.2f}, "
              f"L26 {d.reverse_effect[26]:.2f}; induces a flip in {d.reverse_flipped[18]:.0%} at L18, "
              f"{d.reverse_flipped[26]:.0%} at L26 (max {d.reverse_flipped.max():.0%} at "
              f"L{int(d.reverse_flipped.idxmax())})")
        e = s2[s2.pair == cond]
        if e.empty:
            continue
        a = e[e.component == "attn"].set_index("layer_1idx")
        m = e[e.component == "mlp"].set_index("layer_1idx")
        key = [l for l in (big - 1, big, big + 1) if l in a.index]
        print("  Components: " + "; ".join(f"L{l} attn {a.recovery[l]:+.2f}, MLP {m.recovery[l]:+.2f}"
                                          for l in STAGE2_LAYERS))
        a_key, m_key = a.recovery[key].sum(), m.recovery[key].sum()
        print(f"  At the key layers {ranges_str(key)}: attention {a_key:+.2f} vs MLP {m_key:+.2f} (summed) → "
              f"{'attention' if a_key > m_key else 'MLP'} drives it; strongest single component: "
              f"{max(((l, 'attn', a.recovery[l]) for l in a.index), key=lambda t: t[2])[:2]} / "
              f"{max(((l, 'mlp', m.recovery[l]) for l in m.index), key=lambda t: t[2])[:2]}")


def ranges_str(ls):
    return f"L{ls[0]}–{ls[-1]}" if len(ls) > 1 else f"L{ls[0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--max-pairs", type=int, default=None, help="per pair type (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/patch_residual")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    recs = [json.loads(line) for line in open(args.records)]
    right_flip = [r for r in recs if r["correct"] and r["assert"]["flipped"]]
    pairs = {"doubt": [r for r in right_flip if not r["doubt"]["flipped"]],
             "neutral": [r for r in right_flip if not r["neutral"]["flipped"]]}
    print(f"Initially right and flipped under assert: {len(right_flip)}")
    print(f"Pairs: doubt→assert {len(pairs['doubt'])} (held under doubt), "
          f"neutral→assert {len(pairs['neutral'])} (held under neutral; neutral flips "
          f"{sum(r['neutral']['flipped'] for r in recs)} times in all {len(recs)} records)")
    if args.max_pairs:
        rng = np.random.default_rng(args.seed)
        pairs = {c: [p[i] for i in sorted(rng.choice(len(p), min(args.max_pairs, len(p)), replace=False))]
                 for c, p in pairs.items()}
        print(f"Test run: {args.max_pairs} pairs per type")

    items = {}
    for name in sorted({r["dataset"] for r in right_flip}):
        for x in load_dataset("allenai/ai2_arc", name, split="test"):
            items[(name, x["id"])] = x
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    runner = Phase1Runner(torch.bfloat16, device)
    patcher = Patcher(runner)
    n_layers = runner.n_layers
    print(f"Model on {device}, {n_layers} layers")

    pair_rows, s1_rows, s2_rows = [], [], []
    for cond, plist in pairs.items():
        t0 = time.time()
        for k, rec in enumerate(plist):
            base, s1, s2 = run_pair(patcher, rec, items[(rec["dataset"], rec["id"])], cond, n_layers)
            pair_rows.append(base)
            if s1 is None:
                print(f"  SKIP row {rec['row']} ({PAIR_TYPES[cond]}): clean run gave a different answer "
                      f"(source ok={base['src_ok']}, assert ok={base['assert_ok']})")
                continue
            s1_rows += s1
            s2_rows += s2
            if (k + 1) % 25 == 0 or k + 1 == len(plist):
                print(f"  {PAIR_TYPES[cond]}: {k + 1}/{len(plist)} pairs ({time.time() - t0:.0f}s)")

    pairs_df = pd.DataFrame(pair_rows)
    ok = pairs_df.dropna(subset=["source_clean"])
    for cond, label in PAIR_TYPES.items():
        p = pairs_df[pairs_df.pair == cond]
        q = ok[ok.pair == cond]
        print(f"{label}: {len(q)}/{len(p)} pairs reproduce the recorded answers ({len(p) - len(q)} skipped); "
              f"clean metric source {q.source_clean.mean():+.2f}, assert {q.assert_clean.mean():+.2f}")
    s1_long, s2_long = pd.DataFrame(s1_rows), pd.DataFrame(s2_rows)
    last = s1_long[s1_long.layer_1idx == n_layers]
    print(f"Sanity check, patching L{n_layers} residual: max |recovery − 1| "
          f"{np.abs(last.recovery - 1).max():.2g}, max |reverse effect − 1| {np.abs(last.reverse_effect - 1).max():.2g}")
    for c in ("reverted", "reverse_flipped"):
        s1_long[c] = s1_long[c].astype(float)
    s2_long["reverted"] = s2_long["reverted"].astype(float)
    s1 = agg(s1_long, ["pair", "layer", "layer_1idx"], ["recovery", "reverted", "reverse_effect", "reverse_flipped"])
    s2 = agg(s2_long, ["pair", "layer", "layer_1idx", "component"], ["recovery", "reverted"])

    pairs_df.to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)
    for cond in PAIR_TYPES:
        s1[s1.pair == cond].to_csv(os.path.join(args.out_dir, f"stage1_resid_{cond}.csv"), index=False)
        s2[s2.pair == cond].to_csv(os.path.join(args.out_dir, f"stage2_components_{cond}.csv"), index=False)
        s1_long[s1_long.pair == cond].to_csv(os.path.join(args.out_dir, f"stage1_resid_{cond}_per_pair.csv"), index=False)
        s2_long[s2_long.pair == cond].to_csv(os.path.join(args.out_dir, f"stage2_components_{cond}_per_pair.csv"), index=False)
    plot_stage1(s1, os.path.join(args.out_dir, "stage1_resid_recovery.png"))
    plot_stage2(s2, s1, os.path.join(args.out_dir, "stage2_attn_vs_mlp.png"))
    print(f"\nWrote CSVs, plots and run_log.txt to {args.out_dir}/")
    summarize(s1, s2)


if __name__ == "__main__":
    main()
