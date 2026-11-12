"""
Head-level activation patching at the final position, for the layers where patch_residual.py found the
flip being written into the residual stream (L13–17).

Gemma2Attention concatenates the per-head outputs, head-major (heads × head_dim), into the input
of o_proj, so head h of a layer is the slice [h·head_dim, (h+1)·head_dim) of that input. A forward
pre-hook on o_proj captures or overwrites that slice at the final position.

Pairs are the verified pairs from results/patch_residual/pairs.csv (initially right, flipped under assert,
held under doubt / neutral); prompts, the metric and the Patcher come from patch_residual.py.
  recovery = (patched − assert_clean) / (source_clean − assert_clean)

Stage 0  consistency: patching all heads of a layer together must reproduce patch_residual.py's stage-2
         attention-only recovery (L13–17, 10 pairs per type). Stops if it doesn't.
Stage 1  screen every head in L13–17, source → assert, on a seeded sample of 150 pairs per type.
Stage 2  top 8 heads per type (union), all pairs, forward (source → assert) and reverse
         (assert → source); cumulative top-k (k = 1, 2, 3, 5, 8) per type.
Stage 3  attention of the final position for the top heads, on 50 pairs per type, by token group:
         turn-2 user message, previous answer "(X)", question + options, template/other.

Layers are 1-indexed in every output; heads are numbered 0–7 within a layer (L17.H3 = layer 17, head 3).

Usage:
    python patch_heads.py --max-pairs 10 --out-dir results/patch_heads/test_10   # quick test
    python patch_heads.py                                                          # full run
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from patch_residual import COLORS, PAIR_TYPES, Patcher, agg, build_prompts, metric
from runner import FINAL_PREFILL, LETTERS, Phase1Runner, format_question
from collect import CONDITIONS
from probe_initial import BLUE, INK, MUTED
from probe_baselines import ORANGE, Tee, style

SCREEN_LAYERS = range(13, 18)  # 1-indexed
TOP_N = 8
KS = [1, 2, 3, 5, 8]
RESID_REF = {"doubt": 16, "neutral": 14}  # residual-stream reference layer from patch_residual.py
N_SCREEN, N_ATTN, N_CHECK = 150, 50, 10
GROUPS = {"turn2_msg": "turn-2 user message", "prev_answer": "previous answer (X)",
          "question": "question + options", "other": "template / other"}
ASSERT_GRAY = "#8a8985"


def hname(layer, head):
    """0-indexed layer, head → label with 1-indexed layer."""
    return f"L{layer + 1}.H{head}"


class HeadPatcher(Patcher):
    """patch_residual.Patcher plus capture/patching of the o_proj input ("head", layer), i.e. the
    concatenated per-head attention outputs, at the final position."""

    def __init__(self, runner):
        super().__init__(runner)
        self.head_dim = runner.model.model.layers[0].self_attn.head_dim
        for i, layer in enumerate(runner.model.model.layers):
            layer.self_attn.o_proj.register_forward_pre_hook(self._head_hook(i))

    def _head_hook(self, i):
        def hook(module, args):
            x = args[0]
            if self.capture is not None:
                self.capture[("head", i)] = x[0, -1].detach().clone()
            todo = self.patches.get(("head", i))
            if not todo:
                return None
            x = x.clone()
            for h, v in todo:
                x[0, -1, h * self.head_dim:(h + 1) * self.head_dim] = v
            return (x,) + tuple(args[1:])
        return hook

    def patched_heads(self, text, configs, source):
        """One single run per config (a list of (layer, head)), each head patched with its slice of
        source; returns (len(configs), 4) letter logits."""
        D, out = self.head_dim, []
        for heads in configs:
            p = {}
            for l, h in heads:
                p.setdefault(("head", l), []).append((h, source[("head", l)][h * D:(h + 1) * D]))
            self.patches = p
            logits, _, _ = self.runner.run(text)
            out.append(logits.float().numpy())
        self.patches = {}
        return np.array(out)


def setup_pair(patcher, rec, item, cond):
    """Clean source and assert runs; None if they don't reproduce the recorded answers."""
    prompts = build_prompts(patcher.runner, item, rec["initial"], cond)
    src_logits, src_cap = patcher.clean(prompts[cond])
    asr_logits, asr_cap = patcher.clean(prompts["assert"])
    if (LETTERS[int(src_logits.argmax())] != rec[cond]["final"]
            or LETTERS[int(asr_logits.argmax())] != rec["assert"]["final"]):
        return None
    orig = LETTERS.index(rec["initial"])
    m_src, m_asr = float(metric(src_logits, orig)[0]), float(metric(asr_logits, orig)[0])
    return {"rec": rec, "item": item, "cond": cond, "orig": orig, "prompts": prompts,
            "src_cap": src_cap, "asr_cap": asr_cap, "m_src": m_src, "m_asr": m_asr, "gap": m_src - m_asr}


def forward(patcher, P, configs):
    """source → assert: recovery and whether the answer reverts to the original."""
    logits = patcher.patched_heads(P["prompts"]["assert"], configs, P["src_cap"])
    m = metric(logits, P["orig"])
    return (m - P["m_asr"]) / P["gap"], logits.argmax(1) == P["orig"]


def reverse(patcher, P, configs):
    """assert → source: normalized drop and whether the answer flips away from the original."""
    logits = patcher.patched_heads(P["prompts"][P["cond"]], configs, P["asr_cap"])
    m = metric(logits, P["orig"])
    return (P["m_src"] - m) / P["gap"], logits.argmax(1) != P["orig"]


def token_groups(tok, text, q, initial, msg):
    """Group label per token of text, plus BOS / prefill masks (both inside "other")."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    qs = text.index(q)
    qe = qs + len(q)
    a_s = text.index(f"({initial})", text.index("<start_of_turn>model", qe))
    ae = a_s + 3
    ms = text.index(msg, ae)
    me = ms + len(msg)
    pf = len(text) - len(FINAL_PREFILL)
    labels, prefill = [], []
    for s, e in enc["offset_mapping"]:
        if s < qe and e > qs:
            labels.append("question")
        elif s < ae and e > a_s:
            labels.append("prev_answer")
        elif s < me and e > ms:
            labels.append("turn2_msg")
        else:
            labels.append("other")
        prefill.append(s >= pf)
    bos = np.array([t == tok.bos_token_id for t in enc["input_ids"]])
    return np.array(labels), bos, np.array(prefill), enc["input_ids"]


@torch.inference_mode()
def attention_run(runner, text):
    ids = runner.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(runner.device)
    out = runner.model(ids, use_cache=False, output_attentions=True)
    final = out.logits[0, -1].float()[runner.letter_ids].cpu().numpy()
    return final, out.attentions


# ---------------------------------------------------------------- plots

def plot_heatmaps(s1, path):
    conds = [c for c in PAIR_TYPES if c in set(s1.pair)]
    vmax = np.abs(s1.recovery).max()
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ee", BLUE])
    fig, axes = plt.subplots(1, len(conds), figsize=(6.2 * len(conds), 3.6), dpi=150, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        d = s1[s1.pair == cond]
        layers, heads = sorted(d.layer_1idx.unique()), sorted(d["head"].unique())
        grid = d.pivot(index="layer_1idx", columns="head", values="recovery").loc[layers, heads].to_numpy()
        rev = d.pivot(index="layer_1idx", columns="head", values="reverted").loc[layers, heads].to_numpy()
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(layers)):
            for j in range(len(heads)):
                ax.text(j, i - 0.12, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color=INK)
                ax.text(j, i + 0.22, f"{rev[i, j]:.0%}", ha="center", va="center", fontsize=6, color=MUTED)
        ax.set_xticks(range(len(heads)), heads)
        ax.set_yticks(range(len(layers)), [f"L{l}" for l in layers])
        ax.set_xlabel("head", color=INK)
        ax.set_ylabel("layer", color=INK)
        ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{PAIR_TYPES[cond]} (n={int(d.n_pairs.iloc[0])}): mean recovery; % reverting below",
                     color=INK, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative(cum, resid, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
    for cond, label in PAIR_TYPES.items():
        d = cum[cum.pair == cond].sort_values("k")
        if d.empty:
            continue
        c = COLORS[cond]
        for ax, col, ls, what in [(a1, "recovery", "-", "source → assert recovery"),
                                  (a1, "reverse_effect", "--", "assert → source effect"),
                                  (a2, "reverted", "-", "reverts to original (source → assert)"),
                                  (a2, "reverse_flipped", "--", "flip induced (assert → source)")]:
            ax.fill_between(d.k, d[col] - d[f"{col}_se"], d[col] + d[f"{col}_se"], color=c, alpha=0.15, linewidth=0)
            ax.plot(d.k, d[col], color=c, linestyle=ls, linewidth=2, marker="o", markersize=4,
                    label=f"{label}: {what}")
        r = resid[cond]
        a1.axhline(r["recovery"], color=c, linestyle=":", linewidth=1.5,
                   label=f"{label}: full residual stream L{RESID_REF[cond]} ({r['recovery']:.2f})")
        a2.axhline(r["reverted"], color=c, linestyle=":", linewidth=1.5,
                   label=f"{label}: residual L{RESID_REF[cond]} reverts ({r['reverted']:.0%})")
    for ax, title, yl in [(a1, "Cumulative top-k head patching (mean ± SE)", "normalized effect"),
                          (a2, "Answer changes", "fraction of pairs")]:
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(KS)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        style(ax, yl)
        ax.set_xlabel("k (top heads patched together)", color=INK)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_attn_groups(s3, head, path):
    conds = [c for c in PAIR_TYPES if c in set(s3.pair)]
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 3.6), dpi=150, squeeze=False, sharey=True)
    x = np.arange(len(GROUPS))
    for ax, cond in zip(axes[0], conds):
        d = s3[(s3.pair == cond) & (s3.head_name == head)]
        for off, run, color, label in [(-0.2, "assert", ASSERT_GRAY, "assert"),
                                       (0.2, cond, COLORS[cond], cond)]:
            e = d[d.run == run].set_index("group").loc[list(GROUPS)]
            ax.bar(x + off, e["mass"], 0.38, yerr=e["mass_se"], color=color, label=label,
                   error_kw={"ecolor": MUTED, "elinewidth": 1}, edgecolor="white", linewidth=1)
        ax.set_xticks(x, [GROUPS[g].replace(" + ", "\n+ ").replace(" (X)", "\n(X)") for g in GROUPS], fontsize=7.5)
        ax.set_title(f"{head}, {PAIR_TYPES[cond]} pairs (n={int(d.n_pairs.iloc[0])})", color=INK, fontsize=9, loc="left")
        style(ax, "attention mass from final position")
        ax.set_xlabel("")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_attn_example(example, heads, path):
    runs = list(example)
    fig, axes = plt.subplots(len(runs), 1, figsize=(16, 2.2 + 0.32 * len(heads) * len(runs)), dpi=150)
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff", BLUE])
    for ax, run in zip(axes, runs):
        e = example[run]
        W = np.array([e["weights"][h] for h in heads])
        vmax = W[:, ~e["bos"]].max()  # BOS soaks up most attention; scale to the rest
        ax.imshow(W, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(heads)), [hname(*h) for h in heads], fontsize=7)
        ax.set_xticks(range(len(e["tokens"])), e["tokens"], rotation=90, fontsize=4.5)
        bounds = [i for i in range(1, len(e["groups"])) if e["groups"][i] != e["groups"][i - 1]]
        for b in bounds:
            ax.axvline(b - 0.5, color=MUTED, linewidth=0.6)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_title(f"{run} run: final-position attention per head (colour scale excludes BOS; "
                     f"BOS mass shown in CSV)", color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--pairs", default="results/patch_residual/pairs.csv")
    ap.add_argument("--max-pairs", type=int, default=None, help="per pair type, all stages (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/patch_heads")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "attn_groups"), exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    recs = {r["row"]: r for r in map(json.loads, open(args.records))}
    pdf = pd.read_csv(args.pairs)
    pdf = pdf[pdf.src_ok & pdf.assert_ok]
    pairs = {c: [recs[r] for r in pdf[pdf.pair == c].row] for c in PAIR_TYPES}
    rng = np.random.default_rng(args.seed)
    n_screen = min(args.max_pairs or N_SCREEN, N_SCREEN)
    screen = {c: sorted(rng.choice(len(p), min(n_screen, len(p)), replace=False)) for c, p in pairs.items()}
    if args.max_pairs:  # test: every stage on the same small sample
        pairs = {c: [p[i] for i in screen[c]] for c, p in pairs.items()}
        screen = {c: list(range(len(p))) for c, p in pairs.items()}
    screened_rows = {pairs[c][i]["row"] for c in pairs for i in screen[c]}
    print(f"Pairs from {args.pairs}: " + ", ".join(f"{PAIR_TYPES[c]} {len(p)}" for c, p in pairs.items())
          + f"; screening sample {n_screen} per type ({len(screened_rows)} distinct questions)")

    items = {}
    for name in sorted({r["dataset"] for p in pairs.values() for r in p}):
        for x in load_dataset("allenai/ai2_arc", name, split="test"):
            items[(name, x["id"])] = x
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    runner = Phase1Runner(torch.bfloat16, device)
    patcher = HeadPatcher(runner)
    cfg, attn0 = runner.model.config, runner.model.model.layers[0].self_attn
    n_heads, D = cfg.num_attention_heads, patcher.head_dim
    print(f"\nModel on {device}. Config: num_attention_heads {n_heads}, num_key_value_heads "
          f"{cfg.num_key_value_heads} (grouped-query: query heads 2k and 2k+1 share KV head k), head_dim {D}, "
          f"d_model {cfg.hidden_size}")
    print(f"o_proj: in_features {attn0.o_proj.in_features} = {n_heads} × {D}; Gemma2Attention.forward "
          f"transposes the attention output to (batch, seq, heads, head_dim) and reshapes it to (batch, seq, "
          f"heads·head_dim), so the o_proj input is the head-major concatenation of per-head outputs")
    assert attn0.o_proj.in_features == n_heads * D

    def P_of(cond, rec):
        P = setup_pair(patcher, rec, items[(rec["dataset"], rec["id"])], cond)
        if P is None:
            print(f"  SKIP row {rec['row']} ({PAIR_TYPES[cond]}): clean run gave a different answer")
        return P

    # ---- stage 0: all heads of a layer == stage-2 attention-only patch
    print("\n=== Stage 0: consistency check (all heads of a layer vs attention-only patch) ===")
    old = pd.concat([pd.read_csv(f"results/patch_residual/stage2_components_{c}_per_pair.csv") for c in PAIR_TYPES])
    old = old[old.component == "attn"].set_index(["pair", "row", "layer_1idx"]).recovery
    check = []
    for cond in PAIR_TYPES:
        for i in screen[cond][:N_CHECK]:
            P = P_of(cond, pairs[cond][i])
            if P is None:
                continue
            ref = pdf[(pdf.pair == cond) & (pdf.row == P["rec"]["row"])].iloc[0]
            rec_heads, _ = forward(patcher, P, [[(l - 1, h) for h in range(n_heads)] for l in SCREEN_LAYERS])
            attn = patcher.patched(P["prompts"]["assert"], [("attn", l - 1) for l in SCREEN_LAYERS], P["src_cap"])
            rec_attn = (metric(attn, P["orig"]) - P["m_asr"]) / P["gap"]
            for j, l in enumerate(SCREEN_LAYERS):
                check.append({"pair": cond, "row": P["rec"]["row"], "layer_1idx": l,
                              "all_heads_recovery": rec_heads[j], "attn_recovery_now": rec_attn[j],
                              "attn_recovery_patching_py": old.loc[(cond, P["rec"]["row"], l)],
                              "clean_diff": max(abs(P["m_src"] - ref.source_clean), abs(P["m_asr"] - ref.assert_clean))})
    check = pd.DataFrame(check)
    check["diff_now"] = (check.all_heads_recovery - check.attn_recovery_now).abs()
    check["diff_patching_py"] = (check.all_heads_recovery - check.attn_recovery_patching_py).abs()
    check.to_csv(os.path.join(args.out_dir, "stage0_consistency.csv"), index=False)
    for l, d in check.groupby("layer_1idx"):
        print(f"  L{l}: all-heads recovery {d.all_heads_recovery.mean():+.3f}, attention-only {d.attn_recovery_now.mean():+.3f} "
              f"(patch_residual.py {d.attn_recovery_patching_py.mean():+.3f}); max |diff| {d.diff_now.max():.2g} "
              f"(vs patch_residual.py {d.diff_patching_py.max():.2g}), {len(d)} pairs")
    print(f"  Clean metrics vs pairs.csv: max |diff| {check.clean_diff.max():.2g}")
    if check.diff_now.max() > 1e-3 or check.diff_patching_py.max() > 1e-3:
        print("  MISMATCH: per-head patching does not reproduce the attention-only patch. Stopping.")
        return
    print("  OK: patching all heads together reproduces the attention-only recovery.")

    # ---- stage 1: screen every head in L13–17
    print(f"\n=== Stage 1: screening {len(SCREEN_LAYERS) * n_heads} heads (L{SCREEN_LAYERS[0]}–{SCREEN_LAYERS[-1]}) ===")
    sites = [(l - 1, h) for l in SCREEN_LAYERS for h in range(n_heads)]
    s1_rows = []
    for cond in PAIR_TYPES:
        t0 = time.time()
        for k, i in enumerate(screen[cond]):
            P = P_of(cond, pairs[cond][i])
            if P is None:
                continue
            rec_, rev_ = forward(patcher, P, [[s] for s in sites])
            s1_rows += [{"pair": cond, "row": P["rec"]["row"], "layer": l, "layer_1idx": l + 1, "head": h,
                         "head_name": hname(l, h), "recovery": rec_[j], "reverted": float(rev_[j])}
                        for j, (l, h) in enumerate(sites)]
            if (k + 1) % 25 == 0 or k + 1 == len(screen[cond]):
                print(f"  {PAIR_TYPES[cond]}: {k + 1}/{len(screen[cond])} pairs ({time.time() - t0:.0f}s)")
    s1_long = pd.DataFrame(s1_rows)
    s1 = agg(s1_long, ["pair", "layer", "layer_1idx", "head", "head_name"], ["recovery", "reverted"])
    s1_long.to_csv(os.path.join(args.out_dir, "stage1_screen_per_pair.csv"), index=False)
    s1.to_csv(os.path.join(args.out_dir, "stage1_screen.csv"), index=False)
    plot_heatmaps(s1, os.path.join(args.out_dir, "stage1_heatmap.png"))
    top = {}
    for cond in PAIR_TYPES:
        d = s1[s1.pair == cond].sort_values("recovery", ascending=False)
        top[cond] = [(int(r.layer), int(r.head)) for r in d.head(TOP_N).itertuples()]
        print(f"  {PAIR_TYPES[cond]} top {TOP_N}: " + ", ".join(
            f"{r.head_name} {r.recovery:+.3f}±{r.recovery_se:.3f} ({r.reverted:.0%})" for r in d.head(TOP_N).itertuples()))
    union = sorted(set(top["doubt"]) | set(top["neutral"]))
    print(f"  Union: {len(union)} heads: {', '.join(hname(*h) for h in union)}")

    # ---- stage 2: top heads on all pairs, both directions; cumulative top-k
    print(f"\n=== Stage 2: {len(union)} heads on all pairs, forward and reverse; cumulative top-k ===")
    s2_rows, cum_rows = [], []
    for cond in PAIR_TYPES:
        t0 = time.time()
        cum_cfg = [top[cond][:k] for k in KS]
        for k, rec in enumerate(pairs[cond]):
            P = P_of(cond, rec)
            if P is None:
                continue
            base = {"pair": cond, "row": rec["row"], "heldout": rec["row"] not in screened_rows}
            f_rec, f_rev = forward(patcher, P, [[h] for h in union])
            r_eff, r_flip = reverse(patcher, P, [[h] for h in union])
            s2_rows += [{**base, "layer_1idx": l + 1, "head": h, "head_name": hname(l, h),
                         "recovery": f_rec[j], "reverted": float(f_rev[j]),
                         "reverse_effect": r_eff[j], "reverse_flipped": float(r_flip[j])}
                        for j, (l, h) in enumerate(union)]
            f_rec, f_rev = forward(patcher, P, cum_cfg)
            r_eff, r_flip = reverse(patcher, P, cum_cfg)
            cum_rows += [{**base, "k": kk, "heads": " ".join(hname(*h) for h in cum_cfg[j]),
                          "recovery": f_rec[j], "reverted": float(f_rev[j]),
                          "reverse_effect": r_eff[j], "reverse_flipped": float(r_flip[j])}
                         for j, kk in enumerate(KS)]
            if (k + 1) % 50 == 0 or k + 1 == len(pairs[cond]):
                print(f"  {PAIR_TYPES[cond]}: {k + 1}/{len(pairs[cond])} pairs ({time.time() - t0:.0f}s)")
    cols = ["recovery", "reverted", "reverse_effect", "reverse_flipped"]
    s2_long, cum_long = pd.DataFrame(s2_rows), pd.DataFrame(cum_rows)
    s2 = agg(s2_long, ["pair", "layer_1idx", "head", "head_name"], cols)
    cum = agg(cum_long, ["pair", "k", "heads"], cols)
    s2_ho = agg(s2_long[s2_long.heldout], ["pair", "layer_1idx", "head", "head_name"], cols) \
        if s2_long.heldout.any() else None
    cum_ho = agg(cum_long[cum_long.heldout], ["pair", "k", "heads"], cols) if cum_long.heldout.any() else None
    s2_long.to_csv(os.path.join(args.out_dir, "stage2_heads_per_pair.csv"), index=False)
    cum_long.to_csv(os.path.join(args.out_dir, "stage2_cumulative_per_pair.csv"), index=False)
    s2.to_csv(os.path.join(args.out_dir, "stage2_heads.csv"), index=False)
    cum.to_csv(os.path.join(args.out_dir, "stage2_cumulative.csv"), index=False)
    if s2_ho is not None:
        s2_ho.to_csv(os.path.join(args.out_dir, "stage2_heads_heldout.csv"), index=False)
        cum_ho.to_csv(os.path.join(args.out_dir, "stage2_cumulative_heldout.csv"), index=False)
    resid = {}
    for cond in PAIR_TYPES:
        r = pd.read_csv(f"results/patch_residual/stage1_resid_{cond}.csv")
        resid[cond] = r[r.layer_1idx == RESID_REF[cond]].iloc[0]
    plot_cumulative(cum, resid, os.path.join(args.out_dir, "stage2_cumulative_k.png"))

    # ---- stage 3: attention patterns
    print(f"\n=== Stage 3: attention of the final position for {len(union)} heads ===")
    s3_rows, example, shape_noted = [], None, False
    for cond in PAIR_TYPES:
        sub = sorted(np.random.default_rng(args.seed + 1).choice(screen[cond], min(N_ATTN, len(screen[cond])), replace=False))
        for i in sub:
            rec = pairs[cond][i]
            item = items[(rec["dataset"], rec["id"])]
            prompts = build_prompts(runner, item, rec["initial"], cond)
            ex = {}
            for run in (cond, "assert"):
                logits, atts = attention_run(runner, prompts[run])
                assert LETTERS[int(logits.argmax())] == rec[run]["final"], f"row {rec['row']} {run}: answer changed"
                labels, bos, prefill, ids = token_groups(runner.tok, prompts[run], format_question(item),
                                                         rec["initial"], CONDITIONS[run])
                if not shape_noted:
                    print(f"  attentions: {len(atts)} layers, each {tuple(atts[0].shape)} = (batch, query heads, "
                          f"query pos, key pos); repeat_kv expands the {cfg.num_key_value_heads} KV heads before the "
                          f"softmax, so weights are per query head. Rows sum to {atts[0][0, :, -1].float().sum(-1).mean():.3f}")
                    print(f"  token groups ({len(ids)} tokens, {run} run): "
                          + ", ".join(f"{g} {int((labels == g).sum())}" for g in GROUPS)
                          + f" (other includes BOS {int(bos.sum())} and prefill {int(prefill.sum())})")
                    print("    " + " | ".join(f"{g}: {runner.tok.decode([t for t, lab in zip(ids, labels) if lab == g])!r}"[:160]
                                              for g in ("turn2_msg", "prev_answer")))
                    shape_noted = True
                assert atts[0].shape[-1] == len(ids)
                w_by_head = {}
                for l, h in union:
                    w = atts[l][0, h, -1].float().cpu().numpy()
                    w_by_head[(l, h)] = w
                    for g in GROUPS:
                        m = labels == g
                        s3_rows.append({"pair": cond, "row": rec["row"], "run": run, "head_name": hname(l, h),
                                        "layer_1idx": l + 1, "head": h, "group": g, "mass": w[m].sum(),
                                        "per_token": w[m].mean(), "n_tokens": int(m.sum()),
                                        "bos_mass": w[bos].sum() if g == "other" else np.nan,
                                        "prefill_mass": w[prefill].sum() if g == "other" else np.nan})
                ex[run] = {"weights": w_by_head, "bos": bos, "groups": labels,
                           "tokens": [runner.tok.decode([t]).replace("\n", "\\n") for t in ids]}
            if example is None and cond == "doubt":
                example = (rec["row"], ex)
        print(f"  {PAIR_TYPES[cond]}: {len(sub)} pairs")
    s3_long = pd.DataFrame(s3_rows)
    s3 = agg(s3_long, ["pair", "run", "head_name", "layer_1idx", "head", "group"], ["mass", "per_token", "bos_mass", "prefill_mass"])
    s3_long.to_csv(os.path.join(args.out_dir, "stage3_attention_per_pair.csv"), index=False)
    s3.to_csv(os.path.join(args.out_dir, "stage3_attention.csv"), index=False)
    for l, h in union:
        plot_attn_groups(s3, hname(l, h), os.path.join(args.out_dir, "attn_groups", f"{hname(l, h)}.png"))
    ex_heads = top["doubt"]
    plot_attn_example(example[1], ex_heads, os.path.join(args.out_dir, f"attn_example_row{example[0]}.png"))
    print(f"\nWrote CSVs, heatmap, cumulative plot, attention charts and run_log.txt to {args.out_dir}/")

    summarize(s1, s2, cum, s2_ho, cum_ho, s3_long, top, resid, n_heads)


def summarize(s1, s2, cum, s2_ho, cum_ho, s3_long, top, resid, n_heads):
    print("\n=== SUMMARY (layers 1-indexed, heads 0–7) ===")
    a = s1[s1.pair == "doubt"].set_index("head_name").recovery
    b = s1[s1.pair == "neutral"].set_index("head_name").recovery.loc[a.index]
    shared = [hname(*h) for h in top["doubt"] if h in top["neutral"]]
    print("\n1. Which heads matter (screen, mean recovery from patching one head):")
    for cond in PAIR_TYPES:
        d = s1[s1.pair == cond].sort_values("recovery", ascending=False)
        print(f"  {PAIR_TYPES[cond]}: " + ", ".join(f"{r.head_name} {r.recovery:+.2f}" for r in d.head(5).itertuples())
              + f"; {int((d.recovery > 0.05).sum())}/{len(d)} heads above 0.05, most negative "
              f"{d.iloc[-1].head_name} {d.iloc[-1].recovery:+.2f}")
    print(f"  Shared between the two top-{TOP_N} lists: {len(shared)} ({', '.join(shared) or 'none'}); "
          f"Spearman correlation of all {len(a)} head recoveries across pair types: {spearmanr(a, b)[0]:.2f}")

    print("\n2. Top-k heads together vs the full residual stream:")
    for cond in PAIR_TYPES:
        d = cum[cum.pair == cond].set_index("k")
        r = resid[cond]
        ho = cum_ho[cum_ho.pair == cond].set_index("k") if cum_ho is not None else None
        r17 = pd.read_csv(f"results/patch_residual/stage1_resid_{cond}.csv").set_index("layer_1idx").loc[SCREEN_LAYERS[-1]]
        print(f"  {PAIR_TYPES[cond]} (residual stream L{RESID_REF[cond]}: recovery {r.recovery:.2f}, reverts {r.reverted:.0%}; "
              f"L{SCREEN_LAYERS[-1]}, after all screened heads: recovery {r17.recovery:.2f}, reverts {r17.reverted:.0%}):")
        for k in KS:
            print(f"    k={k}: recovery {d.recovery[k]:+.2f} ± {d.recovery_se[k]:.2f} "
                  f"({d.recovery[k] / r.recovery:.0%} of L{RESID_REF[cond]}, {d.recovery[k] / r17.recovery:.0%} of "
                  f"L{SCREEN_LAYERS[-1]} residual), reverts {d.reverted[k]:.0%}"
                  + (f"; held-out pairs {ho.recovery[k]:+.2f} (n={int(ho.n_pairs[k])})" if ho is not None else ""))
        print(f"    heads: {d.heads[KS[-1]]}")

    print("\n3. Necessary and sufficient? Forward (source → assert): restoring the head's held-answer output in "
          "the assert run; a revert means its assert-run output is needed for the flip. Reverse (assert → "
          "source): writing the assert-run output into the held run; a flip means it is enough to cause one.")
    for cond in PAIR_TYPES:
        d = s2[s2.pair == cond].set_index("head_name")
        print(f"  {PAIR_TYPES[cond]} (all {int(d.n_pairs.iloc[0])} pairs), single heads:")
        for l, h in top[cond][:3]:
            r = d.loc[hname(l, h)]
            print(f"    {hname(l, h)}: forward recovery {r.recovery:+.2f}, reverts {r.reverted:.0%}; "
                  f"reverse effect {r.reverse_effect:+.2f}, flips {r.reverse_flipped:.0%}")
        c = cum[cum.pair == cond].set_index("k")
        k8 = c.loc[KS[-1]]
        def word(x):
            return "largely" if x >= 0.5 else "partly" if x >= 0.1 else "not"
        print(f"    top {KS[-1]} together: forward reverts {k8.reverted:.0%} (recovery {k8.recovery:+.2f}); reverse "
              f"flips {k8.reverse_flipped:.0%} (effect {k8.reverse_effect:+.2f}) → {word(k8.reverted)} necessary, "
              f"{word(k8.reverse_flipped)} sufficient")
        if s2_ho is not None:
            h = cum_ho[cum_ho.pair == cond].set_index("k").loc[KS[-1]]
            print(f"    held-out pairs only (n={int(h.n_pairs)}): reverts {h.reverted:.0%}, reverse flips {h.reverse_flipped:.0%}")

    print("\n4. Attention of the final position (mass on each token group, assert vs source run):")
    for cond in PAIR_TYPES:
        d = s3_long[s3_long.pair == cond]
        print(f"  {PAIR_TYPES[cond]} pairs (n={d.row.nunique()}):")
        for l, h in top[cond][:5]:
            e = d[d.head_name == hname(l, h)]
            m = e.pivot_table(index=["row", "group"], columns="run", values="mass").reset_index()
            m["diff"] = m["assert"] - m[cond]
            g = m.groupby("group")
            mean, se = g[["assert", cond, "diff"]].mean(), g["diff"].std() / np.sqrt(g.size())
            o = e[(e.group == "other")].groupby("run")[["bos_mass", "prefill_mass"]].mean()
            big = mean.drop(columns="diff").mean(axis=1).drop("other").idxmax()
            dg = mean["diff"].abs().idxmax()
            print(f"    {hname(l, h)}: " + ", ".join(f"{GROUPS[x].split(' ')[0]} {mean.loc[x, 'assert']:.2f}/{mean.loc[x, cond]:.2f}"
                                                   for x in GROUPS)
                  + f" (assert/{cond}; of template: BOS {o.bos_mass['assert']:.2f}/{o.bos_mass[cond]:.2f}, prefill "
                  f"'Final answer: (' {o.prefill_mass['assert']:.2f}/{o.prefill_mass[cond]:.2f}). Main content target: {GROUPS[big]}; "
                  f"largest change: {GROUPS[dg]} {mean.loc[dg, 'diff']:+.2f} ± {se[dg]:.2f}")
    print("\nNotes: the top heads were chosen on the screening sample, which is part of 'all pairs' in stage 2; "
          "held-out numbers exclude every screened question. Head recoveries are not additive: heads patched "
          "together interact through later layers.")


if __name__ == "__main__":
    main()
