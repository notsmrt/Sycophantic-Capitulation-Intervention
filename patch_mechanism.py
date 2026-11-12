"""
What do the top heads push toward, and where does L16.H3 get the pushback content from?

Part 1  direct logit attribution (DLA) at the final position, for every head, in the assert, doubt
        and neutral runs of every pair question (union of the doubt→assert and neutral→assert pairs).
        A head's output is its slice of the o_proj input times the matching o_proj columns, then
        post_attention_layernorm, then the final norm and the four letter rows of lm_head:
            DLA_h = W_letters · [(1+w_f) ⊙ s_f · (1+w_a) ⊙ s_a · W_O[:, h] z_h]
        s_a = 1/RMS of the whole attention output of that layer, s_f = 1/RMS of the final residual,
        both taken from the actual run. Approximations: the two RMSNorms are linearized at the
        run's own scale (exact for the sum of all heads of a layer, checked below; a head's share
        assumes the others stay put), the final 30·tanh(x/30) soft-cap is ignored (pre-cap logit
        units), and only the direct path to the logits is counted, not effects via later layers.
        Reported: original letter minus the mean of the other three, and the same for the letter
        the model switches to under assert.

Part 2  doubt→assert pairs: patch the residual stream (output of layers 1–16) from the doubt run
        into the assert run at position groups aligned from the end of the prompt: each token of
        the prefill "Final answer: (" (the last is the final position), the three non-final prefill
        tokens together, the chat-template tokens between the message and the prefill, and the
        turn-2 user message (the last 5 tokens; "That's incorrect." is 5 tokens, "Are you sure
        about that?" 6). Reports recovery (patch_residual.py's metric) and how far L16.H3's final-position
        output moves toward its doubt-run value.
        Then patch only L16.H3's key/value inputs at those positions. L16.H3 is query head 3 and
        shares KV head 1 with head 2, so: (1) run assert with KV head 1's k_proj/v_proj output
        (pre-RoPE) patched from the doubt run, record L16.H3's output at the final position;
        (2) run assert again with only that head output written in. Variants: K+V, V only, K only.

Usage:
    python patch_mechanism.py --max-pairs 10 --out-dir results/patch_mechanism/test_10   # quick test
    python patch_mechanism.py                                                           # full run
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

from patch_heads import ASSERT_GRAY, HeadPatcher, hname, setup_pair, token_groups
from patch_residual import COLORS, PAIR_TYPES, agg, build_prompts, metric
from runner import LETTERS, Phase1Runner, format_question
from collect import CONDITIONS
from probe_initial import BLUE, INK, MUTED
from probe_baselines import ORANGE, Tee, style

AQUA, YELLOW = "#1baf7a", "#eda100"
NEG_HEADS = ["L16.H6", "L17.H2"]  # pushed against recovery in patch_heads stage 1
H3 = (15, 3)                      # L16.H3, 0-indexed layer
PATCH_LAYERS = range(1, 17)       # 1-indexed
N_PREFILL, N_TEMPLATE = 4, 5      # "Final answer: (" tokens; "<end_of_turn>\n<start_of_turn>model\n"
COND_COLORS = {"assert": ASSERT_GRAY, "doubt": BLUE, "neutral": ORANGE}


class MechPatcher(HeadPatcher):
    """HeadPatcher plus full-sequence capture/patching of layer outputs ("rfull", layer) and of
    L16's k_proj / v_proj outputs ("k" / "v", layer) at arbitrary positions."""

    def __init__(self, runner, kv_layer):
        super().__init__(runner)
        self.full = None        # dict to fill with full-sequence tensors on the next run, or None
        self.pos_patches = {}   # key -> list of (position, feature slice, vector)
        for i, layer in enumerate(runner.model.model.layers):
            layer.register_forward_hook(self._full_hook(("rfull", i)))
        attn = runner.model.model.layers[kv_layer].self_attn
        attn.k_proj.register_forward_hook(self._full_hook(("k", kv_layer)))
        attn.v_proj.register_forward_hook(self._full_hook(("v", kv_layer)))

    def _full_hook(self, key):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.full is not None:
                self.full[key] = h[0].detach().clone()
            todo = self.pos_patches.get(key)
            if not todo:
                return None
            h = h.clone()
            for pos, sl, v in todo:
                h[0, pos, sl] = v
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def go(self, text, pos=None, heads=None, full=False):
        """One unbatched run with optional position patches and head patches; returns letter logits,
        final-position captures and (if full) full-sequence captures."""
        self.capture, self.full = {}, ({} if full else None)
        self.pos_patches, self.patches = pos or {}, heads or {}
        logits, _, _ = self.runner.run(text)
        out = (logits.float().numpy(), self.capture, self.full)
        self.capture, self.full, self.pos_patches, self.patches = None, None, {}, {}
        return out


# ---------------------------------------------------------------- part 1: DLA

class DLA:
    """Per-head direct logit attribution onto the four letters at the final position."""

    def __init__(self, runner):
        m = runner.model.model
        self.layers, self.n = m.layers, len(m.layers)
        a = m.layers[0].self_attn
        self.H, self.D = runner.model.config.num_attention_heads, a.head_dim
        self.WO = [l.self_attn.o_proj.weight.float() for l in m.layers]                  # (d, H·D)
        self.wa = [1 + l.post_attention_layernorm.weight.float() for l in m.layers]
        self.eps_a = [l.post_attention_layernorm.eps for l in m.layers]
        self.U = runner.W_letters.float() * (1 + m.norm.weight.float())                  # (4, d)
        self.eps_f = m.norm.eps

    @torch.inference_mode()
    def __call__(self, cap):
        """(n_layers, H, 4) letter DLA, plus the max relative error of the per-head sum vs the
        captured attention output and the linearized final logits (pre soft-cap)."""
        rf = cap[("resid", self.n - 1)].float()
        s_f = torch.rsqrt(rf.pow(2).mean() + self.eps_f)
        out, err = [], 0.0
        for l in range(self.n):
            z = cap[("head", l)].float().view(self.H, self.D)
            per = torch.einsum("dhk,hk->hd", self.WO[l].view(-1, self.H, self.D), z)        # (H, d)
            o = per.sum(0)
            s_a = torch.rsqrt(o.pow(2).mean() + self.eps_a[l])
            normed = per * s_a * self.wa[l]
            ref = cap[("attn", l)].float()
            err = max(err, ((normed.sum(0) - ref).norm() / ref.norm()).item())
            out.append((normed @ self.U.T) * s_f)
        lin = (self.U @ rf) * s_f
        return torch.stack(out).cpu().numpy(), err, lin.cpu().numpy()


def lead(x, k):
    """letter k minus the mean of the other three; x (..., 4)."""
    return x[..., k] - np.delete(x, k, axis=-1).mean(-1)


# ---------------------------------------------------------------- plots

def plot_dla_bars(d, heads, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), dpi=150, sharex=True)
    x = np.arange(len(heads))
    for ax, col, title in [(axes[0], "dla_orig", "toward the original (correct) answer"),
                           (axes[1], "dla_target", "toward the letter chosen under assert")]:
        for j, cond in enumerate(COND_COLORS):
            e = d[d.cond == cond].set_index("head_name").loc[heads]
            ax.bar(x + (j - 1) * 0.27, e[col], 0.25, yerr=e[f"{col}_se"], color=COND_COLORS[cond],
                   label=f"{cond} run", edgecolor="white", linewidth=1, error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(f"Direct logit attribution {title}: letter minus mean of the other three "
                     f"(n={int(d.n_pairs.iloc[0])} questions, mean ± SE)", color=INK, fontsize=9, loc="left")
        style(ax, "logit (pre soft-cap)")
        ax.set_xlabel("")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3)
    axes[1].set_xticks(x, heads)
    axes[1].set_xlabel("head", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dla_heatmap(diffs, path):
    vmax = max(np.abs(g).max() for _, g in diffs)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ee", BLUE])
    fig, axes = plt.subplots(1, len(diffs), figsize=(4.2 * len(diffs), 7.5), dpi=150, squeeze=False)
    for ax, (title, g) in zip(axes[0], diffs):
        im = ax.imshow(g, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(g.shape[0]), [f"L{l + 1}" for l in range(g.shape[0])], fontsize=7)
        ax.set_xticks(range(g.shape[1]), range(g.shape[1]))
        ax.set_xlabel("head", color=INK)
        ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, color=INK, fontsize=8.5, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


GROUP_STYLE = {"final '('": (INK, "-"), "prefill, non-final": (BLUE, "-"), "template": (AQUA, "-"),
               "message": (ORANGE, "-"), "'Final'": (BLUE, ":"), "' answer'": (YELLOW, "-"), "':'": (BLUE, "--")}


def plot_positions(p2, n, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=150, sharex=True)
    panels = [["final '('", "prefill, non-final", "template", "message"], ["'Final'", "' answer'", "':'"]]
    for col, groups in enumerate(panels):
        for row, (val, yl) in enumerate([("recovery", "recovery (final answer)"),
                                         ("h3_shift", "L16.H3 output shift toward doubt run")]):
            ax = axes[row, col]
            for g in groups:
                d = p2[p2.group == g].sort_values("layer_1idx")
                c, ls = GROUP_STYLE[g]
                ax.fill_between(d.layer_1idx, d[val] - d[f"{val}_se"], d[val] + d[f"{val}_se"], color=c, alpha=0.15, linewidth=0)
                ax.plot(d.layer_1idx, d[val], color=c, linestyle=ls, linewidth=2, marker="o", markersize=3, label=g)
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
            style(ax, yl)
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="center left" if col == 0 else "upper left")
            if row == 0:
                ax.set_title(("Position groups" if col == 0 else "Single prefill tokens (non-final)")
                             + f": doubt → assert residual patch (n={n}, mean ± SE)", color=INK, fontsize=9, loc="left")
    for ax in axes[1]:
        ax.text(16.3, 0.97, "L16 output comes after L16.H3,\nso its shift there is 0", color=MUTED, fontsize=7,
                ha="right", va="top", transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_kv(kv, ref, path):
    sets = ["prefill, non-final (3)", "prefill (4)", "message + template + prefill"]
    variants = ["K+V", "V only", "K only"]
    colors = [BLUE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    x = np.arange(len(sets))
    w = 0.8 / len(variants)
    for j, v in enumerate(variants):
        e = kv[kv.variant == v].set_index("positions").loc[sets]
        ax.bar(x + (j - (len(variants) - 1) / 2) * w, e.recovery, w * 0.92, yerr=e.recovery_se, color=colors[j],
               label=v, edgecolor="white", error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.axhline(ref, color=INK, linestyle="--", linewidth=1.2, label=f"whole L16.H3 output patched ({ref:.2f})")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x, sets)
    style(ax, "recovery")
    ax.set_xlabel("positions whose key/value input is taken from the doubt run", color=INK)
    ax.set_title(f"L16.H3 key/value-input patching, doubt → assert (n={int(kv.n_pairs.iloc[0])}, mean ± SE)",
                 color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records.jsonl")
    ap.add_argument("--pairs", default="results/patch_residual/pairs.csv")
    ap.add_argument("--max-pairs", type=int, default=None, help="per pair type (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/patch_mechanism")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "run_log.txt"))

    recs = {r["row"]: r for r in map(json.loads, open(args.records))}
    pdf = pd.read_csv(args.pairs)
    pdf = pdf[pdf.src_ok & pdf.assert_ok]
    pairs = {c: [recs[r] for r in pdf[pdf.pair == c].row] for c in PAIR_TYPES}
    if args.max_pairs:
        rng = np.random.default_rng(args.seed)
        pairs = {c: [p[i] for i in sorted(rng.choice(len(p), min(args.max_pairs, len(p)), replace=False))]
                 for c, p in pairs.items()}
    in_set = {c: {r["row"] for r in p} for c, p in pairs.items()}
    union = sorted(set().union(*in_set.values()))
    print(f"Pairs: " + ", ".join(f"{PAIR_TYPES[c]} {len(p)}" for c, p in pairs.items())
          + f"; {len(union)} distinct questions")

    hp = pd.read_csv("results/patch_heads/stage2_heads.csv")
    top_heads = sorted(set(hp.head_name) | set(NEG_HEADS), key=lambda s: (int(s[1:].split(".")[0]), int(s.split("H")[1])))
    print(f"Top heads (patch_heads stage 2 union + {', '.join(NEG_HEADS)}): {', '.join(top_heads)}")

    items = {}
    for name in sorted({recs[r]["dataset"] for r in union}):
        for x in load_dataset("allenai/ai2_arc", name, split="test"):
            items[(name, x["id"])] = x
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    runner = Phase1Runner(torch.bfloat16, device)
    patcher = MechPatcher(runner, H3[0])
    dla = DLA(runner)
    D = patcher.head_dim
    kv_head = H3[1] // (dla.H // runner.model.config.num_key_value_heads)
    h3_sl = slice(H3[1] * D, (H3[1] + 1) * D)
    print(f"Model on {device}; L16.H3 = query head {H3[1]}, reads KV head {kv_head} "
          f"(shared with query head {kv_head * 2 + (1 - H3[1] % 2)})")

    # ---- part 1
    print("\n=== Part 1: direct logit attribution at the final position ===")
    rows, errs, cap_err, skipped, t0 = [], [], [], 0, time.time()
    for k, r in enumerate(union):
        rec = recs[r]
        item = items[(rec["dataset"], rec["id"])]
        prompts = {**build_prompts(runner, item, rec["initial"], "doubt"),
                   **build_prompts(runner, item, rec["initial"], "neutral")}
        orig, target = LETTERS.index(rec["initial"]), LETTERS.index(rec["assert"]["final"])
        for cond in ("assert", "doubt", "neutral"):
            logits, cap = patcher.clean(prompts[cond])
            if LETTERS[int(logits.argmax())] != rec[cond]["final"]:
                skipped += 1
                continue
            g, e, lin = dla(cap)
            errs.append(e)
            cap_err.append(np.abs(30 * np.tanh(lin / 30) - logits).max())
            lo, lt = lead(g, orig), lead(g, target)
            for l in range(dla.n):
                for h in range(dla.H):
                    rows.append({"row": r, "in_doubt_pairs": r in in_set["doubt"], "in_neutral_pairs": r in in_set["neutral"],
                                 "cond": cond, "layer_1idx": l + 1, "head": h, "head_name": hname(l, h),
                                 "dla_orig": lo[l, h], "dla_target": lt[l, h]})
        if (k + 1) % 100 == 0 or k + 1 == len(union):
            print(f"  {k + 1}/{len(union)} questions ({time.time() - t0:.0f}s)")
    p1 = pd.DataFrame(rows)
    print(f"  Runs not reproducing the recorded answer (skipped): {skipped}")
    print(f"  Check: sum of per-head contributions vs captured post_attention_layernorm output, max relative "
          f"error {max(errs):.2g} (bf16 rounding); soft-capped linearized final logits vs real logits, max |diff| "
          f"{max(cap_err):.2g}")
    p1.to_csv(os.path.join(args.out_dir, "part1_dla_per_run.csv"), index=False)
    sets = {"union": p1, "doubt pairs": p1[p1.in_doubt_pairs], "neutral pairs": p1[p1.in_neutral_pairs]}
    p1_agg = pd.concat([agg(d, ["cond", "layer_1idx", "head", "head_name"], ["dla_orig", "dla_target"]).assign(set=s)
                        for s, d in sets.items()])
    p1_agg.to_csv(os.path.join(args.out_dir, "part1_dla.csv"), index=False)
    plot_dla_bars(p1_agg[p1_agg.set == "union"], top_heads, os.path.join(args.out_dir, "part1_dla_top_heads.png"))
    diffs = []
    for cond in ("doubt", "neutral"):
        d = sets[f"{cond} pairs"]
        piv = d.pivot_table(index=["row", "layer_1idx", "head"], columns="cond", values="dla_orig").dropna()
        g = (piv["assert"] - piv[cond]).groupby(level=["layer_1idx", "head"]).mean().unstack().to_numpy()
        diffs.append((f"assert − {cond} run, original-letter DLA\n({PAIR_TYPES[cond]} questions; "
                      f"orange = assert pushes away more)", g))
    plot_dla_heatmap(diffs, os.path.join(args.out_dir, "part1_dla_assert_minus_source.png"))

    # ---- part 2
    print("\n=== Part 2: doubt → assert position patching (layers 1–16) and L16.H3 key/value inputs ===")
    p2_rows, kv_rows, check, t0 = [], [], [], time.time()
    old = pd.read_csv("results/patch_residual/stage1_resid_doubt_per_pair.csv").set_index(["row", "layer_1idx"]).recovery
    for k, rec in enumerate(pairs["doubt"]):
        item = items[(rec["dataset"], rec["id"])]
        P = setup_pair(patcher, rec, item, "doubt")
        if P is None:
            print(f"  SKIP row {rec['row']}: clean run gave a different answer")
            continue
        src_t, asr_t = P["prompts"]["doubt"], P["prompts"]["assert"]
        _, _, full = patcher.go(src_t, full=True)
        q = format_question(item)
        lab_s, _, _, ids_s = token_groups(runner.tok, src_t, q, rec["initial"], CONDITIONS["doubt"])
        lab_a, _, _, ids_a = token_groups(runner.tok, asr_t, q, rec["initial"], CONDITIONS["assert"])
        n_tail = N_PREFILL + N_TEMPLATE
        assert ids_s[-n_tail:] == ids_a[-n_tail:], "template/prefill tokens differ"
        n_msg = min((lab_s == "turn2_msg").sum(), (lab_a == "turn2_msg").sum())
        assert all(lab_s[-n_tail - 1 - i] == "turn2_msg" and lab_a[-n_tail - 1 - i] == "turn2_msg" for i in range(n_msg))
        Ts, Ta = len(ids_s), len(ids_a)
        groups = {"'Final'": [-4], "' answer'": [-3], "':'": [-2], "final '('": [-1],
                  "prefill, non-final": [-4, -3, -2], "template": list(range(-n_tail, -N_PREFILL)),
                  "message": list(range(-n_tail - n_msg, -n_tail))}
        z_asr, z_src = P["asr_cap"][("head", H3[0])][h3_sl].float(), P["src_cap"][("head", H3[0])][h3_sl].float()
        dz = z_src - z_asr

        def shift(cap):
            return float(((cap[("head", H3[0])][h3_sl].float() - z_asr) @ dz / (dz @ dz)).item())

        def rec_of(logits):
            return float((metric(logits, P["orig"])[0] - P["m_asr"]) / P["gap"])

        for L in PATCH_LAYERS:
            src_l = full[("rfull", L - 1)]
            for g, offs in groups.items():
                logits, cap, _ = patcher.go(asr_t, pos={("rfull", L - 1): [(Ta + o, slice(None), src_l[Ts + o]) for o in offs]})
                p2_rows.append({"row": rec["row"], "layer_1idx": L, "group": g, "n_positions": len(offs),
                                "recovery": rec_of(logits), "reverted": float(logits.argmax() == P["orig"]),
                                "h3_shift": shift(cap) if L < H3[0] + 1 else 0.0})
                if g == "final '('":
                    check.append(abs(p2_rows[-1]["recovery"] - old.loc[(rec["row"], L)]))

        # L16.H3 key/value inputs
        kv_sl = slice(kv_head * D, (kv_head + 1) * D)
        logits, _, _ = patcher.go(asr_t, heads={("head", H3[0]): [(H3[1], z_src.to(z_asr.device).to(torch.bfloat16))]})
        whole = rec_of(logits)
        pos_sets = {"prefill (4)": groups["'Final'"] + groups["' answer'"] + groups["':'"] + groups["final '('"],
                    "prefill, non-final (3)": groups["prefill, non-final"],
                    "message + template + prefill": groups["message"] + groups["template"] + [-4, -3, -2, -1]}
        for ps, offs in pos_sets.items():
            for variant, keys in [("K+V", ("k", "v")), ("V only", ("v",)), ("K only", ("k",))]:
                pos = {(kk, H3[0]): [(Ta + o, kv_sl, full[(kk, H3[0])][Ts + o, kv_sl]) for o in offs] for kk in keys}
                l_shared, cap, _ = patcher.go(asr_t, pos=pos)
                z3 = cap[("head", H3[0])][h3_sl]
                l_only, _, _ = patcher.go(asr_t, heads={("head", H3[0]): [(H3[1], z3)]})
                kv_rows.append({"row": rec["row"], "positions": ps, "variant": variant,
                                "recovery": rec_of(l_only), "reverted": float(l_only.argmax() == P["orig"]),
                                "h3_shift": shift(cap), "recovery_kv_head_shared": rec_of(l_shared),
                                "whole_h3_recovery": whole})
        if (k + 1) % 25 == 0 or k + 1 == len(pairs["doubt"]):
            print(f"  {k + 1}/{len(pairs['doubt'])} pairs ({time.time() - t0:.0f}s)")
    print(f"  Check: patching the final position here vs patch_residual.py stage 1, max |Δrecovery| {max(check):.2g}")
    p2_long, kv_long = pd.DataFrame(p2_rows), pd.DataFrame(kv_rows)
    p2 = agg(p2_long, ["group", "layer_1idx"], ["recovery", "reverted", "h3_shift"])
    kv = agg(kv_long, ["positions", "variant"], ["recovery", "reverted", "h3_shift", "recovery_kv_head_shared", "whole_h3_recovery"])
    p2_long.to_csv(os.path.join(args.out_dir, "part2_positions_per_pair.csv"), index=False)
    kv_long.to_csv(os.path.join(args.out_dir, "part2_h3_kv_per_pair.csv"), index=False)
    p2.to_csv(os.path.join(args.out_dir, "part2_positions.csv"), index=False)
    kv.to_csv(os.path.join(args.out_dir, "part2_h3_kv.csv"), index=False)
    n2 = p2_long.row.nunique()
    plot_positions(p2, n2, os.path.join(args.out_dir, "part2_position_patching.png"))
    plot_kv(kv, kv_long.whole_h3_recovery.mean(), os.path.join(args.out_dir, "part2_h3_kv.png"))
    print(f"\nWrote CSVs, plots and run_log.txt to {args.out_dir}/")

    summarize(p1_agg, sets, top_heads, p2, kv, kv_long, n2)


def summarize(p1_agg, sets, top_heads, p2, kv, kv_long, n2):
    print("\n=== SUMMARY (layers 1-indexed, heads 0–7) ===")
    print("\nPart 1: what each head writes directly onto the answer letters at the final position "
          "(original letter / letter chosen under assert, each minus the mean of the other three; "
          "pre soft-cap logits, all pair questions):")
    u = p1_agg[p1_agg.set == "union"].set_index(["head_name", "cond"])
    for h in top_heads:
        a, d, n = u.loc[(h, "assert")], u.loc[(h, "doubt")], u.loc[(h, "neutral")]
        diffs = []
        for cond in ("doubt", "neutral"):
            s = sets[f"{cond} pairs"]
            piv = s[s.head_name == h].pivot_table(index="row", columns="cond", values="dla_orig").dropna()
            dd = piv["assert"] - piv[cond]
            diffs.append(f"assert−{cond} {dd.mean():+.2f}±{dd.std() / np.sqrt(len(dd)):.2f}")
        if a.dla_orig < -0.1 and a.dla_target > 0.1:
            verdict = "pushes toward the new answer under assert"
        elif a.dla_orig > 0.1:
            verdict = "supports the original answer under assert"
        else:
            verdict = "little direct effect under assert"
        print(f"  {h}: original {a.dla_orig:+.2f} / {d.dla_orig:+.2f} / {n.dla_orig:+.2f} (assert/doubt/neutral), "
              f"new letter {a.dla_target:+.2f} / {d.dla_target:+.2f} / {n.dla_target:+.2f}; {', '.join(diffs)} → {verdict}")
    a = p1_agg[(p1_agg.set == "doubt pairs")]
    piv = a.pivot_table(index="head_name", columns="cond", values="dla_orig")
    big = (piv["assert"] - piv["doubt"]).sort_values()
    print("  Largest assert−doubt differences in original-letter DLA over all 208 heads (doubt pairs): "
          + ", ".join(f"{h} {v:+.2f}" for h, v in big.head(5).items())
          + " … " + ", ".join(f"{h} {v:+.2f}" for h, v in big.tail(3).items()))

    print(f"\nPart 2 (doubt→assert, n={n2}): residual stream patched from the doubt run at aligned positions.")
    for g in ["final '('", "prefill, non-final", "'Final'", "' answer'", "':'", "template", "message"]:
        d = p2[p2.group == g].set_index("layer_1idx")
        best = d.recovery.idxmax()
        h3 = d.h3_shift.loc[:15]
        print(f"  {g:20s}: recovery " + ", ".join(f"L{l} {d.recovery[l]:+.2f}" for l in (4, 8, 12, 13, 14, 15, 16))
              + f" (max L{best} {d.recovery[best]:+.2f}, reverts {d.reverted[best]:.0%}); L16.H3 shift max "
              f"L{h3.idxmax()} {h3.max():+.2f}")
    peaks = {g: p2[p2.group == g].set_index("layer_1idx").recovery.idxmax()
             for g in ["message", "template", "prefill, non-final", "final '('"]}
    print("  → Layer where patching each group recovers most (where the doubt-vs-assert content sits): "
          + ", ".join(f"{g} L{l}" for g, l in peaks.items()))
    for g in ["template", "prefill, non-final"]:
        s = p2[p2.group == g].set_index("layer_1idx").h3_shift.loc[:15]
        on = [l for l in s.index if s[l] >= 0.5 * s.max()]
        print(f"    {g}: patching it moves L16.H3's output ≥ half of its peak ({s.max():+.2f} at L{s.idxmax()}) "
              f"at L{on[0]}–{on[-1]}")
    print("  L16.H3 key/value inputs only (head output rerouted so only L16.H3 changes):")
    for r in kv.itertuples():
        print(f"    {r.positions:30s} {r.variant:7s}: recovery {r.recovery:+.2f} ± {r.recovery_se:.2f}, reverts {r.reverted:.0%}, "
              f"L16.H3 shift {r.h3_shift:+.2f} (KV head shared with H2: {r.recovery_kv_head_shared:+.2f})")
    print(f"    reference: whole L16.H3 output patched {kv_long.whole_h3_recovery.mean():+.2f}")


if __name__ == "__main__":
    main()
