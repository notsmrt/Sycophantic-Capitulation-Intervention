"""
Confidence-matched check: is the model more willing to abandon wrong answers
only because it was less confident in them?

Works on Phase 1 results.jsonl and Phase 2 data/records.jsonl.

Usage:
    pip install statsmodels
    python confidence_regression.py                               # Phase 1 results.jsonl
    python confidence_regression.py --path data/records.jsonl
    python confidence_regression.py --path data/records.jsonl --dataset ARC-Challenge
    python confidence_regression.py --path data/records.jsonl --dataset-control  # + dataset covariate
    python confidence_regression.py --path data/records.jsonl --quadratic        # + confidence² covariate

Three tests per condition:
  1. Flip rates for right vs wrong answers within confidence bins.
  2. Logistic regression: flipped ~ correct + confidence.
     If 'correct' still predicts flipping with confidence controlled,
     the model carries a correctness signal its confidence doesn't show.
  3. Among flipped runs only: late-layer score ~ correct + confidence.
     Tests whether "pushed further when wrong" survives the same control.
"""
import argparse
import json

import numpy as np
import statsmodels.api as sm

LETTERS = ["A", "B", "C", "D"]
LATE_LAYERS = range(18, 25)  # 0-indexed → layers 19–25, after the decision zone


def confidence(rec):
    """Initial confidence on a log scale. Uses the logit margin if saved, else log-odds."""
    if "initial_margin" in rec:
        return rec["initial_margin"]
    p = min(max(rec["initial_conf"], 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def late_score(rec, cond):
    lens = np.array(rec[cond]["lens"])
    init = LETTERS.index(rec["initial"])
    other = np.delete(lens, init, axis=1).max(axis=1)
    return float((lens[:, init] - other)[list(LATE_LAYERS)].mean())


def binned_rates(recs, cond, n_bins):
    conf = np.array([confidence(r) for r in recs])
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    print(f"\n  Flip rate by confidence bin (quantiles of all questions):")
    print(f"  {'confidence range':>22}{'right n':>9}{'right flip':>12}{'wrong n':>9}{'wrong flip':>12}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = [r for r, c in zip(recs, conf) if lo <= c <= hi]
        out = []
        for want in (True, False):
            g = [r for r in rows if r["correct"] == want]
            rate = f"{np.mean([r[cond]['flipped'] for r in g]):.1%}" if g else "—"
            out += [len(g), rate]
        print(f"  {lo:>10.2f} to {hi:<8.2f}{out[0]:>9}{out[1]:>12}{out[2]:>9}{out[3]:>12}")
    print("  (Bins with few wrong answers are unreliable; the regression below uses all data.)")


def design(recs, quadratic, dataset_control):
    """Covariates besides 'correct', as (names, columns). Confidence is z-scored."""
    conf = np.array([confidence(r) for r in recs])
    conf_z = (conf - conf.mean()) / conf.std()
    names, cols = ["confidence (z)"], [conf_z]
    if quadratic:
        names.append("confidence² (z²)")
        cols.append(conf_z ** 2)
    if dataset_control:
        ds = np.array([r.get("dataset") == "ARC-Challenge" for r in recs], dtype=float)
        if 0 < ds.sum() < len(ds):
            names.append("ARC-Challenge")
            cols.append(ds)
    return names, cols


def fit(y, correct, covs, kind):
    X = sm.add_constant(np.column_stack([correct.astype(float)] + covs))
    if kind == "logit":
        return sm.Logit(y, X).fit(disp=0)
    return sm.OLS(y, X).fit()


def report(model, kind, cov_names):
    names = ["intercept", "correct"] + cov_names
    ci = model.conf_int()
    for i, name in enumerate(names[1:], start=1):
        coef, (lo, hi), p = model.params[i], ci[i], model.pvalues[i]
        extra = f"   odds ratio {np.exp(coef):.2f}" if kind == "logit" else ""
        print(f"    {name:18} {coef:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  p={p:.2g}{extra}")


def verdict(model, kind, controls):
    """One-sentence reading of the 'correct' coefficient, based on the actual fit."""
    coef, p = model.params[1], model.pvalues[1]
    sig = p < 0.05
    status = f"{'significant' if sig else 'not significant'} (p={p:.2g}), {'negative' if coef < 0 else 'positive'}"
    if kind == "logit":
        if not sig:
            meaning = (f"no evidence that correctness affects flipping after controlling for {controls} "
                       f"(odds ratio {np.exp(coef):.2f}).")
        elif coef < 0:
            meaning = (f"right answers are abandoned less than wrong ones at equal {controls} "
                       f"(odds ratio {np.exp(coef):.2f}), so the model carries a correctness "
                       "signal its confidence doesn't show.")
        else:
            meaning = (f"right answers are abandoned MORE than wrong ones at equal {controls} "
                       f"(odds ratio {np.exp(coef):.2f}).")
    else:
        if not sig:
            meaning = (f"no evidence that wrong answers are pushed further than right ones "
                       f"after controlling for {controls}.")
        elif coef > 0:
            meaning = (f"wrong answers are pushed further from the initial answer than right "
                       f"ones, even at equal {controls}.")
        else:
            meaning = (f"right answers are pushed further from the initial answer than wrong "
                       f"ones, even at equal {controls}.")
    print(f"    Verdict: 'correct' is {status}: {meaning}")
    return {"coef": float(coef), "p": float(p), "significant": bool(sig)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results.jsonl")
    ap.add_argument("--conds", nargs="+", default=["assert", "doubt"])
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--dataset", choices=["ARC-Easy", "ARC-Challenge", "all"], default="all")
    ap.add_argument("--dataset-control", action="store_true",
                    help="add dataset (ARC-Challenge vs ARC-Easy) as a covariate in both regressions")
    ap.add_argument("--quadratic", action="store_true",
                    help="add confidence² as a covariate in both regressions")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.path)]
    if args.dataset != "all":
        recs = [r for r in recs if r.get("dataset") == args.dataset]
        if not recs:
            raise SystemExit(f"No records with dataset == {args.dataset} in {args.path}")
    datasets = sorted({r.get("dataset", "?") for r in recs})
    print(f"Dataset filter: {args.dataset}  (present: {', '.join(datasets)})")
    if args.dataset_control and len(datasets) < 2:
        print("NOTE: --dataset-control requested but only one dataset is present, so dataset is "
              "constant and cannot be a covariate; it is omitted (results equal the uncontrolled fit).")
    controls = "confidence" + (" and confidence²" if args.quadratic else "") + \
        (" and dataset" if args.dataset_control and len(datasets) > 1 else "")
    summary = {}
    conf_all = np.array([confidence(r) for r in recs])
    correct_all = np.array([r["correct"] for r in recs])
    print(f"{len(recs)} questions: {correct_all.sum()} right, {(~correct_all).sum()} wrong")
    print(f"Mean confidence (log scale): right {conf_all[correct_all].mean():.2f}, "
          f"wrong {conf_all[~correct_all].mean():.2f}")

    for cond in args.conds:
        print(f"\n=== {cond.upper()} ===")
        binned_rates(recs, cond, args.bins)

        y = np.array([r[cond]["flipped"] for r in recs], dtype=float)
        names, covs = design(recs, args.quadratic, args.dataset_control)
        print(f"\n  Logistic regression: flipped ~ correct + {' + '.join(names)}")
        m = fit(y, correct_all, covs, "logit")
        report(m, "logit", names)
        summary[cond] = {"flip": verdict(m, "logit", controls)}

        flipped = [r for r in recs if r[cond]["flipped"]]
        if len(flipped) > 10 and len({r["correct"] for r in flipped}) == 2:
            s = np.array([late_score(r, cond) for r in flipped])
            c = np.array([r["correct"] for r in flipped])
            names, covs = design(flipped, args.quadratic, args.dataset_control)
            print(f"\n  Among flipped runs (n={len(flipped)}): late-layer score ~ correct + {' + '.join(names)}")
            m = fit(s, c, covs, "ols")
            report(m, "ols", names)
            summary[cond]["late"] = verdict(m, "ols", controls)

    print(f"\n=== SUMMARY (controls: {controls}; n={len(recs)}) ===")
    for cond, res in summary.items():
        f = res["flip"]
        line = (f"  {cond}: 'correct' → flipping {'survives' if f['significant'] else 'does NOT survive'} "
                f"(coef {f['coef']:+.3f}, OR {np.exp(f['coef']):.2f}, p={f['p']:.2g})")
        if "late" in res:
            l = res["late"]
            line += (f"; late-layer push {'significant' if l['significant'] else 'not significant'} "
                     f"(p={l['p']:.2g})")
        print(line)


if __name__ == "__main__":
    main()
