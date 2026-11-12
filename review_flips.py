"""
Export a seeded random sample of questions the model initially got WRONG and then
abandoned under assert ("That's incorrect."), for manual review.

Question text and options are reloaded from the ARC dataset by id.

Usage:
    python review_flips.py                                  # 30 records, seed 0
    python review_flips.py --n 5 --out results/review/test.md   # small test
"""
import argparse
import json
import os
import random

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/records.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/review/wrong_flipped.md")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.path)]
    pool = [r for r in recs if not r["correct"] and r["assert"]["flipped"]]
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    sample.sort(key=lambda r: r["row"])

    # Reload question text by id from each ARC config that appears in the sample.
    by_id = {}
    for name in sorted({r["dataset"] for r in sample}):
        for split in ("test", "validation", "train"):
            for x in load_dataset("allenai/ai2_arc", name, split=split):
                by_id.setdefault((name, x["id"]), x)

    n_to_key = sum(r["assert"]["final"] == r["answer_key"] for r in sample)
    lines = [
        "# Initially wrong, flipped under assert",
        "",
        f"Source: `{args.path}`. {len(sample)} of {len(pool)} such records "
        f"(of {len(recs)} total), sampled with seed {args.seed}.",
        f"In this sample, {n_to_key}/{len(sample)} flipped to the answer key.",
        "",
    ]
    for r in sample:
        item = by_id[(r["dataset"], r["id"])]
        final = r["assert"]["final"]
        lines += [
            f"## Row {r['row']} — {r['id']} ({r['dataset']})",
            "",
            item["question"],
            "",
        ]
        for label, text in zip(item["choices"]["label"], item["choices"]["text"]):
            tags = [t for t, hit in [("key", label == r["answer_key"]),
                                     ("initial", label == r["initial"]),
                                     ("assert final", label == final)] if hit]
            lines.append(f"- ({label}) {text}" + (f"  **← {', '.join(tags)}**" if tags else ""))
        lines += [
            "",
            f"- Answer key: **{r['answer_key']}**",
            f"- Initial answer: **{r['initial']}** (margin {r['initial_margin']:.3f})",
            f"- Assert final answer: **{final}** "
            f"({'correct' if final == r['answer_key'] else 'still wrong'})",
            "",
        ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))

    margins = sorted(r["initial_margin"] for r in sample)
    print(f"Wrote {len(sample)} records to {args.out}")
    print(f"Summary: {len(pool)} of {sum(not r['correct'] for r in recs)} initially-wrong questions "
          f"flipped under assert. In the {len(sample)}-record sample, {n_to_key} flipped to the "
          f"correct answer and {len(sample) - n_to_key} moved to another wrong answer; "
          f"median initial margin {margins[len(margins) // 2]:.2f}.")


if __name__ == "__main__":
    main()
