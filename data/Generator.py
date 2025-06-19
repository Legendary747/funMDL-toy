#!/usr/bin/env python
"""
Generate a control dataset to probe the Lexical-Overlap Heuristic,
mirroring the HANS paper (SO_Swap / Passive_Swap / Passive_Entail).

Usage:
    python generate_overlap_control.py --eval-size 500 --train-size 200
"""
import json, random, argparse
import pandas as pd
from pathlib import Path

random.seed(42)

# ---------- verb dictionary: base -> past ----------
VERBS = {
    "pay": "paid", "chase": "chased", "praise": "praised", "beat": "beat",
    "help": "helped", "blame": "blamed", "call": "called", "follow": "followed",
    "visit": "visited", "meet": "met", "invite": "invited", "teach": "taught",
    "bring": "brought", "send": "sent", "show": "showed", "choose": "chose",
    "hold": "held", "build": "built", "catch": "caught", "buy": "bought",
    "write": "wrote", "read": "read", "open": "opened", "close": "closed",
    "carry": "carried", "raise": "raised", "offer": "offered",
    "deliver": "delivered", "clean": "cleaned", "repair": "repaired"
}

# ---------- name list ----------
NAMES = pd.read_csv("boy_names_2001.csv")["Name"].str.strip().tolist()[:30]

def gen_triplet():
    """Return (A, B, verb_base, verb_past) with A != B."""
    A, B = random.sample(NAMES, 2)
    base, past = random.choice(list(VERBS.items()))
    return A, B, base, past


def build_examples(n_pairs: int, eval: bool = False):
    examples = []
    for _ in range(n_pairs):
        A, B, base, past = gen_triplet()

        # premise用被动句确保词汇一致
        premise = f"{A} {base} {B} before dinner."

        # Passive_Swap (non-entail, 主宾调换)
        examples.append({
            "premise": premise,
            "hypothesis": f"{B} was {past} by {A} before dinner.",
            "label": "non_entailment",
            "subcase": "Passive_Swap"
        })

        # Passive_Entail (entail, 与premise完全相同)
        examples.append({
            "premise": premise,
            "hypothesis": f"{A} was {past} by {B} before dinner.",
            "label": "entailment",
            "subcase": "Passive_Entail"
        })

        # Passive_Swap (non-entail, 主宾调换)
        examples.append({
            "premise": premise,
            "hypothesis": f"{B} {base} {A} before dinner.",
            "label": "non_entailment",
            "subcase": "Simple_Swap"
        })

    random.shuffle(examples)
    return examples


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"✔ Saved {len(rows):>4} lines ➜ {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-size", type=int, default=500,
                        help="Number of premise templates for evaluation split")
    parser.add_argument("--train-size", type=int, default=500,
                        help="Templates for optional augmentation split")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Output folder")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    # 1) evaluation set: SO_Swap + Passive_Swap only
    eval_rows = build_examples(args.eval_size, eval=True)
    save_jsonl(eval_rows, out_dir / "lexical_overlap_control_eval.jsonl")

    # 2) (optional) augmentation set
    if args.train_size > 0:
        train_rows = build_examples(args.train_size, eval=False)
        save_jsonl(train_rows, out_dir / "lexical_overlap_control_train.jsonl")
