#!/usr/bin/env python
"""
Evaluate random-guess baseline on lexical-overlap control set.
"""

import argparse, pathlib, random
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------- utils -----------------------------------------------------------
def load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_json(p, lines=True) if p.suffix.endswith("jsonl") else pd.read_csv(p)

def map_label(col: pd.Series) -> pd.Series:
    return col.map({"non_entailment": 0, "entailment": 1})

# ---------- main ------------------------------------------------------------
def main(args):
    random.seed(42)

    df = load_table(args.eval_file)
    y_true = map_label(df["label"]).tolist()
    y_pred = [random.choice([0, 1]) for _ in y_true]

    print("\n=== RANDOM BASELINE EVALUATION ===")
    print(classification_report(
        y_true, y_pred, digits=3,
        target_names=["non_entailment", "entailment"]
    ))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # -------- per-subcase accuracy -----------------------------------------
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    print("\n=== PER-SUBCASE ACCURACY ===")
    for sub in sorted(df["subcase"].unique()):
        sub_df = df[df["subcase"] == sub]
        acc = accuracy_score(sub_df["y_true"], sub_df["y_pred"])
        print(f"{sub:15s}:  accuracy = {acc:.3f}  "
              f"({len(sub_df)} samples)")

# ---------- entry -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default="../data/data/lexical_overlap_control_eval.jsonl")
    args = ap.parse_args()
    main(args)
