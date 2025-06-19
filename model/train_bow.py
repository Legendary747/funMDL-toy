#!/usr/bin/env python
"""
Train a unigram Bag-of-Words + LogisticRegression baseline,
then evaluate on a lexical-overlap control set.

Example:
    python train_bow.py \
        --train-file ../data/mnli_train.csv \
        --eval-file  ../data/lexical_overlap_control_eval.jsonl
"""

import argparse
import os
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ---------- helper ----------------------------------------------------------
def load_table(path: str) -> pd.DataFrame:
    """Auto-detect CSV vs JSONL and return DataFrame."""
    p = pathlib.Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in {".jsonl", ".json"}:
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported file type: {p.suffix!r}")


def concat_prem_hypo(df: pd.DataFrame) -> pd.Series:
    """Premise + [SEP] + Hypothesis string."""
    return df["premise"] + " [SEP] " + df["hypothesis"]


def map_label(series: pd.Series) -> pd.Series:
    """Map textual label → int(0/1)."""
    return series.map({
        "non_entailment": 0, "contradiction": 0, "neutral": 0,
        "entailment": 1
    })


# ---------- main ------------------------------------------------------------
def main():
    # 1 load training data
    train_df = load_table(args.train_file)
    X_all = concat_prem_hypo(train_df).to_numpy()
    y_all = map_label(train_df["label"]).to_numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all,
        test_size=args.test_size,
        stratify=y_all,
        random_state=42
    )

    # 2) vectorise (unigram) & train
    vec = CountVectorizer(ngram_range=(1, 1))
    X_tr_vec = vec.fit_transform(X_tr)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        verbose=1
    ).fit(X_tr_vec, y_tr)

    # 3) quick validation on held-out slice
    print("\n=== Validation on held-out slice of TRAIN ===")
    val_preds = clf.predict(vec.transform(X_val))
    print(classification_report(
        y_val, val_preds, digits=3,
        target_names=["non_entailment", "entailment"]
    ))

    # 4) load control evaluation set
    eval_df = load_table(args.eval_file)
    X_eval = concat_prem_hypo(eval_df).to_numpy()
    y_eval = map_label(eval_df["label"]).to_numpy()

    # 5) batch predict with progress bar
    print("\n=== Evaluation on CONTROL set ===")
    batch = 1024
    preds = []
    for i in tqdm(range(0, len(X_eval), batch), desc="Predicting"):
        preds.extend(clf.predict(vec.transform(X_eval[i:i + batch])))

    print(classification_report(
        y_eval, preds, digits=3,
        target_names=["non_entailment", "entailment"]
    ))
    # convert eval_df to include preds
    eval_df["y_true"] = y_eval
    eval_df["y_pred"] = preds

    # print per-subcase accuracy
    print("\n=== PER-SUBCASE ACCURACY ===")
    for sub in sorted(eval_df["subcase"].unique()):
        sub_df = eval_df[eval_df["subcase"] == sub]
        acc = (sub_df["y_true"] == sub_df["y_pred"]).mean()
        print(f"{sub:15s}:  accuracy = {acc:.3f}  ({len(sub_df)} samples)")


# ---------- entry point -----------------------------------------------------
if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default="../data/data/lexical_overlap_control_train.jsonl",
        help="CSV or JSONL used for training"
    )
    parser.add_argument(
        "--eval-file",
        default="../data/data/lexical_overlap_control_eval.jsonl",
        help="CSV or JSONL used ONLY for final evaluation"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="fraction of TRAIN reserved as validation split"
    )
    args = parser.parse_args()
    main()
