#!/usr/bin/env python
"""
Zero-shot evaluate BERT-MNLI on lexical‐overlap control set
and print per-subcase accuracy.
"""

import argparse, pathlib, torch, pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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
    device = 0 if torch.cuda.is_available() else -1
    print("CUDA available:", torch.cuda.is_available(), "device:", device)

    tok = AutoTokenizer.from_pretrained("huggingface/distilbert-base-uncased-finetuned-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("huggingface/distilbert-base-uncased-finetuned-mnli")
    print(model.config.id2label)
    # model.config.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    # model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    # print(model.config.id2label)

    nli = pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=device,
        top_k=None,           # 返回 3 类全部分数
        batch_size=args.batch
    )

    df      = load_table(args.eval_file)
    text_a  = df["premise"].astype(str).tolist()
    text_b  = df["hypothesis"].astype(str).tolist()
    y_true  = map_label(df["label"]).tolist()

    preds = []
    for i in tqdm(range(0, len(text_a), args.batch), desc="Predict"):
        batch = [{"text": a, "text_pair": b}
                 for a, b in zip(text_a[i:i+args.batch], text_b[i:i+args.batch])]
        scores = nli(batch)
        for sc in scores:
            top = max(sc, key=lambda d: d["score"])["label"].upper()
            # print(top)
            preds.append(1 if "ENTAI" in top else 0)

    # -------- overall metrics ----------------------------------------------
    print("\n=== OVERALL ===")
    print(classification_report(
        y_true, preds, digits=3,
        target_names=["non_entailment", "entailment"]
    ))
    print("Confusion matrix:\n", confusion_matrix(y_true, preds))

    # -------- per-subcase accuracy -----------------------------------------
    df["y_true"] = y_true
    df["y_pred"] = preds
    print("\n=== PER-SUBCASE ACCURACY ===")
    for sub in sorted(df["subcase"].unique()):
        sub_df = df[df["subcase"] == sub]
        acc = accuracy_score(sub_df["y_true"], sub_df["y_pred"])
        print(f"{sub:15s}:  accuracy = {acc:.3f}  "
              f"({len(sub_df)} samples)")

# ---------- entry -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="huggingface/distilbert-base-uncased-finetuned-mnli")
    ap.add_argument("--eval-file", default="../data/data/lexical_overlap_control_eval.jsonl")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    main(args)
