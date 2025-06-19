# funMDL-toy

This is a simple but carefully designed project to test syntactic heuristics in Natural Language Inference (NLI) models using a minimal, focused control dataset.

The project implements the idea of building a **control set** to evaluate a specific hypothesis from the [HANS paper (McCoy et al., 2019)][^1], without relying on the full dataset. This allows for lightweight and interpretable testing.

## 💡 Motivation

This project was built as part of an assignment to create a **precise control dataset** for testing a single linguistic hypothesis. I was inspired by the way the [HANS dataset][^1] reveals how large language models can perform well for the wrong reasons, due to reliance on shallow heuristics.

I am also personally interested in pursuing NLP-related topics in my master's thesis, so this was a good opportunity to deepen my understanding of syntactic generalization in NLI models.

## 🗂 Project Structure

```bash
funMDL-toy/
├── data/
│ ├── boy_names_2001.csv
│ ├── lexical_overlap_control_eval.jsonl
│ ├── lexical_overlap_control_train.jsonl
│ └── Generator.py
│
├── model/
│ ├── debug_Bert.py
│ ├── how_about_better_model.py
│ └── random_guess.py
│ └── train_bow.py
│
└── README.md
```


- **`data/Generator.py`**: Script that generates the control dataset with subcases like `Passive_Entail`, `Passive_Swap`, `Simple_Swap`, etc.
- **`boy_names_2001.csv`**: List of names used to construct natural-sounding examples. Taken from [this repo](https://github.com/aruljohn/popular-baby-names).
- **`model/train_bow.py`**: Trains and evaluates a simple Bag-of-Words logistic regression classifier.
- **`model/how_about_better_model.py`**: Evaluates pre-trained transformer models (e.g. BERT, DistilBERT) on the control set using HuggingFace pipelines.
- **`model/debug_Bert.py`**: Utility script for debugging tokenizer behavior and label mappings.
- **`model/random_guess.py`**: Most basic baseline.

## 📊 Example Use

To run evaluation on the control set using a BERT model:

```bash
# via Cli
python model/how_about_better_model.py \
  --model huggingface/distilbert-base-uncased-finetuned-mnli \
  --eval-file data/lexical_overlap_control_eval.jsonl
```

Or just run via Pycharm / VS code

## 📁 Dataset Details
The dataset consists of NLI examples designed to evaluate lexical overlap heuristics, using passive/active constructions and subject-object swaps.

Each example includes:

- premise: a sentence like "John helped Bob before dinner."

- hypothesis: a constructed sentence (entailing or not)

- label: "entailment" or "non_entailment"

- subcase: one of Passive_Entail, Passive_Swap, Simple_Swap

## 📜 Reference

[^1]: McCoy, R. T., Pavlick, E., & Linzen, T. (2019). **Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in NLI.** *ACL 2019*.