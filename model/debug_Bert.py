from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch, json, tqdm, sklearn.metrics as skm, collections

device = 0 if torch.cuda.is_available() else -1
tok  = AutoTokenizer.from_pretrained("huggingface/distilbert-base-uncased-finetuned-mnli")
bert = AutoModelForSequenceClassification.from_pretrained("huggingface/distilbert-base-uncased-finetuned-mnli")
nli  = pipeline("text-classification", model=bert, tokenizer=tok,
                device=device, batch_size=32, truncation=True, top_k=None)

print(nli({'text':"Alice was helped by Bob before dinner.",
           'text_pair':"Bob was helped by Alice before dinner."}))
# ➜ [{'label': 'ENTAILMENT', 'score': 0.99…}]

