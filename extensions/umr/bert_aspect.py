"""BERT fine-tune for AMR → UMR aspect prediction.

Replaces the sklearn LogisticRegression classifier in neural_aspect.py with a
small BERT encoder fine-tuned on UMR 2.0 English gold annotations.

Input representation:
    "{sentence_text} [SEP] {target_concept}"

Output: aspect label (state / activity / performance / habitual / process /
inceptive / endeavor / generic).

Why BERT over sklearn:
- Captures sentence-level context (a verb's aspect can depend on temporal
  adverbs, modifiers, surrounding clause structure)
- Handles OOV PropBank frames via WordPiece tokenization
- Should push gold-accuracy from sklearn's 77.9% toward ~85%+ given enough
  data — though UMR English gold is small (~1500 examples).

Usage:
    PYTHONPATH=. python -m extensions.umr.bert_aspect \\
        --epochs 5 --batch-size 16 --model distilbert-base-uncased \\
        --out-model extensions/umr/bert_aspect_clf
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import penman

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_text_examples(sentences) -> Tuple[List[str], List[str]]:
    """For every gold-aspect-annotated node, return (input_text, label).

    Input format: "{sentence_text} [SEP] {target_concept}"
    """
    from extensions.umr.loader import UMRSentence

    texts: List[str] = []
    labels: List[str] = []
    for sent in sentences:
        try:
            g = penman.decode(sent.sentence_umr_penman)
        except Exception:
            continue
        gold: Dict[str, str] = {}
        for s, role, t in g.triples:
            if role == ":aspect":
                gold[s] = t
        if not gold:
            continue
        # For each gold-annotated node, extract concept
        concept_of = {s: t for s, role, t in g.triples if role == ":instance"}
        for var, label in gold.items():
            concept = concept_of.get(var)
            if not concept:
                continue
            sentence_text = sent.text or sent.english_translation or ""
            # Mark the target by inserting tags around its concept
            input_text = f"{sentence_text} [SEP] target: {concept}"
            texts.append(input_text)
            labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_eval(
    texts: List[str],
    labels: List[str],
    model_name: str = "distilbert-base-uncased",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
    max_length: int = 128,
    seed: int = 42,
    out_dir: str = "extensions/umr/bert_aspect_clf",
) -> dict:
    """Fine-tune BERT on (text, label) pairs and evaluate on a held-out 20%."""
    import numpy as np
    import torch
    from sklearn.metrics import (
        classification_report,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Drop very rare labels
    label_counts = Counter(labels)
    keep = {l for l, c in label_counts.items() if c >= 3}
    pairs = [(t, l) for t, l in zip(texts, labels) if l in keep]
    texts, labels = zip(*pairs)
    texts, labels = list(texts), list(labels)
    print(f"After filter: {len(texts)} examples, labels = {sorted(set(labels))}")

    # Encode labels
    sorted_labels = sorted(set(labels))
    lab2id = {l: i for i, l in enumerate(sorted_labels)}
    id2lab = {i: l for l, i in lab2id.items()}
    y = [lab2id[l] for l in labels]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=seed, stratify=y,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(sorted_labels)
    ).to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_ds = TextDataset(X_train, y_train)
    test_ds = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps,
    )

    # Class-balanced loss
    class_counts = Counter(y_train)
    weights = torch.tensor([
        1.0 / class_counts[i] for i in range(len(sorted_labels))
    ], dtype=torch.float32, device=device)
    weights = weights / weights.sum() * len(sorted_labels)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels_t = batch["label"].to(device)
            out = model(input_ids=input_ids, attention_mask=mask)
            loss = loss_fn(out.logits, labels_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{epochs}  train_loss={total_loss/len(train_loader):.4f}")

    # Eval
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(batch["label"].cpu().tolist())

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average="micro", zero_division=0
    )
    report = classification_report(
        all_true, all_preds,
        target_names=[id2lab[i] for i in range(len(sorted_labels))],
        zero_division=0,
    )
    print()
    print(report)
    print(f"macro F1 = {macro_f1:.3f}, micro F1 = {micro_f1:.3f}")

    # Save
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    Path(out_path / "label_map.json").write_text(json.dumps(lab2id, indent=2))
    print(f"Saved model to {out_path}")

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "labels": sorted_labels,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "classification_report": report,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr-root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-model", default="extensions/umr/bert_aspect_clf")
    ap.add_argument("--out-report", default="extensions/reports/bert_aspect_report.json")
    args = ap.parse_args()

    from extensions.umr.loader import load_umr_dataset
    sents = load_umr_dataset(args.umr_root, args.language)
    print(f"Loaded {len(sents)} sentences from {args.language}")

    texts, labels = build_text_examples(sents)
    print(f"Built {len(texts)} (text, label) pairs")
    print(f"Label distribution: {Counter(labels).most_common()}")

    result = train_and_eval(
        texts, labels,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        out_dir=args.out_model,
    )
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(result, indent=2))
    print(f"Report saved to {args.out_report}")


if __name__ == "__main__":
    main()
