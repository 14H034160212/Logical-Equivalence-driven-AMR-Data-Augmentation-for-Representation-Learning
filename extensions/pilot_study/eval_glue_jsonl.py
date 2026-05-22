"""Eval-only loop over a JSONL of (sentence1, sentence2, label) using a
local pretrained sequence-classification checkpoint. Used to cross-evaluate
the v5- vs v6-trained DeBERTa-large checkpoints on each other's
validation sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--val-jsonl", required=True, type=Path)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.to(device).eval()

    records = []
    for line in args.val_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        records.append((r["sentence1"], r["sentence2"], int(r["label"])))

    n_correct = 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]
            s1 = [b[0] for b in batch]
            s2 = [b[1] for b in batch]
            labels = torch.tensor([b[2] for b in batch], device=device)
            enc = tokenizer(
                s1, s2,
                truncation=True, padding=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            out = model(**enc)
            preds = out.logits.argmax(dim=-1)
            n_correct += (preds == labels).sum().item()
            n += len(batch)

    acc = n_correct / max(1, n)
    print(json.dumps({
        "model_dir": str(args.model_dir),
        "val_jsonl": str(args.val_jsonl),
        "n": n,
        "correct": n_correct,
        "accuracy": acc,
    }, indent=2))


if __name__ == "__main__":
    main()
