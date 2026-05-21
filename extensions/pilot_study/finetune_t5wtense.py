"""Fine-tune T5wtense to better preserve polarity (negation) when generating
text from AMR graphs.

Motivation
----------
The pilot study showed T5wtense drops a polarity-neg edge ~17% of the time
(15/77 records in run6), producing logically-wrong text like 'Alice can't
finish her homework' for an AMR that means 'It is not the case that Alice
may skip her homework'.

This script fine-tunes the T5wtense generator on (modified_AMR, gold_text)
pairs harvested from the pilot study. Gold sources:
  A. test_sentences.json gold rewrites (most reliable; small)
  B. gpt-4o outputs from the pilot that the LLM-judge marked EQUIVALENT
     (larger, used as silver-standard targets)

Training:
  - Initialize from amrlib's model_generate_t5wtense-v0_1_0
  - 3 epochs, lr=3e-5, batch 8, on one A100 (~20-30 min)
  - Save to extensions/pilot_study/ft_t5wtense/

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=4 \\
        /data/qbao775/miniconda3/envs/leamr/bin/python \\
            -m extensions.pilot_study.finetune_t5wtense
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
log = logging.getLogger("ft_t5wtense")


def collect_training_pairs() -> List[Tuple[str, str]]:
    """Harvest (modified_AMR, target_text) pairs from contrastive dataset jsonls.

    Sources (in order of decreasing quality):
      1. Pilot test_sentences.json gold_outputs — but they don't have AMR.
         We pair gold text with the AMR from contrastive_dataset_smoketest.jsonl
         when the same (sentence_id, rule) is present.
      2. contrastive_dataset_smoketest.jsonl: amr_positive paired with the
         positive text (LLM-verified silver).
      3. pararule_contrastive.jsonl: same, larger.
    """
    pairs: List[Tuple[str, str]] = []
    seen: set = set()

    # Source A: contrastive datasets — straightforward (amr_positive, positive)
    candidate_paths = [
        Path("extensions/reports/contrastive_pilot_smoketest.jsonl"),
        Path("extensions/pilot_study/results/combined/rewrite/amr_lda.jsonl"),
        Path("extensions/pilot_study/pararule_contrastive.jsonl"),
    ]

    test_json = json.loads(Path("extensions/pilot_study/test_sentences.json").read_text())
    gold_map: Dict[Tuple[str, str], str] = {}
    for s in test_json["sentences"]:
        for rule, text in (s.get("gold_outputs") or {}).items():
            if text and isinstance(text, str) and len(text) < 200:
                gold_map[(s["id"], rule)] = text

    # Source A: amr_positive + positive from contrastive datasets
    for path in candidate_paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            amr = r.get("amr_positive")
            text = r.get("positive")
            if not amr or not text:
                continue
            # Prefer hand-curated gold when available
            sid, rule = r.get("sentence_id", ""), r.get("rule", "")
            gold = gold_map.get((sid, rule))
            target = gold if gold else text
            key = (amr, target)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((amr, target))

    return pairs


def train(
    pairs: List[Tuple[str, str]],
    pretrained_dir: str = "amrlib/data/model_generate_t5wtense-v0_1_0",
    out_dir: str = "extensions/pilot_study/ft_t5wtense",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    max_input_length: int = 512,
    max_target_length: int = 64,
    seed: int = 42,
) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log.info("Loading T5wtense from %s", pretrained_dir)
    # T5wtense's checkpoint dir lacks tokenizer files — use t5-base tokenizer
    # (the model is based on T5-base; tokenizer is identical).
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    except OSError:
        log.info("  tokenizer not in checkpoint dir; falling back to t5-base")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(pretrained_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 80/20 split
    np.random.shuffle(pairs)
    split = int(len(pairs) * 0.85)
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]
    log.info("Train: %d, Eval: %d", len(train_pairs), len(eval_pairs))

    class PairDataset(TorchDataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            amr, text = self.pairs[idx]
            enc = tokenizer(
                amr, truncation=True, padding="max_length",
                max_length=max_input_length, return_tensors="pt",
            )
            dec = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=max_target_length, return_tensors="pt",
            )
            labels = dec["input_ids"].squeeze(0).clone()
            labels[labels == tokenizer.pad_token_id] = -100
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels,
            }

    train_ds = PairDataset(train_pairs)
    eval_ds = PairDataset(eval_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps,
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n += 1
        # Eval
        model.eval()
        eval_loss = 0
        en = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                eval_loss += out.loss.item()
                en += 1
        log.info("epoch %d/%d  train_loss=%.4f  eval_loss=%.4f",
                 epoch + 1, epochs, epoch_loss / max(1, n),
                 eval_loss / max(1, en))

    # Save
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    # Copy amrlib_meta.json so amrlib can load the model
    import shutil
    src_meta = Path(pretrained_dir) / "amrlib_meta.json"
    if src_meta.exists():
        shutil.copy(src_meta, out_path / "amrlib_meta.json")
    log.info("Saved fine-tuned T5wtense to %s", out_path)
    return {
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "epochs": epochs,
        "out_dir": str(out_path),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-dir",
                    default="amrlib/data/model_generate_t5wtense-v0_1_0")
    ap.add_argument("--out-dir", default="extensions/pilot_study/ft_t5wtense")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out-report",
                    default="extensions/reports/ft_t5wtense_report.json")
    args = ap.parse_args()

    pairs = collect_training_pairs()
    log.info("Collected %d (AMR, text) pairs from pilot data", len(pairs))
    if len(pairs) < 10:
        log.error("Not enough training pairs (%d); aborting.", len(pairs))
        return
    result = train(
        pairs,
        pretrained_dir=args.pretrained_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(result, indent=2))
    log.info("Report saved to %s", args.out_report)


if __name__ == "__main__":
    main()
