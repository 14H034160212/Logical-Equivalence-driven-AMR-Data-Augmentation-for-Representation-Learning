"""Hybrid validator using BERT (DistilBERT) instead of sklearn for the NN fallback.

Same eval protocol as validate_hybrid.py — rule first, NN-on-abstain.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import penman

warnings.filterwarnings("ignore")

from .converter import infer_aspect
from .loader import UMRSentence, load_umr_dataset
from .validate_converter import amr_like_from_umr, extract_gold_annotations


def load_bert_classifier(model_dir: str):
    """Load the fine-tuned BERT model + tokenizer + label map."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    lab_map = json.loads(Path(f"{model_dir}/label_map.json").read_text())
    id2lab = {v: k for k, v in lab_map.items()}
    return model, tokenizer, id2lab, device


def predict_aspect_bert(
    model, tokenizer, id2lab, device,
    sentence_text: str, concept: str,
    confidence_threshold: float = 0.0,
    max_length: int = 128,
) -> Optional[str]:
    """Predict aspect via BERT given (sentence, target concept)."""
    import torch

    text = f"{sentence_text} [SEP] target: {concept}"
    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
    top_idx = int(probs.argmax().item())
    top_prob = float(probs[top_idx].item())
    if top_prob < confidence_threshold:
        return None
    return id2lab[top_idx]


def evaluate(
    sentences: List[UMRSentence],
    bert_model, tokenizer, id2lab, device,
    confidence_threshold: float = 0.0,
) -> dict:
    """Evaluate rule-first, BERT-fallback hybrid."""
    n_with_gold = 0
    gold_correct = 0
    gold_wrong = 0
    gold_abstain = 0
    gold_by_source = Counter()
    rule_count = 0
    bert_count = 0
    bert_abstain_count = 0
    confusion = defaultdict(int)

    for sent in sentences:
        gold_aspect, _ = extract_gold_annotations(sent.sentence_umr_penman)
        if not gold_aspect:
            continue
        n_with_gold += 1
        amr_like = amr_like_from_umr(sent.sentence_umr_penman)
        if amr_like is None:
            continue
        concept_of = {s: t for s, r, t in amr_like.triples if r == ":instance"}

        for var, concept in concept_of.items():
            gold = gold_aspect.get(var)
            rule_pred = infer_aspect(amr_like, var, concept)
            if rule_pred is not None:
                pred = rule_pred
                rule_count += 1
                source = "rule"
            else:
                pred = predict_aspect_bert(
                    bert_model, tokenizer, id2lab, device,
                    sent.text or sent.english_translation or "",
                    concept,
                    confidence_threshold=confidence_threshold,
                )
                if pred is None:
                    bert_abstain_count += 1
                    source = "abstain"
                else:
                    bert_count += 1
                    source = "bert"

            if gold:
                gold_by_source[source] += 1
                if pred is None:
                    gold_abstain += 1
                elif pred == gold:
                    gold_correct += 1
                else:
                    gold_wrong += 1
                    confusion[f"{gold}→{pred or 'None'}"] += 1

    n_gold = gold_correct + gold_wrong + gold_abstain
    gold_acc = gold_correct / n_gold if n_gold else 0.0
    return {
        "n_sentences": n_with_gold,
        "n_gold_nodes": n_gold,
        "gold_accuracy": gold_acc,
        "gold_correct": gold_correct,
        "gold_wrong": gold_wrong,
        "gold_abstain": gold_abstain,
        "gold_by_source": dict(gold_by_source),
        "rule_count": rule_count,
        "bert_count": bert_count,
        "bert_abstain_count": bert_abstain_count,
        "top_confusions": sorted(confusion.items(), key=lambda x: -x[1])[:10],
        "confidence_threshold": confidence_threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr-root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--bert-model", default="extensions/umr/bert_aspect_clf")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--threshold-sweep", action="store_true",
                    help="evaluate at multiple thresholds")
    ap.add_argument("--out-json", default="extensions/reports/hybrid_bert_aspect_report.json")
    args = ap.parse_args()

    sents = load_umr_dataset(args.umr_root, args.language)
    if args.max_sentences:
        sents = sents[: args.max_sentences]
    print(f"Loaded {len(sents)} sentences")

    print(f"Loading BERT classifier from {args.bert_model}")
    bert_model, tokenizer, id2lab, device = load_bert_classifier(args.bert_model)

    thresholds = (0.0, 0.3, 0.5, 0.7, 0.9) if args.threshold_sweep else (0.0,)
    rows = []
    for thr in thresholds:
        m = evaluate(sents, bert_model, tokenizer, id2lab, device,
                     confidence_threshold=thr)
        m["confidence_threshold"] = thr
        rows.append(m)
        print(f"\nthr={thr:.2f}  gold_acc={m['gold_accuracy']:.3f}  "
              f"correct={m['gold_correct']}  wrong={m['gold_wrong']}  "
              f"abstain={m['gold_abstain']}  by={m['gold_by_source']}")

    best = max(rows, key=lambda x: x["gold_accuracy"])
    print()
    print(f"BEST gold-accuracy: {best['gold_accuracy']:.3f} at threshold {best['confidence_threshold']}")
    print(f"  Top confusions: {best['top_confusions'][:5]}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"runs": rows, "best": best}, indent=2, default=str))
    print(f"\nReport saved to {args.out_json}")


if __name__ == "__main__":
    main()
