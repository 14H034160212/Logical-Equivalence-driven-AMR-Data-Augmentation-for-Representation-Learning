"""Validate the HYBRID AMR→UMR converter (rule + NN).

The hybrid policy:
  1. Try the rule-based converter first (fast, deterministic).
  2. If the rule abstains (returns None), fall back to the NN classifier.
  3. If NN confidence is below `nn_threshold`, abstain (return None).

We evaluate this hybrid on the same UMR 2.0 English gold set used for the
pure-rule baseline (extensions/umr/validate_converter.py).
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import penman

from .converter import infer_aspect
from .loader import UMRSentence, load_umr_dataset
from .neural_aspect import _node_features, vectorize
from .validate_converter import amr_like_from_umr, extract_gold_annotations


def predict_aspect_hybrid(
    g: penman.Graph, node: str, concept: str,
    clf, vocab, nn_threshold: float = 0.0,
) -> Optional[str]:
    """Try rule first, then NN with confidence threshold."""
    rule_pred = infer_aspect(g, node, concept)
    if rule_pred is not None:
        return rule_pred
    # NN fallback
    if clf is None:
        return None
    feats = _node_features(g, node, concept)
    X, _ = vectorize([feats], vocab=vocab)
    proba = clf.predict_proba(X)[0]
    top_idx = int(np.argmax(proba))
    top_proba = proba[top_idx]
    if top_proba < nn_threshold:
        return None
    return clf.classes_[top_idx]


def evaluate(
    sentences: List[UMRSentence],
    clf, vocab,
    nn_threshold: float = 0.0,
) -> dict:
    """Two metrics:

    A. **Full** (the strict-eval used in the rule-only baseline):
       - FP counts predictions on nodes that have no gold annotation.
       - Useful but punishes systems that try to predict on more nodes.

    B. **Gold-conditional** (recall on gold-annotated nodes only):
       - For each node that HAS a gold annotation, did we predict it correctly?
       - This is the "fair" comparison — both rule and hybrid get scored
         only on the subset gold actually annotated.
    """
    full = {"tp": 0, "fp": 0, "fn": 0,
            "confusion": defaultdict(int),
            "gold_only": Counter(),
            "pred_only": Counter()}
    # Gold-conditional metrics (only nodes with gold annotation)
    gold_only_correct = 0
    gold_only_wrong = 0
    gold_only_abstain = 0
    gold_only_by_source = Counter()
    rule_used = 0
    nn_used = 0
    nn_abstained = 0
    n_processed = 0

    for sent in sentences:
        gold_aspect, _ = extract_gold_annotations(sent.sentence_umr_penman)
        if not gold_aspect:
            continue
        n_processed += 1
        amr_like = amr_like_from_umr(sent.sentence_umr_penman)
        if amr_like is None:
            continue
        concept_of = {s: t for s, r, t in amr_like.triples if r == ":instance"}
        for var, concept in concept_of.items():
            gold = gold_aspect.get(var)
            rule_pred = infer_aspect(amr_like, var, concept)
            if rule_pred is not None:
                pred = rule_pred
                rule_used += 1
                source = "rule"
            else:
                pred = predict_aspect_hybrid(
                    amr_like, var, concept, clf, vocab, nn_threshold,
                )
                if pred is None:
                    nn_abstained += 1
                    source = "abstain"
                else:
                    nn_used += 1
                    source = "nn"
            # Full eval: count everything
            if gold and pred:
                if gold == pred:
                    full["tp"] += 1
                else:
                    full["fp"] += 1
                    full["fn"] += 1
                    full["confusion"][f"{gold}→{pred}"] += 1
            elif gold and not pred:
                full["fn"] += 1
                full["gold_only"][gold] += 1
            elif pred and not gold:
                full["fp"] += 1
                full["pred_only"][pred] += 1
            # Gold-conditional eval: only nodes that gold annotates
            if gold:
                gold_only_by_source[source] += 1
                if pred is None:
                    gold_only_abstain += 1
                elif pred == gold:
                    gold_only_correct += 1
                else:
                    gold_only_wrong += 1

    tp, fp, fn = full["tp"], full["fp"], full["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    n_gold = gold_only_correct + gold_only_wrong + gold_only_abstain
    gold_acc = gold_only_correct / n_gold if n_gold else 0.0

    return {
        "n_sentences": n_processed,
        "n_gold_nodes": n_gold,
        "rule_used": rule_used,
        "nn_used": nn_used,
        "nn_abstained": nn_abstained,
        "full_precision": p, "full_recall": r, "full_f1": f1,
        "gold_accuracy": gold_acc,
        "gold_correct": gold_only_correct,
        "gold_wrong": gold_only_wrong,
        "gold_abstain": gold_only_abstain,
        "gold_by_source": dict(gold_only_by_source),
        "top_confusions": sorted(full["confusion"].items(), key=lambda x: -x[1])[:10],
        "nn_threshold": nn_threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr-root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--model", default="extensions/umr/aspect_classifier.pkl")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--nn-threshold", type=float, default=0.0)
    ap.add_argument("--out-json", default="extensions/reports/hybrid_aspect_report.json")
    args = ap.parse_args()

    sents = load_umr_dataset(args.umr_root, args.language)
    if args.max_sentences:
        sents = sents[: args.max_sentences]
    print(f"Loaded {len(sents)} sentences")

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    clf, vocab = bundle["clf"], bundle["vocab"]

    # Threshold sweep — two metric columns side by side
    print()
    print("Threshold sweep:")
    print(f"  {'thr':>5}  {'full_F1':>8}  {'gold_acc':>9}  {'rule':>6}  {'nn':>6}  {'abst':>6}")
    rows = []
    for thr in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        m = evaluate(sents, clf, vocab, nn_threshold=thr)
        print(f"  {thr:>5.2f}  {m['full_f1']:>8.3f}  {m['gold_accuracy']:>9.3f}  "
              f"{m['rule_used']:>6d}  {m['nn_used']:>6d}  {m['nn_abstained']:>6d}")
        rows.append(m)

    best = max(rows, key=lambda x: x["gold_accuracy"])
    print()
    print(f"Best gold-accuracy: {best['gold_accuracy']:.3f} at threshold {best['nn_threshold']}")
    print(f"  n_gold_nodes={best['n_gold_nodes']}  correct={best['gold_correct']}  "
          f"wrong={best['gold_wrong']}  abstain={best['gold_abstain']}")
    print(f"  by source: {best['gold_by_source']}")
    print(f"Top confusions: {best['top_confusions'][:5]}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"runs": rows, "best": best}, f, indent=2, default=str)
    print(f"Saved report to {args.out_json}")


if __name__ == "__main__":
    main()
