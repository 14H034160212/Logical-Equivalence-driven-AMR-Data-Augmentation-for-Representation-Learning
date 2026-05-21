"""Validate the rule-based AMR→UMR converter against UMR 2.0 gold annotations.

Approach:
  1. Load gold UMR sentences from umr-data
  2. For each sentence, parse the gold UMR penman and extract:
        gold_aspect    : {node_var → aspect-label}
        gold_modal     : {node_var → modal-strength-label}
  3. Strip those :aspect / :modal-strength edges to produce "AMR-like" input
  4. Run convert_amr_to_umr() on the AMR-like input
  5. Compare predicted :aspect / :modal-strength to gold per-node
  6. Report precision / recall / per-label confusion matrix

Usage:
    PYTHONPATH=. python -m extensions.umr.validate_converter \\
        --umr-root extensions/umr/umr-data \\
        --max-sentences 1000 \\
        --out-json extensions/umr/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import penman

from .converter import (
    convert_amr_to_umr, infer_aspect, infer_modal_strength, _is_root_event,
)
from .loader import load_umr_dataset, UMRSentence


# Strip UMR-only attributes to produce an AMR-like graph for re-conversion.
UMR_ONLY_ROLES = {":aspect", ":modal-strength", ":ref-number", ":ref-person",
                  ":animacy", ":wiki", ":quant"}


def amr_like_from_umr(umr_penman: str) -> Optional[penman.Graph]:
    """Strip UMR-specific role triples to get an AMR-like graph."""
    try:
        g = penman.decode(umr_penman)
    except Exception:
        return None
    new_triples = [(s, r, t) for s, r, t in g.triples if r not in UMR_ONLY_ROLES]
    return penman.Graph(new_triples, top=g.top)


def extract_gold_annotations(umr_penman: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Pull out gold :aspect and :modal-strength annotations as
    {node_var: label} dicts."""
    aspect: Dict[str, str] = {}
    modal: Dict[str, str] = {}
    try:
        g = penman.decode(umr_penman)
    except Exception:
        return aspect, modal
    for s, role, t in g.triples:
        if role == ":aspect":
            aspect[s] = t
        elif role == ":modal-strength":
            modal[s] = t
    return aspect, modal


def evaluate(
    sentences: List[UMRSentence],
    max_sentences: Optional[int] = None,
) -> dict:
    """Run the converter on each gold UMR sentence and tally accuracy."""
    n_processed = 0
    aspect_stats = {"tp": 0, "fp": 0, "fn": 0,
                    "confusion": defaultdict(int),  # (gold, pred) -> count
                    "gold_only": Counter(),
                    "pred_only": Counter()}
    modal_stats = {"tp": 0, "fp": 0, "fn": 0,
                   "confusion": defaultdict(int),
                   "gold_only": Counter(),
                   "pred_only": Counter()}
    error_examples = {"aspect": [], "modal": []}

    for sent in sentences:
        if max_sentences and n_processed >= max_sentences:
            break
        gold_aspect, gold_modal = extract_gold_annotations(sent.sentence_umr_penman)
        if not gold_aspect and not gold_modal:
            continue  # nothing to compare against
        n_processed += 1

        amr_like = amr_like_from_umr(sent.sentence_umr_penman)
        if amr_like is None:
            continue

        # Concept lookup: var → concept string
        concept_of = {s: t for s, r, t in amr_like.triples if r == ":instance"}

        # Predict aspect for every concept node and compare
        for var, concept in concept_of.items():
            pred = infer_aspect(amr_like, var, concept)
            gold = gold_aspect.get(var)
            if gold and pred:
                if gold == pred:
                    aspect_stats["tp"] += 1
                else:
                    aspect_stats["fp"] += 1
                    aspect_stats["fn"] += 1
                    aspect_stats["confusion"][f"{gold}→{pred}"] += 1
                    if len(error_examples["aspect"]) < 10:
                        error_examples["aspect"].append({
                            "sent": sent.text[:80],
                            "concept": concept,
                            "gold": gold,
                            "pred": pred,
                        })
            elif gold and not pred:
                aspect_stats["fn"] += 1
                aspect_stats["gold_only"][gold] += 1
            elif pred and not gold:
                aspect_stats["fp"] += 1
                aspect_stats["pred_only"][pred] += 1

        # Same for modal-strength (only root-level events)
        for var, concept in concept_of.items():
            if not _is_root_event(amr_like, var):
                # Skip non-root; converter only annotates root-level events
                pred = None
            else:
                pred = infer_modal_strength(amr_like, var, concept)
            gold = gold_modal.get(var)
            if gold and pred:
                if gold == pred:
                    modal_stats["tp"] += 1
                else:
                    modal_stats["fp"] += 1
                    modal_stats["fn"] += 1
                    modal_stats["confusion"][f"{gold}→{pred}"] += 1
            elif gold and not pred:
                modal_stats["fn"] += 1
                modal_stats["gold_only"][gold] += 1
            elif pred and not gold:
                modal_stats["fp"] += 1
                modal_stats["pred_only"][pred] += 1

    def to_metrics(s):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f1,
                "confusion": dict(s["confusion"]),
                "gold_only": dict(s["gold_only"]),
                "pred_only": dict(s["pred_only"])}

    return {
        "n_sentences_with_gold": n_processed,
        "aspect": to_metrics(aspect_stats),
        "modal_strength": to_metrics(modal_stats),
        "error_examples": error_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr-root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--out-json", type=Path, default=Path("extensions/umr/validation_report.json"))
    args = ap.parse_args()

    sents = load_umr_dataset(args.umr_root, args.language)
    print(f"Loaded {len(sents)} sentences.")
    report = evaluate(sents, max_sentences=args.max_sentences)
    args.out_json.write_text(json.dumps(report, indent=2))
    print()
    print(f"Validated on {report['n_sentences_with_gold']} sentences with gold annotations.")
    print()
    for cat in ("aspect", "modal_strength"):
        m = report[cat]
        print(f"=== {cat.upper()} ===")
        print(f"  tp={m['tp']}  fp={m['fp']}  fn={m['fn']}")
        print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")
        if m['confusion']:
            print(f"  top confusions: {sorted(m['confusion'].items(), key=lambda x: -x[1])[:5]}")
        if m['gold_only']:
            top_gold = sorted(m['gold_only'].items(), key=lambda x: -x[1])[:5]
            print(f"  gold-only (rule missed): {dict(top_gold)}")
        if m['pred_only']:
            top_pred = sorted(m['pred_only'].items(), key=lambda x: -x[1])[:5]
            print(f"  pred-only (false positives): {dict(top_pred)}")
        print()
    print(f"Full report: {args.out_json}")


if __name__ == "__main__":
    main()
