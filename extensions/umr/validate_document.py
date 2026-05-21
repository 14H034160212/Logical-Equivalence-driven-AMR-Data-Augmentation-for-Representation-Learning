"""Validate document-level UMR derivation against gold UMR 2.0 annotations.

The derivation pipeline is in extensions/umr/document.py. This script
groups UMR sentences by doc_id, runs the derivation, and compares the
predicted (src, relation, tgt) triples to gold per-document.

Reports precision/recall/F1 per category (:temporal, :modal, :coref).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import penman

from .document import parse_document_annotation, derive_doc_relations
from .loader import load_umr_dataset


def normalize_rel(rel: str) -> str:
    """Map UMR document-level rel variants to canonical."""
    r = rel.lstrip(":")
    # Map to a common set
    if r in ("full-affirmative", "fullaff", "full_affirmative"):
        return ":full-affirmative"
    if r in ("partial-affirmative", "partialaff"):
        return ":partial-affirmative"
    if r in ("full-negative", "fullneg"):
        return ":full-negative"
    if r in ("partial-negative", "partialneg"):
        return ":partial-negative"
    if r in ("neutral-affirmative",):
        return ":neutral-affirmative"
    if r in ("neutral-negative",):
        return ":neutral-negative"
    return ":" + r


def evaluate_documents(sentences, max_docs: int = None):
    """Group by doc_id, derive relations, compare to gold."""
    # Group sentences by doc
    by_doc: Dict[str, list] = defaultdict(list)
    for s in sentences:
        by_doc[s.doc_id].append(s)
    docs = list(by_doc.items())
    if max_docs:
        docs = docs[:max_docs]
    print(f"Evaluating {len(docs)} documents")

    counts = {
        "temporal": {"tp": 0, "fp": 0, "fn": 0,
                     "rel_confusion": Counter(),
                     "gold_rel_dist": Counter(),
                     "pred_rel_dist": Counter()},
        "modal":    {"tp": 0, "fp": 0, "fn": 0,
                     "rel_confusion": Counter(),
                     "gold_rel_dist": Counter(),
                     "pred_rel_dist": Counter()},
        "coref":    {"tp": 0, "fp": 0, "fn": 0,
                     "rel_confusion": Counter(),
                     "gold_rel_dist": Counter(),
                     "pred_rel_dist": Counter()},
    }

    n_with_gold = 0
    for doc_id, doc_sents in docs:
        # Parse each gold UMR penman into a Graph, collect gold relations
        amr_graphs: List[Tuple[str, penman.Graph]] = []
        gold_temporal: List[Tuple[str, str, str]] = []
        gold_modal: List[Tuple[str, str, str]] = []
        gold_coref: List[Tuple[str, str, str]] = []
        for s in doc_sents:
            try:
                g = penman.decode(s.sentence_umr_penman)
            except Exception:
                continue
            # Gold UMR variables already encode sentence index in the name
            # (e.g., s1l, s2p), so we pass empty prefix here — using the bare
            # variables matches gold's namespace directly.
            amr_graphs.append(("", g))
            doc_lines = getattr(s, "_document_lines", [])
            if doc_lines:
                doc_rels = parse_document_annotation(doc_lines)
                for src, rel, tgt in doc_rels.temporal:
                    gold_temporal.append((src, normalize_rel(rel), tgt))
                for src, rel, tgt in doc_rels.modal:
                    gold_modal.append((src, normalize_rel(rel), tgt))
                for src, rel, tgt in doc_rels.coref:
                    gold_coref.append((src, normalize_rel(rel), tgt))

        if not (gold_temporal or gold_modal or gold_coref):
            continue
        n_with_gold += 1

        # Derive predicted relations (no prefix stripping needed — gold vars
        # already encode sentence index)
        pred_all = derive_doc_relations(amr_graphs)
        pred_unprefixed = pred_all

        # Categorize predictions
        pred_temporal = [t for t in pred_unprefixed if t[1] in (
            ":before", ":after", ":overlap", ":contained", ":depends-on")]
        pred_modal = [t for t in pred_unprefixed if t[1] in (
            ":full-affirmative", ":partial-affirmative", ":full-negative",
            ":partial-negative", ":neutral-affirmative", ":neutral-negative")]
        pred_coref = [t for t in pred_unprefixed if t[1] in (
            ":same-entity", ":same-event")]

        # Inverse-relation table: (A :rel B) <=> (B :inverse_rel A)
        INVERSE_REL = {
            ":before": ":after",
            ":after": ":before",
            ":overlap": ":overlap",
            ":same-entity": ":same-entity",
            ":same-event": ":same-event",
            ":full-affirmative": ":full-affirmative",
            ":partial-affirmative": ":partial-affirmative",
            ":full-negative": ":full-negative",
        }

        def matches(p_src, p_rel, p_tgt, gold_set):
            # Exact match
            if (p_src, p_rel, p_tgt) in gold_set:
                return True
            # Inverted match: (A :before B) ≡ (B :after A)
            inv = INVERSE_REL.get(p_rel)
            if inv and (p_tgt, inv, p_src) in gold_set:
                return True
            return False

        def count_pair_overlap(gold, pred, bucket):
            gold_set = {(s, r, t) for s, r, t in gold}
            # Track which gold triples got hit (for fn computation)
            hit_gold: Set[Tuple[str, str, str]] = set()
            for s, r, t in pred:
                bucket["pred_rel_dist"][r] += 1
                if matches(s, r, t, gold_set):
                    bucket["tp"] += 1
                    if (s, r, t) in gold_set:
                        hit_gold.add((s, r, t))
                    else:
                        inv = INVERSE_REL.get(r)
                        if inv and (t, inv, s) in gold_set:
                            hit_gold.add((t, inv, s))
                else:
                    bucket["fp"] += 1
                    # Confusion: same pair endpoints, different relation
                    same_pair = [(gs, gr, gt) for (gs, gr, gt) in gold
                                 if {gs, gt} == {s, t}]
                    if same_pair:
                        bucket["rel_confusion"][f"{same_pair[0][1]}→{r}"] += 1
            for s, r, t in gold:
                bucket["gold_rel_dist"][r] += 1
                if (s, r, t) not in hit_gold:
                    bucket["fn"] += 1

        count_pair_overlap(gold_temporal, pred_temporal, counts["temporal"])
        count_pair_overlap(gold_modal, pred_modal, counts["modal"])
        count_pair_overlap(gold_coref, pred_coref, counts["coref"])

    print(f"\nDocuments with gold doc-level annotations: {n_with_gold}")
    out = {"n_documents_evaluated": n_with_gold}
    for cat, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print()
        print(f"=== {cat.upper()} ===")
        print(f"  tp={tp}  fp={fp}  fn={fn}")
        print(f"  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
        if c["rel_confusion"]:
            print(f"  rel confusions: {c['rel_confusion'].most_common(5)}")
        out[cat] = {"tp": tp, "fp": fp, "fn": fn,
                    "precision": p, "recall": r, "f1": f1,
                    "rel_confusion": dict(c["rel_confusion"]),
                    "gold_rel_dist": dict(c["gold_rel_dist"]),
                    "pred_rel_dist": dict(c["pred_rel_dist"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr-root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--out-json", default="extensions/reports/document_umr_report.json")
    args = ap.parse_args()

    sents = load_umr_dataset(args.umr_root, args.language)
    print(f"Loaded {len(sents)} sentences")
    report = evaluate_documents(sents, max_docs=args.max_docs)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {args.out_json}")


if __name__ == "__main__":
    main()
