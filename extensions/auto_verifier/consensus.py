"""Multi-verifier consensus engine.

Strategy
--------
1. Each verifier produces an independent VerifierVerdict.
2. Verdicts with label UNKNOWN, PARSE_FAILED, or UNGRAMMATICAL are dropped
   from the consensus vote (treated as abstentions).
3. Among the remaining verdicts, count votes for EQUIVALENT vs NOT_EQUIVALENT.
4. Majority label wins. Ties → "needs_human_review = True".
5. Unanimity (all verifiers vote the same way) → high confidence, no review.
6. Non-unanimous → "needs_human_review = True" (the spot-check set).

Anti-circularity report
-----------------------
The engine also reports per-verifier breakdowns, so the paper can show:

  "Even when restricting to the LLM-judge verifiers (V2, V3) that are not
  derived from our AMR pipeline, AMR-LDA still achieves X% equivalence vs
  best LLM-rewriter Y%."

This defeats the "AMR-verifier favors AMR-LDA" attack.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence

from .types import ConsensusResult, Label, VerifierVerdict


def _vote(verdicts: Sequence[VerifierVerdict]) -> tuple:
    """Return (label, n_votes_for_majority, n_total_votes, unanimous)."""
    # Treat only EQUIVALENT and NOT_EQUIVALENT as votes; others abstain.
    valid_labels = {Label.EQUIVALENT, Label.NOT_EQUIVALENT}
    votes = [v.label for v in verdicts if v.label in valid_labels]
    if not votes:
        return Label.UNKNOWN, 0, 0, False

    counter = Counter(votes)
    top_label, top_count = counter.most_common(1)[0]
    return top_label, top_count, len(votes), top_count == len(votes)


def aggregate(
    item_id: str,
    input_sentence: str,
    candidate_sentence: str,
    rule: str,
    verdicts: List[VerifierVerdict],
) -> ConsensusResult:
    """Run consensus over a list of verdicts."""
    label, top_count, total, unanimous = _vote(verdicts)
    needs_review = (not unanimous) or (total < len(verdicts))
    # Confidence: fraction of valid votes that agree with majority, weighted
    # by per-verdict confidence.
    if total == 0:
        confidence = 0.0
    else:
        agreeing_confidence = sum(
            v.confidence for v in verdicts if v.label == label
        )
        confidence = agreeing_confidence / max(1, total)

    return ConsensusResult(
        item_id=item_id,
        input_sentence=input_sentence,
        candidate_sentence=candidate_sentence,
        rule=rule,
        verdicts=verdicts,
        majority_label=label,
        unanimity=unanimous,
        needs_human_review=needs_review,
        confidence=confidence,
    )


def summary_by_verifier(results: List[ConsensusResult]) -> dict:
    """Compute per-verifier equivalence rate.

    Used to demonstrate anti-circularity: AMR-LDA outputs should win not only
    under the AMR verifier but also under LLM-judge verifiers.

    Returns a dict: {verifier_name: {"equivalent": X, "not_equivalent": Y,
                                      "abstain": Z, "rate": X/(X+Y)}}
    """
    by_v: dict = {}
    for cr in results:
        for v in cr.verdicts:
            bucket = by_v.setdefault(
                v.verifier_name, {"equivalent": 0, "not_equivalent": 0, "abstain": 0}
            )
            if v.label == Label.EQUIVALENT:
                bucket["equivalent"] += 1
            elif v.label == Label.NOT_EQUIVALENT:
                bucket["not_equivalent"] += 1
            else:
                bucket["abstain"] += 1
    for name, b in by_v.items():
        decided = b["equivalent"] + b["not_equivalent"]
        b["equivalence_rate"] = (b["equivalent"] / decided) if decided > 0 else 0.0
    return by_v


def summary_by_system(
    results: List[ConsensusResult], system_of: dict
) -> dict:
    """Compute per-system equivalence rate using consensus label.

    `system_of` is a map from item_id -> system_name (e.g., "amr_lda",
    "gpt-4o", "claude-opus-4-7", etc.). Items requiring human review are
    counted as "pending".
    """
    by_s: dict = {}
    for cr in results:
        sys_name = system_of.get(cr.item_id, "unknown")
        b = by_s.setdefault(
            sys_name,
            {"equivalent": 0, "not_equivalent": 0, "pending_review": 0, "n": 0},
        )
        b["n"] += 1
        if cr.needs_human_review:
            b["pending_review"] += 1
            continue
        if cr.majority_label == Label.EQUIVALENT:
            b["equivalent"] += 1
        elif cr.majority_label == Label.NOT_EQUIVALENT:
            b["not_equivalent"] += 1
    for name, b in by_s.items():
        decided = b["equivalent"] + b["not_equivalent"]
        b["equivalence_rate_decided"] = (
            (b["equivalent"] / decided) if decided > 0 else 0.0
        )
        b["pending_fraction"] = b["pending_review"] / b["n"] if b["n"] > 0 else 0.0
    return by_s


def review_queue(results: List[ConsensusResult]) -> List[ConsensusResult]:
    """Return only items that need human spot-check."""
    return [r for r in results if r.needs_human_review]
