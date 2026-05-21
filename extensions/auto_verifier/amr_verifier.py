"""V1: AMR structural verifier.

Given (input_sentence S, candidate_sentence S', rule_name R), this verifier:
  1. Parses S to AMR graph G.
  2. Parses S' to AMR graph G'.
  3. Applies rule R to G to compute the EXPECTED graph G_expected.
  4. Compares G_expected vs G' via triple-overlap F1 ("simplified SMATCH").
  5. Returns EQUIVALENT if F1 >= threshold, else NOT_EQUIVALENT.

Parser interface
----------------
The actual AMR parser (e.g., amrlib's parse_xfm_bartlarge) is an injectable
dependency so the verifier can be unit-tested without loading a 500MB model.
Implement `AMRParserProtocol.parse(sentence: str) -> str` for your parser.

For the user's existing setup, we provide an AmrlibParser thin wrapper that
matches the original logical_equivalence_functions.py invocation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol, Set, Tuple

import penman

from ..logic_rules import get_rule
from .types import Label, VerifierVerdict


log = logging.getLogger(__name__)


class AMRParserProtocol(Protocol):
    def parse(self, sentence: str) -> str:  # returns penman string
        ...


class AmrlibParser:
    """Thin wrapper over amrlib's parse_xfm_bartlarge (matches the user's existing setup)."""

    def __init__(self, model_dir: str = "amrlib/parse_xfm_bart_large_v0_1_0"):
        try:
            import amrlib  # noqa: F401
            from amrlib.models.parse_xfm.inference import Inference
        except ImportError as e:
            raise RuntimeError(
                "amrlib not installed. pip install amrlib or use a different parser."
            ) from e
        self._impl = Inference(model_dir)

    def parse(self, sentence: str) -> str:
        graphs = self._impl.parse_sents([sentence])
        return graphs[0] if graphs else ""


class MockParser:
    """In-memory parser for unit tests: returns the penman string from a dict lookup."""

    def __init__(self, table: dict):
        self.table = table

    def parse(self, sentence: str) -> str:
        return self.table.get(sentence, "")


def _canonical_triples(g: penman.Graph) -> Set[Tuple[str, str, str]]:
    """Canonicalize triples by replacing variable names with their AMR concepts.

    This is a poor-man's SMATCH that avoids variable-renaming ambiguity. It
    will mis-handle co-reference (when two nodes have the same concept), but
    that's an acceptable approximation for the pilot.
    """
    var_to_concept: dict = {}
    for s, role, t in g.triples:
        if role == ":instance":
            var_to_concept[s] = t

    def remap(x: str) -> str:
        return var_to_concept.get(x, x)

    canonical: Set[Tuple[str, str, str]] = set()
    for s, role, t in g.triples:
        canonical.add((remap(s), role, remap(t)))
    return canonical


def triple_f1(g_gold: penman.Graph, g_pred: penman.Graph) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 over canonicalized triple sets."""
    gold = _canonical_triples(g_gold)
    pred = _canonical_triples(g_pred)
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    overlap = gold & pred
    p = len(overlap) / len(pred)
    r = len(overlap) / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


class AMRVerifier:
    """V1: parse both sentences, apply rule to input, compare to candidate."""

    NAME = "amr_struct"

    def __init__(
        self,
        parser: AMRParserProtocol,
        threshold: float = 0.60,
        roundtrip_baseline: Optional[float] = None,
    ):
        # Default lowered from 0.85 → 0.60 because the AMR-to-text (T5wtense)
        # round-trip introduces lexical drift (e.g., "LED" → "compact fluorescent
        # light") even when the logical transformation is correct. A stricter
        # threshold under-counts true equivalences. 0.60 is calibrated against
        # the pilot data — see extensions/auto_verifier/results/run3 for source.
        self.parser = parser
        self.threshold = threshold
        self.roundtrip_baseline = roundtrip_baseline

    def verify(
        self, input_sentence: str, candidate_sentence: str, rule: str
    ) -> VerifierVerdict:
        try:
            input_penman = self.parser.parse(input_sentence)
            candidate_penman = self.parser.parse(candidate_sentence)
        except Exception as e:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.PARSE_FAILED,
                confidence=0.0,
                error=f"parse error: {e}",
            )
        if not input_penman or not candidate_penman:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.PARSE_FAILED,
                confidence=0.0,
                error="empty parser output",
            )

        try:
            g_input = penman.decode(input_penman)
            g_candidate = penman.decode(candidate_penman)
        except Exception as e:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.PARSE_FAILED,
                confidence=0.0,
                error=f"penman decode error: {e}",
            )

        try:
            rule_obj = get_rule(rule)
        except KeyError:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.UNKNOWN,
                confidence=0.0,
                error=f"unknown rule: {rule}",
            )

        results = rule_obj.apply(g_input)
        if not results or results[0].positive_graph is None:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.NOT_EQUIVALENT,
                confidence=0.5,
                details={"reason": "rule did not fire on input"},
            )

        # Compute structural similarity against EVERY positive result the rule
        # could produce (some rules have multiple applicable matches).
        best_f1 = 0.0
        best_p = best_r = 0.0
        for res in results:
            if res.positive_graph is None:
                continue
            try:
                g_expected = penman.decode(res.positive_graph)
            except Exception:
                continue
            p, r, f1 = triple_f1(g_expected, g_candidate)
            if f1 > best_f1:
                best_f1, best_p, best_r = f1, p, r

        label = Label.EQUIVALENT if best_f1 >= self.threshold else Label.NOT_EQUIVALENT
        # Confidence = absolute distance from threshold, scaled
        gap = abs(best_f1 - self.threshold)
        confidence = min(1.0, 0.5 + 2 * gap)

        return VerifierVerdict(
            verifier_name=self.NAME,
            label=label,
            confidence=confidence,
            score=best_f1,
            details={
                "precision": f"{best_p:.3f}",
                "recall": f"{best_r:.3f}",
                "f1": f"{best_f1:.3f}",
                "threshold": f"{self.threshold:.3f}",
            },
        )
