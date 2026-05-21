"""V4: Pure SMATCH verifier — no rule reasoning, just AMR similarity.

This is the most conservative verifier: it parses both sentences to AMR and
checks whether the resulting graphs are structurally similar enough to call
them "the same proposition expressed differently."

It is NOT rule-aware. It will count two unrelated sentences as NOT_EQUIVALENT
even if the rule would have produced a transformation. But it provides an
independent signal that does NOT use our LogicRule code path, which is the
critical anti-circularity property.

Note: For a "real" SMATCH F1, install the `smatch` package (`pip install
smatch`). The fallback here is the same triple-F1 used by AMRVerifier.
"""

from __future__ import annotations

import logging
from typing import Optional

import penman

from .amr_verifier import AMRParserProtocol, triple_f1
from .types import Label, VerifierVerdict


log = logging.getLogger(__name__)


def _try_real_smatch() -> Optional[callable]:
    """Attempt to use the `smatch` package's reference implementation.

    The `smatch` PyPI package historically diverges from amrlib's penman API
    on whitespace/format. In practice it has been unreliable on our parser
    output, returning F1=0.0 on visibly-similar graphs. So we disable it by
    default and fall back to the deterministic triple-F1 implementation in
    `amr_verifier.triple_f1`. Set the env var `USE_REAL_SMATCH=1` to opt in.
    """
    import os

    if os.environ.get("USE_REAL_SMATCH", "0") != "1":
        return None

    try:
        from smatch import get_amr_match, compute_f  # type: ignore
    except ImportError:
        return None

    def _real_smatch(g1: penman.Graph, g2: penman.Graph) -> float:
        amr1 = penman.encode(g1)
        amr2 = penman.encode(g2)
        try:
            best_match_num, test_triple_num, gold_triple_num = get_amr_match(
                amr1, amr2
            )
            _, _, f = compute_f(best_match_num, test_triple_num, gold_triple_num)
            return f
        except Exception:
            return 0.0

    return _real_smatch


class SmatchVerifier:
    """Rule-agnostic AMR similarity verifier."""

    NAME = "smatch_struct"

    def __init__(
        self,
        parser: AMRParserProtocol,
        equivalent_threshold: float = 0.7,
        non_equivalent_threshold: float = 0.45,
    ):
        """
        Parameters
        ----------
        equivalent_threshold : a SMATCH F1 above this counts as EQUIVALENT
        non_equivalent_threshold : below this, NOT_EQUIVALENT
        in-between : UNKNOWN (signals to the consensus engine that this verifier
                     abstains)
        """
        self.parser = parser
        self.eq_thr = equivalent_threshold
        self.neq_thr = non_equivalent_threshold
        self._real_smatch = _try_real_smatch()

    def _compute_f1(self, g1: penman.Graph, g2: penman.Graph) -> float:
        if self._real_smatch is not None:
            return self._real_smatch(g1, g2)
        return triple_f1(g1, g2)[2]

    def verify(
        self, input_sentence: str, candidate_sentence: str, rule: str
    ) -> VerifierVerdict:
        try:
            p1 = self.parser.parse(input_sentence)
            p2 = self.parser.parse(candidate_sentence)
        except Exception as e:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.PARSE_FAILED,
                confidence=0.0,
                error=str(e),
            )
        if not p1 or not p2:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.PARSE_FAILED,
                confidence=0.0,
                error="empty parser output",
            )

        try:
            g1 = penman.decode(p1)
            g2 = penman.decode(p2)
        except Exception as e:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.PARSE_FAILED,
                confidence=0.0,
                error=str(e),
            )

        score = self._compute_f1(g1, g2)

        if score >= self.eq_thr:
            label, conf = Label.EQUIVALENT, min(1.0, 0.5 + score)
        elif score <= self.neq_thr:
            label, conf = Label.NOT_EQUIVALENT, min(1.0, 0.5 + (1 - score))
        else:
            label, conf = Label.UNKNOWN, 0.3

        return VerifierVerdict(
            verifier_name=self.NAME,
            label=label,
            confidence=conf,
            score=score,
            details={
                "smatch_f1": f"{score:.3f}",
                "eq_threshold": f"{self.eq_thr:.2f}",
                "neq_threshold": f"{self.neq_thr:.2f}",
                "real_smatch_available": str(self._real_smatch is not None),
            },
        )
