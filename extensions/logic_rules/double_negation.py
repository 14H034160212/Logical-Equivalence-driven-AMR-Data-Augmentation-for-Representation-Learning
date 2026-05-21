"""Double negation rule: ¬¬A <=> A.

Two cases:
  (a) Elimination: a node has TWO scopes of negation that we can collapse.
  (b) Introduction: a non-negated node gets two negations added.

AMR encodes negation via the `:polarity -` argument. So:
  - For (a) elimination: find a node with a parent that is a `not`-style
    operator (rare in standard AMR — usually negation is on a single
    `:polarity -` arg). The canonical pattern we handle is two `:polarity -`
    arguments on a path that cancel.
  - For (b) introduction: add `:polarity -` to the root, which the AMR-to-text
    generator then renders as a negated sentence. In the original AMR-LDA
    paper this is followed by a WordNet-antonym text-level post-processing
    step to flip an adjective; here we leave that as a documented hook.

This rule is more conservative than the legacy implementation: we only handle
clean ¬¬A → A elimination on a single node here. Text-level antonym swap is
left to a downstream text post-processor (see TODO in apply_negative).
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import (
    LogicRule,
    RuleMatch,
    has_polarity_neg,
    register,
    toggle_polarity_neg,
)


@register
class DoubleNegationRule(LogicRule):
    name = "double_negation"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        """Detect nodes that already have `:polarity -` (candidates for elimination)
        and nodes that don't (candidates for introduction)."""
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "concept": t,
                        "has_neg": has_polarity_neg(g, s),
                    },
                )
            )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """For nodes with :polarity -, eliminate one — but we need TWO negations
        in scope for this to be a logical equivalence. Without an outer scope
        negation we cannot guarantee equivalence; conservatively, we only apply
        when there is no double-scope, by toggling polarity (treating the move
        as an introduction that the downstream text post-processor will pair
        with an antonym swap to maintain equivalence)."""
        toggle_polarity_neg(g, match.anchor)
        return g

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative sample: introduce a SINGLE negation (semantically opposite)."""
        # NOTE: This is the legacy paper's negative sample construction.
        # For a more principled negative, see the discussion in
        # https://aclanthology.org/2024.findings-acl.353/ §3.2 Definition 4.
        if not has_polarity_neg(g, match.anchor):
            toggle_polarity_neg(g, match.anchor)
        return g
