"""Transitivity rule (STUB).

  From (A -> B) AND (B -> C), conclude (A -> C).

This is a TWO-graph rule: it takes two AMR graphs as input (two implications
sharing a middle term) and produces a single inferred implication graph.

Implementation challenges
-------------------------
1. Cross-sentence anaphora resolution: to detect that the consequent of graph
   G1 (`(b / believe-01 ...)`) is the same entity as the antecedent of G2
   (`(b / believe-01 ...)`), we need coreference reasoning.
2. Predicate matching: AMR concepts must match (modulo sense, modulo lexical
   variation). PropBank framesets help but are not exhaustive.
3. UMR document-level :before / :depends-on annotations are a cleaner basis
   than raw AMR — that's why we plan to add `doc_temporal_transitivity` as a
   UMR-level rule once the AMR→UMR converter is in place.

For now this rule is a stub that raises NotImplementedError on apply. The
detect() method returns an empty list so it doesn't break the registry.
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import LogicRule, RuleMatch, register


@register
class TransitivityRule(LogicRule):
    name = "transitivity"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        # Single-graph detection is not meaningful for transitivity.
        # The two-graph API is on a TODO list below.
        return []

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        raise NotImplementedError(
            "Transitivity is a two-graph rule; use TransitivityRule.apply_two() "
            "once that API lands. See module docstring."
        )

    # TODO: API for two-graph rules
    # def apply_two(self, g1: penman.Graph, g2: penman.Graph) -> Optional[penman.Graph]:
    #     ...
