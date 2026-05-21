"""Commutative rule: (A and B) <=> (B and A); (A or B) <=> (B or A).

Ported from legacy/amr_lda/logical_equivalence_functions.py:commutative.

We treat both `and` and `or` AMR roots, although the legacy paper version only
fired on `and`. For `or`, the rule is also classically valid and harmless.
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import (
    LogicRule,
    RuleMatch,
    has_polarity_neg,
    register,
    swap_roles,
    toggle_polarity_neg,
)


@register
class CommutativeRule(LogicRule):
    name = "commutative"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":instance" or t not in ("and", "or"):
                continue
            anchor = s
            op1 = g.edges(source=anchor, role=":op1")
            op2 = g.edges(source=anchor, role=":op2")
            if len(op1) == 1 and len(op2) == 1:
                matches.append(
                    RuleMatch(
                        anchor=anchor,
                        extras={
                            "op1_target": op1[0].target,
                            "op2_target": op2[0].target,
                            "connective": t,
                        },
                    )
                )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        a, b = swap_roles(g, match.anchor, ":op1", ":op2")
        return g if a is not None else None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative sample: flip polarity on op2 target (NOT logically equivalent)."""
        op2 = match.extras["op2_target"]
        toggle_polarity_neg(g, op2)
        return g
