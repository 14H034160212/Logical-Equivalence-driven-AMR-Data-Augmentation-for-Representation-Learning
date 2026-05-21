"""De Morgan's law: ¬(A ∧ B) <=> (¬A) ∨ (¬B); ¬(A ∨ B) <=> (¬A) ∧ (¬B).

In AMR, negation is encoded as `:polarity -` on the node that the negation
scopes over. We detect nodes whose instance is `and` or `or` AND that have a
`:polarity -` argument (i.e., the whole conjunction/disjunction is negated).
We rewrite by:
  1. Removing the outer `:polarity -`.
  2. Flipping the connective (and ↔ or).
  3. Adding `:polarity -` to each of the two operand targets.

This is the canonical De Morgan transformation. For a non-equivalent negative
sample, we only flip the connective without distributing the negation.
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import (
    LogicRule,
    RuleMatch,
    has_polarity_neg,
    register,
    replace_instance,
    toggle_polarity_neg,
)


@register
class DeMorganRule(LogicRule):
    name = "de_morgan"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []

        # --- Pattern A: direct (and/or :polarity- :op1 X :op2 Y) ---
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            if t not in ("and", "or"):
                continue
            if not has_polarity_neg(g, s):
                continue
            op1 = g.edges(source=s, role=":op1")
            op2 = g.edges(source=s, role=":op2")
            if len(op1) == 1 and len(op2) == 1:
                matches.append(
                    RuleMatch(
                        anchor=s,
                        extras={
                            "pattern": "direct",
                            "connective": t,
                            "op1_target": op1[0].target,
                            "op2_target": op2[0].target,
                        },
                    )
                )

        # --- Pattern B: outer wrapper with :polarity- and a child and/or ---
        # E.g., (case-04 :polarity- :ARG1 (and :op1 X :op2 Y))
        # E.g., (verb-XX :polarity- :ARG0 (and :op1 X :op2 Y))
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            if not has_polarity_neg(g, s):
                continue
            # Look at every outgoing edge from s; if any target is itself an and/or
            # with op1 and op2, that's a Pattern B match.
            for src, edge_role, target in g.triples:
                if src != s or edge_role in (":instance", ":polarity"):
                    continue
                target_concept = None
                for s2, r2, t2 in g.triples:
                    if s2 == target and r2 == ":instance":
                        target_concept = t2
                        break
                if target_concept not in ("and", "or"):
                    continue
                op1 = g.edges(source=target, role=":op1")
                op2 = g.edges(source=target, role=":op2")
                if len(op1) == 1 and len(op2) == 1:
                    matches.append(
                        RuleMatch(
                            anchor=target,  # The and/or node, not the wrapper
                            extras={
                                "pattern": "wrapped",
                                "wrapper": s,
                                "wrapper_role": edge_role,
                                "connective": target_concept,
                                "op1_target": op1[0].target,
                                "op2_target": op2[0].target,
                            },
                        )
                    )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        anchor = match.anchor
        old_conn = match.extras["connective"]
        new_conn = "or" if old_conn == "and" else "and"
        pattern = match.extras.get("pattern", "direct")

        if pattern == "direct":
            # (and/or :polarity- :op1 X :op2 Y) → (or/and :op1 ¬X :op2 ¬Y)
            if has_polarity_neg(g, anchor):
                toggle_polarity_neg(g, anchor)
            replace_instance(g, anchor, new_conn)
            toggle_polarity_neg(g, match.extras["op1_target"])
            toggle_polarity_neg(g, match.extras["op2_target"])
            return g

        if pattern == "wrapped":
            # (wrapper :polarity- :role (and :op1 X :op2 Y))
            # → (wrapper :role (or :op1 ¬X :op2 ¬Y))   (negation pushed into operands)
            wrapper = match.extras["wrapper"]
            if has_polarity_neg(g, wrapper):
                toggle_polarity_neg(g, wrapper)
            replace_instance(g, anchor, new_conn)
            toggle_polarity_neg(g, match.extras["op1_target"])
            toggle_polarity_neg(g, match.extras["op2_target"])
            return g

        return None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: flip the connective without distributing negation.
        Result is NOT logically equivalent; useful as a contrastive sample."""
        anchor = match.anchor
        old_conn = match.extras["connective"]
        new_conn = "or" if old_conn == "and" else "and"
        pattern = match.extras.get("pattern", "direct")

        if pattern == "direct":
            if has_polarity_neg(g, anchor):
                toggle_polarity_neg(g, anchor)
            replace_instance(g, anchor, new_conn)
            return g
        if pattern == "wrapped":
            wrapper = match.extras["wrapper"]
            if has_polarity_neg(g, wrapper):
                toggle_polarity_neg(g, wrapper)
            replace_instance(g, anchor, new_conn)
            # deliberately NOT toggling op1/op2 polarities
            return g
        return None
