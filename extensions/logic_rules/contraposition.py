"""Contraposition rule: (A -> B) <=> (not B -> not A).

Supports two AMR encodings of conditionals:

1. `have-condition-91` form (PropBank frame): the original legacy paper uses
   this. Root is the `have-condition-91` node; :ARG1 is the consequent, :ARG2
   is the antecedent.

2. `:condition` form (role-based): the standard parse_xfm_bart_large output
   for "If A, then B" sentences. Root is B (consequent); B has an outgoing
   `:condition` edge to A (antecedent). This is what real AMR parsers tend
   to produce on `If X, then Y` sentences.

The detect/apply methods handle both styles via a `style` flag in extras.
"""

from __future__ import annotations

import copy
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
class ContrapositionRule(LogicRule):
    name = "contraposition"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []

        # --- Style 1: have-condition-91 (legacy) ---
        for s, role, t in g.triples:
            if role != ":instance" or t != "have-condition-91":
                continue
            arg1s = g.edges(source=s, role=":ARG1")
            arg2s = g.edges(source=s, role=":ARG2")
            if len(arg1s) == 1 and len(arg2s) == 1:
                matches.append(
                    RuleMatch(
                        anchor=s,
                        extras={
                            "style": "have_condition_91",
                            "arg1_target": arg1s[0].target,
                            "arg2_target": arg2s[0].target,
                        },
                    )
                )

        # --- Style 2: :condition role (parse_xfm_bart_large default) ---
        for s, role, t in g.triples:
            if role != ":condition":
                continue
            # s = consequent node, t = antecedent node
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "style": "condition_role",
                        "consequent": s,
                        "antecedent": t,
                    },
                )
            )

        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        style = match.extras.get("style", "have_condition_91")

        if style == "have_condition_91":
            anchor = match.anchor
            arg1, arg2 = match.extras["arg1_target"], match.extras["arg2_target"]
            a, b = swap_roles(g, anchor, ":ARG1", ":ARG2")
            if a is None:
                return None
            toggle_polarity_neg(g, arg1)
            toggle_polarity_neg(g, arg2)
            return g

        if style == "condition_role":
            cons = match.extras["consequent"]
            ant = match.extras["antecedent"]
            # Remove the original (cons :condition ant) edge
            if (cons, ":condition", ant) in g.triples:
                g.triples.remove((cons, ":condition", ant))
            else:
                return None
            # Reverse the condition direction
            g.triples.append((ant, ":condition", cons))
            # Toggle polarity on both sides — implements ¬B → ¬A
            toggle_polarity_neg(g, cons)
            toggle_polarity_neg(g, ant)
            # Re-root the graph at the new top (formerly antecedent)
            # penman.Graph stores the top in `_top` (penman >= 1.1); fall back if missing.
            try:
                g._top = ant
            except AttributeError:
                pass
            return g

        return None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Single-side polarity flip — NOT equivalent (contrastive negative)."""
        style = match.extras.get("style", "have_condition_91")

        if style == "have_condition_91":
            toggle_polarity_neg(g, match.extras["arg2_target"])
            return g

        if style == "condition_role":
            toggle_polarity_neg(g, match.extras["antecedent"])
            return g

        return None
