"""Implication rule: (A -> B) <=> (not A or B).

Three styles handled:

1. `have-condition-91` root (legacy AMR-LDA convention):
       (A -> B) becomes (or :op1 ¬A :op2 B) by relabeling root + flipping polarity.
2. `or` root (the reverse direction):
       (¬A or B) becomes (A -> B) — relabel + polarity flip.
3. `:condition` role (parse_xfm_bart_large standard output):
       (B :condition A) becomes a fresh `or` node with op1=¬A, op2=B.
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


def _rename_role(g: penman.Graph, parent: str, old: str, new: str) -> None:
    for i, (s, role, t) in enumerate(g.triples):
        if s == parent and role == old:
            g.triples[i] = (s, new, t)


def _fresh_var(g: penman.Graph, prefix: str = "or") -> str:
    """Mint a variable name that doesn't collide with existing instances."""
    used = {s for s, role, t in g.triples if role == ":instance"}
    if prefix not in used:
        return prefix
    i = 1
    while f"{prefix}{i}" in used:
        i += 1
    return f"{prefix}{i}"


@register
class ImplicationRule(LogicRule):
    name = "implication"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []

        # --- Style 1 + 2: have-condition-91 / or roots ---
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            if t == "have-condition-91":
                arg1 = g.edges(source=s, role=":ARG1")
                arg2 = g.edges(source=s, role=":ARG2")
                if len(arg1) == 1 and len(arg2) == 1:
                    matches.append(
                        RuleMatch(
                            anchor=s,
                            extras={
                                "form": "cond_to_or",
                                "arg1_target": arg1[0].target,
                                "arg2_target": arg2[0].target,
                            },
                        )
                    )
            elif t == "or":
                op1 = g.edges(source=s, role=":op1")
                op2 = g.edges(source=s, role=":op2")
                if len(op1) == 1 and len(op2) == 1:
                    matches.append(
                        RuleMatch(
                            anchor=s,
                            extras={
                                "form": "or_to_cond",
                                "op1_target": op1[0].target,
                                "op2_target": op2[0].target,
                            },
                        )
                    )

        # --- Style 3: :condition role ---
        for s, role, t in g.triples:
            if role != ":condition":
                continue
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "form": "condition_role_to_or",
                        "consequent": s,
                        "antecedent": t,
                    },
                )
            )

        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        anchor = match.anchor
        form = match.extras["form"]

        if form == "cond_to_or":
            replace_instance(g, anchor, "or")
            _rename_role(g, anchor, ":ARG2", ":op1")
            _rename_role(g, anchor, ":ARG1", ":op2")
            toggle_polarity_neg(g, match.extras["arg2_target"])
            return g

        if form == "or_to_cond":
            replace_instance(g, anchor, "have-condition-91")
            _rename_role(g, anchor, ":op1", ":ARG2")
            _rename_role(g, anchor, ":op2", ":ARG1")
            toggle_polarity_neg(g, match.extras["op1_target"])
            return g

        if form == "condition_role_to_or":
            cons = match.extras["consequent"]
            ant = match.extras["antecedent"]
            # Remove the original :condition edge
            if (cons, ":condition", ant) in g.triples:
                g.triples.remove((cons, ":condition", ant))
            else:
                return None
            # Create a fresh `or` node and re-root
            or_var = _fresh_var(g, "or")
            g.triples.insert(0, (or_var, ":instance", "or"))
            g.triples.append((or_var, ":op1", ant))
            g.triples.append((or_var, ":op2", cons))
            # Negate the antecedent (¬A)
            toggle_polarity_neg(g, ant)
            try:
                g._top = or_var
            except AttributeError:
                pass
            return g

        return None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        anchor = match.anchor
        form = match.extras["form"]

        if form == "cond_to_or":
            replace_instance(g, anchor, "or")
            _rename_role(g, anchor, ":ARG2", ":op1")
            _rename_role(g, anchor, ":ARG1", ":op2")
            return g

        if form == "or_to_cond":
            replace_instance(g, anchor, "have-condition-91")
            _rename_role(g, anchor, ":op1", ":ARG2")
            _rename_role(g, anchor, ":op2", ":ARG1")
            return g

        if form == "condition_role_to_or":
            # Same structure but skip the polarity flip — not equivalent.
            cons = match.extras["consequent"]
            ant = match.extras["antecedent"]
            if (cons, ":condition", ant) in g.triples:
                g.triples.remove((cons, ":condition", ant))
            else:
                return None
            or_var = _fresh_var(g, "or")
            g.triples.insert(0, (or_var, ":instance", "or"))
            g.triples.append((or_var, ":op1", ant))
            g.triples.append((or_var, ":op2", cons))
            # NO polarity toggle — non-equivalent
            try:
                g._top = or_var
            except AttributeError:
                pass
            return g

        return None
