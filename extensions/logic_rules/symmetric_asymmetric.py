"""Symmetric / Asymmetric relation rules.

  - Symmetric predicate p:   p(X, Y) <=> p(Y, X).
  - Asymmetric predicate p:  p(X, Y) entails ¬p(Y, X), so swapping the
    arguments produces a NON-equivalent (typically false-by-default) graph.

We expose two rules so callers can choose precisely which transformation to
apply. The shared lexicon SYMMETRIC_FRAMES / ASYMMETRIC_FRAMES is the source
of truth — extend it as needed for a particular corpus.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import penman

from .base import LogicRule, RuleMatch, register, swap_roles


SYMMETRIC_FRAMES: Set[str] = {
    # Marriage / partnership
    "marry-01",  # X marries Y <=> Y marries X
    # Family (symmetric only when looked at as "siblings")
    "sibling-01",
    # Geometry / relations
    "equal-01",
    "intersect-01",
    "border-01",
    "neighbor-01",
    # Mathematical / set
    "intersect-01",
    "overlap-01",
    "coincide-01",
    # Social
    "befriend-01",
    "agree-01",  # X agrees with Y <=> Y agrees with X (sometimes; pragmatic)
}


ASYMMETRIC_FRAMES: Set[str] = {
    # Comparative
    "exceed-01",
    "outrank-01",
    "outperform-01",
    "surpass-01",
    # Hierarchical
    "lead-01",
    "manage-01",
    "supervise-01",
    "report-01",  # report to
    # Family
    "parent-01",
    "child-01",
    "mother-01",
    "father-01",
    # Possession / authorship
    "own-01",
    "author-01",
    "create-01",
    # Causation
    "cause-01",
    "result-01",
}


def _find_arg_pair(g: penman.Graph, anchor: str):
    """Try common argument-pair conventions for binary relations.

    Returns (role_a, role_b) when both are present exactly once on `anchor`.
    The list is in priority order — :ARG0/:ARG1 is the most common, but some
    PropBank frames (e.g., marry-01 as parsed by parse_xfm) use :ARG1/:ARG2,
    and copular frames like have-degree-91 use :ARG1 + :ARG4 (subject + object
    of comparison).
    """
    candidates = [
        (":ARG0", ":ARG1"),
        (":ARG1", ":ARG2"),
        (":ARG1", ":ARG4"),  # have-degree-91 comparatives
        (":ARG0", ":ARG2"),
    ]
    for ra, rb in candidates:
        a_edges = g.edges(source=anchor, role=ra)
        b_edges = g.edges(source=anchor, role=rb)
        if len(a_edges) == 1 and len(b_edges) == 1:
            return ra, rb
    return None, None


# Additional asymmetric frame to handle "X is taller than Y" / comparative degrees.
ASYMMETRIC_FRAMES.add("have-degree-91")


@register
class SymmetricRule(LogicRule):
    name = "symmetric"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":instance" or t not in SYMMETRIC_FRAMES:
                continue
            ra, rb = _find_arg_pair(g, s)
            if ra is not None:
                matches.append(
                    RuleMatch(
                        anchor=s,
                        extras={"concept": t, "role_a": ra, "role_b": rb},
                    )
                )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        a, b = swap_roles(
            g, match.anchor, match.extras["role_a"], match.extras["role_b"]
        )
        return g if a is not None else None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        from .base import toggle_polarity_neg

        edges = g.edges(source=match.anchor, role=match.extras["role_a"])
        if edges:
            toggle_polarity_neg(g, edges[0].target)
            return g
        return None


@register
class AsymmetricRule(LogicRule):
    name = "asymmetric"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":instance" or t not in ASYMMETRIC_FRAMES:
                continue
            ra, rb = _find_arg_pair(g, s)
            if ra is not None:
                matches.append(
                    RuleMatch(
                        anchor=s,
                        extras={"concept": t, "role_a": ra, "role_b": rb},
                    )
                )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """For asymmetric p(X, Y), the equivalent transformation is
        ¬p(Y, X) — swap args AND negate the predicate."""
        from .base import toggle_polarity_neg

        a, b = swap_roles(
            g, match.anchor, match.extras["role_a"], match.extras["role_b"]
        )
        if a is None:
            return None
        toggle_polarity_neg(g, match.anchor)
        return g

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: swap args WITHOUT negating — produces p(Y, X) which is
        false-by-default for asymmetric predicates."""
        a, b = swap_roles(
            g, match.anchor, match.extras["role_a"], match.extras["role_b"]
        )
        return g if a is not None else None
