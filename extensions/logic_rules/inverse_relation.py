"""Inverse relation rule: p(X, Y) <=> p'(Y, X) for predicates with a lexical inverse.

Examples:
  - parent_of(X, Y) <=> child_of(Y, X)
  - buy(X, Y, Z) <=> sell(Y, X, Z)  (X is buyer of Z from Y / Y is seller of Z to X)
  - own(X, Y)    <=> belong_to(Y, X)

In AMR, predicates are PropBank framesets like `buy-01`, `sell-01`. We maintain
an inverse-frame dictionary and rewrite the predicate plus swap :ARG0 / :ARG1
(or other argument roles as appropriate).

To extend: add entries to INVERSE_FRAMES. For an empirical scale-up, parse the
PropBank frameset XMLs to extract `relsubject:` / `relparticular:` annotations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import penman

from .base import LogicRule, RuleMatch, register, swap_roles


# Manually curated inverse frame pairs.
# Format: forward_concept -> (inverse_concept, role_swap)
# role_swap is a tuple (role_a, role_b) meaning: under the inversed predicate,
# swap roles a and b. Most binary inverses use (":ARG0", ":ARG1").
INVERSE_FRAMES: Dict[str, Tuple[str, Tuple[str, str]]] = {
    # Buying / selling / trading
    "buy-01": ("sell-01", (":ARG0", ":ARG2")),
    "sell-01": ("buy-01", (":ARG0", ":ARG2")),
    "rent-01": ("rent-02", (":ARG0", ":ARG2")),
    "lease-01": ("lease-02", (":ARG0", ":ARG2")),
    # Family relations (frame-level — these are rare; most family relations
    # come through have-rel-role-91 which is handled separately below)
    "parent-01": ("child-01", (":ARG0", ":ARG1")),
    "child-01": ("parent-01", (":ARG0", ":ARG1")),
    "mother-01": ("son-01", (":ARG0", ":ARG1")),
    "father-01": ("son-01", (":ARG0", ":ARG1")),
    # Ownership
    "own-01": ("belong-01", (":ARG0", ":ARG1")),
    "belong-01": ("own-01", (":ARG0", ":ARG1")),
    "possess-01": ("belong-01", (":ARG0", ":ARG1")),
    # Giving / receiving / sending
    "give-01": ("receive-01", (":ARG0", ":ARG2")),
    "receive-01": ("give-01", (":ARG0", ":ARG2")),
    "send-01": ("receive-01", (":ARG0", ":ARG2")),
    "deliver-01": ("receive-01", (":ARG0", ":ARG2")),
    "lend-01": ("borrow-01", (":ARG0", ":ARG2")),
    "borrow-01": ("lend-01", (":ARG0", ":ARG2")),
    # Teaching / learning
    "teach-01": ("learn-01", (":ARG0", ":ARG2")),
    "learn-01": ("teach-01", (":ARG0", ":ARG2")),
    "instruct-01": ("learn-01", (":ARG0", ":ARG2")),
    # Voice-only (passive identity)
    "discover-01": ("discover-01", (":ARG0", ":ARG1")),
    "invent-01": ("invent-01", (":ARG0", ":ARG1")),
    "create-01": ("create-01", (":ARG0", ":ARG1")),
    "write-01": ("write-01", (":ARG0", ":ARG1")),
    "build-01": ("build-01", (":ARG0", ":ARG1")),
    # Order / sequence
    "lead-01": ("follow-01", (":ARG0", ":ARG1")),
    "follow-01": ("lead-01", (":ARG0", ":ARG1")),
    "precede-01": ("follow-01", (":ARG0", ":ARG1")),
    "succeed-01": ("precede-01", (":ARG0", ":ARG1")),
    # Praise / appreciate
    "praise-01": ("appreciate-01", (":ARG0", ":ARG1")),
    "appreciate-01": ("praise-01", (":ARG0", ":ARG1")),
    "criticize-01": ("criticize-01", (":ARG0", ":ARG1")),
    # Defense / attack
    "attack-01": ("defend-01", (":ARG0", ":ARG1")),
    "defend-01": ("attack-01", (":ARG0", ":ARG1")),
    # Emotional / cognitive (active to passive)
    "love-01": ("love-01", (":ARG0", ":ARG1")),
    "hate-01": ("hate-01", (":ARG0", ":ARG1")),
    "like-01": ("like-01", (":ARG0", ":ARG1")),
    "trust-01": ("trust-01", (":ARG0", ":ARG1")),
    "know-01": ("know-01", (":ARG0", ":ARG1")),
    "remember-01": ("remember-01", (":ARG0", ":ARG1")),
    # Award / receive award
    "award-01": ("award-01", (":ARG1", ":ARG2")),  # X awarded to Y ↔ Y received X
    # Employment
    "employ-01": ("work-09", (":ARG0", ":ARG1")),  # X employs Y ↔ Y works for X
    "hire-01": ("work-09", (":ARG0", ":ARG1")),
    # Travel / location
    "visit-01": ("visit-01", (":ARG0", ":ARG1")),
    "see-01": ("see-01", (":ARG0", ":ARG1")),
}


# Role-name inverses for have-rel-role-91 and have-org-role-91.
# These frames encode "X's <role> is Y" — the inverse swaps X↔Y and inverts
# the role name (e.g., 'mother' ↔ 'son'). Gender-specific inverses approximate
# (we don't know the gender of the inverse).
ROLE_INVERSES: Dict[str, str] = {
    "mother": "son",
    "father": "son",
    "parent": "child",
    "son": "parent",
    "daughter": "parent",
    "child": "parent",
    "husband": "wife",
    "wife": "husband",
    "sibling": "sibling",
    "brother": "sibling",
    "sister": "sibling",
    "grandfather": "grandson",
    "grandmother": "grandson",
    "grandson": "grandparent",
    "granddaughter": "grandparent",
    "uncle": "nephew",
    "aunt": "nephew",
    "nephew": "uncle",
    "niece": "uncle",
    "cousin": "cousin",
    "friend": "friend",
    "neighbor": "neighbor",
    "mentor": "mentee",
    "mentee": "mentor",
    "boss": "employee",
    "employee": "boss",
    "manager": "report",
    "teacher": "student",
    "student": "teacher",
    "doctor": "patient",
    "patient": "doctor",
    "boyfriend": "girlfriend",
    "girlfriend": "boyfriend",
    "client": "provider",
    "customer": "provider",
    # Organizational
    "capital": "country",  # capital of -> country whose capital is
    "president": "country",
    "ceo": "company",
    "manager": "team",
}


@register
class InverseRelationRule(LogicRule):
    name = "inverse_relation"

    INVERSE_FRAMES = INVERSE_FRAMES  # exposed for monkey-patching/extension
    ROLE_INVERSES = ROLE_INVERSES

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []

        # --- Pattern A: PropBank frame inverses (buy/sell, give/receive, etc.) ---
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            if t not in self.INVERSE_FRAMES:
                continue
            new_concept, (ra, rb) = self.INVERSE_FRAMES[t]
            ra_edges = g.edges(source=s, role=ra)
            rb_edges = g.edges(source=s, role=rb)
            if len(ra_edges) == 1 and len(rb_edges) == 1:
                matches.append(
                    RuleMatch(
                        anchor=s,
                        extras={
                            "pattern": "frame",
                            "old_concept": t,
                            "new_concept": new_concept,
                            "role_a": ra,
                            "role_b": rb,
                        },
                    )
                )

        # --- Pattern B: have-rel-role-91 / have-org-role-91 ---
        # Frame: (have-rel-role-91 :ARG0 X :ARG1 Y :ARG2 (role / mother))
        # Means: "X's mother is Y" — invert to "Y's son is X" by swapping
        # :ARG0/:ARG1 and replacing the :ARG2 role concept.
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            if t not in ("have-rel-role-91", "have-org-role-91"):
                continue
            arg0 = g.edges(source=s, role=":ARG0")
            arg1 = g.edges(source=s, role=":ARG1")
            arg2 = g.edges(source=s, role=":ARG2")
            if not (len(arg0) == 1 and len(arg1) == 1 and len(arg2) == 1):
                continue
            role_node = arg2[0].target
            # Find the role concept
            role_concept = None
            for s2, r2, t2 in g.triples:
                if s2 == role_node and r2 == ":instance":
                    role_concept = t2
                    break
            if role_concept is None or role_concept not in self.ROLE_INVERSES:
                continue
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "pattern": "role_role",
                        "frame": t,
                        "role_node": role_node,
                        "role_concept": role_concept,
                        "inverse_role": self.ROLE_INVERSES[role_concept],
                    },
                )
            )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        pattern = match.extras.get("pattern", "frame")

        if pattern == "frame":
            anchor = match.anchor
            new_concept = match.extras["new_concept"]
            ra = match.extras["role_a"]
            rb = match.extras["role_b"]
            for i, (s, role, t) in enumerate(g.triples):
                if s == anchor and role == ":instance":
                    g.triples[i] = (s, role, new_concept)
                    break
            a, b = swap_roles(g, anchor, ra, rb)
            return g if a is not None else None

        if pattern == "role_role":
            anchor = match.anchor
            role_node = match.extras["role_node"]
            inv_role = match.extras["inverse_role"]
            # Swap :ARG0 / :ARG1 on the have-rel-role-91 node
            a, b = swap_roles(g, anchor, ":ARG0", ":ARG1")
            if a is None:
                return None
            # Replace the role concept on the :ARG2 sub-node
            for i, (s, role, t) in enumerate(g.triples):
                if s == role_node and role == ":instance":
                    g.triples[i] = (s, role, inv_role)
                    return g
            return None

        return None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: swap arguments without changing the predicate.
        For asymmetric predicates this produces a non-equivalent graph;
        for symmetric ones it's trivially equivalent and not a useful negative."""
        pattern = match.extras.get("pattern", "frame")
        if pattern == "frame":
            ra = match.extras["role_a"]
            rb = match.extras["role_b"]
            a, b = swap_roles(g, match.anchor, ra, rb)
            return g if a is not None else None
        if pattern == "role_role":
            # Swap the args but DO NOT change the role concept — produces
            # nonsensical "Tom Cruise's mother is Mary"-style for the wrong direction
            a, b = swap_roles(g, match.anchor, ":ARG0", ":ARG1")
            return g if a is not None else None
        return None
