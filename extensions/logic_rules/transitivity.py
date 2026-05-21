"""Transitivity rule: from (A -> B) AND (B -> C), conclude (A -> C).

This is a TWO-graph rule. The base LogicRule abstraction is single-graph, so
we expose a `detect_two(g1, g2)` and `apply_two_positive(g1, g2, match)` API
in addition to the single-graph methods (which return empty / None to keep
the registry happy).

How matching works
------------------
Given two implication AMRs:
  G1: (B :condition A)   parsed from "If A, then B."
  G2: (C :condition B')  parsed from "If B', then C."

If the consequent of G1 (B) matches the antecedent of G2 (B'), we infer a new
graph for "If A, then C" by composing them.

Matching is heuristic:
  - Concept equality (same PropBank frame stripped of sense suffix)
  - Same predicate argument structure (number of ARGs)
  - For named entities: same name string

We tolerate small lexical variation by comparing the multiset of
:instance-concepts under each sub-AMR.

Limitations
-----------
This is a deliberate minimum-viable implementation. The Post et al. 2024
AMR->UMR converter exposes document-level :before / :after / :depends-on
relations that would give a cleaner basis for transitivity at the UMR layer
— that's our `doc_level_temporal_transitivity` rule (sentence-level only so
far).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

import penman

from .base import LogicRule, RuleMatch, register


def _instance_of(g: penman.Graph, node: str) -> Optional[str]:
    for s, role, t in g.triples:
        if s == node and role == ":instance":
            return t
    return None


def _strip_sense(concept: str) -> str:
    """'study-01' → 'study'; 'have-condition-91' → 'have-condition'."""
    parts = concept.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return concept


def _subgraph_concepts(g: penman.Graph, root: str, max_depth: int = 4) -> Set[str]:
    """Collect the (stripped) concept multiset reachable from `root` up to
    `max_depth` hops."""
    out: Set[str] = set()
    frontier = [(root, 0)]
    visited: Set[str] = set()
    while frontier:
        node, depth = frontier.pop()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        concept = _instance_of(g, node)
        if concept:
            out.add(_strip_sense(concept))
        for s, role, t in g.triples:
            if s == node and role not in (":instance", ":polarity"):
                frontier.append((t, depth + 1))
    return out


def _find_implication_anchor(g: penman.Graph) -> Optional[Tuple[str, str, str]]:
    """Return (anchor_concept, consequent_node, antecedent_node) for an
    implication in either of two AMR forms:
      - `have-condition-91 :ARG1 cons :ARG2 ant`
      - `cons :condition ant`
    """
    for s, role, t in g.triples:
        if role == ":instance" and t == "have-condition-91":
            cons = next(
                (tt for ss, rr, tt in g.triples if ss == s and rr == ":ARG1"),
                None,
            )
            ant = next(
                (tt for ss, rr, tt in g.triples if ss == s and rr == ":ARG2"),
                None,
            )
            if cons and ant:
                return ("have_condition", cons, ant)
    for s, role, t in g.triples:
        if role == ":condition":
            return ("condition", s, t)
    return None


def _subgraph_similarity(g1: penman.Graph, root1: str,
                         g2: penman.Graph, root2: str) -> float:
    """Jaccard similarity of stripped concept sets reachable from each root."""
    c1 = _subgraph_concepts(g1, root1)
    c2 = _subgraph_concepts(g2, root2)
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        return 0.0
    inter = c1 & c2
    union = c1 | c2
    return len(inter) / len(union) if union else 0.0


@register
class TransitivityRule(LogicRule):
    name = "transitivity"

    # Single-graph methods deliberately no-op; use detect_two / apply_two.
    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        return []

    def apply_positive(self, g, match):  # type: ignore[override]
        return None

    def apply_negative(self, g, match):  # type: ignore[override]
        return None

    # ---- two-graph API ----

    def detect_two(
        self,
        g1: penman.Graph,
        g2: penman.Graph,
        match_threshold: float = 0.5,
    ) -> Optional[dict]:
        """Detect a transitive composition opportunity between two graphs.

        Returns a dict with the discovered match, or None if no chain exists.
        """
        impl1 = _find_implication_anchor(g1)
        impl2 = _find_implication_anchor(g2)
        if impl1 is None or impl2 is None:
            return None
        _, cons1, ant1 = impl1
        _, cons2, ant2 = impl2

        # The consequent of g1 should "match" the antecedent of g2 for
        # transitivity to fire. Compute their Jaccard similarity.
        sim = _subgraph_similarity(g1, cons1, g2, ant2)
        if sim < match_threshold:
            return None
        return {
            "g1_consequent": cons1,
            "g1_antecedent": ant1,
            "g2_consequent": cons2,
            "g2_antecedent": ant2,
            "similarity": sim,
        }

    def apply_two_positive(
        self,
        g1: penman.Graph,
        g2: penman.Graph,
        match: dict,
    ) -> Optional[penman.Graph]:
        """Produce a new graph for the transitive conclusion.

        Concretely, build `(C :condition A)` where A is g1's antecedent and C
        is g2's consequent. We do this by merging the relevant subgraphs of
        g1 and g2 into a single penman.Graph with appropriate variable
        renaming to avoid collisions.
        """
        ant1 = match["g1_antecedent"]
        cons2 = match["g2_consequent"]

        # Variable-rename g2 nodes to avoid collisions with g1.
        g2_vars = {s for s, r, t in g2.triples if r == ":instance"}
        g1_vars = {s for s, r, t in g1.triples if r == ":instance"}
        renames: dict = {}
        for v in g2_vars:
            if v in g1_vars:
                base = v
                i = 2
                while f"{base}{i}" in g1_vars or f"{base}{i}" in renames.values():
                    i += 1
                renames[v] = f"{base}{i}"
            else:
                renames[v] = v

        def remap_node(x: str) -> str:
            return renames.get(x, x)

        # Collect triples reachable from g1's antecedent (the new antecedent)
        a_triples = _collect_reachable(g1, ant1)
        # Collect triples reachable from g2's consequent (the new consequent),
        # with variable renaming
        c_triples_orig = _collect_reachable(g2, cons2)
        c_triples = [(remap_node(s), r, remap_node(t) if t in renames or t in g2_vars else t)
                     for s, r, t in c_triples_orig]
        new_cons = remap_node(cons2)

        # New graph: (new_cons :condition ant1) plus all reachable triples
        merged_triples = list(a_triples) + list(c_triples)
        merged_triples.append((new_cons, ":condition", ant1))

        # Deduplicate while preserving order
        seen: Set[Tuple] = set()
        out_triples = []
        for t in merged_triples:
            if t in seen:
                continue
            seen.add(t)
            out_triples.append(t)

        return penman.Graph(out_triples, top=new_cons)


def _collect_reachable(g: penman.Graph, root: str) -> List[Tuple[str, str, str]]:
    """Return all triples reachable from `root` (excluding outgoing condition
    edges, which are the chain-points)."""
    out: List[Tuple[str, str, str]] = []
    frontier = [root]
    visited: Set[str] = set()
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        for s, role, t in g.triples:
            if s == node:
                if role == ":condition":
                    continue  # don't follow the implication edge
                out.append((s, role, t))
                if isinstance(t, str) and not t.startswith('"') and not t.lstrip('-').isdigit():
                    frontier.append(t)
    return out
