"""Abstract base class for logical equivalence rules over AMR / UMR graphs.

This refactor extracts the pattern that is currently duplicated across
`contraposition`, `commutative`, `implication`, `demorgan`, etc. in the original
`logical_equivalence_functions.py` into a single extensible `LogicRule` interface.

Each rule is responsible for:
  1. detect(g)         -> a list of applicable Match points in the graph
  2. apply_positive    -> produce a logically equivalent graph from a match
  3. apply_negative    -> produce a non-equivalent graph (for contrastive negative
                          sample mining; used in AMR-LDA's stage-1 training)

The legacy bulk API
    fn(graphs, sentence_list, logic_word_list)
        -> (return_graphs, label_list, sentence_and_tag_list)
is reproduced by `LogicRule.transform_batch(...)` so existing call sites in
`BERT/utils_multiple_choice.py` and the script generators don't break.

New rules — De Morgan, Transitivity, Inverse Relation, Symmetric/Asymmetric,
Predicate Implication, UMR-level Aspect/Modal/Temporal — slot in as separate
subclasses in sibling files.
"""

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import penman


@dataclass
class RuleMatch:
    """A position in a graph where a rule can fire.

    `anchor` is the node id of the construction the rule pivots on (e.g. the
    `have-condition-91` node for contraposition). `extras` carries any
    rule-specific metadata so subclasses don't have to re-traverse the graph.
    """

    anchor: str
    extras: dict = field(default_factory=dict)


@dataclass
class RuleResult:
    """One rule application produces a positive and (optionally) negative sample.

    A `None` for either field means the rule could not produce that variant —
    e.g., for `commutative` we always produce both, but for `transitivity`
    the negative sample is constructed differently and may be absent.
    """

    positive_graph: Optional[str] = None  # penman-encoded string
    negative_graph: Optional[str] = None
    rule_name: str = ""
    explanation: str = ""


class LogicRule(abc.ABC):
    """Abstract base. Subclasses must set `name` and implement detect/apply_*."""

    name: str = "base"

    # --- core API --------------------------------------------------------

    @abc.abstractmethod
    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        """Return all positions in `g` where this rule fires (may be empty)."""

    @abc.abstractmethod
    def apply_positive(self, g: penman.Graph, match: RuleMatch) -> Optional[penman.Graph]:
        """Produce a logically equivalent graph. Return None if not applicable."""

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Produce a logically NON-equivalent graph for contrastive training.

        Default: return None (no negative sample). Override for rules that have
        a principled negative construction (e.g., contraposition can produce
        partial-negation forms).
        """
        return None

    # --- driver ----------------------------------------------------------

    def apply(self, g: penman.Graph) -> List[RuleResult]:
        """Run the rule on a graph, returning all (positive, negative) pairs."""
        results: List[RuleResult] = []
        for m in self.detect(g):
            pos = self.apply_positive(copy.deepcopy(g), m)
            neg = self.apply_negative(copy.deepcopy(g), m)
            results.append(
                RuleResult(
                    positive_graph=penman.encode(pos) if pos is not None else None,
                    negative_graph=penman.encode(neg) if neg is not None else None,
                    rule_name=self.name,
                )
            )
        return results

    # --- legacy bulk API for backward compat with original codebase ------

    def transform_batch(
        self,
        graphs: Sequence[Optional[str]],
        sentence_list: Sequence[str],
        logic_word_list: Sequence[str],
    ) -> Tuple[List[str], List[int], List[List[str]]]:
        """Reproduce the original `contraposition(graphs, ...)` style signature.

        Returns
        -------
        return_list : list of penman-encoded graphs (positives and negatives interleaved)
        label_list  : 1 for positive (equivalent), 0 for negative (non-equivalent)
        sentence_and_tag_list : metadata mirroring the original API
        """
        return_list: List[str] = []
        label_list: List[int] = []
        sentence_and_tag_list: List[List[str]] = []

        for idx, graph_str in enumerate(graphs):
            if graph_str is None:
                continue
            try:
                g = penman.decode(graph_str)
            except Exception:
                continue
            for res in self.apply(g):
                if res.positive_graph is not None:
                    return_list.append(res.positive_graph)
                    label_list.append(1)
                    sentence_and_tag_list.append(
                        [sentence_list[idx], logic_word_list[idx]]
                    )
                if res.negative_graph is not None:
                    return_list.append(res.negative_graph)
                    label_list.append(0)
                    sentence_and_tag_list.append(
                        [sentence_list[idx], logic_word_list[idx]]
                    )
        return return_list, label_list, sentence_and_tag_list

    # --- introspection ---------------------------------------------------

    def is_applicable(self, g: penman.Graph) -> bool:
        return bool(self.detect(g))

    def __repr__(self) -> str:
        return f"<LogicRule {self.name}>"


# ---------------------------------------------------------------------------
# Helper utilities shared across subclasses
# ---------------------------------------------------------------------------


def has_polarity_neg(g: penman.Graph, node: str) -> bool:
    return (node, ":polarity", "-") in g.triples


def toggle_polarity_neg(g: penman.Graph, node: str) -> None:
    """In-place toggle of `:polarity -` on `node`."""
    triple = (node, ":polarity", "-")
    if triple in g.triples:
        g.triples.remove(triple)
    else:
        g.triples.append(triple)


def negate_with_demorgan(g: penman.Graph, node: str) -> None:
    """Apply ¬ to `node` and distribute over conjunction/disjunction via De Morgan.

    - ¬(A ∧ B) → (¬A) ∨ (¬B)  → in AMR: switch `and` → `or`, recurse into each :op
    - ¬(A ∨ B) → (¬A) ∧ (¬B)  → switch `or` → `and`, recurse into each :op
    - ¬atom    → toggle `:polarity -` on the atom

    The distributed form keeps the polarity count aligned with how T5wtense
    actually renders negated conjunctions ("not A or not B"), so the
    self-consistency check passes on conjunctive-antecedent contraposition
    cases like S008, S028, S045.
    """
    concept = find_instance_target(g, node)
    if concept == "and":
        replace_instance(g, node, "or")
        for s, role, t in list(g.triples):
            if s == node and role.startswith(":op"):
                negate_with_demorgan(g, t)
    elif concept == "or":
        replace_instance(g, node, "and")
        for s, role, t in list(g.triples):
            if s == node and role.startswith(":op"):
                negate_with_demorgan(g, t)
    else:
        toggle_polarity_neg(g, node)


def find_instance_target(g: penman.Graph, node: str) -> Optional[str]:
    """Return the AMR concept (e.g. 'have-condition-91') for a given node."""
    for s, role, t in g.triples:
        if s == node and role == ":instance":
            return t
    return None


def replace_instance(g: penman.Graph, node: str, new_concept: str) -> None:
    """In-place change of the concept for `node`."""
    for i, (s, role, t) in enumerate(g.triples):
        if s == node and role == ":instance":
            g.triples[i] = (s, role, new_concept)
            return


def swap_roles(
    g: penman.Graph, node: str, role_a: str, role_b: str
) -> Tuple[Optional[str], Optional[str]]:
    """Swap the targets of two roles on the same parent node. Returns the old targets."""
    a_target = b_target = None
    for s, r, t in g.triples:
        if s == node and r == role_a:
            a_target = t
        elif s == node and r == role_b:
            b_target = t
    if a_target is None or b_target is None:
        return a_target, b_target
    g.triples.remove((node, role_a, a_target))
    g.triples.remove((node, role_b, b_target))
    g.triples.append((node, role_a, b_target))
    g.triples.append((node, role_b, a_target))
    return a_target, b_target


# ---------------------------------------------------------------------------
# Registry — populated by subclass modules at import time
# ---------------------------------------------------------------------------


_REGISTRY: dict = {}


def register(rule_cls):
    """Class decorator that auto-registers a LogicRule subclass by `name`."""
    inst = rule_cls()
    _REGISTRY[inst.name] = inst
    return rule_cls


def get_rule(name: str) -> LogicRule:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown rule: {name}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_rules() -> List[LogicRule]:
    return list(_REGISTRY.values())


def rule_names() -> List[str]:
    return sorted(_REGISTRY.keys())
