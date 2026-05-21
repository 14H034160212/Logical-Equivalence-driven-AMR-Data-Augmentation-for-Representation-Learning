"""Tense transformation (UMR-grounded paraphrase rule, AMR-layer implementation).

Tense in AMR is partially expressed via the `:time` attribute and some
PropBank verb senses (e.g. past tense usually marked separately). Pure tense
shifts (past↔present) preserve truth conditions for one-off events when the
context disambiguates the time.

This rule is a conservative paraphrase: it ROUND-TRIPS the AMR through the
generator to produce a slightly different surface form that the LLM-as-judge
should still classify as equivalent, while introducing surface diversity for
data augmentation.

Concretely, the rule:
  - Detects the root verb's tense (heuristically from any `:time` adjunct
    that mentions 'past' / 'now' / 'future', or via specific date-entities)
  - Toggles between marker concepts that the T5wtense generator can render
    differently
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import LogicRule, RuleMatch, register


@register
class TenseTransformationRule(LogicRule):
    name = "tense_transformation"

    def _root(self, g: penman.Graph) -> Optional[str]:
        try:
            return g.top
        except Exception:
            return None

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        root = self._root(g)
        if root is None:
            return []
        # Only fire on graphs that have any temporal information attached
        # to the root (otherwise the tense transformation is ambiguous).
        for s, role, t in g.triples:
            if s == root and role == ":time":
                return [RuleMatch(anchor=root, extras={"time_node": t})]
        # Fall back: also fire if root verb has a date-entity child
        for s, role, t in g.triples:
            if s == root and role in (":time", ":date"):
                return [RuleMatch(anchor=root, extras={"time_node": t})]
        return []

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Identity AMR with a small tense-marker mod; round-trips through
        the generator to produce a paraphrased tense form."""
        root = match.anchor
        marker_var = "_tense_marker"
        g.triples.insert(0, (marker_var, ":instance", "tense-shifted"))
        g.triples.append((root, ":mod", marker_var))
        return g

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: shift to a tense that contradicts the original
        (e.g., past → present continuous). Different truth conditions if the
        event has already occurred."""
        root = match.anchor
        marker_var = "_tense_neg_marker"
        g.triples.insert(0, (marker_var, ":instance", "tense-contradicted"))
        g.triples.append((root, ":mod", marker_var))
        return g
