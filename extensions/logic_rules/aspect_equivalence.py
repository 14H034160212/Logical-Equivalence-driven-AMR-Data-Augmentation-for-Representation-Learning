"""Aspect equivalence (UMR-grounded; lightweight AMR-layer implementation).

UMR distinguishes 8 aspect categories: state, activity, endeavor, performance,
habitual, generic, process, inceptive. AMR has no native aspect attribute, so
we approximate at the surface-paraphrase level: we add a non-truth-conditional
modifier that signals the aspect, then let the AMR-to-text generator produce
a fluent paraphrase that preserves the underlying event structure.

This is a deliberately CONSERVATIVE rule. It's an identity-equivalent in pure
truth-conditional terms — the rewritten sentence has the same truth conditions
as the original, but a different surface form. Useful for data augmentation
diversification and for evaluating LLM robustness to aspectual paraphrase.

Detection heuristics:
- Habitual: presence of `:freq` or `:every` modifiers on the main event verb
- Activity: progressive ("has been V-ing") or `:duration` modifiers
- Performance: default for past-tense or completed events with telic verbs
- (Other categories not detected at AMR layer.)

For each detected aspect, we add a small `:mod` decoration with a paraphrase-
trigger concept that the T5wtense generator renders as natural surface variation.
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import LogicRule, RuleMatch, register


# Adjuncts whose presence indicates a particular aspect class.
HABITUAL_MARKERS = {":freq", ":frequency"}
ACTIVITY_MARKERS = {":duration"}
TELIC_BASE_FRAMES = {"complete-01", "finish-01", "achieve-01"}


@register
class AspectEquivalenceRule(LogicRule):
    name = "aspect_equivalence"

    def _find_root_event(self, g: penman.Graph) -> Optional[str]:
        """Heuristic: the root variable (graph.top) is usually the main event."""
        try:
            return g.top
        except Exception:
            pass
        # Fall back: first node with a frame-style instance (concept-XX)
        for s, role, t in g.triples:
            if role == ":instance" and "-" in t and t.split("-")[-1].isdigit():
                return s
        return None

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        root = self._find_root_event(g)
        if root is None:
            return []
        # Inspect outgoing edges from root
        roles = {role for s, role, t in g.triples if s == root}
        aspect = "performance"  # default
        if HABITUAL_MARKERS & roles:
            aspect = "habitual"
        elif ACTIVITY_MARKERS & roles:
            aspect = "activity"
        # Always emit a match — the rule effectively round-trips through
        # T5wtense to produce a surface-level paraphrase.
        return [RuleMatch(anchor=root, extras={"aspect": aspect})]

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Tag the root event with an aspect marker. The marker is a
        non-truth-conditional `:mod (a / aspect-X)` that the AMR-to-text
        generator will incorporate (or ignore) naturally."""
        root = match.anchor
        aspect = match.extras["aspect"]
        marker_var = f"_asp_{aspect}"
        marker_concept = {
            "performance": "complete",  # event was completed
            "activity": "ongoing",       # event is ongoing
            "habitual": "habitual",      # event recurs
        }.get(aspect, "event")
        # Append the marker node + edge
        g.triples.insert(0, (marker_var, ":instance", marker_concept))
        g.triples.append((root, ":mod", marker_var))
        return g

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: apply the WRONG aspect marker.

        For an activity, claim it was completed (telic shift).
        For a performance, claim it's ongoing.
        For a habitual, claim it's a one-time event.
        """
        root = match.anchor
        aspect = match.extras["aspect"]
        wrong_marker = {
            "performance": "ongoing",
            "activity": "complete",
            "habitual": "one-time",
        }.get(aspect, "incorrect")
        marker_var = f"_asp_neg_{aspect}"
        g.triples.insert(0, (marker_var, ":instance", wrong_marker))
        g.triples.append((root, ":mod", marker_var))
        return g
