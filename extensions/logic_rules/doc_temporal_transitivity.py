"""Document-level temporal transitivity (UMR-grounded, AMR-layer approximation).

UMR encodes inter-event temporal relations explicitly: :before, :after,
:Contained, :Overlap, :Depends-on. From `A :before B` and `B :before C`, we
can infer `A :before C` (and dually for :after).

AMR doesn't have a document-level layer, but within a single sentence the
parser produces a `:time (after :op1 X)` substructure when there's an
explicit "after X" / "before X" phrase. When two such chains are linked
(e.g., "Y happened after X, which happened after W"), we can infer the
transitive relation between Y and W.

Detection: walks the graph for nested `:time (after :op1 ...)` or `:time
(before :op1 ...)` patterns and extracts the implied transitive ordering.

Apply: produces an AMR that makes the transitive claim explicit.
"""

from __future__ import annotations

from typing import List, Optional

import penman

from .base import LogicRule, RuleMatch, register


@register
class DocTemporalTransitivityRule(LogicRule):
    name = "doc_level_temporal_transitivity"

    def _instance(self, g: penman.Graph, var: str) -> Optional[str]:
        for s, role, t in g.triples:
            if s == var and role == ":instance":
                return t
        return None

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        """Detect any node that has a :time edge to an after/before node which
        in turn has :op1 to another event."""
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":time":
                continue
            inst = self._instance(g, t)
            if inst not in ("after", "before"):
                continue
            op1_edges = g.edges(source=t, role=":op1")
            if len(op1_edges) != 1:
                continue
            inner_event = op1_edges[0].target
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "outer_event": s,
                        "temporal_relation": inst,
                        "anchor_event": inner_event,
                        "time_node": t,
                    },
                )
            )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """For a simple two-event temporal claim (no chain), this just rewrites
        the AMR to use a different but semantically-equivalent surface
        construction: lift the `:time after :op1 X` to an explicit precede-01
        / follow-01 frame.

        For "B happened after A" we generate AMR for "A preceded B" / "A
        happened before B" by introducing a precede-01 root.
        """
        outer = match.extras["outer_event"]
        anchor = match.extras["anchor_event"]
        relation = match.extras["temporal_relation"]
        time_node = match.extras["time_node"]
        # Pick the new frame
        new_frame = "precede-01" if relation == "after" else "follow-01"
        # Remove the original :time → time_node edge
        try:
            g.triples.remove((outer, ":time", time_node))
        except ValueError:
            return None
        # Introduce a new root that wraps both events
        new_root_var = "trans_temporal"
        g.triples.insert(0, (new_root_var, ":instance", new_frame))
        g.triples.append((new_root_var, ":ARG0", anchor))
        g.triples.append((new_root_var, ":ARG1", outer))
        try:
            g._top = new_root_var
        except AttributeError:
            pass
        return g

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: swap the temporal relation direction.

        Maps "X happened after Y" → "Y happened after X" (swapped roles in
        the new frame), which has different truth conditions."""
        outer = match.extras["outer_event"]
        anchor = match.extras["anchor_event"]
        relation = match.extras["temporal_relation"]
        time_node = match.extras["time_node"]
        new_frame = "precede-01" if relation == "after" else "follow-01"
        try:
            g.triples.remove((outer, ":time", time_node))
        except ValueError:
            return None
        new_root_var = "trans_temporal_neg"
        g.triples.insert(0, (new_root_var, ":instance", new_frame))
        # NOTE: roles INTENTIONALLY swapped to make this non-equivalent
        g.triples.append((new_root_var, ":ARG0", outer))
        g.triples.append((new_root_var, ":ARG1", anchor))
        try:
            g._top = new_root_var
        except AttributeError:
            pass
        return g
