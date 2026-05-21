"""Modal strength inversion (UMR-grounded, implemented at the AMR layer).

UMR distinguishes 6 modal strengths (FullAff, PrtAff, NeutAff, NeutNeg, PrtNeg,
FullNeg). AMR has no native modal-strength attribute, but PropBank framesets
for modal verbs encode roughly equivalent distinctions:

  FullAff: obligate-01, require-01, need-01      (must, has to, needs)
  PrtAff : possible-01, permit-01                (may, can, is allowed to)
  FullNeg: prohibit-01, forbid-01                (must not, is not allowed to)
  PrtNeg : (PRT-AFF of negated proposition)      (need not)

Classical modal-strength equivalence:
  FullAff(P) ≡ ¬PrtAff(¬P)         "must P"          ≡ "cannot not P"
  PrtAff(P)  ≡ ¬FullAff(¬P)         "may P"           ≡ "is not forbidden P"
  FullNeg(P) ≡ FullAff(¬P)          "must not P"      ≡ "must (not P)"
  PrtNeg(P)  ≡ PrtAff(¬P)           "need not P"      ≡ "may (not P)"

This rule applies the FullAff↔FullNeg(¬·) and PrtAff↔PrtNeg(¬·) duals: it
swaps the modal frame for its dual and toggles the polarity of the embedded
proposition.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import penman

from .base import (
    LogicRule,
    RuleMatch,
    has_polarity_neg,
    register,
    replace_instance,
    toggle_polarity_neg,
)


# Map: source modal frame → (dual frame, candidate roles pointing at the action argument)
# The parser may place the action under any of these roles for the same frame
# (e.g., permit-01's action may be :ARG1 OR :ARG2), so we try each.
MODAL_DUALS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "obligate-01": ("possible-01", (":ARG2", ":ARG1")),
    "require-01":  ("possible-01", (":ARG1", ":ARG2")),
    "need-01":     ("possible-01", (":ARG1", ":ARG2")),
    "must-01":     ("possible-01", (":ARG1", ":ARG2")),
    "possible-01": ("obligate-01", (":ARG1", ":ARG2")),
    "permit-01":   ("prohibit-01", (":ARG1", ":ARG2")),  # parser uses :ARG1
    "allow-01":    ("prohibit-01", (":ARG1", ":ARG2")),
    "prohibit-01": ("permit-01",   (":ARG1", ":ARG2")),
    "forbid-01":   ("permit-01",   (":ARG1", ":ARG2")),
    "recommend-01": ("recommend-01", (":ARG1", ":ARG2")),
    "suggest-01":   ("suggest-01",   (":ARG1", ":ARG2")),
    "mandate-01":   ("possible-01",  (":ARG1", ":ARG2")),  # "It is mandatory that..."
    "mandatory-01": ("possible-01",  (":ARG1", ":ARG2")),
}


@register
class ModalStrengthInversionRule(LogicRule):
    """Implements modal-strength duality at the AMR layer."""

    name = "modal_strength_inversion"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":instance" or t not in MODAL_DUALS:
                continue
            dual, candidate_roles = MODAL_DUALS[t]
            chosen_role = None
            chosen_target = None
            for r in candidate_roles:
                edges = g.edges(source=s, role=r)
                if len(edges) == 1:
                    chosen_role = r
                    chosen_target = edges[0].target
                    break
            if chosen_role is None:
                continue
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "old_modal": t,
                        "new_modal": dual,
                        "action_role": chosen_role,
                        "action_target": chosen_target,
                    },
                )
            )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Apply the modal duality: swap the frame and toggle the embedded
        proposition's polarity AND the modal's polarity."""
        anchor = match.anchor
        action = match.extras["action_target"]
        new_modal = match.extras["new_modal"]
        replace_instance(g, anchor, new_modal)
        toggle_polarity_neg(g, anchor)   # outer modal gets ¬
        toggle_polarity_neg(g, action)   # embedded proposition gets ¬
        return g

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: swap the frame WITHOUT flipping any polarity.
        This produces a graph with different modal strength (modal drift) —
        a typical LLM-error pattern."""
        replace_instance(g, match.anchor, match.extras["new_modal"])
        return g
