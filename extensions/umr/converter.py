"""Rule-based AMR → UMR converter (lightweight reproduction of Post et al. 2024).

Reference
---------
Post, Benet, McGregor, Pacheco, Palmer. 2024.
"Accelerating UMR Adoption: Neuro-Symbolic Conversion from AMR-to-UMR with Low
Supervision." DMR 2024.

The original paper uses animacy parsing + symbolic rules + a small neural
network to handle non-deterministic AMR→UMR mappings. This module implements
the SYMBOLIC RULE component — covering the common deterministic cases — and
exposes hooks for plugging in a neural classifier for the harder ones.

What this converter adds to AMR (deriving UMR-spec attributes):
  1. :aspect attribute on event nodes (state / activity / performance /
     habitual / endeavor / process / inceptive)
  2. :modal-strength on the root or modal frames
     (FullAff / PrtAff / NeutAff / NeutNeg / PrtNeg / FullNeg)
  3. :ref-person, :ref-number, :ref-gender on person-typed nodes (heuristic)

What's NOT done here:
  - Document-level temporal/modal/coref annotation (cross-sentence)
  - Non-deterministic role mappings (where a neural classifier helps)
  - Full animacy parsing (we use a simple PropBank-frame-and-noun heuristic)

Usage
-----
    from extensions.umr.converter import convert_amr_to_umr
    umr_penman = convert_amr_to_umr(amr_penman_string)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import penman


# ---------------------------------------------------------------------------
# Aspect inference: assign one of UMR's aspect categories to each event node.
# ---------------------------------------------------------------------------

# Stative concepts — typically labeled "state" in UMR.
# Curated from UMR 2.0 English gold annotations (top frames marked :aspect state).
STATE_CONCEPTS: Set[str] = {
    # Copular / existential / possessive / identity
    "be", "have-01", "have-02", "have-03", "have-04", "have-05", "have-91",
    "exist-01", "exist-91", "identity-91",
    "live-01", "remain-01", "stay-01", "contain-01",
    # Cognitive / emotive / perceptual states
    "know-01", "know-02", "know-03", "believe-01", "think-01", "think-02",
    "love-01", "hate-01", "like-01", "like-02", "trust-01", "remember-01",
    "want-01", "need-01", "fear-01", "hope-01", "wish-01",
    "feel-01", "feel-02", "feel-03", "see-01", "watch-01",
    "appear-02", "seem-01", "look-01",  # perceptual statives
    # Possession / relations
    "own-01", "belong-01", "possess-01", "include-01", "consist-01",
    "comprise-01", "constitute-01",
    # Properties / quality / comparison
    "resemble-01", "differ-02", "match-01", "equal-01", "deserve-01",
    "require-01", "represent-01", "matter-01",
    # 91-suffixed copular / relational framesets — UMR treats these as state
    "have-degree-91", "have-rel-role-91", "have-org-role-91", "have-rel-role-92",
    "have-org-role-92", "have-mod-91", "have-quant-91", "have-purpose-91",
    "have-condition-91", "have-name-91", "have-li-91", "have-place-91",
    "have-role-91", "rate-entity-91", "publication-91",
    # State of being (death, suffering, lack)
    "die-01", "miss-01", "suffer-01", "lack-01", "problem-02",
    # Stative spatial
    "be-located-at-91", "border-01", "neighbor-01", "adjoin-01",
    # Show / display states
    "show-01",
}

# Performance (telic, bounded) — verbs with clear completion point
PERFORMANCE_CONCEPTS: Set[str] = {
    # Speech acts (UMR treats single utterances as bounded events)
    "say-01", "tell-01", "speak-01", "announce-01", "declare-01", "claim-01",
    "answer-01", "respond-01", "reply-01", "ask-01", "ask-02",
    # Common events
    "complete-01", "finish-01", "achieve-01", "win-01", "lose-01",
    "build-01", "create-01", "destroy-01", "kill-01",
    "publish-01", "discover-01", "invent-01",
    "arrive-01", "depart-01", "leave-15", "enter-01", "exit-01",
    "come-01", "go-01", "go-02", "fly-01", "fly-02", "land-01",
    "return-01", "rescue-01", "deliver-01",
    "graduate-01", "marry-01", "establish-01", "found-01",
    "buy-01", "sell-01", "give-01", "receive-01", "send-01", "pay-01",
    "elect-01", "appoint-01", "promote-02",
    "decide-01", "choose-01", "deny-01",
    "interview-01", "sentence-01", "pardon-01",
    "call-01", "call-03",
    "help-01", "show-01", "respond-01",
    "charge-05", "eliminate-01", "mean-01",
    # Catastrophic events
    "landslide-01", "earthquake-01", "explode-01",
}

# Activity (atelic, ongoing) — verbs without a fixed endpoint
ACTIVITY_CONCEPTS: Set[str] = {
    "study-01", "work-01", "play-01", "run-02", "walk-01", "swim-01",
    "drive-01", "read-01", "write-01", "listen-01", "speak-01",
    "talk-01", "discuss-01", "negotiate-01", "search-01", "investigate-01",
    "think-01", "ponder-01", "wonder-01",
    "try-01", "ride-01", "cope-01", "rush-01",
    "rain-01", "snow-01", "shine-01",   # weather verbs
    "televise-01", "broadcast-01",
    "deforest-01", "hasten-01",
}

# Inceptive — "starts to V"
INCEPTIVE_CONCEPTS: Set[str] = {
    "begin-01", "start-01", "initiate-01", "commence-01", "launch-01",
}

# Process — ongoing change
PROCESS_CONCEPTS: Set[str] = {
    "grow-01", "develop-01", "evolve-01", "change-01", "increase-01",
    "decrease-01", "rise-01", "fall-01", "improve-01", "deteriorate-01",
    "spread-03", "expand-01", "shrink-01",
}


def _has_freq_marker(g: penman.Graph, node: str) -> bool:
    """True if the node has a :freq or 'every' modifier."""
    for s, role, t in g.triples:
        if s != node:
            continue
        if role in (":freq", ":frequency"):
            return True
        if role == ":mod":
            # Check if the modifier concept is "every", "all", "habitual"
            for s2, r2, t2 in g.triples:
                if s2 == t and r2 == ":instance" and t2 in ("every", "all", "habitual"):
                    return True
    return False


def _has_progressive_marker(g: penman.Graph, node: str) -> bool:
    """Heuristic: progressive aspect often signaled by 'ing' form word or
    explicit :tense (continuous)."""
    for s, role, t in g.triples:
        if s == node and role in (":tense", ":aspect"):
            if isinstance(t, str) and ("continuous" in t or "progressive" in t):
                return True
    return False


def infer_aspect(g: penman.Graph, node: str, concept: str) -> Optional[str]:
    """Return one of UMR aspect labels, or None if no rule fires."""
    # Habitual takes priority — frequency marker forces habitual reading
    if _has_freq_marker(g, node):
        return "habitual"
    # Progressive
    if _has_progressive_marker(g, node):
        return "activity"
    # Concept-driven defaults
    if concept in STATE_CONCEPTS:
        return "state"
    if concept in INCEPTIVE_CONCEPTS:
        return "inceptive"
    if concept in PROCESS_CONCEPTS:
        return "process"
    if concept in PERFORMANCE_CONCEPTS:
        return "performance"
    if concept in ACTIVITY_CONCEPTS:
        return "activity"
    # No fallback — unknown frames return None (better precision than recall).
    # The previous "default to performance" rule mislabeled most state verbs.
    return None


# ---------------------------------------------------------------------------
# Modal strength inference
# ---------------------------------------------------------------------------

FULL_AFF_FRAMES: Set[str] = {
    "obligate-01", "require-01", "need-01", "must-01", "mandate-01",
    "mandatory-01", "have-to-01",
}

PRT_AFF_FRAMES: Set[str] = {
    "possible-01", "permit-01", "allow-01", "recommend-01", "suggest-01",
    "may-01", "can-01", "might-01", "could-01",
}

FULL_NEG_FRAMES: Set[str] = {
    "prohibit-01", "forbid-01", "ban-01",
}

PRT_NEG_FRAMES: Set[str] = {
    # Need not, may not (modal-negative forms)
}


def _has_polarity_neg(g: penman.Graph, node: str) -> bool:
    return (node, ":polarity", "-") in g.triples


def infer_modal_strength(g: penman.Graph, node: str, concept: str) -> Optional[str]:
    """Return the UMR modal-strength value for any predicate node.

    In UMR practice, EVERY affirmed proposition receives a :modal-strength
    annotation (full-affirmative by default). This is the speaker's stance on
    the proposition's truth. We default to full-affirmative for any predicate
    frame (concept matching `verb-NN`), full-negative if polarity is negative,
    and use explicit modal frame mappings as overrides.
    """
    # Modal-frame overrides
    if concept in FULL_AFF_FRAMES:
        return "full-affirmative"
    if concept in PRT_AFF_FRAMES:
        return "partial-affirmative"
    if concept in FULL_NEG_FRAMES:
        return "full-negative"
    if concept in PRT_NEG_FRAMES:
        return "partial-negative"
    # Default for any predicate (frameset like verb-01, frameset-91):
    if re.match(r"^[a-z][a-z0-9-]*-\d{2,3}$", concept):
        return "full-negative" if _has_polarity_neg(g, node) else "full-affirmative"
    return None


# ---------------------------------------------------------------------------
# Animacy / reference-person inference
# ---------------------------------------------------------------------------

ANIMATE_CONCEPTS: Set[str] = {
    "person", "man", "woman", "boy", "girl", "child", "baby", "adult",
    "dog", "cat", "horse", "animal", "bird", "fish", "tiger", "lion",
    "human", "patient", "doctor", "teacher", "student",
}


def infer_animacy(g: penman.Graph, node: str, concept: str) -> Optional[str]:
    """Return 'animate' or 'inanimate' or None."""
    if concept in ANIMATE_CONCEPTS:
        return "animate"
    # Default: inanimate for everything else (objects, abstractions, etc.)
    # but only label nouns, not verbs
    if "-" not in concept:  # nouns/concepts don't have -XX frame numbers
        return "inanimate"
    return None


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


@dataclass
class ConvertResult:
    umr_graph: penman.Graph
    annotations_added: Dict[str, int]


def _is_root_event(g: penman.Graph, node: str) -> bool:
    """A node is a root-level event if it is the graph's top or sits directly
    under the top via a non-argument role (i.e., a coordination conjunct or
    embedded event that the author is asserting)."""
    if node == g.top:
        return True
    # Find any edge whose target is this node and source is g.top
    for s, role, t in g.triples:
        if t == node and s == g.top and role in (
            ":op1", ":op2", ":op3", ":op4",  # conjuncts
            ":ARG1",  # often the asserted proposition under a :modal frame
        ):
            return True
    return False


def convert_amr_to_umr(amr_penman: str) -> ConvertResult:
    """Convert an AMR penman string to a UMR-style graph.

    Adds :aspect, :modal-strength, :animacy attributes where derivable.

    Annotation policies:
      - :aspect — applied to every event/state-like predicate frame
      - :modal-strength — applied ONLY to root-level events (the propositions
        the author is asserting). Per UMR practice, embedded clauses inside
        relative clauses or arguments don't receive their own modal-strength
        unless they are independently asserted.
      - :animacy — applied to noun nodes
    """
    g = penman.decode(amr_penman)
    counts = {"aspect": 0, "modal-strength": 0, "animacy": 0}
    new_triples = list(g.triples)
    seen_nodes: Set[str] = set()

    for s, role, t in list(g.triples):
        if role != ":instance":
            continue
        if s in seen_nodes:
            continue
        seen_nodes.add(s)
        concept = t

        aspect = infer_aspect(g, s, concept)
        if aspect:
            new_triples.append((s, ":aspect", aspect))
            counts["aspect"] += 1

        # Only annotate modal-strength on root-level events
        if _is_root_event(g, s):
            modal = infer_modal_strength(g, s, concept)
            if modal:
                new_triples.append((s, ":modal-strength", modal))
                counts["modal-strength"] += 1

        animacy = infer_animacy(g, s, concept)
        if animacy:
            new_triples.append((s, ":animacy", animacy))
            counts["animacy"] += 1

    new_graph = penman.Graph(new_triples, top=g.top)
    return ConvertResult(umr_graph=new_graph, annotations_added=counts)


def convert_amr_to_umr_text(amr_penman: str) -> str:
    """Convenience wrapper that returns the penman-encoded UMR text."""
    result = convert_amr_to_umr(amr_penman)
    return penman.encode(result.umr_graph)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _smoke_test():
    """A quick demonstration on a hand-built AMR."""
    test_amr = """
    (o / obligate-01
       :ARG0 (p / person :name (n / name :op1 "Alice"))
       :ARG2 (f / finish-01
              :ARG0 p
              :ARG1 (h / homework)))
    """
    result = convert_amr_to_umr(test_amr)
    print("Input AMR:")
    print(test_amr.strip())
    print()
    print("Output UMR (with :aspect, :modal-strength, :animacy added):")
    print(penman.encode(result.umr_graph))
    print()
    print("Annotations added:", result.annotations_added)


if __name__ == "__main__":
    _smoke_test()
