"""Predicate implication: if predicate p semantically entails predicate q, then
p(X) entails q(X) (one-directional, not equivalence).

  - "barks(X)" entails "makes_sound(X)" (via WordNet hypernym)
  - "robin(X)" entails "bird(X)" (via WordNet hypernym)

We use WordNet's `hypernyms()` to find the IMMEDIATE hypernym (parent in the
taxonomy) of the predicate's stem. Only immediate hypernyms are used to keep
the entailment chain short and verifiable.

This is a unidirectional rule: P -> Q, not P <=> Q. So `apply_positive` produces
a sentence that is entailed by the input (Q is a NECESSARY consequence of P).
For RL verifier purposes, it serves as a "subset of truth conditions" check.
"""

from __future__ import annotations

import re
from typing import List, Optional

import penman

from .base import LogicRule, RuleMatch, register


# Try wordnet; fall back to a manual mini-dictionary if data isn't downloaded.
try:
    from nltk.corpus import wordnet as _wn

    _WN_IMPORTED = True
except ImportError:
    _wn = None
    _WN_IMPORTED = False


def _hypernym_stem(stem: str) -> Optional[str]:
    """Return the immediate hypernym lemma for `stem`, or None if unknown."""
    if _WN_IMPORTED:
        try:
            synsets = _wn.synsets(stem, pos=_wn.NOUN) or _wn.synsets(
                stem, pos=_wn.VERB
            )
            if synsets:
                hypers = synsets[0].hypernyms()
                if hypers:
                    return hypers[0].lemmas()[0].name().replace("_", "-")
        except LookupError:
            # wordnet corpus not downloaded — fall through to manual map
            pass
    return _MANUAL_HYPERNYMS.get(stem)


# Fallback hypernym map — used when NLTK WordNet is unavailable or a corpus
# specific mapping is preferred.
_MANUAL_HYPERNYMS = {
    "bark": "make-sound",
    "meow": "make-sound",
    "shout": "speak",
    "whisper": "speak",
    "robin": "bird",
    "sparrow": "bird",
    "dog": "animal",
    "cat": "animal",
    "rose": "flower",
    "tulip": "flower",
    "run": "move",
    "walk": "move",
    "sprint": "run",
}


_FRAME_STEM_RE = re.compile(r"^(.+?)-(\d+)$")


def _split_frame(concept: str) -> tuple:
    """Split 'bark-01' -> ('bark', '01'); 'bird' -> ('bird', None)."""
    m = _FRAME_STEM_RE.match(concept)
    if m:
        return m.group(1), m.group(2)
    return concept, None


@register
class PredicateImplicationRule(LogicRule):
    name = "predicate_implication"

    def detect(self, g: penman.Graph) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            stem, _sense = _split_frame(t)
            hyper = _hypernym_stem(stem)
            if hyper is None:
                continue
            matches.append(
                RuleMatch(
                    anchor=s,
                    extras={
                        "old_concept": t,
                        "stem": stem,
                        "sense": _sense,
                        "hypernym_stem": hyper,
                    },
                )
            )
        return matches

    def apply_positive(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Replace the concept with its hypernym (preserving sense suffix
        if present)."""
        anchor = match.anchor
        hyper_stem = match.extras["hypernym_stem"]
        sense = match.extras["sense"]
        new_concept = f"{hyper_stem}-{sense}" if sense else hyper_stem

        for i, (s, role, t) in enumerate(g.triples):
            if s == anchor and role == ":instance":
                g.triples[i] = (s, role, new_concept)
                return g
        return None

    def apply_negative(
        self, g: penman.Graph, match: RuleMatch
    ) -> Optional[penman.Graph]:
        """Negative: replace with a co-hyponym (sibling, not hypernym).
        Without WordNet sibling access, this is left as a stub returning None."""
        return None
