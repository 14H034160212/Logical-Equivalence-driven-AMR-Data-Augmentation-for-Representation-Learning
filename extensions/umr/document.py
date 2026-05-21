"""Document-level UMR annotation parser + AMR-side derivation rules.

UMR encodes cross-sentence relationships in three families:

  :temporal   — relations between events (`:before` / `:after` / `:overlap` /
                `:depends-on` / `:contained`)
  :modal      — author / speaker stance subgraph (`:full-affirmative` /
                `:partial-affirmative` / `:full-negative` / ...)
  :coref      — `:same-entity` / `:same-event` across sentences

This module:
  1. Parses the `# document level annotation` section of a UMR file into
     structured triples.
  2. Implements a basic rule-based derivation pipeline from a sequence of
     AMR-style sentences to derived temporal/coref relations. Used as a
     baseline for the heavier neural document-level methods that papers
     like Post et al. 2024 would plug in for the hard cases.

Derivation rules used:
  - **Temporal sequencing from sentence order**: if sentence k mentions event
    A and sentence k+1 mentions event B, and no explicit temporal marker
    contradicts, hypothesize A :before B (weak).
  - **Temporal sequencing from `:time after/before` markers**: extract the
    explicit relations directly.
  - **Coreference by name match**: two nodes in different sentences with the
    same `:name :op1 "X"` value are :same-entity.
  - **Modal author affirmation**: every root event without `:polarity-` gets
    :full-affirmative from the author by default; with `:polarity-`,
    :full-negative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import penman


# ---------------------------------------------------------------------------
# Parsing the gold UMR document-level section
# ---------------------------------------------------------------------------


TRIPLE_RE = re.compile(r"\(\s*([\w\-]+)\s+(:[\w\-]+)\s+([\w\-]+)\s*\)")


@dataclass
class DocRelations:
    """Parsed document-level relations for a single sentence block."""

    sent_id: str
    temporal: List[Tuple[str, str, str]] = field(default_factory=list)  # (src, rel, tgt)
    modal: List[Tuple[str, str, str]] = field(default_factory=list)
    coref: List[Tuple[str, str, str]] = field(default_factory=list)


def parse_document_annotation(doc_lines: List[str]) -> DocRelations:
    """Parse the lines of a `# document level annotation` block.

    Returns a DocRelations holding the extracted triples per category.
    """
    text = "\n".join(doc_lines)
    out = DocRelations(sent_id="")
    # Each category is in its own `:temporal ((...))` block.
    for category in ("temporal", "modal", "coref"):
        # Find :<category> followed by an outer paren group
        idx = text.find(f":{category}")
        if idx < 0:
            continue
        # Walk forward to find the start of the outer paren group
        start = text.find("(", idx)
        if start < 0:
            continue
        # Match the outer parens
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        body = text[start:end + 1]
        # Extract individual triples within the body
        triples = TRIPLE_RE.findall(body)
        for src, rel, tgt in triples:
            if category == "temporal":
                out.temporal.append((src, rel, tgt))
            elif category == "modal":
                out.modal.append((src, rel, tgt))
            elif category == "coref":
                out.coref.append((src, rel, tgt))
    return out


# ---------------------------------------------------------------------------
# Derivation rules from a sequence of sentence-level AMR graphs
# ---------------------------------------------------------------------------


def _root_event(g: penman.Graph) -> Optional[str]:
    return g.top


def _instance(g: penman.Graph, node: str) -> Optional[str]:
    for s, role, t in g.triples:
        if s == node and role == ":instance":
            return t
    return None


def _has_polarity_neg(g: penman.Graph, node: str) -> bool:
    return (node, ":polarity", "-") in g.triples


def _name_of_entity(g: penman.Graph, node: str) -> Optional[str]:
    """Extract the first :op1 string from a :name subtree on `node`."""
    for s, role, t in g.triples:
        if s == node and role == ":name":
            for s2, r2, t2 in g.triples:
                if s2 == t and r2 == ":op1":
                    return t2.strip('"') if isinstance(t2, str) else str(t2)
    return None


def derive_doc_relations(
    graphs: List[Tuple[str, penman.Graph]],
    weak_temporal_from_order: bool = True,
) -> List[Tuple[str, str, str]]:
    """Derive document-level :temporal / :modal / :coref triples from a
    sequence of (sentence_id_prefix, AMR_graph) pairs.

    The graphs should be parsed AMR (or AMR-like) sentence graphs in narrative
    order. Returns a flat list of (src_node, relation, tgt_node) triples in
    UMR document-level format.
    """
    relations: List[Tuple[str, str, str]] = []

    # --- Modal: author affirms each root event ---
    for prefix, g in graphs:
        root = _root_event(g)
        if root is None:
            continue
        rel = ":full-negative" if _has_polarity_neg(g, root) else ":full-affirmative"
        relations.append(("author", rel, f"{prefix}{root}"))

    # --- Explicit :time after/before within a sentence ---
    for prefix, g in graphs:
        for s, role, t in g.triples:
            if role != ":time":
                continue
            inst = _instance(g, t)
            if inst not in ("after", "before"):
                continue
            # Find :op1 of the time node
            for s2, r2, t2 in g.triples:
                if s2 == t and r2 == ":op1":
                    target_event = t2
                    relations.append((
                        f"{prefix}{s}",
                        f":{inst}",
                        f"{prefix}{target_event}",
                    ))
                    break

    # --- Weak: sentence-order temporal sequencing ---
    if weak_temporal_from_order and len(graphs) >= 2:
        for i in range(len(graphs) - 1):
            prev_prefix, prev_g = graphs[i]
            next_prefix, next_g = graphs[i + 1]
            prev_root = _root_event(prev_g)
            next_root = _root_event(next_g)
            if prev_root and next_root:
                relations.append((
                    f"{prev_prefix}{prev_root}",
                    ":before",
                    f"{next_prefix}{next_root}",
                ))

    # --- Coreference: same name across sentences = :same-entity ---
    name_to_nodes: Dict[str, List[str]] = {}
    for prefix, g in graphs:
        for s, role, t in g.triples:
            if role != ":instance":
                continue
            name = _name_of_entity(g, s)
            if name:
                name_to_nodes.setdefault(name, []).append(f"{prefix}{s}")
    for name, nodes in name_to_nodes.items():
        if len(nodes) >= 2:
            # Link consecutive mentions
            for i in range(len(nodes) - 1):
                relations.append((nodes[i], ":same-entity", nodes[i + 1]))

    return relations


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------


def _smoke_test():
    """Demo: two simple AMR sentences with shared entity 'Alice'."""
    g1 = penman.decode("""
    (s / study-01
        :ARG0 (p1 / person :name (n1 / name :op1 "Alice")))
    """)
    g2 = penman.decode("""
    (p / pass-07
        :ARG0 (p2 / person :name (n2 / name :op1 "Alice"))
        :ARG1 (e / exam))
    """)
    rels = derive_doc_relations([("s1", g1), ("s2", g2)])
    print("Derived document-level relations:")
    for src, rel, tgt in rels:
        print(f"  {src}  {rel}  {tgt}")


if __name__ == "__main__":
    _smoke_test()
