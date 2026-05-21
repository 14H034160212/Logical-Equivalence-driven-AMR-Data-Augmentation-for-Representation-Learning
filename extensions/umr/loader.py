"""Loader for the UMR 2.0 corpus (umr4nlp/umr-data).

UMR files are organized in blocks separated by 80 hash signs ('#' x 80). Each
block has 5 sections, separated by blank lines or by 79-hash-sign delimiters:

    1. Meta information (block tag, sentence id)
    2. Sentence info (words, morphemes, POS tags, English translations)
    3. Sentence-level UMR annotation (Penman notation, augmented with
       :aspect, :modal-strength, :ref-number, :ref-person, etc.)
    4. Alignment info (UMR concept node -> token index span)
    5. Document-level annotation
         (a) temporal relations: :Before, :After, :Contained, :Overlap,
             :Depends-on
         (b) modal dependencies
         (c) coreference relations: :same-entity, :same-event

This loader does *not* attempt to fully parse the sentence-level Penman with
the standard `penman` library (it would fail on UMR-specific attributes); we
keep the raw penman string and provide regex helpers to extract the UMR-only
fields. For full structured parsing, see `umr_to_amr_subset()` which strips
UMR extensions and returns a vanilla AMR.

Usage
-----

    from extensions.umr.loader import load_umr_dataset, UMRSentence

    sentences = load_umr_dataset(language="english")
    for s in sentences[:3]:
        print(s.sent_id, s.text, len(s.aspect_attrs), len(s.modal_attrs))
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UMRSentence:
    """One sentence-level block from a UMR file."""

    sent_id: str
    doc_id: str
    text: str
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    english_translation: Optional[str] = None
    sentence_umr_penman: str = ""
    alignment: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)
    # UMR-specific attributes extracted from the sentence-level penman:
    aspect_attrs: Dict[str, str] = field(default_factory=dict)
    modal_attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class UMRDocument:
    """A full UMR document with sentence- and document-level annotations."""

    doc_id: str
    sentences: List[UMRSentence] = field(default_factory=list)
    # Document-level temporal/modal/coref relations as raw triples (source, role, target)
    temporal_rels: List[Tuple[str, str, str]] = field(default_factory=list)
    modal_rels: List[Tuple[str, str, str]] = field(default_factory=list)
    coref_rels: List[Tuple[str, str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


BLOCK_DELIM = "#" * 80
SECTION_DELIM_RE = re.compile(r"^#+\s*$", re.MULTILINE)


def _split_blocks(text: str) -> List[str]:
    """Split a UMR file into sentence blocks at the 80-hash delimiter."""
    parts = text.split(BLOCK_DELIM)
    return [p.strip() for p in parts if p.strip()]


def _parse_sentence_block(block: str, doc_id: str) -> Optional[UMRSentence]:
    """Parse one sentence block. Returns None if the block has no UMR penman."""
    # Detect sections by looking for canonical line prefixes.
    sent_id = ""
    text = ""
    tokens: List[str] = []
    pos_tags: List[str] = []
    english: Optional[str] = None
    umr_penman_lines: List[str] = []
    alignment_lines: List[str] = []
    in_umr = False
    in_alignment = False

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# :: snt"):
            sent_id = stripped.replace("# :: snt", "").strip()
        elif stripped.startswith("Words:"):
            tokens = stripped[len("Words:"):].split()
        elif stripped.startswith("Word IDs:"):
            pass  # ignore
        elif stripped.startswith("POS:"):
            pos_tags = stripped[len("POS:"):].split()
        elif stripped.startswith("English translation:"):
            english = stripped[len("English translation:"):].strip()
        elif stripped.startswith("Sentence Level Annotation"):
            in_umr = True
            in_alignment = False
            continue
        elif stripped.startswith("Alignment"):
            in_umr = False
            in_alignment = True
            continue
        elif stripped.startswith("Document Level Annotation"):
            in_umr = False
            in_alignment = False
            continue
        elif in_umr:
            umr_penman_lines.append(line)
        elif in_alignment:
            alignment_lines.append(line)

    umr_penman = "\n".join(umr_penman_lines).strip()
    if not umr_penman:
        return None

    text = " ".join(tokens) if tokens else (english or "")
    return UMRSentence(
        sent_id=sent_id or f"{doc_id}_unknown",
        doc_id=doc_id,
        text=text,
        tokens=tokens,
        pos_tags=pos_tags,
        english_translation=english,
        sentence_umr_penman=umr_penman,
        alignment=_parse_alignment(alignment_lines),
        aspect_attrs=_extract_attr(umr_penman, "aspect"),
        modal_attrs=_extract_attr(umr_penman, "modal-strength"),
    )


def _parse_alignment(lines: List[str]) -> List[Tuple[str, Tuple[int, int]]]:
    """Parse alignment lines like 'x0: 1-1' into (node_var, (start, end))."""
    out: List[Tuple[str, Tuple[int, int]]] = []
    for line in lines:
        m = re.match(r"^\s*([^:\s]+)\s*:\s*(\d+)-(\d+)\s*$", line)
        if m:
            out.append((m.group(1), (int(m.group(2)), int(m.group(3)))))
    return out


_ATTR_RE_TEMPLATE = re.compile(r":(\w+)\s+([^\s)]+)")


def _extract_attr(penman_text: str, attr_name: str) -> Dict[str, str]:
    """Extract UMR-specific attributes like `:aspect activity` from raw penman text.

    Returns a dict mapping the parent node variable to the attribute value.
    Best-effort: the regex walks lines and looks for the most recent `(var /`
    declaration to associate each attribute with a node.
    """
    out: Dict[str, str] = {}
    current_var: Optional[str] = None
    var_decl_re = re.compile(r"\(([a-zA-Z][\w\d]*)\s*/")
    attr_pattern = re.compile(rf":{re.escape(attr_name)}\s+([^\s)]+)")
    for line in penman_text.splitlines():
        decls = var_decl_re.findall(line)
        if decls:
            current_var = decls[-1]
        for m in attr_pattern.finditer(line):
            if current_var is not None:
                out[current_var] = m.group(1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_umr_dataset(
    umr_data_root: str = "extensions/umr/umr-data",
    language: str = "english",
) -> List[UMRSentence]:
    """Load all sentence-level UMR annotations for a given language.

    Expects the umr4nlp/umr-data repository cloned at `umr_data_root`. See
    `download_umr_data.sh` for the recommended setup.
    """
    root = Path(umr_data_root) / language
    if not root.exists():
        raise FileNotFoundError(
            f"UMR data root not found: {root}. Run extensions/umr/download_umr_data.sh first."
        )
    sentences: List[UMRSentence] = []
    for fp in sorted(root.rglob("*.txt")):
        doc_id = fp.stem
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for block in _split_blocks(text):
            s = _parse_sentence_block(block, doc_id=doc_id)
            if s is not None:
                sentences.append(s)
    return sentences


def aspect_modal_stats(sentences: List[UMRSentence]) -> Dict[str, Dict[str, int]]:
    """Compute distribution of UMR aspect and modal-strength values.

    Useful as the first sanity-check after loading: confirms which attributes
    are populated in the English portion of UMR 2.0 (the field that motivates
    our Tense/Modal rules).
    """
    aspect_counts: Dict[str, int] = {}
    modal_counts: Dict[str, int] = {}
    n_sents_with_aspect = 0
    n_sents_with_modal = 0
    for s in sentences:
        if s.aspect_attrs:
            n_sents_with_aspect += 1
        if s.modal_attrs:
            n_sents_with_modal += 1
        for v in s.aspect_attrs.values():
            aspect_counts[v] = aspect_counts.get(v, 0) + 1
        for v in s.modal_attrs.values():
            modal_counts[v] = modal_counts.get(v, 0) + 1
    return {
        "totals": {
            "n_sentences": len(sentences),
            "n_sentences_with_aspect": n_sents_with_aspect,
            "n_sentences_with_modal": n_sents_with_modal,
        },
        "aspect_distribution": aspect_counts,
        "modal_distribution": modal_counts,
    }


def dump_stats(sentences: List[UMRSentence], out_path: str) -> None:
    stats = aspect_modal_stats(sentences)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote stats: {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="extensions/umr/umr-data")
    ap.add_argument("--language", default="english")
    ap.add_argument("--stats-out", default="extensions/umr/english_stats.json")
    args = ap.parse_args()

    sentences = load_umr_dataset(args.root, args.language)
    print(f"loaded {len(sentences)} sentences from {args.language}")
    dump_stats(sentences, args.stats_out)
