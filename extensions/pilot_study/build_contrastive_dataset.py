"""Build a contrastive (anchor, positive, negative) dataset using the LogicRule
framework + parse_xfm_bart_large AMR parser + T5wtense AMR-to-text generator.

For each input sentence and each applicable rule:
  - apply_positive  → a logically EQUIVALENT sentence (label 1)
  - apply_negative  → a NON-equivalent sentence (label 0)

The output JSONL is suitable as training data for:
  - the AMR-LDA contrastive pretraining objective (original paper)
  - the RL verifier-shaped reward learning (this paper's extension)
  - generic sentence-embedding contrastive learning (SimCSE-style)

Each output record:
  {
    "anchor": "If Alice studies hard, then she passes the exam.",
    "positive": "If Alice does not pass the exam, then she does not study hard.",
    "negative": "If Alice studies hard, then she does not pass the exam.",
    "rule": "contraposition",
    "amr_anchor": "(...)",
    "amr_positive": "(...)",
    "amr_negative": "(...)",
    "self_check_positive": "ok" | "self_check_failed: ...",
    "self_check_negative": "ok" | "self_check_failed: ..."
  }

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \\
        python -m extensions.pilot_study.build_contrastive_dataset \\
            --input-sentences extensions/pilot_study/test_sentences.json \\
            --output extensions/pilot_study/contrastive_dataset.jsonl \\
            --rules contraposition commutative implication de_morgan
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("contrastive_dataset")


def load_input_sentences(path: Path) -> List[dict]:
    """Read sentences from one of three supported formats.

    - test_sentences.json: pilot study format ({"sentences": [{text, id, ...}]})
    - plain JSONL with "text" or "sentence" keys
    - .txt with one sentence per line
    """
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if "sentences" in data:
            return [
                {"id": s["id"], "text": s["text"],
                 "applicable_rules": s.get("applicable_rules", [])}
                for s in data["sentences"]
            ]
        if isinstance(data, list):
            return [{"id": str(i), "text": s} if isinstance(s, str)
                    else {"id": s.get("id", str(i)), "text": s.get("text", s.get("sentence", ""))}
                    for i, s in enumerate(data)]
    if path.suffix == ".jsonl":
        out = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append({"id": d.get("id", str(i)),
                            "text": d.get("text", d.get("sentence", "")),
                            "applicable_rules": d.get("applicable_rules", [])})
        return out
    # .txt: one sentence per line
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if line:
            out.append({"id": str(i), "text": line})
    return out


def _count_polarity(g: penman.Graph) -> int:
    return sum(1 for s, r, t in g.triples if r == ":polarity" and t == "-")


def _self_consistency_check(parser, generated_text: str, expected_amr: str) -> Tuple[bool, str]:
    """Reused from generate_amr_lda.py."""
    try:
        check = parser.parse_sents([generated_text])[0]
        if not check:
            return False, "empty_parse"
        cg = penman.decode(check)
        eg = penman.decode(expected_amr)
    except Exception as e:
        return False, f"check_failed: {e}"
    n_e = _count_polarity(eg)
    n_c = _count_polarity(cg)
    if (n_e % 2) != (n_c % 2):
        return False, f"parity_flipped (expected {n_e}, got {n_c})"
    if n_e > 0 and n_c == 0:
        return False, f"polarity_dropped (expected {n_e})"
    if abs(n_c - n_e) >= 2:
        return False, f"drift (expected {n_e}, got {n_c})"
    return True, "ok"


# Map from test_sentences.json compound names to LogicRule names
RULE_DISPATCH: Dict[str, List[str]] = {
    "contraposition": ["contraposition"],
    "commutative": ["commutative"],
    "implication": ["implication"],
    "double_negation": ["double_negation"],
    "de_morgan": ["de_morgan"],
    "transitivity": ["transitivity"],
    "inverse_relation": ["inverse_relation"],
    "symmetric_asymmetric": ["symmetric", "asymmetric"],
    "predicate_implication": ["predicate_implication"],
    "aspect_equivalence": ["aspect_equivalence"],
    "modal_strength_inversion": ["modal_strength_inversion"],
    "doc_level_temporal_transitivity": ["doc_level_temporal_transitivity"],
    "tense_transformation": ["tense_transformation"],
}


def build_one(
    parser, gtos, sentence: str, rule_name: str, sent_id: str,
    enable_self_check: bool = True,
) -> Optional[dict]:
    """Generate one contrastive record for (sentence, rule).

    Returns None if the rule did not fire on the input AMR.
    """
    from extensions.logic_rules import get_rule

    try:
        amr_anchor = parser.parse_sents([sentence])[0]
    except Exception as e:
        return None
    if not amr_anchor:
        return None
    try:
        g = penman.decode(amr_anchor)
    except Exception:
        return None

    try:
        rule = get_rule(rule_name)
    except KeyError:
        return None

    results = rule.apply(g)
    if not results:
        return None
    # We need both a positive AND a negative; iterate until we find a match
    # that has both.
    for res in results:
        if res.positive_graph is None or res.negative_graph is None:
            continue

        try:
            pos_text, _ = gtos.generate([res.positive_graph])
            neg_text, _ = gtos.generate([res.negative_graph])
        except Exception:
            continue
        if not pos_text or not neg_text:
            continue
        pos_text, neg_text = pos_text[0], neg_text[0]

        # Self-consistency check
        if enable_self_check:
            pos_ok, pos_reason = _self_consistency_check(parser, pos_text, res.positive_graph)
            neg_ok, neg_reason = _self_consistency_check(parser, neg_text, res.negative_graph)
            sc_pos = "ok" if pos_ok else f"self_check_failed: {pos_reason}"
            sc_neg = "ok" if neg_ok else f"self_check_failed: {neg_reason}"
        else:
            sc_pos = sc_neg = "skipped"

        return {
            "sentence_id": sent_id,
            "rule": rule_name,
            "anchor": sentence,
            "positive": pos_text,
            "negative": neg_text,
            "amr_anchor": amr_anchor,
            "amr_positive": res.positive_graph,
            "amr_negative": res.negative_graph,
            "self_check_positive": sc_pos,
            "self_check_negative": sc_neg,
        }
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-sentences", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--rules", nargs="*", default=None,
        help="rules to apply (default: all applicable_rules per sentence)",
    )
    ap.add_argument(
        "--parse-model",
        default="amrlib/data/model_parse_xfm_bart_large-v0_1_0",
    )
    ap.add_argument(
        "--gen-model",
        default="amrlib/data/model_generate_t5wtense-v0_1_0",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument(
        "--no-self-check", action="store_true",
        help="skip the T5wtense self-consistency check (faster, lower quality)",
    )
    args = ap.parse_args()

    import amrlib
    from amrlib.models.parse_xfm.inference import Inference

    log.info("Loading parser %s", args.parse_model)
    parser = Inference(args.parse_model, batch_size=1, num_beams=1)
    log.info("Loading generator %s", args.gen_model)
    gtos = amrlib.load_gtos_model(args.gen_model)

    sentences = load_input_sentences(args.input_sentences)
    if args.ids:
        sentences = [s for s in sentences if s["id"] in args.ids]
    if args.limit:
        sentences = sentences[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_ok = 0
    n_passed_self_check = 0
    from collections import Counter
    status_counter: Counter = Counter()

    with open(args.output, "w") as fout:
        for s in sentences:
            sent_id = s["id"]
            sent_text = s["text"]
            applicable = s.get("applicable_rules", [])
            # Filter rules
            rule_labels = args.rules if args.rules else applicable
            # Map compound labels to LogicRule names
            rule_names: List[str] = []
            for label in rule_labels:
                rule_names.extend(RULE_DISPATCH.get(label, [label]))

            for rname in rule_names:
                n_total += 1
                rec = build_one(
                    parser, gtos, sent_text, rname, sent_id,
                    enable_self_check=not args.no_self_check,
                )
                if rec is None:
                    status_counter["no_fire"] += 1
                    continue
                n_ok += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec["self_check_positive"] == "ok" and rec["self_check_negative"] == "ok":
                    n_passed_self_check += 1
                    status_counter["both_passed"] += 1
                elif rec["self_check_positive"] != "ok" and rec["self_check_negative"] != "ok":
                    status_counter["both_failed"] += 1
                else:
                    status_counter["one_failed"] += 1
                if n_ok % 20 == 0:
                    log.info("  built %d / %d so far", n_ok, n_total)

    log.info("")
    log.info("Done. %d records written to %s", n_ok, args.output)
    log.info("  total attempted: %d", n_total)
    log.info("  rule fired: %d (%.1f%%)", n_ok, 100 * n_ok / max(1, n_total))
    log.info("  both pos+neg passed self-check: %d (%.1f%%)",
             n_passed_self_check, 100 * n_passed_self_check / max(1, n_ok))
    log.info("  status counts: %s", dict(status_counter))


if __name__ == "__main__":
    main()
