"""Generate the AMR-LDA reference outputs for the pilot study.

Pipeline:
  1. Read test_sentences.json
  2. For each sentence S, parse to AMR via parse_xfm_bart_large
  3. For each applicable rule R that has an active implementation:
       apply rule R to AMR(S) → modified AMR(S')
       generate text S' via t5wtense (AMR-to-text)
  4. Emit (sentence_id, rule, input, output, ...) to JSONL just like the LLM baseline.

These outputs become the AMR-LDA system in the auto-verifier comparison
("our method" vs "LLM-as-rewriter").

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \\
        python -m extensions.pilot_study.generate_amr_lda \\
            --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \\
            --gen-model amrlib/data/model_generate_t5wtense-v0_1_0 \\
            --out extensions/pilot_study/results/combined/rewrite/amr_lda.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import penman

# Suppress noise
warnings.filterwarnings("ignore")

log = logging.getLogger("amr_lda_gen")


# Map from test_sentences.json `applicable_rules` entry → LogicRule name.
# The pilot JSON uses some compound names; we dispatch them here.
RULE_DISPATCH: Dict[str, List[str]] = {
    "contraposition": ["contraposition"],
    "commutative": ["commutative"],
    "implication": ["implication"],
    "double_negation": ["double_negation"],
    "de_morgan": ["de_morgan"],
    "transitivity": ["transitivity"],  # currently stub
    "inverse_relation": ["inverse_relation"],
    "symmetric_asymmetric": ["symmetric", "asymmetric"],
    "predicate_implication": ["predicate_implication"],
    # UMR-level rules (AMR-layer approximations)
    "aspect_equivalence": ["aspect_equivalence"],
    "modal_strength_inversion": ["modal_strength_inversion"],
    "doc_level_temporal_transitivity": ["doc_level_temporal_transitivity"],
    "tense_transformation": ["tense_transformation"],
}


def load_models(parse_dir: str, gen_dir: str):
    """Load AMR parser and AMR-to-text generator. Heavy: ~2GB RAM."""
    import amrlib
    from amrlib.models.parse_xfm.inference import Inference

    log.info("Loading AMR parser from %s ...", parse_dir)
    parser = Inference(parse_dir, batch_size=1, num_beams=1)
    log.info("Loading AMR-to-text generator from %s ...", gen_dir)
    gtos = amrlib.load_gtos_model(gen_dir)
    return parser, gtos


def _count_polarity_neg(g: penman.Graph) -> int:
    return sum(1 for s, role, t in g.triples
               if role == ":polarity" and t == "-")


def _self_consistency_check(
    parser, generated_text: str, expected_amr: str
) -> Tuple[bool, str]:
    """Re-parse the generated text and verify key structural features survived.

    Specifically we count :polarity- edges. T5wtense sometimes drops negations,
    rendering "It is not the case that Alice may skip" → "Alice can finish",
    which loses the equivalence. By re-parsing and counting polarities, we
    detect this pattern.

    Returns (passed, reason).
    """
    try:
        check_graph_str = parser.parse_sents([generated_text])[0]
        if not check_graph_str:
            return False, "check_parse_empty"
        check_g = penman.decode(check_graph_str)
        expected_g = penman.decode(expected_amr)
    except Exception as e:
        return False, f"check_parse_failed: {e}"

    n_expected = _count_polarity_neg(expected_g)
    n_check = _count_polarity_neg(check_g)
    # Require parity preservation (even ↔ even, odd ↔ odd) AND no absolute
    # zero-loss when expected has at least one. Parity is the critical
    # truth-condition signal: dropping ONE negation flips truth conditions
    # for negation-toggling rules (contraposition, double_neg, modal_strength).
    if (n_expected % 2) != (n_check % 2):
        return False, f"polarity_parity_flipped: expected {n_expected}, got {n_check}"
    if n_expected > 0 and n_check == 0:
        return False, f"polarity_completely_dropped: expected {n_expected}, got 0"
    if abs(n_check - n_expected) >= 2:
        return False, f"polarity_drift: expected {n_expected}, got {n_check}"
    return True, "ok"


def amr_lda_one(
    parser, gtos, sentence: str, rule_name: str, rule_objects: dict,
    enable_self_check: bool = True,
) -> Tuple[Optional[str], str]:
    """Run the full AMR-LDA pipeline on one (sentence, rule).

    Returns (generated_text, status_message). generated_text is None if the
    rule did not fire on the parsed AMR.

    With self-consistency check (default): the generator output is re-parsed
    to AMR and the polarity count is compared against the expected post-rule
    AMR. On mismatch we retry with num_beams=4 (more diverse generation) and,
    if still inconsistent, we report `self_check_failed` and return the best
    candidate anyway.
    """
    from extensions.logic_rules import get_rule

    try:
        graph_str = parser.parse_sents([sentence])[0]
    except Exception as e:
        return None, f"parse_failed: {e}"
    if not graph_str:
        return None, "parse_empty"

    try:
        g = penman.decode(graph_str)
    except Exception as e:
        return None, f"penman_decode_failed: {e}"

    try:
        rule = get_rule(rule_name)
    except KeyError:
        return None, f"unknown_rule: {rule_name}"

    results = rule.apply(g)
    if not results or results[0].positive_graph is None:
        return None, f"rule_did_not_fire: {rule_name}"

    modified_amr = results[0].positive_graph

    try:
        sents, _ = gtos.generate([modified_amr])
    except Exception as e:
        return None, f"generation_failed: {e}"
    if not sents:
        return None, "generation_empty"

    text = sents[0]
    if not enable_self_check:
        return text, "ok"

    passed, reason = _self_consistency_check(parser, text, modified_amr)
    if passed:
        return text, "ok"
    # Re-try with greedier generation (different num_beams). If gtos doesn't
    # expose beam_size cleanly, we just try the same once more — many T5
    # variants are non-deterministic enough that a second pass differs.
    try:
        sents2, _ = gtos.generate([modified_amr])
        text2 = sents2[0] if sents2 else None
        if text2:
            passed2, reason2 = _self_consistency_check(parser, text2, modified_amr)
            if passed2:
                return text2, "ok_retry"
            # Pick the candidate with closer polarity count
            return text, f"self_check_failed: {reason}"
    except Exception:
        pass
    return text, f"self_check_failed: {reason}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-model", required=True, type=str)
    ap.add_argument("--gen-model", required=True, type=str)
    ap.add_argument(
        "--test-sentences",
        default="extensions/pilot_study/test_sentences.json",
        type=Path,
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="limit n sentences (debug)")
    ap.add_argument(
        "--ids", nargs="*", default=None, help="filter to specific sentence IDs"
    )
    ap.add_argument(
        "--no-self-check",
        action="store_true",
        help="disable T5wtense self-consistency check (faster but lower quality)",
    )
    args = ap.parse_args()

    parser, gtos = load_models(args.parse_model, args.gen_model)

    with open(args.test_sentences) as f:
        sentences = json.load(f)["sentences"]
    if args.ids:
        sentences = [s for s in sentences if s["id"] in args.ids]
    if args.limit:
        sentences = sentences[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total, n_ok, n_no_fire = 0, 0, 0
    status_counter: Dict[str, int] = {}

    with open(args.out, "w") as fout:
        for s in sentences:
            sentence = s["text"]
            for rule_label in s.get("applicable_rules", []):
                rule_targets = RULE_DISPATCH.get(rule_label, [])
                if not rule_targets:
                    rec = {
                        "sentence_id": s["id"],
                        "input": sentence,
                        "rule": rule_label,
                        "model": "amr_lda",
                        "output": None,
                        "gold": s.get("gold_outputs", {}).get(rule_label),
                        "status": "rule_not_implemented",
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_total += 1
                    status_counter["rule_not_implemented"] = status_counter.get("rule_not_implemented", 0) + 1
                    continue
                # Try each candidate rule (e.g., symmetric then asymmetric); take first that fires.
                output, status = None, "rule_did_not_fire"
                for rname in rule_targets:
                    output, status = amr_lda_one(
                        parser, gtos, sentence, rname, {},
                        enable_self_check=not args.no_self_check,
                    )
                    if output is not None:
                        break
                rec = {
                    "sentence_id": s["id"],
                    "input": sentence,
                    "rule": rule_label,
                    "model": "amr_lda",
                    "output": output if output is not None else f"[{status}]",
                    "gold": s.get("gold_outputs", {}).get(rule_label),
                    "status": status,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
                if status == "ok":
                    n_ok += 1
                else:
                    n_no_fire += 1
                status_counter[status] = status_counter.get(status, 0) + 1
                if n_total % 10 == 0:
                    log.info("  processed %d items (ok=%d, no_fire=%d)", n_total, n_ok, n_no_fire)

    log.info("AMR-LDA generation done: %d total, %d ok, %d failed", n_total, n_ok, n_no_fire)
    log.info("Status counts: %s", status_counter)
    log.info("Output written to %s", args.out)


if __name__ == "__main__":
    main()
