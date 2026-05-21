"""AMR-LDA generator v2: best-of-N sampling + self-check picking.

The v1 generator ([generate_amr_lda.py](generate_amr_lda.py)) uses a single
greedy T5wtense pass per rule application. When the AMR contains polarity
nodes (common after contraposition / De Morgan / modal inversion), the
generator drops them ~17% of the time — this is the run6 self-check
failure pattern.

v2 fix: sample K candidates per rule application with different beam sizes
and pick the one whose self-consistency check passes. If none pass, return
the highest-confidence candidate with a self_check_failed status (same as v1).

This is a cheap quality improvement — multiple T5wtense passes are still
~1s each, and we typically only need 2-3 attempts to find a polarity-
preserving candidate.

Usage (drop-in replacement for v1):
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \\
        python -m extensions.pilot_study.generate_amr_lda_v2 \\
            --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \\
            --gen-model amrlib/data/model_generate_t5wtense-v0_1_0 \\
            --out extensions/pilot_study/results/combined/rewrite/amr_lda_v2.jsonl \\
            --num-samples 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("amr_lda_v2")


def _count_polarity(g: penman.Graph) -> int:
    return sum(1 for s, r, t in g.triples if r == ":polarity" and t == "-")


def _self_consistency_check(parser, text: str, expected_amr: str) -> Tuple[bool, str]:
    """Returns (passed, reason)."""
    try:
        check = parser.parse_sents([text])[0]
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
        return False, f"polarity_dropped"
    if abs(n_c - n_e) >= 2:
        return False, f"drift (expected {n_e}, got {n_c})"
    return True, "ok"


def amr_lda_one_v2(
    parser, gtos, sentence: str, rule_name: str,
    num_samples: int = 4,
) -> Tuple[Optional[str], str, dict]:
    """v2: try multiple generations, pick the one passing self-check.

    Returns (text, status, metadata).
    Metadata includes: which sample passed, all candidates, etc.
    """
    from extensions.logic_rules import get_rule

    try:
        amr_str = parser.parse_sents([sentence])[0]
    except Exception as e:
        return None, f"parse_failed: {e}", {}
    if not amr_str:
        return None, "parse_empty", {}

    try:
        g = penman.decode(amr_str)
    except Exception as e:
        return None, f"penman_decode_failed: {e}", {}

    try:
        rule = get_rule(rule_name)
    except KeyError:
        return None, f"unknown_rule: {rule_name}", {}

    results = rule.apply(g)
    if not results or results[0].positive_graph is None:
        return None, f"rule_did_not_fire: {rule_name}", {}

    modified_amr = results[0].positive_graph

    # Multi-sample: try `num_samples` generations with diverse sampling
    candidates = []
    seen_texts = set()
    for sample_idx in range(num_samples):
        try:
            # Use sampling with different parameters per attempt for diversity
            if sample_idx == 0:
                # First attempt: greedy / beam search (high precision)
                sents, _ = gtos.generate([modified_amr])
            else:
                # Subsequent attempts: nucleus sampling for diversity
                sents, _ = gtos.generate(
                    [modified_amr],
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7 + 0.1 * sample_idx,
                )
        except Exception:
            continue
        if not sents or not sents[0]:
            continue
        text = sents[0]
        if text in seen_texts:
            continue  # dedupe identical outputs
        seen_texts.add(text)
        passed, reason = _self_consistency_check(parser, text, modified_amr)
        candidates.append({
            "text": text,
            "sample_idx": sample_idx,
            "passed_self_check": passed,
            "reason": reason,
        })
        if passed:
            # Early-exit on first passing candidate
            return text, "ok", {
                "n_samples_tried": sample_idx + 1,
                "n_candidates": len(candidates),
                "all_candidates": candidates,
            }

    # No candidate passed — return the first one with the failure flagged
    if not candidates:
        return None, "no_candidates", {}
    first = candidates[0]
    return first["text"], f"self_check_failed: {first['reason']}", {
        "n_samples_tried": num_samples,
        "n_candidates": len(candidates),
        "all_candidates": candidates,
    }


RULE_DISPATCH = {
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-model", required=True)
    ap.add_argument("--gen-model", required=True)
    ap.add_argument("--test-sentences", default="extensions/pilot_study/test_sentences.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=4,
                    help="best-of-N samples per rule application")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import amrlib
    from amrlib.models.parse_xfm.inference import Inference

    log.info("Loading parser %s", args.parse_model)
    parser = Inference(args.parse_model, batch_size=1, num_beams=4)
    log.info("Loading generator %s", args.gen_model)
    gtos = amrlib.load_gtos_model(args.gen_model)

    sentences = json.loads(Path(args.test_sentences).read_text())["sentences"]
    if args.limit:
        sentences = sentences[: args.limit]

    status_counter: Counter = Counter()
    n_total = 0
    samples_per_pass = []  # track how many samples needed to pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout:
        for s in sentences:
            sentence = s["text"]
            for rule_label in s.get("applicable_rules", []):
                rule_names = RULE_DISPATCH.get(rule_label, [])
                if not rule_names:
                    rec = {
                        "sentence_id": s["id"],
                        "input": sentence,
                        "rule": rule_label,
                        "model": "amr_lda_v2",
                        "output": None,
                        "gold": s.get("gold_outputs", {}).get(rule_label),
                        "status": "rule_not_implemented",
                        "n_samples_tried": 0,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    status_counter["rule_not_implemented"] += 1
                    n_total += 1
                    continue
                output, status, meta = None, "rule_did_not_fire", {}
                for rname in rule_names:
                    output, status, meta = amr_lda_one_v2(
                        parser, gtos, sentence, rname,
                        num_samples=args.num_samples,
                    )
                    if output is not None:
                        break
                rec = {
                    "sentence_id": s["id"],
                    "input": sentence,
                    "rule": rule_label,
                    "model": "amr_lda_v2",
                    "output": output if output is not None else f"[{status}]",
                    "gold": s.get("gold_outputs", {}).get(rule_label),
                    "status": status,
                    "n_samples_tried": meta.get("n_samples_tried", 0),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                status_counter[status.split(":")[0]] += 1
                if status == "ok":
                    samples_per_pass.append(meta.get("n_samples_tried", 1))
                n_total += 1
                if n_total % 10 == 0:
                    log.info("  %d items, ok=%d", n_total,
                             status_counter.get("ok", 0))

    log.info("Done. %d records, status: %s", n_total, dict(status_counter))
    if samples_per_pass:
        avg = sum(samples_per_pass) / len(samples_per_pass)
        log.info("  avg samples needed for self-check pass: %.2f", avg)
        log.info("  first-sample pass rate: %d/%d (%.1f%%)",
                 sum(1 for s in samples_per_pass if s == 1),
                 len(samples_per_pass),
                 100 * sum(1 for s in samples_per_pass if s == 1) / len(samples_per_pass))


if __name__ == "__main__":
    main()
