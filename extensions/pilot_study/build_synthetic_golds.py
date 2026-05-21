"""Materialize (modified_AMR, hand-derived_gold) pairs for the SELF_CHECK
failure cases that test_sentences.json does NOT have a gold for, but which
admit a clean canonical-form rewrite under their logic rule.

For each (sentence_id, rule) below the rewrite was derived by hand from the
formal rule:
  - implication:        P → Q  ≡  ¬P ∨ Q
  - contraposition:     P → Q  ≡  ¬Q → ¬P   (and De Morgan when P is a conjunction)
  - de_morgan:          ¬A ∧ ¬B ≡ ¬(A ∨ B)
  - modal_strength_inv: □P ≡ ¬◇¬P;  ◇P ≡ ¬□¬P

These complement the test_sentences.json golds harvested by
build_failure_set_golds.py.

Output: extensions/pilot_study/synthetic_golds.jsonl
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("synth_golds")


SYNTHETIC_GOLDS = {
    ("S004", "implication"):
        "The water does not reach 100 degrees Celsius at sea level, or it boils.",
    ("S008", "contraposition"):
        "If a patient does not recover within two weeks, then they did not take "
        "the medication or did not follow the doctor's instructions.",
    ("S014", "de_morgan"):
        "It is not the case that the manager attended the meeting or the "
        "assistant attended the meeting.",
    ("S026", "implication"):
        "A country does not produce more goods than it consumes, or it exports "
        "the surplus to trading partners.",
    ("S045", "contraposition"):
        "If the flight does not depart on time, then some passenger does not "
        "board by 8 AM or the crew is not ready.",
    ("S040", "modal_strength_inversion"):
        "It is not possible that Alice does not finish her homework before dinner.",
    ("S041", "modal_strength_inversion"):
        "It is not necessary that visitors not use the WiFi after registering "
        "at the front desk.",
}


RULE_DISPATCH = {
    "contraposition": ["contraposition"],
    "de_morgan": ["de_morgan"],
    "implication": ["implication"],
    "modal_strength_inversion": ["modal_strength_inversion"],
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from amrlib.models.parse_xfm.inference import Inference
    from extensions.logic_rules import get_rule

    sents = json.loads(
        Path("extensions/pilot_study/test_sentences.json").read_text()
    )["sentences"]
    by_id = {s["id"]: s for s in sents}

    log.info("Loading parser…")
    parser = Inference(
        "amrlib/data/model_parse_xfm_bart_large-v0_1_0",
        batch_size=1, num_beams=1,
    )

    out_path = Path("extensions/pilot_study/synthetic_golds.jsonl")
    n_written = 0
    with open(out_path, "w") as fout:
        for (sid, rule), gold in SYNTHETIC_GOLDS.items():
            s = by_id.get(sid)
            if not s:
                log.warning("unknown sid %s", sid)
                continue
            graph_str = parser.parse_sents([s["text"]])[0]
            if not graph_str:
                log.warning("parse_empty for %s", sid)
                continue
            g = penman.decode(graph_str)
            rule_name = RULE_DISPATCH.get(rule, [rule])[0]
            try:
                rule_obj = get_rule(rule_name)
            except KeyError:
                log.warning("unknown rule %s, skipping %s", rule_name, sid)
                continue
            results = rule_obj.apply(g)
            if not results or results[0].positive_graph is None:
                log.warning("rule_did_not_fire %s/%s", sid, rule_name)
                continue
            amr_positive = results[0].positive_graph
            fout.write(
                json.dumps(
                    {
                        "sentence_id": sid,
                        "rule": rule,
                        "anchor": s["text"],
                        "amr_positive": amr_positive,
                        "positive": gold,
                        "source": "hand_derived",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
            log.info("✓ %s/%s -> %s", sid, rule, gold[:60])

    log.info("Wrote %d synthetic-gold pairs to %s", n_written, out_path)


if __name__ == "__main__":
    main()
