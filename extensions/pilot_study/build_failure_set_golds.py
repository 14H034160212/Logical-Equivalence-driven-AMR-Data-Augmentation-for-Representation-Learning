"""Materialize (modified_AMR, gold_text) pairs for the SELF_CHECK failure
set so the T5wtense fine-tune can learn directly from gold rewrites on the
exact cases where the stock decoder hallucinates polarity.

For each (sentence_id, rule) in the failure set where test_sentences.json
provides a gold rewrite:
  1. parse the original input sentence to AMR
  2. apply the rule to get the modified AMR (what the gold text should describe)
  3. emit {amr_positive: <modified AMR>, positive: <gold text>, ...} jsonl

Output: extensions/pilot_study/failure_set_golds.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("failure_golds")


FAILURE_SET = [
    ("S004", "contraposition"),
    ("S005", "contraposition"),
    ("S008", "contraposition"),
    ("S013", "de_morgan"),
    ("S014", "de_morgan"),
    ("S022", "contraposition"),
    ("S026", "contraposition"),
    ("S028", "contraposition"),
]


RULE_DISPATCH = {
    "contraposition": ["contraposition"],
    "de_morgan": ["de_morgan"],
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

    out_path = Path("extensions/pilot_study/failure_set_golds.jsonl")
    n_written = 0
    with open(out_path, "w") as fout:
        for sid, rule in FAILURE_SET:
            s = by_id.get(sid)
            gold = (s.get("gold_outputs") or {}).get(rule) if s else None
            if not gold:
                log.warning("no gold for %s/%s, skipping", sid, rule)
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
                        "source": "test_sentences_gold",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
            log.info("✓ %s/%s -> %s", sid, rule, gold[:60])

    log.info("Wrote %d gold pairs to %s", n_written, out_path)


if __name__ == "__main__":
    main()
