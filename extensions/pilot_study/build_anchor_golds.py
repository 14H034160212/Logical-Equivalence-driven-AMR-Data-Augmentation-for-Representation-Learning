"""For each (sentence_id, rule) where v3 regressed (stock was OK, v3 broke
it), use the stock generator's correct output as a gold anchor for the
next fine-tune. This guards the rules without explicit hand-derived golds
(notably double_negation) from drifting under the silver pairs.

Reads stock outputs from
extensions/pilot_study/results/combined/rewrite/amr_lda.jsonl and parses
the input + applies the rule fresh to materialise the modified AMR
(which the stock jsonl doesn't carry).

Output: extensions/pilot_study/anchor_golds.jsonl
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("anchor_golds")

# (sentence_id, rule) — items where v3 regressed vs stock
REGRESSED = [
    ("S001", "implication"),
    ("S005", "double_negation"),
    ("S006", "contraposition"),
    ("S042", "double_negation"),
]

RULE_DISPATCH = {
    "contraposition": "contraposition",
    "implication": "implication",
    "double_negation": "double_negation",
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from amrlib.models.parse_xfm.inference import Inference
    from extensions.logic_rules import get_rule

    stock_rows = {}
    for line in Path(
        "extensions/pilot_study/results/combined/rewrite/amr_lda.jsonl"
    ).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("output"):
                stock_rows[(r["sentence_id"], r["rule"])] = r

    log.info("Loading parser…")
    parser = Inference(
        "amrlib/data/model_parse_xfm_bart_large-v0_1_0",
        batch_size=1, num_beams=1,
    )

    out_path = Path("extensions/pilot_study/anchor_golds.jsonl")
    n_written = 0
    with open(out_path, "w") as fout:
        for sid, rule in REGRESSED:
            row = stock_rows.get((sid, rule))
            if not row:
                log.warning("no stock-OK row for %s/%s", sid, rule)
                continue
            text = row["input"]
            target = row["output"]
            graph_str = parser.parse_sents([text])[0]
            if not graph_str:
                log.warning("parse_empty for %s", sid)
                continue
            g = penman.decode(graph_str)
            rule_name = RULE_DISPATCH.get(rule, rule)
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
                        "anchor": text,
                        "amr_positive": amr_positive,
                        "positive": target,
                        "source": "stock_correct",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
            log.info("✓ %s/%s -> %s", sid, rule, target[:60])

    log.info("Wrote %d anchor-gold pairs to %s", n_written, out_path)


if __name__ == "__main__":
    main()
