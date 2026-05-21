"""Analyze AMR-LDA outputs by self-consistency status.

Shows quality difference between:
  - ok items (passed self-consistency check)
  - self_check_failed items (T5wtense dropped/flipped a polarity)
  - rule_did_not_fire items (excluded from coverage)

Used to demonstrate the value of the self-consistency check.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite", type=Path, required=True,
                    help="path to amr_lda.jsonl")
    ap.add_argument("--verifier-items", type=Path, required=True,
                    help="path to per_item.jsonl from auto_verifier")
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    rewrites = []
    with open(args.rewrite) as f:
        for line in f:
            rewrites.append(json.loads(line))

    verdicts = {}
    with open(args.verifier_items) as f:
        for line in f:
            d = json.loads(line)
            verdicts[d["item_id"]] = d

    # Group by status category
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rewrites:
        st = r.get("status", "ok")
        if st.startswith("rule_did_not_fire") or st.startswith("rule_not_implemented"):
            cat = "rule_did_not_fire"
        elif st.startswith("self_check_failed"):
            cat = "self_check_failed"
        elif st == "ok" or st == "ok_retry":
            cat = "ok"
        else:
            cat = "other"
        groups[cat].append(r)

    # For each group, compute verifier consensus stats
    lines = ["# Self-Consistency Check — Quality Breakdown", ""]
    lines.append("Demonstrates the value of the T5wtense self-consistency check by")
    lines.append("comparing quality of items that passed vs failed the polarity")
    lines.append("parity check.")
    lines.append("")
    lines.append("| Status group | N | V1 AMR EQ | V2 LLM EQ | Consensus EQ | Consensus NEQ | Pending |")
    lines.append("|---|---|---|---|---|---|---|")

    for cat in ["ok", "self_check_failed", "rule_did_not_fire"]:
        recs = groups.get(cat, [])
        if not recs:
            continue
        n = len(recs)
        v1_eq = 0
        v2_eq = 0
        cons_eq = 0
        cons_neq = 0
        pending = 0
        for r in recs:
            iid = f"amr_lda::{r['sentence_id']}::{r['rule']}"
            v = verdicts.get(iid)
            if v is None:
                continue
            for vd in v["verdicts"]:
                if vd["verifier_name"] == "amr_struct" and vd["label"] == "EQUIVALENT":
                    v1_eq += 1
                if vd["verifier_name"] == "llm_gpt-4o-mini" and vd["label"] == "EQUIVALENT":
                    v2_eq += 1
            if v.get("needs_human_review"):
                pending += 1
            elif v["majority_label"] == "EQUIVALENT":
                cons_eq += 1
            elif v["majority_label"] == "NOT_EQUIVALENT":
                cons_neq += 1
        lines.append(f"| {cat} | {n} | {v1_eq} ({v1_eq*100//max(1,n)}%) | "
                     f"{v2_eq} ({v2_eq*100//max(1,n)}%) | "
                     f"{cons_eq} | {cons_neq} | {pending} |")
    lines.append("")

    # Detail the self_check_failed items — these are the headline of our story
    sc_failed = groups.get("self_check_failed", [])
    if sc_failed:
        lines.append("## Items flagged by self-consistency check")
        lines.append("")
        lines.append(
            "These outputs had a polarity-parity mismatch between the rule-applied "
            "AMR (expected) and the AMR re-parsed from T5wtense's generated text. "
            "Each one is a known generator failure — the rule worked correctly but "
            "the text generator dropped or flipped a negation."
        )
        lines.append("")
        lines.append("| ID | Rule | Input | Output | Status |")
        lines.append("|---|---|---|---|---|")
        for r in sc_failed:
            inp = r['input'][:60].replace("|", "\\|") + ("..." if len(r['input']) > 60 else "")
            out = (r['output'] or "")[:80].replace("|", "\\|")
            st = r['status'].replace("self_check_failed: ", "")
            lines.append(f"| {r['sentence_id']} | {r['rule']} | _{inp}_ | **{out}** | {st} |")

    args.out_md.write_text("\n".join(lines))
    print(f"Wrote self-check analysis to {args.out_md}")
    # Also print headline
    print()
    for cat in ["ok", "self_check_failed", "rule_did_not_fire"]:
        recs = groups.get(cat, [])
        print(f"  {cat}: {len(recs)} items")


if __name__ == "__main__":
    main()
