"""Before/after comparison report for AMR-LDA pipeline improvements.

Given two verifier runs (before patches, after patches), highlight:
  - Items where coverage was added (rule_did_not_fire → ok)
  - Items where verifier verdict flipped (NEQ → EQ or vice versa)
  - Net change in per-system equivalence rate

Usage:
    PYTHONPATH=/path/to/repo python -m extensions.pilot_study.before_after_diff \\
        --before extensions/auto_verifier/results/run3_clean/per_item.jsonl \\
        --after extensions/auto_verifier/results/run4_clean/per_item.jsonl \\
        --out-md extensions/pilot_study/results/combined/BEFORE_AFTER.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load(path: Path) -> Dict[str, dict]:
    items = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                items[d["item_id"]] = d
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument(
        "--system-filter",
        type=str,
        default="amr_lda",
        help="only report on items from this system",
    )
    args = ap.parse_args()

    before = load(args.before)
    after = load(args.after)

    common_ids = set(before) & set(after)
    only_after = set(after) - set(before)
    only_before = set(before) - set(after)

    sys_filter = args.system_filter

    # Counts
    coverage_added: List[dict] = []  # was no real output, now has
    verdict_flip_pos: List[dict] = []  # was NEQ, now EQ
    verdict_flip_neg: List[dict] = []  # was EQ, now NEQ
    unchanged_eq: List[dict] = []
    unchanged_neq: List[dict] = []

    for iid in sorted(common_ids):
        if not iid.startswith(sys_filter + "::"):
            continue
        b = before[iid]
        a = after[iid]
        if b["majority_label"] == "NOT_EQUIVALENT" and a["majority_label"] == "EQUIVALENT":
            verdict_flip_pos.append(a)
        elif b["majority_label"] == "EQUIVALENT" and a["majority_label"] == "NOT_EQUIVALENT":
            verdict_flip_neg.append(a)
        elif a["majority_label"] == "EQUIVALENT":
            unchanged_eq.append(a)
        elif a["majority_label"] == "NOT_EQUIVALENT":
            unchanged_neq.append(a)

    lines: List[str] = []
    lines.append(f"# Before/After Diff — {sys_filter}")
    lines.append("")
    lines.append(f"Comparing **{args.before.parent.name}** (before) vs **{args.after.parent.name}** (after).")
    lines.append("")
    lines.append(f"- Coverage delta: {len(only_after)} new items judged | {len(only_before)} dropped")
    lines.append(f"- Flips NEQ → EQ: **{len(verdict_flip_pos)}**")
    lines.append(f"- Flips EQ → NEQ: **{len(verdict_flip_neg)}** (regression)")
    lines.append(f"- Stable EQ: {len(unchanged_eq)}")
    lines.append(f"- Stable NEQ: {len(unchanged_neq)}")
    lines.append("")

    if verdict_flip_pos:
        lines.append(f"## Items the patch RECOVERED ({len(verdict_flip_pos)} items)")
        lines.append("")
        for it in verdict_flip_pos[:30]:
            lines.append(f"**{it['item_id']}** ({it['rule']})  ")
            lines.append(f"Input: _{it['input_sentence']}_  ")
            lines.append(f"Output: **{it['candidate_sentence']}**  ")
            v = " · ".join(f"{vd['verifier_name']}={vd['label']}" for vd in it['verdicts'])
            lines.append(f"Verdicts: {v}  ")
            lines.append("")

    if verdict_flip_neg:
        lines.append(f"## Regressions ({len(verdict_flip_neg)} items, the patch made these WORSE)")
        lines.append("")
        for it in verdict_flip_neg[:30]:
            lines.append(f"**{it['item_id']}** ({it['rule']})")
            lines.append(f"Input: _{it['input_sentence']}_  ")
            lines.append(f"Output: **{it['candidate_sentence']}**  ")
            v = " · ".join(f"{vd['verifier_name']}={vd['label']}" for vd in it['verdicts'])
            lines.append(f"Verdicts: {v}  ")
            lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"Wrote diff to {args.out_md}")
    print(f"Recovered: {len(verdict_flip_pos)}, regressed: {len(verdict_flip_neg)}, "
          f"new items: {len(only_after)}")


if __name__ == "__main__":
    main()
