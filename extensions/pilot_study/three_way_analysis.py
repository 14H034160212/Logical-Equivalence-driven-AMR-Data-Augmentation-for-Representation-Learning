"""Three-way comparison: AMR-LDA vs gpt-4o vs gpt-4o-mini.

Properly handles AMR-LDA's "rule did not fire" / "rule not implemented" cases
by separating COVERAGE (did the system produce output?) from QUALITY (given
output, is it logically equivalent?).

Reads three sources:
  - Pilot rewrite outputs:  extensions/pilot_study/results/combined/rewrite/*.jsonl
  - Auto-verifier consensus:   extensions/auto_verifier/results/<run>/per_item.jsonl

Usage:
    PYTHONPATH=/path/to/repo python -m extensions.pilot_study.three_way_analysis \\
        --rewrite-dir extensions/pilot_study/results/combined/rewrite/ \\
        --verifier-items extensions/auto_verifier/results/run3_clean/per_item.jsonl \\
        --out-md extensions/pilot_study/results/combined/THREE_WAY.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def load_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


SENTINEL_PREFIXES = ("[rule_did_not_fire", "[rule_not_implemented", "[parse_failed",
                     "[generation_failed")


def has_real_output(rec: dict) -> bool:
    """An item has a 'real' output if it's not a status sentinel."""
    out = rec.get("output")
    if out is None:
        return False
    if any(out.startswith(p) for p in SENTINEL_PREFIXES):
        return False
    status = rec.get("status", "ok")
    if status != "ok" and status != "":
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite-dir", type=Path, required=True)
    ap.add_argument("--verifier-items", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    # Load all candidate outputs
    rewrites: Dict[str, List[dict]] = {}
    for jf in sorted(args.rewrite_dir.glob("*.jsonl")):
        rewrites[jf.stem] = load_jsonl(jf)

    # Load verifier verdicts and index by item_id
    verdicts = load_jsonl(args.verifier_items)
    verdict_by_id = {v["item_id"]: v for v in verdicts}

    # Per-system stats
    stats = {}
    for model, records in rewrites.items():
        n_total = len(records)
        n_with_output = sum(1 for r in records if has_real_output(r))
        # Map (sentence_id, rule) -> rec
        # For verifier verdicts, the item_id format is "<system>::<sentence_id>::<rule>"
        n_eq = 0
        n_neq = 0
        n_pending = 0
        n_no_judgment = 0
        for r in records:
            if not has_real_output(r):
                n_no_judgment += 1
                continue
            iid = f"{model}::{r['sentence_id']}::{r['rule']}"
            v = verdict_by_id.get(iid)
            if v is None:
                n_no_judgment += 1
                continue
            if v.get("needs_human_review"):
                n_pending += 1
            elif v["majority_label"] == "EQUIVALENT":
                n_eq += 1
            elif v["majority_label"] == "NOT_EQUIVALENT":
                n_neq += 1
            else:
                n_no_judgment += 1
        stats[model] = {
            "n_total": n_total,
            "n_with_output": n_with_output,
            "coverage": n_with_output / n_total if n_total else 0.0,
            "n_eq": n_eq,
            "n_neq": n_neq,
            "n_pending": n_pending,
            "n_no_judgment": n_no_judgment,
            "decided": n_eq + n_neq,
            "quality_given_decided": n_eq / (n_eq + n_neq) if (n_eq + n_neq) else 0.0,
            "overall_eq_over_total": n_eq / n_total if n_total else 0.0,
        }

    # Per-rule breakdown per system
    per_rule_stats: Dict[str, Dict[str, dict]] = defaultdict(dict)
    all_rules = set()
    for model, records in rewrites.items():
        by_rule: Dict[str, dict] = defaultdict(lambda: {"n": 0, "n_with_output": 0,
                                                         "n_eq": 0, "n_neq": 0,
                                                         "n_pending": 0})
        for r in records:
            rule = r.get("rule", "unknown")
            all_rules.add(rule)
            by_rule[rule]["n"] += 1
            if not has_real_output(r):
                continue
            by_rule[rule]["n_with_output"] += 1
            iid = f"{model}::{r['sentence_id']}::{rule}"
            v = verdict_by_id.get(iid)
            if v is None:
                continue
            if v.get("needs_human_review"):
                by_rule[rule]["n_pending"] += 1
            elif v["majority_label"] == "EQUIVALENT":
                by_rule[rule]["n_eq"] += 1
            elif v["majority_label"] == "NOT_EQUIVALENT":
                by_rule[rule]["n_neq"] += 1
        per_rule_stats[model] = dict(by_rule)

    # ---- Render markdown ----
    lines = ["# Three-way Comparison — AMR-LDA vs gpt-4o vs gpt-4o-mini", ""]
    lines.append("Auto-generated by `extensions/pilot_study/three_way_analysis.py`.")
    lines.append("")
    lines.append("## Headline summary")
    lines.append("")
    lines.append(
        "We separate **coverage** (did the system produce a real output?) from "
        "**quality** (given a real output and a decided verifier verdict, was it "
        "logically equivalent?)."
    )
    lines.append("")
    lines.append("| System | Total | With output | Coverage | EQ | NEQ | Pending | "
                 "Quality (EQ/decided) | Overall (EQ/total) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in sorted(stats.keys()):
        s = stats[model]
        lines.append(
            f"| {model} | {s['n_total']} | {s['n_with_output']} | "
            f"{s['coverage']:.1%} | {s['n_eq']} | {s['n_neq']} | {s['n_pending']} | "
            f"{s['quality_given_decided']:.1%} | {s['overall_eq_over_total']:.1%} |"
        )
    lines.append("")
    lines.append("### Reading the headline")
    lines.append("")
    lines.append(
        "- **Coverage** measures the fraction of items the system produced ANY output for. "
        "LLMs always produce 100% (they paraphrase no matter what); AMR-LDA's coverage is "
        "limited by parser + dictionary gaps."
    )
    lines.append(
        "- **Quality (EQ/decided)** measures, for items the verifiers AGREED on, "
        "what fraction were judged logically equivalent. This is the strict comparison."
    )
    lines.append(
        "- **Overall (EQ/total)** combines both: out of ALL items, how many had both "
        "(a) the system producing output and (b) verifiers agreeing it's equivalent."
    )
    lines.append("")

    lines.append("## Per-rule breakdown")
    lines.append("")
    models = sorted(per_rule_stats.keys())
    lines.append("| Rule | " + " | ".join(f"{m} (cov / qual)" for m in models) + " |")
    lines.append("|" + "---|" * (len(models) + 1))
    for rule in sorted(all_rules):
        cells = [rule]
        for m in models:
            r = per_rule_stats[m].get(rule, {"n": 0, "n_with_output": 0,
                                              "n_eq": 0, "n_neq": 0, "n_pending": 0})
            n = r["n"]
            cov = r["n_with_output"] / n if n else 0.0
            decided = r["n_eq"] + r["n_neq"]
            qual = r["n_eq"] / decided if decided else 0.0
            if n == 0:
                cells.append("—")
            else:
                cells.append(f"{cov:.0%} / {qual:.0%} (n={n})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Failure modes section ----
    lines.append("## AMR-LDA failure modes")
    lines.append("")
    amr_records = rewrites.get("amr_lda", [])
    status_counter = Counter(r.get("status", "?") for r in amr_records if not has_real_output(r))
    if status_counter:
        lines.append("Distribution of failures when AMR-LDA did not produce real output:")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|---|---|")
        for status, cnt in status_counter.most_common():
            lines.append(f"| `{status}` | {cnt} |")
        lines.append("")
        lines.append(
            "These breakdowns suggest where the AMR-LDA pipeline most needs improvement: "
            "more inverse-frame dictionary entries, handling parser AMR-style variations "
            "for De Morgan, and implementing the UMR-level rules (aspect / modal / temporal / tense)."
        )
        lines.append("")

    lines.append("## What the AMR-LDA outputs actually look like")
    lines.append("")
    amr_ok = [r for r in amr_records if has_real_output(r)]
    for r in amr_ok[:12]:
        lines.append(f"**[{r['sentence_id']}/{r['rule']}]**  ")
        lines.append(f"Input:  _{r['input']}_  ")
        lines.append(f"AMR-LDA: **{r['output']}**  ")
        gpt4o_match = next(
            (g for g in rewrites.get("gpt-4o", [])
             if g["sentence_id"] == r["sentence_id"] and g["rule"] == r["rule"]),
            None,
        )
        if gpt4o_match:
            lines.append(f"gpt-4o:  _{gpt4o_match['output']}_  ")
        if r.get("gold"):
            lines.append(f"gold:    _{r['gold']}_  ")
        lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"Wrote 3-way analysis to {args.out_md}")
    # Also print headline numbers to stdout
    print()
    print("HEADLINE:")
    for model in sorted(stats.keys()):
        s = stats[model]
        print(
            f"  {model:15s}  cov={s['coverage']:.1%}  quality={s['quality_given_decided']:.1%}  "
            f"overall={s['overall_eq_over_total']:.1%}  (eq={s['n_eq']} neq={s['n_neq']} pending={s['n_pending']})"
        )


if __name__ == "__main__":
    main()
