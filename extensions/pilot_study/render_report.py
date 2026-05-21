"""Render a publication-style markdown report from the pilot + auto-verifier outputs.

Usage:
    PYTHONPATH=/path/to/repo python render_report.py \\
        --pilot-summary results/combined/summary.json \\
        --verifier-dir ../auto_verifier/results/run1 \\
        --out report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render(
    pilot_summary_path: Path,
    verifier_dir: Optional[Path],
    out_path: Path,
):
    lines: List[str] = []
    lines.append("# Pilot Study Results — AMR-LDA Extension")
    lines.append("")
    lines.append(
        "Auto-generated report from the LLM-as-rewriter pilot + multi-verifier "
        "consensus pipeline. Numbers are placeholder-real (single-seed, two-model)."
    )
    lines.append("")

    # ---- Section 1: pilot summary ----
    if pilot_summary_path.exists():
        summary = json.loads(pilot_summary_path.read_text())
        lines.append("## 1. Pilot rewrite results (exact-match against gold)")
        lines.append("")
        lines.append("| Model | Records | Exact match | LLM-as-judge equivalence |")
        lines.append("|---|---|---|---|")
        for model, m in summary.get("models", {}).items():
            j = summary.get("judge", {}).get(model, {})
            decided = j.get("equivalent", 0) + j.get("not_equivalent", 0)
            jr = (
                j.get("equivalent", 0) / decided if decided > 0 else 0.0
            )
            n = 0
            for r, c in summary.get("per_rule", {}).get(model, {}).items():
                n += c.get("total", 0)
            lines.append(
                f"| {model} | {n} | {m['exact_match_rate']:.1%} | "
                f"{jr:.1%} (eq={j.get('equivalent',0)} neq={j.get('not_equivalent',0)} skip={j.get('skipped',0)}) |"
            )
        lines.append("")

        lines.append("### Per-rule breakdown (LLM-judge equivalence rate)")
        lines.append("")
        # Collect all rules
        all_rules: set = set()
        for model in summary.get("per_rule", {}):
            all_rules.update(summary["per_rule"][model].keys())

        models = list(summary.get("per_rule", {}).keys())
        header = "| Rule | " + " | ".join(models) + " |"
        sep = "|---|" + "|".join(["---"] * len(models)) + "|"
        lines.append(header)
        lines.append(sep)
        for rule in sorted(all_rules):
            row = [rule]
            for m in models:
                cell = summary["per_rule"][m].get(rule, {})
                total = cell.get("total", 0)
                if total:
                    eq = cell.get("judge_equivalent", 0)
                    row.append(f"{eq/total:.0%} (n={total})")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ---- Section 2: multi-verifier consensus ----
    if verifier_dir is not None and verifier_dir.exists():
        lines.append("## 2. Multi-verifier consensus")
        lines.append("")
        bv = verifier_dir / "summary_by_verifier.json"
        bs = verifier_dir / "summary_by_system.json"
        if bv.exists():
            by_v = json.loads(bv.read_text())
            lines.append("### Per-verifier rates (anti-circularity table)")
            lines.append("")
            lines.append("| Verifier | EQUIVALENT | NOT_EQUIVALENT | Abstain | Rate |")
            lines.append("|---|---|---|---|---|")
            for name, b in by_v.items():
                lines.append(
                    f"| {name} | {b['equivalent']} | {b['not_equivalent']} | "
                    f"{b['abstain']} | {b['equivalence_rate']:.1%} |"
                )
            lines.append("")
        if bs.exists():
            by_s = json.loads(bs.read_text())
            lines.append("### Per-system consensus rates")
            lines.append("")
            lines.append("| System | Total | EQ | NEQ | Pending review | Rate (decided) |")
            lines.append("|---|---|---|---|---|---|")
            for name, b in by_s.items():
                lines.append(
                    f"| {name} | {b['n']} | {b['equivalent']} | {b['not_equivalent']} | "
                    f"{b['pending_review']} | {b['equivalence_rate_decided']:.1%} |"
                )
            lines.append("")

        review_q = verifier_dir / "review_queue.jsonl"
        review = load_jsonl(review_q)
        if review:
            lines.append(f"### Review queue ({len(review)} items requiring human spot-check)")
            lines.append("")
            lines.append(
                "Each item below had at least one verifier disagree with the majority. "
                "These are the only items needing human eyes."
            )
            lines.append("")
            lines.append("| # | Rule | Input → Candidate | Verdicts |")
            lines.append("|---|---|---|---|")
            for i, item in enumerate(review[:30]):  # cap display at 30
                input_s = item['input_sentence'][:60] + ('...' if len(item['input_sentence']) > 60 else '')
                cand_s = item['candidate_sentence'][:60] + ('...' if len(item['candidate_sentence']) > 60 else '')
                verdicts = " · ".join(
                    f"{v['verifier_name']}={v['label']}" for v in item['verdicts']
                )
                lines.append(
                    f"| {i+1} | {item['rule']} | _{input_s}_ → **{cand_s}** | {verdicts} |"
                )
            if len(review) > 30:
                lines.append(f"| ... | ... | _({len(review)-30} more)_ | ... |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Generated by `extensions/pilot_study/render_report.py`")
    out_path.write_text("\n".join(lines))
    print(f"report written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-summary", type=Path, required=True)
    ap.add_argument("--verifier-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    render(args.pilot_summary, args.verifier_dir, args.out)


if __name__ == "__main__":
    main()
