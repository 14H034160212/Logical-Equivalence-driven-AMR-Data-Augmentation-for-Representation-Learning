"""Analyze the pilot study results.

Given a results dir (output of run_llm_baseline.py rewrite mode), produce a
summary report comparing each model's outputs to the gold standard.

Two metrics:
  - Exact match against gold (string equality after light normalization)
  - LLM-as-judge semantic equivalence (uses OPENAI_API_KEY)

Usage:
    PYTHONPATH=/path/to/repo python analyze_results.py \\
        --results-dir extensions/pilot_study/results/<ts>/rewrite/ \\
        --judge gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def normalize(s: str) -> str:
    """Light normalization for exact-match comparison."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".!?")
    return s


def load_results(results_dir: Path) -> Dict[str, List[dict]]:
    """Load per-model JSONL files into {model_name: [records]}."""
    out: Dict[str, List[dict]] = {}
    for jsonl in sorted(results_dir.glob("*.jsonl")):
        model = jsonl.stem
        records = []
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        out[model] = records
    return out


def exact_match_rate(records: List[dict]) -> float:
    if not records:
        return 0.0
    n_match = 0
    n_with_gold = 0
    for r in records:
        gold = r.get("gold")
        out = r.get("output")
        if not gold or not out:
            continue
        n_with_gold += 1
        if normalize(out) == normalize(gold):
            n_match += 1
    return n_match / n_with_gold if n_with_gold else 0.0


def llm_judge_equivalence(
    records: List[dict],
    judge_model: str,
    api_key: Optional[str] = None,
) -> Dict[str, int]:
    """Use an LLM as judge to score each record as EQUIVALENT or NOT_EQUIVALENT."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai>=1.0")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required for LLM-as-judge")
    client = OpenAI(api_key=api_key)

    counts: Dict[str, int] = Counter()
    judge_prompt = (
        "You are an expert in formal logic. Judge whether two natural language "
        "sentences are LOGICALLY EQUIVALENT under classical propositional or "
        "first-order logic. Two sentences are logically equivalent if and only if "
        "they have the same truth value in every possible model. Be strict: "
        "differences in modal strength, quantifier scope, or pragmatic implicature "
        "that change truth conditions count as NOT equivalent.\n\n"
        "Reply with exactly one of: EQUIVALENT, NOT_EQUIVALENT."
    )
    for r in records:
        gold = r.get("gold")
        out = r.get("output")
        if not gold or not out:
            counts["skipped"] += 1
            continue
        if normalize(out) == normalize(gold):
            counts["equivalent"] += 1  # exact match → trivially equivalent
            r["judge_verdict"] = "EQUIVALENT (exact)"
            continue
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": f"Sentence A: {gold}\nSentence B: {out}\n\nAre A and B logically equivalent?"},
                ],
                temperature=0.0,
                max_tokens=20,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            if "NOT" in verdict or "NOT_EQUIVALENT" in verdict:
                counts["not_equivalent"] += 1
                r["judge_verdict"] = "NOT_EQUIVALENT"
            elif "EQUIVALENT" in verdict:
                counts["equivalent"] += 1
                r["judge_verdict"] = "EQUIVALENT"
            else:
                counts["unclear"] += 1
                r["judge_verdict"] = f"UNCLEAR ({verdict[:30]})"
        except Exception as e:
            counts["error"] += 1
            r["judge_verdict"] = f"ERROR: {e}"
    return counts


def per_rule_breakdown(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """Group records by rule and compute exact-match + judge verdict counts."""
    by_rule: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    for r in records:
        rule = r.get("rule", "unknown")
        gold = r.get("gold")
        out = r.get("output")
        if not gold or not out:
            by_rule[rule]["no_gold"] += 1
            continue
        by_rule[rule]["total"] += 1
        if normalize(out) == normalize(gold):
            by_rule[rule]["exact_match"] += 1
        verdict = r.get("judge_verdict", "")
        if verdict.startswith("EQUIVALENT"):
            by_rule[rule]["judge_equivalent"] += 1
        elif verdict.startswith("NOT_EQUIVALENT"):
            by_rule[rule]["judge_not_equivalent"] += 1
    return dict(by_rule)


def print_report(by_model: Dict[str, List[dict]], judge_results: Dict[str, Dict[str, int]]) -> None:
    print()
    print("=" * 80)
    print("SUMMARY  —  exact-match against gold + LLM-as-judge equivalence")
    print("=" * 80)
    for model, records in by_model.items():
        em = exact_match_rate(records)
        j = judge_results.get(model, {})
        total_judged = j.get("equivalent", 0) + j.get("not_equivalent", 0)
        judge_rate = (
            j.get("equivalent", 0) / total_judged if total_judged > 0 else 0.0
        )
        print(
            f"  {model:25s}  N={len(records):3d}  "
            f"exact_match={em:6.2%}   judge_equiv={judge_rate:6.2%}   "
            f"(eq={j.get('equivalent', 0)} neq={j.get('not_equivalent', 0)} "
            f"unclear={j.get('unclear', 0)} skip={j.get('skipped', 0)})"
        )
    print()
    print("=" * 80)
    print("PER-RULE BREAKDOWN (judge_equiv rate)")
    print("=" * 80)
    for model, records in by_model.items():
        print(f"\n{model}:")
        by_rule = per_rule_breakdown(records)
        for rule, counts in sorted(by_rule.items()):
            total = counts.get("total", 0)
            if not total:
                continue
            eq = counts.get("judge_equivalent", 0)
            em = counts.get("exact_match", 0)
            print(
                f"  {rule:30s}  N={total:3d}  exact={em/total:5.2%}  judge={eq/total:5.2%}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--judge", type=str, default=None, help="LLM model name for judge, e.g. gpt-4o-mini")
    ap.add_argument("--skip-judge", action="store_true", help="exact-match only, skip LLM judge")
    ap.add_argument("--out-summary", type=Path, default=None, help="write JSON summary to this path")
    args = ap.parse_args()

    by_model = load_results(args.results_dir)
    if not by_model:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded results for {len(by_model)} models:")
    for m, rs in by_model.items():
        print(f"  {m}: {len(rs)} records")

    judge_results = {}
    if not args.skip_judge and args.judge:
        print(f"\nRunning LLM-as-judge with {args.judge}...")
        for model, records in by_model.items():
            counts = llm_judge_equivalence(records, args.judge)
            judge_results[model] = counts

    print_report(by_model, judge_results)

    if args.out_summary:
        summary = {
            "models": {m: {"exact_match_rate": exact_match_rate(rs)} for m, rs in by_model.items()},
            "per_rule": {m: per_rule_breakdown(rs) for m, rs in by_model.items()},
            "judge": judge_results,
        }
        args.out_summary.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {args.out_summary}")


if __name__ == "__main__":
    main()
