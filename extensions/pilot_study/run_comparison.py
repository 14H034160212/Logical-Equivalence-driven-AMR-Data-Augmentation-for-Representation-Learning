"""Compare multiple auto-verifier runs and produce a longitudinal trajectory table.

Given multiple per_item.jsonl files (one per run), produce a single summary
showing how AMR-LDA's coverage and quality evolved across patches.

Usage:
    PYTHONPATH=. python -m extensions.pilot_study.run_comparison \\
        --runs run3:extensions/auto_verifier/results/run3_clean \\
               run4:extensions/auto_verifier/results/run4_clean \\
               run5:extensions/auto_verifier/results/run5_clean \\
        --out-md extensions/pilot_study/results/combined/TRAJECTORY.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


SENTINEL_PREFIXES = ("[rule_did_not_fire", "[rule_not_implemented",
                     "[parse_failed", "[generation_failed")


def load_run(run_dir: Path) -> Dict[str, dict]:
    per_item = run_dir / "per_item.jsonl"
    items: Dict[str, dict] = {}
    if not per_item.exists():
        return items
    with open(per_item) as f:
        for line in f:
            d = json.loads(line)
            items[d["item_id"]] = d
    return items


def load_candidates(rewrite_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    if not rewrite_dir.exists():
        return out
    for jl in sorted(rewrite_dir.glob("*.jsonl")):
        out[jl.stem] = [json.loads(line) for line in open(jl)]
    return out


def has_real_output(rec: dict) -> bool:
    out = rec.get("output")
    if out is None:
        return False
    if any(out.startswith(p) for p in SENTINEL_PREFIXES):
        return False
    return (rec.get("status", "ok") in ("ok", ""))


def per_system_stats(verdicts_by_id, candidates):
    stats = {}
    for system, recs in candidates.items():
        n_total = len(recs)
        n_cov = sum(1 for r in recs if has_real_output(r))
        n_eq = n_neq = n_pending = 0
        for r in recs:
            if not has_real_output(r):
                continue
            iid = f"{system}::{r['sentence_id']}::{r['rule']}"
            v = verdicts_by_id.get(iid)
            if v is None:
                continue
            if v.get("needs_human_review"):
                n_pending += 1
            elif v["majority_label"] == "EQUIVALENT":
                n_eq += 1
            elif v["majority_label"] == "NOT_EQUIVALENT":
                n_neq += 1
        decided = n_eq + n_neq
        stats[system] = {
            "n_total": n_total,
            "coverage": n_cov / n_total if n_total else 0.0,
            "quality": n_eq / decided if decided else 0.0,
            "overall": n_eq / n_total if n_total else 0.0,
            "n_eq": n_eq, "n_neq": n_neq, "n_pending": n_pending,
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="format: name:path  e.g. run3:results/run3_clean")
    ap.add_argument("--rewrite-dir", type=Path,
                    default="extensions/pilot_study/results/combined/rewrite")
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    run_entries = []
    for spec in args.runs:
        name, _, path = spec.partition(":")
        run_entries.append((name, Path(path)))

    # Candidates are shared across runs in the live setup, but AMR-LDA may
    # have been re-generated between runs, so we load them snapshotted from
    # the rewrite-dir. For now we trust the LATEST rewrite-dir state — and
    # let the verifier verdict tell us coverage/quality at each run.
    candidates_now = load_candidates(args.rewrite_dir)

    rows: List[Dict] = []
    for name, run_dir in run_entries:
        verdicts = load_run(run_dir)
        # Reconstruct per-run system stats. We assume the candidate set in
        # each run was whatever was on disk at time of running; here we use
        # the current snapshot as a proxy.
        stats = per_system_stats(verdicts, candidates_now)
        for system, s in stats.items():
            rows.append({"run": name, "system": system, **s})

    # Render markdown
    lines = ["# AMR-LDA Improvement Trajectory", ""]
    lines.append("Per-run coverage / quality / overall numbers across the pilot.")
    lines.append("")

    # Get unique systems
    systems = sorted({r["system"] for r in rows})
    runs = [name for name, _ in run_entries]

    for metric, title in [
        ("coverage", "Coverage (real outputs / total items)"),
        ("quality", "Quality (EQ / decided items)"),
        ("overall", "Overall (EQ / total items)"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| System | " + " | ".join(runs) + " |")
        lines.append("|" + "---|" * (len(runs) + 1))
        for sys in systems:
            cells = [sys]
            for run in runs:
                val = next(
                    (r[metric] for r in rows if r["run"] == run and r["system"] == sys),
                    0.0,
                )
                cells.append(f"{val:.1%}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"Trajectory written to {args.out_md}")
    for row in rows:
        print(f"  {row['run']:10s} {row['system']:15s}  "
              f"cov={row['coverage']:.1%} qual={row['quality']:.1%} overall={row['overall']:.1%}")


if __name__ == "__main__":
    main()
