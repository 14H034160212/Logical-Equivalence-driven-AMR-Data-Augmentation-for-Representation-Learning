"""Re-aggregate consensus from existing per_item.jsonl, optionally dropping verifiers.

Use this when one verifier turned out to be broken or untrustworthy and you
want to re-compute the consensus without re-running the full pipeline.

Usage:
    PYTHONPATH=/path/to/repo python -m extensions.auto_verifier.reaggregate \\
        --in-dir extensions/auto_verifier/results/run2 \\
        --out-dir extensions/auto_verifier/results/run2_v1v2 \\
        --drop-verifiers smatch_struct
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--drop-verifiers",
        nargs="*",
        default=[],
        help="verifier_name values to exclude from consensus",
    )
    args = ap.parse_args()

    drop = set(args.drop_verifiers)
    in_per_item = args.in_dir / "per_item.jsonl"
    items = []
    with open(in_per_item) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # Convert "rule did not fire on input" verdicts to UNKNOWN abstentions.
    # The AMR-struct verifier returned NOT_EQUIVALENT in this case which is
    # too aggressive — it should abstain.
    for item in items:
        for v in item["verdicts"]:
            reason = (v.get("details") or {}).get("reason", "")
            if "did not fire" in reason:
                v["label"] = "UNKNOWN"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_v = {}
    by_s = {}
    review = []
    new_items = []

    for item in items:
        new_verdicts = [v for v in item["verdicts"] if v["verifier_name"] not in drop]
        item = dict(item)
        item["verdicts"] = new_verdicts

        # Recompute consensus
        votes = [v["label"] for v in new_verdicts if v["label"] in ("EQUIVALENT", "NOT_EQUIVALENT")]
        if not votes:
            item["majority_label"] = "UNKNOWN"
            item["unanimity"] = False
            item["needs_human_review"] = True
            item["confidence"] = 0.0
        else:
            c = Counter(votes)
            top, count = c.most_common(1)[0]
            unanimous = count == len(votes)
            item["majority_label"] = top
            item["unanimity"] = unanimous
            item["needs_human_review"] = not unanimous
            agree_conf = sum(v["confidence"] for v in new_verdicts if v["label"] == top)
            item["confidence"] = agree_conf / len(votes)

        new_items.append(item)

        # Aggregate by verifier
        for v in new_verdicts:
            b = by_v.setdefault(v["verifier_name"], {"equivalent": 0, "not_equivalent": 0, "abstain": 0})
            if v["label"] == "EQUIVALENT":
                b["equivalent"] += 1
            elif v["label"] == "NOT_EQUIVALENT":
                b["not_equivalent"] += 1
            else:
                b["abstain"] += 1

        # Aggregate by system (extracted from item_id prefix)
        sys = item["item_id"].split("::")[0]
        s = by_s.setdefault(sys, {"equivalent": 0, "not_equivalent": 0, "pending_review": 0, "n": 0})
        s["n"] += 1
        if item["needs_human_review"]:
            s["pending_review"] += 1
        else:
            if item["majority_label"] == "EQUIVALENT":
                s["equivalent"] += 1
            elif item["majority_label"] == "NOT_EQUIVALENT":
                s["not_equivalent"] += 1

    for b in by_v.values():
        decided = b["equivalent"] + b["not_equivalent"]
        b["equivalence_rate"] = (b["equivalent"] / decided) if decided > 0 else 0.0
    for s in by_s.values():
        decided = s["equivalent"] + s["not_equivalent"]
        s["equivalence_rate_decided"] = (s["equivalent"] / decided) if decided > 0 else 0.0
        s["pending_fraction"] = s["pending_review"] / s["n"] if s["n"] > 0 else 0.0

    # Write outputs
    with open(args.out_dir / "per_item.jsonl", "w") as f:
        for it in new_items:
            f.write(json.dumps(it) + "\n")
    with open(args.out_dir / "summary_by_verifier.json", "w") as f:
        json.dump(by_v, f, indent=2)
    with open(args.out_dir / "summary_by_system.json", "w") as f:
        json.dump(by_s, f, indent=2)
    with open(args.out_dir / "review_queue.jsonl", "w") as f:
        for it in new_items:
            if it["needs_human_review"]:
                f.write(json.dumps(it) + "\n")

    print(f"Wrote re-aggregated results to {args.out_dir}")
    print()
    print("PER-VERIFIER:")
    for name, b in by_v.items():
        print(f"  {name:30s}  eq={b['equivalent']:>4d}  neq={b['not_equivalent']:>4d}  abstain={b['abstain']:>4d}  rate={b['equivalence_rate']:.3f}")
    print()
    print("PER-SYSTEM:")
    for name, b in by_s.items():
        print(f"  {name:30s}  eq={b['equivalent']:>4d}  neq={b['not_equivalent']:>4d}  pending={b['pending_review']:>4d}  rate={b['equivalence_rate_decided']:.3f}")
    print()
    review_count = sum(1 for it in new_items if it["needs_human_review"])
    print(f"Review queue: {review_count} / {len(new_items)} = {review_count*100/len(new_items):.1f}%")


if __name__ == "__main__":
    main()
