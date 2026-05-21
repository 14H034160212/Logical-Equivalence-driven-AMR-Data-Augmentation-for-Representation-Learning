"""Orchestrate the auto-verifier pipeline over the pilot study outputs.

Inputs
------
- Outputs from `pilot_study/run_llm_baseline.py --mode rewrite` for each
  candidate system (AMR-LDA, GPT-4o, Claude, DeepSeek, Llama, ...).
- These are JSONL files at:
  extensions/pilot_study/results/<timestamp>/rewrite/<system>.jsonl
  with one record per (sentence_id, rule).

Outputs
-------
- Per-item ConsensusResult records
- Per-system equivalence-rate table
- Per-verifier equivalence-rate breakdown
- Review queue (items needing human spot-check)

Usage
-----
    # Dry-run with the MockParser (no real AMR parser needed)
    python -m extensions.auto_verifier.run_auto_verify \\
        --candidates-dir extensions/pilot_study/results/<ts>/rewrite/ \\
        --mock-parser \\
        --skip-llm-judges \\
        --out-dir extensions/auto_verifier/results/<ts>

    # Real run (requires amrlib + API keys)
    python -m extensions.auto_verifier.run_auto_verify \\
        --candidates-dir extensions/pilot_study/results/<ts>/rewrite/ \\
        --amrlib-model-dir /path/to/parse_xfm_bart_large_v0_1_0 \\
        --llm-judges gpt-4o claude-opus-4-7 \\
        --out-dir extensions/auto_verifier/results/<ts>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .amr_verifier import AMRVerifier, AmrlibParser, MockParser
from .consensus import (
    aggregate,
    review_queue,
    summary_by_system,
    summary_by_verifier,
)
from .llm_verifier import LLMVerifier
from .smatch_verifier import SmatchVerifier
from .types import ConsensusResult, Label, VerifierVerdict


log = logging.getLogger(__name__)


def load_candidates(candidates_dir: Path) -> List[dict]:
    """Load all candidate-system outputs into one flat list of records.

    Each record has: sentence_id, input, rule, model (= system name), output, gold.
    """
    records: List[dict] = []
    for jsonl in sorted(candidates_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["system"] = r.get("model") or jsonl.stem
                records.append(r)
    return records


def serialize_consensus(cr: ConsensusResult) -> dict:
    return {
        "item_id": cr.item_id,
        "input_sentence": cr.input_sentence,
        "candidate_sentence": cr.candidate_sentence,
        "rule": cr.rule,
        "majority_label": cr.majority_label.value,
        "unanimity": cr.unanimity,
        "needs_human_review": cr.needs_human_review,
        "confidence": round(cr.confidence, 4),
        "verdicts": [
            {
                "verifier_name": v.verifier_name,
                "label": v.label.value,
                "confidence": round(v.confidence, 4),
                "score": v.score,
                "details": v.details,
                "error": v.error,
            }
            for v in cr.verdicts
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--amrlib-model-dir",
        type=str,
        default=None,
        help="path to amrlib parse_xfm model directory",
    )
    ap.add_argument(
        "--mock-parser",
        action="store_true",
        help="use MockParser instead of amrlib (testing only)",
    )
    ap.add_argument(
        "--mock-parser-table",
        type=Path,
        default=None,
        help="optional JSON file mapping sentence -> penman for MockParser",
    )
    ap.add_argument(
        "--llm-judges",
        nargs="*",
        default=[],
        help="LLM model names to use as judges (e.g., gpt-4o claude-opus-4-7)",
    )
    ap.add_argument(
        "--skip-llm-judges",
        action="store_true",
        help="disable LLM-as-judge verifiers (uses only AMR + SMATCH)",
    )
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=None, help="limit n records (debug)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    # --- set up parser ---
    if args.mock_parser:
        table = {}
        if args.mock_parser_table and args.mock_parser_table.exists():
            table = json.loads(args.mock_parser_table.read_text())
        parser = MockParser(table)
        log.info("using MockParser (table size=%d)", len(table))
    else:
        if not args.amrlib_model_dir:
            print(
                "ERROR: --amrlib-model-dir required unless --mock-parser is set",
                file=sys.stderr,
            )
            sys.exit(2)
        parser = AmrlibParser(args.amrlib_model_dir)
        log.info("using AmrlibParser at %s", args.amrlib_model_dir)

    # --- set up verifiers ---
    verifiers = [
        AMRVerifier(parser=parser, threshold=args.threshold),
        SmatchVerifier(parser=parser),
    ]
    if not args.skip_llm_judges:
        for m in args.llm_judges:
            try:
                verifiers.append(LLMVerifier(m))
                log.info("added LLM judge: %s", m)
            except Exception as e:
                log.warning("could not add LLM judge %s: %s", m, e)

    # --- run verification ---
    candidates = load_candidates(args.candidates_dir)
    if args.limit:
        candidates = candidates[: args.limit]
    log.info("verifying %d candidate items with %d verifiers", len(candidates), len(verifiers))

    results: List[ConsensusResult] = []
    system_of: Dict[str, str] = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item_path = out_dir / "per_item.jsonl"

    with open(per_item_path, "w") as fout:
        for i, rec in enumerate(candidates):
            item_id = f"{rec['system']}::{rec['sentence_id']}::{rec['rule']}"
            system_of[item_id] = rec["system"]
            verdicts: List[VerifierVerdict] = []
            for v in verifiers:
                try:
                    verdicts.append(
                        v.verify(rec["input"], rec["output"], rec["rule"])
                    )
                except Exception as e:
                    verdicts.append(
                        VerifierVerdict(
                            verifier_name=getattr(v, "NAME", v.__class__.__name__),
                            label=Label.UNKNOWN,
                            confidence=0.0,
                            error=f"verifier exception: {e}",
                        )
                    )
            cr = aggregate(
                item_id=item_id,
                input_sentence=rec["input"],
                candidate_sentence=rec["output"],
                rule=rec["rule"],
                verdicts=verdicts,
            )
            results.append(cr)
            fout.write(json.dumps(serialize_consensus(cr), ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                log.info("  processed %d/%d", i + 1, len(candidates))

    # --- summaries ---
    by_verifier = summary_by_verifier(results)
    by_system = summary_by_system(results, system_of)
    review = review_queue(results)

    with open(out_dir / "summary_by_verifier.json", "w") as f:
        json.dump(by_verifier, f, indent=2)
    with open(out_dir / "summary_by_system.json", "w") as f:
        json.dump(by_system, f, indent=2)
    with open(out_dir / "review_queue.jsonl", "w") as f:
        for cr in review:
            f.write(json.dumps(serialize_consensus(cr), ensure_ascii=False) + "\n")

    # --- print headline ---
    print()
    print("=" * 60)
    print("PER-VERIFIER EQUIVALENCE RATES")
    print("=" * 60)
    for name, b in by_verifier.items():
        print(
            f"  {name:30s}  eq={b['equivalent']:>4d}  neq={b['not_equivalent']:>4d}  "
            f"abstain={b['abstain']:>4d}  rate={b['equivalence_rate']:.3f}"
        )
    print()
    print("=" * 60)
    print("PER-SYSTEM EQUIVALENCE RATES (consensus)")
    print("=" * 60)
    for name, b in by_system.items():
        print(
            f"  {name:30s}  eq={b['equivalent']:>4d}  neq={b['not_equivalent']:>4d}  "
            f"pending={b['pending_review']:>4d}  rate={b['equivalence_rate_decided']:.3f}"
        )
    print()
    print(f"Review queue: {len(review)} / {len(results)} items "
          f"({100*len(review)/max(1,len(results)):.1f}%)")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
