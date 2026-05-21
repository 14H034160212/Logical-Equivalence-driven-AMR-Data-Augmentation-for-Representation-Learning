"""Extract individual sentences from PARARULE-Plus for contrastive dataset generation.

PARARULE-Plus records have format:
  {"id": "...", "context": "Sent1. Sent2. ... Rule1. Rule2. ...", "questions": [...]}

The "context" is a multi-sentence paragraph mixing facts ("Harry is strong.")
and rules ("Strong people are smart."). We want to extract those individual
sentences for use as inputs to the contrastive dataset builder.

Output: a JSONL where each record is `{"id": ..., "text": <single sentence>}`,
ready to be consumed by build_contrastive_dataset.py.

Usage:
    python -m extensions.pilot_study.prep_pararule_sentences \\
        --input PARARULE-Plus-main/Depth2/PARARULE_Plus_Depth2_shuffled_train.jsonl \\
        --output extensions/pilot_study/pararule_sentences.jsonl \\
        --limit 1000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of sentences extracted")
    ap.add_argument("--dedup", action="store_true",
                    help="drop duplicate sentences")
    args = ap.parse_args()

    seen = set()
    n_written = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            context = rec.get("context", "")
            sentences = SENT_SPLIT_RE.split(context)
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if not sent or len(sent) < 5:
                    continue
                if args.dedup and sent in seen:
                    continue
                seen.add(sent)
                fout.write(json.dumps({
                    "id": f"{rec['id']}_{i}",
                    "text": sent,
                    "source": rec.get("id"),
                }, ensure_ascii=False) + "\n")
                n_written += 1
                if args.limit and n_written >= args.limit:
                    break
            if args.limit and n_written >= args.limit:
                break
    print(f"Wrote {n_written} sentences to {args.output}")


if __name__ == "__main__":
    main()
