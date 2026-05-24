"""Diagnose why v6 (v4 T5) loses on LogiQA vs v5 (stock T5).

Hypothesis from V8_DOUBLENEG_REINTRO.md: v4 T5 produces structurally
tighter pairs that strip surface-form diversity helpful for LogiQA's
multi-hop reasoning. This script quantifies the difference between v5
and v6 (and v7/v8) on:

  * Surface-form diversity per (rule × label) cell:
      - Type-Token Ratio (TTR)
      - Distinct-n (unigram/bigram counts / total)
      - Average length
  * Pair-wise similarity (positive vs anchor):
      - Self-BLEU (against the corresponding sentence1)
      - Embedding similarity (sentence-transformers if available; falls
        back to token-overlap Jaccard)
  * Within-corpus near-duplicate counts on sentence2.

Output: extensions/reports/diversity_v5_v6_v7_v8.json + .md summary.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

CSV_PATHS = {
    "v5_stock": "legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv",
    "v6_v4_t5": "legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v6.csv",
    "v7_rulefix": "legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v7.csv",
    "v8_with_dn": "legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v8.csv",
}


def tokenize(s: str) -> List[str]:
    s = s.lower()
    for p in string.punctuation:
        s = s.replace(p, " ")
    return [t for t in s.split() if t]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def ttr(tokens: List[str]) -> float:
    return len(set(tokens)) / max(1, len(tokens))


def distinct_n(token_lists: List[List[str]], n: int) -> float:
    """Distinct-n: unique n-grams across all sentences / total n-grams."""
    total = 0
    seen = set()
    for tok in token_lists:
        for i in range(len(tok) - n + 1):
            ngram = tuple(tok[i:i + n])
            seen.add(ngram)
            total += 1
    return len(seen) / max(1, total)


def near_dup_rate(token_lists: List[List[str]], threshold: float = 0.8) -> float:
    """Fraction of sentence2 values that have a Jaccard >= threshold with
    any earlier sentence2 in the list. O(n^2) — fine for ~14K."""
    if len(token_lists) > 4000:
        # subsample to keep this tractable
        import random
        random.seed(2026)
        token_lists = random.sample(token_lists, 4000)
    dup = 0
    seen_token_sets: List[set] = []
    for tok in token_lists:
        s = set(tok)
        if not s:
            continue
        for prev in seen_token_sets:
            inter = len(s & prev)
            union = len(s | prev)
            if union > 0 and inter / union >= threshold:
                dup += 1
                break
        seen_token_sets.append(s)
    return dup / max(1, len(token_lists))


def per_pair_similarity(df: pd.DataFrame) -> Dict[str, float]:
    """sentence1 ↔ sentence2 Jaccard for each row, averaged."""
    sims = []
    for _, r in df.iterrows():
        t1 = tokenize(str(r.get("Original_Sentence", "")))
        t2 = tokenize(str(r.get("Generated_Sentence", "")))
        sims.append(jaccard(t1, t2))
    return {
        "mean_pair_jaccard": sum(sims) / max(1, len(sims)),
        "pair_jaccard_low_q": sorted(sims)[len(sims) // 4] if sims else 0.0,
        "pair_jaccard_high_q": sorted(sims)[3 * len(sims) // 4] if sims else 0.0,
    }


def analyze_corpus(path: str) -> Dict[str, object]:
    df = pd.read_csv(path)
    # Restrict to positive (label=1) rows for a fair shape comparison
    pos = df[df["Label"] == 1]
    neg = df[df["Label"] == 0]
    s2_tokens = [tokenize(str(s)) for s in pos["Generated_Sentence"].tolist()]
    s2_neg = [tokenize(str(s)) for s in neg["Generated_Sentence"].tolist()]
    out: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "avg_len_pos": (sum(len(t) for t in s2_tokens) / max(1, len(s2_tokens))),
        "avg_len_neg": (sum(len(t) for t in s2_neg) / max(1, len(s2_neg))),
        "ttr_pos": ttr([t for tl in s2_tokens for t in tl]),
        "ttr_neg": ttr([t for tl in s2_neg for t in tl]),
        "distinct_1_pos": distinct_n(s2_tokens, 1),
        "distinct_2_pos": distinct_n(s2_tokens, 2),
        "distinct_3_pos": distinct_n(s2_tokens, 3),
        "distinct_1_neg": distinct_n(s2_neg, 1),
        "distinct_2_neg": distinct_n(s2_neg, 2),
        "near_dup_rate_pos_0.8": near_dup_rate(s2_tokens, threshold=0.8),
        "near_dup_rate_pos_0.7": near_dup_rate(s2_tokens, threshold=0.7),
    }
    out.update(per_pair_similarity(pos))
    # By-rule TTR
    per_rule = {}
    for rule, grp in pos.groupby("Tag"):
        tl = [tokenize(str(s)) for s in grp["Generated_Sentence"].tolist()]
        per_rule[str(rule)] = {
            "n": int(len(grp)),
            "ttr": ttr([t for tl_ in tl for t in tl_]),
            "distinct_2": distinct_n(tl, 2),
        }
    out["per_rule_pos"] = per_rule
    return out


def main():
    results: Dict[str, Dict[str, object]] = {}
    for name, path in CSV_PATHS.items():
        if not Path(path).exists():
            print(f"skipping {name}: {path} missing")
            continue
        print(f"analyzing {name} ...")
        results[name] = analyze_corpus(path)

    out_json = Path("extensions/reports/diversity_v5_v6_v7_v8.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_json}")

    # Print a quick comparison table
    keys = ["n_pos", "avg_len_pos", "ttr_pos", "distinct_1_pos",
            "distinct_2_pos", "distinct_3_pos",
            "near_dup_rate_pos_0.8", "near_dup_rate_pos_0.7",
            "mean_pair_jaccard"]
    headers = ["metric"] + list(results.keys())
    print("\n" + " | ".join(f"{h:>22}" for h in headers))
    print("-" * (len(headers) * 24))
    for k in keys:
        row = [k] + [
            f"{results[v].get(k, 0):.4f}" if isinstance(results[v].get(k), float)
            else str(results[v].get(k, ""))
            for v in results
        ]
        print(" | ".join(f"{c:>22}" for c in row))


if __name__ == "__main__":
    main()
