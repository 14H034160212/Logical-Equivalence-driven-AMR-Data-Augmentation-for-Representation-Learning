"""v14 — Logic-Equivalent Rule Composition (LeRC).

Novel direction (see index.md "Proposed novel direction"): treat the
14 rules in extensions/logic_rules/ as an algebra of
equivalence-preserving operators and *compose* them to produce K
structurally-distinct but logically-equivalent modified AMRs per
anchor. Each is fed to v4 T5 to yield K surface variants of the same
logical content.

Unlike v9/v11/v12, no sampling and no verifier filter are needed —
correctness is guaranteed by composition of equivalence-preserving
rules.

Composition templates (positive variants for Label=1 rows):

  T1: [contraposition]                — same as v6
  T2: [implication]                   — same as v6
  T3: [commutative]                   — same as v6
  T4: [contraposition, commutative]   — NEW: contrapose then commute
  T5: [implication, commutative]      — NEW: rewrite as ¬P∨Q then commute
  T6: [contraposition, double_negation] — NEW: contrapose then add ¬¬

For Label=0 (negatives), the original v6 negative is kept (one negative
per anchor) to avoid expanding the negative-sample distribution and
keeping the label balance similar to v6.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("v14_lerc")


COMPOSITION_TEMPLATES = [
    ["contraposition"],
    ["implication"],
    ["commutative"],
    ["contraposition", "commutative"],
    ["implication", "commutative"],
]
# Note: [contraposition, double_negation] was tested in smoke and
# dropped — double_negation.apply_positive toggles the existing
# :polarity rather than adding ¬¬, so on a graph that already has a
# negation it removes it, breaking equivalence preservation. Kept
# only compositions that are *provably* equivalence-preserving.


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def apply_composition(graph: penman.Graph, template: List[str]) -> Optional[penman.Graph]:
    """Apply rules in sequence to `graph`, returning the final graph if all
    rules in the template successfully fire, else None.

    Each rule's apply_positive must succeed; if any step yields no result,
    the entire composition is discarded (we don't want partial paths).
    """
    from extensions.logic_rules import get_rule

    g = copy.deepcopy(graph)
    for rule_name in template:
        try:
            rule = get_rule(rule_name)
        except KeyError:
            return None
        results = rule.apply(g)
        if not results:
            return None
        result = results[0]
        if getattr(result, "positive_graph", None) is None:
            return None
        try:
            g = penman.decode(result.positive_graph)
        except Exception:
            return None
    return g


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv"),
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v14.csv"),
    )
    ap.add_argument(
        "--parse-model",
        default="amrlib/data/model_parse_xfm_bart_large-v0_1_0",
    )
    ap.add_argument(
        "--gen-model", default="extensions/pilot_study/ft_t5wtense_v4",
    )
    ap.add_argument("--parse-batch-size", type=int, default=8)
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import amrlib
    from amrlib.models.parse_xfm.inference import Inference as ParserInf

    log.info("Loading input %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    df = df[df["Tag"].isin(
        ["Contraposition law", "Commutative law", "Implication law"])].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info("Rows: %d", len(df))

    # Parse unique anchors once.
    log.info("Loading parser…")
    parser = ParserInf(args.parse_model, batch_size=args.parse_batch_size, num_beams=1)
    unique_sentences = sorted(set(df["Original_Sentence"]))
    log.info("Parsing %d unique anchors…", len(unique_sentences))
    t0 = time.time()
    parse_cache = {}
    for chunk in batched(unique_sentences, args.parse_batch_size):
        try:
            results = parser.parse_sents(chunk)
        except Exception:
            results = ["" for _ in chunk]
        for s, g in zip(chunk, results):
            parse_cache[s] = g or ""
    log.info("Parse done in %.1f s", time.time() - t0)

    # For each Label=1 row, apply each composition template and collect
    # successful modified AMRs. For Label=0 rows, pass through using the
    # original "Contraposition law" negative (no composition).
    positive_paths = []  # (row_idx, template_tag, modified_amr_str)
    negative_passthrough = []  # (row_idx, original Generated_Sentence)
    tpl_success = {tuple(t): 0 for t in COMPOSITION_TEMPLATES}
    n_pos = n_neg = 0
    for idx, row in df.iterrows():
        graph_str = parse_cache.get(row["Original_Sentence"], "")
        label = int(row["Label"])
        if label == 0:
            negative_passthrough.append((idx, row["Generated_Sentence"]))
            n_neg += 1
            continue
        n_pos += 1
        if not graph_str:
            continue
        try:
            g = penman.decode(graph_str)
        except Exception:
            continue
        for template in COMPOSITION_TEMPLATES:
            modified = apply_composition(g, template)
            if modified is None:
                continue
            tpl_success[tuple(template)] += 1
            tag = "+".join(template)
            positive_paths.append((idx, tag, penman.encode(modified)))

    log.info("Positives: %d rows; %d successful composition paths total",
             n_pos, len(positive_paths))
    log.info("Negatives passthrough: %d", len(negative_passthrough))
    log.info("Composition pass counts:")
    for tpl, cnt in tpl_success.items():
        log.info("  %-50s %d", "+".join(tpl), cnt)

    # Batch-generate text from all positive paths via v4 T5.
    log.info("Loading generator %s", args.gen_model)
    gtos = amrlib.load_gtos_model(args.gen_model)

    log.info("Generating %d positive paths (batch=%d)…",
             len(positive_paths), args.gen_batch_size)
    generated = {}  # (idx, tag) -> text
    t0 = time.time()
    for chunk in batched(positive_paths, args.gen_batch_size):
        amrs = [a for _, _, a in chunk]
        try:
            texts, _ = gtos.generate(amrs)
        except Exception as e:
            log.warning("gen batch failed: %s; one-by-one", e)
            texts = []
            for amr in amrs:
                try:
                    t, _ = gtos.generate([amr])
                    texts.append(t[0] if t else "")
                except Exception:
                    texts.append("")
        for (idx, tag, _), text in zip(chunk, texts):
            generated[(idx, tag)] = (text or "").strip()
    log.info("Generation done in %.1f s", time.time() - t0)

    # Emit rows: for each Label=1 anchor, one row per successful template.
    # For Label=0, keep the original passthrough row.
    out_rows = []
    for (idx, tag, _) in positive_paths:
        row = df.loc[idx]
        text = generated.get((idx, tag), "")
        if not text or len(text) < 4:
            continue
        out_rows.append({
            "Origin": f"{row['Origin']}_v14_{tag}",
            "Original_Sentence": row["Original_Sentence"],
            "Generated_Sentence": text,
            "BLEU_Score": row.get("BLEU_Score", 0),
            "Label": 1,
            "Tag": row["Tag"],
            "logic_words": row.get("logic_words", ""),
        })
    for idx, text in negative_passthrough:
        row = df.loc[idx]
        out_rows.append({
            "Origin": f"{row['Origin']}_v14_neg",
            "Original_Sentence": row["Original_Sentence"],
            "Generated_Sentence": text,
            "BLEU_Score": row.get("BLEU_Score", 0),
            "Label": 0,
            "Tag": row["Tag"],
            "logic_words": row.get("logic_words", ""),
        })

    out_df = pd.DataFrame(out_rows)
    log.info("v14 rows: %d (label=1: %d, label=0: %d)",
             len(out_df), (out_df["Label"] == 1).sum(),
             (out_df["Label"] == 0).sum())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf8")
    log.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
