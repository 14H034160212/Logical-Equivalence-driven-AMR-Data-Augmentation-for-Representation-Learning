"""Regenerate the legacy Synthetic_xfm_t5wtense_logical_equivalence_list.csv
using the v4 T5wtense fine-tune as the AMR-to-text generator.

The legacy v5 dataset (legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv,
14180 rows) is produced by:
  - parse each Original_Sentence with BART-large AMR parser
  - apply one of {contraposition, commutative, implication, double_negation}
  - generate text via stock T5wtense
  - keep both positive (Label=1) and negative (Label=0) AMRs from the rule

We keep the same input sentences, the same parse, and the same rule logic
(via extensions/logic_rules), but swap the generator for the v4 fine-tune.
That isolates the contribution of the generator alone in any downstream
contrastive-pretraining accuracy delta.

Output: legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v6.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import penman

warnings.filterwarnings("ignore")
log = logging.getLogger("v6_gen")


# Map legacy Tag → extensions rule name.
# Double negation is excluded because the legacy paper paired it with a
# WordNet-antonym swap (beautiful → ugly) that extensions/logic_rules
# doesn't replicate; using the extensions version flips Label=1 pairs into
# contradictions and breaks the contrastive signal. Only 182/14180 rows
# (1.3%) of the legacy CSV are double_negation, so dropping them is cheap.
TAG_TO_RULE = {
    "Contraposition law": "contraposition",
    "Commutative law": "commutative",
    "Implication law": "implication",
}


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv", type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv"),
    )
    ap.add_argument(
        "--output-csv", type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v6.csv"),
    )
    ap.add_argument(
        "--parse-model", default="amrlib/data/model_parse_xfm_bart_large-v0_1_0",
    )
    ap.add_argument(
        "--gen-model", default="extensions/pilot_study/ft_t5wtense_v4",
    )
    ap.add_argument("--parse-batch-size", type=int, default=8)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap row count (debug only)")
    args = ap.parse_args()

    from extensions.logic_rules import get_rule
    import amrlib
    from amrlib.models.parse_xfm.inference import Inference

    log.info("Reading %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    log.info("rows=%d  unique sentences=%d  tags=%s",
             len(df), df.Original_Sentence.nunique(),
             dict(df.Tag.value_counts()))

    unique_sents = sorted(df.Original_Sentence.dropna().unique().tolist())
    log.info("Loading parser %s", args.parse_model)
    parser = Inference(args.parse_model, batch_size=args.parse_batch_size, num_beams=1)

    log.info("Parsing %d unique sentences (batch=%d)…",
             len(unique_sents), args.parse_batch_size)
    parse_cache: Dict[str, str] = {}
    t0 = time.time()
    for chunk in batched(unique_sents, args.parse_batch_size):
        try:
            graphs = parser.parse_sents(list(chunk))
        except Exception as e:
            log.warning("parse batch failed: %s; falling back to one-by-one", e)
            graphs = []
            for s in chunk:
                try:
                    graphs.extend(parser.parse_sents([s]))
                except Exception:
                    graphs.append(None)
        for sent, g in zip(chunk, graphs):
            parse_cache[sent] = g
        if len(parse_cache) % 200 == 0:
            elapsed = time.time() - t0
            log.info("  parsed %d / %d (%.1f s)", len(parse_cache), len(unique_sents), elapsed)
    log.info("Parse done in %.1f s", time.time() - t0)

    # For each row, look up parsed graph, apply rule, pick pos/neg AMR.
    log.info("Applying rules and collecting modified AMRs…")
    pending: List[Tuple[int, str]] = []  # (row_index, modified_amr)
    fired = 0
    no_fire = 0
    parse_fail = 0
    for idx, row in df.iterrows():
        sent = row["Original_Sentence"]
        tag = row["Tag"]
        label = int(row["Label"])
        graph_str = parse_cache.get(sent)
        if not graph_str:
            parse_fail += 1
            pending.append((idx, None))
            continue
        rule_name = TAG_TO_RULE.get(tag)
        if not rule_name:
            no_fire += 1
            pending.append((idx, None))
            continue
        try:
            rule_obj = get_rule(rule_name)
            g = penman.decode(graph_str)
            results = rule_obj.apply(g)
        except Exception as e:
            no_fire += 1
            pending.append((idx, None))
            continue
        if not results:
            no_fire += 1
            pending.append((idx, None))
            continue
        # Pick a graph based on label (1=positive, 0=negative).
        chosen = None
        for r in results:
            if label == 1 and getattr(r, "positive_graph", None):
                chosen = r.positive_graph
                break
            if label == 0 and getattr(r, "negative_graph", None):
                chosen = r.negative_graph
                break
        # Fallback to first available graph for either polarity.
        if chosen is None:
            for r in results:
                chosen = (getattr(r, "positive_graph", None)
                          or getattr(r, "negative_graph", None))
                if chosen:
                    break
        if not chosen:
            no_fire += 1
            pending.append((idx, None))
            continue
        fired += 1
        pending.append((idx, chosen))

    log.info("Rule application: %d fired, %d no_fire, %d parse_fail (of %d rows)",
             fired, no_fire, parse_fail, len(df))

    # Batch generate via v4 T5.
    log.info("Loading generator %s", args.gen_model)
    gtos = amrlib.load_gtos_model(args.gen_model)
    generated: Dict[int, str] = {}
    t0 = time.time()
    work = [(idx, amr) for idx, amr in pending if amr is not None]
    log.info("Generating %d AMR→text (batch=%d)…", len(work), args.gen_batch_size)
    for chunk in batched(work, args.gen_batch_size):
        amrs = [amr for _, amr in chunk]
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
        for (idx, _), text in zip(chunk, texts):
            generated[idx] = text
        if len(generated) % 500 == 0:
            elapsed = time.time() - t0
            log.info("  generated %d / %d (%.1f s)", len(generated), len(work), elapsed)
    log.info("Generation done in %.1f s", time.time() - t0)

    # Emit output: same schema as input, only Generated_Sentence swapped.
    out_df = df.copy()
    new_gen = []
    for idx in df.index:
        new_gen.append(generated.get(idx, ""))
    out_df["Generated_Sentence"] = new_gen
    # Drop rows where we couldn't regenerate (kept the original CSV row count
    # intact if we just leave blanks would break contrastive training).
    keep = out_df["Generated_Sentence"].astype(str).str.strip() != ""
    out_df = out_df[keep].reset_index(drop=True)
    log.info("Final rows after filtering: %d", len(out_df))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf8")
    log.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
