"""v9 — sampled-decoding variant of build_v6_contrastive.

Rebuilds Synthetic_xfm_t5wtense_logical_equivalence_list_v9.csv using
v4 T5wtense but with `do_sample=True, temperature=1.0, top_p=0.9` and
`num_return_sequences=N`. Each (anchor, rule) source row produces N
output rows with diverse surface forms. Goal: recover the n-gram
diversity v6 lost (see DIVERSITY_ROOT_CAUSE.md) while keeping v4's
polarity preservation.

Bypasses amrlib's Inference wrapper (which exposes only beam search)
and calls the underlying T5ForConditionalGeneration directly.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import penman
import torch

warnings.filterwarnings("ignore")
log = logging.getLogger("v9_sampled")


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
        "--input-csv",
        type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv"),
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v9.csv"),
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
    ap.add_argument(
        "--num-samples", type=int, default=2,
        help="number of sampled outputs per (anchor, rule, label) row",
    )
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from amrlib.models.parse_xfm.inference import Inference as ParserInf
    from amrlib.models.generate_t5wtense.model_input_helper import ModelInputHelper
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from extensions.logic_rules import get_rule

    log.info("Loading input %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    df = df[df["Tag"].isin(TAG_TO_RULE.keys())].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info("Rows after rule filter: %d", len(df))

    # ---- parse ----
    log.info("Loading parser…")
    parser = ParserInf(args.parse_model,
                       batch_size=args.parse_batch_size, num_beams=1)
    unique_sentences = sorted(set(df["Original_Sentence"]))
    log.info("Parsing %d unique sentences…", len(unique_sentences))
    t0 = time.time()
    parse_cache: Dict[str, str] = {}
    for chunk in batched(unique_sentences, args.parse_batch_size):
        try:
            results = parser.parse_sents(chunk)
        except Exception as e:
            log.warning("parse batch failed: %s; one-by-one", e)
            results = []
            for s in chunk:
                try:
                    results.append(parser.parse_sents([s])[0])
                except Exception:
                    results.append("")
        for s, g in zip(chunk, results):
            parse_cache[s] = g or ""
    log.info("Parse done in %.1f s", time.time() - t0)

    # ---- apply rules ----
    pending: List[Tuple[int, Optional[str]]] = []
    fired = no_fire = parse_fail = 0
    for idx, row in df.iterrows():
        graph_str = parse_cache.get(row["Original_Sentence"], "")
        if not graph_str:
            parse_fail += 1
            pending.append((idx, None))
            continue
        try:
            g = penman.decode(graph_str)
        except Exception:
            parse_fail += 1
            pending.append((idx, None))
            continue
        rule_name = TAG_TO_RULE[row["Tag"]]
        try:
            rule_obj = get_rule(rule_name)
        except KeyError:
            pending.append((idx, None))
            continue
        results = rule_obj.apply(g)
        label = int(row["Label"])
        chosen = None
        for r in results:
            if label == 1 and getattr(r, "positive_graph", None):
                chosen = r.positive_graph
                break
            if label == 0 and getattr(r, "negative_graph", None):
                chosen = r.negative_graph
                break
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
    log.info("Rule applied: fired=%d no_fire=%d parse_fail=%d",
             fired, no_fire, parse_fail)

    # ---- direct T5 sampled generation ----
    log.info("Loading T5 model from %s", args.gen_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5 = T5ForConditionalGeneration.from_pretrained(args.gen_model).to(device)
    t5.eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    max_in_len = t5.config.task_specific_params["translation_amr_to_text"]["max_in_len"]
    max_out_len = t5.config.task_specific_params["translation_amr_to_text"]["max_out_len"]

    work = [(idx, amr) for idx, amr in pending if amr is not None]
    log.info("Sampled-gen on %d items × %d samples each (batch=%d, p=%.2f, T=%.2f)",
             len(work), args.num_samples, args.gen_batch_size, args.top_p, args.temperature)
    samples: Dict[int, List[str]] = {}
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for chunk in batched(work, args.gen_batch_size):
            idxs = [i for i, _ in chunk]
            graph_inputs = []
            for _, amr in chunk:
                try:
                    s = ModelInputHelper(amr).get_tagged_oneline()
                except Exception:
                    s = ModelInputHelper.gstring_to_oneline(amr)
                graph_inputs.append(s)
            enc = tokenizer(
                graph_inputs, padding=True, truncation=True,
                max_length=max_in_len, return_tensors="pt",
            ).to(device)
            outs = t5.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=max_out_len,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                num_return_sequences=args.num_samples,
                early_stopping=True,
            )
            texts = tokenizer.batch_decode(outs, skip_special_tokens=True)
            # texts is len(chunk) * num_samples, grouped as
            # [item0_samp0, item0_samp1, item1_samp0, item1_samp1, …]
            for i, idx in enumerate(idxs):
                samples[idx] = texts[i * args.num_samples : (i + 1) * args.num_samples]
            n_done += len(chunk)
            if n_done % 200 < args.gen_batch_size:
                log.info("  %d / %d done (%.1f s)", n_done, len(work), time.time() - t0)
    log.info("Sampled gen done in %.1f s", time.time() - t0)

    # ---- emit ----
    out_rows = []
    for idx, row in df.iterrows():
        if idx not in samples:
            continue
        for k, text in enumerate(samples[idx]):
            t = (text or "").strip()
            if not t:
                continue
            out_rows.append({
                "Origin": f"{row['Origin']}_v9_s{k}",
                "Original_Sentence": row["Original_Sentence"],
                "Generated_Sentence": t,
                "BLEU_Score": row.get("BLEU_Score", 0),
                "Label": int(row["Label"]),
                "Tag": row["Tag"],
                "logic_words": row.get("logic_words", ""),
            })
    out_df = pd.DataFrame(out_rows)
    log.info("Final v9 rows: %d (input %d, target ~%d)",
             len(out_df), len(df), len(df) * args.num_samples)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf8")
    log.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
