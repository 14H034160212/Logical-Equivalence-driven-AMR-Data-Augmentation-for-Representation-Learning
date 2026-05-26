"""v11 — sampled v4 T5 + AMR-verifier filter.

Same pipeline as build_v9_sampled but each sample is parsed back to
AMR and kept only if its :polarity - count matches the rule-applied
modified AMR's polarity count. This isolates the "diversity without
noise" hypothesis: if generator-verifier co-filtering works, v11 should
combine v4 polarity preservation with v5 surface diversity.

Implements polarity-parity self-check (same as
extensions/pilot_study/generate_amr_lda.py:_self_consistency_check).
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import penman
import torch

warnings.filterwarnings("ignore")
log = logging.getLogger("v11_verified")


TAG_TO_RULE = {
    "Contraposition law": "contraposition",
    "Commutative law": "commutative",
    "Implication law": "implication",
}


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def count_polarity_neg(g: penman.Graph) -> int:
    return sum(1 for s, role, t in g.triples if role == ":polarity" and t == "-")


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
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v11.csv"),
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
        "--num-samples", type=int, default=4,
        help="number of sampled outputs per row (more samples -> higher pass rate)",
    )
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
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

    log.info("Loading parser…")
    parser = ParserInf(args.parse_model,
                       batch_size=args.parse_batch_size, num_beams=1)
    unique_sentences = sorted(set(df["Original_Sentence"]))
    log.info("Parsing %d unique anchors…", len(unique_sentences))
    t0 = time.time()
    parse_cache: Dict[str, str] = {}
    for chunk in batched(unique_sentences, args.parse_batch_size):
        try:
            results = parser.parse_sents(chunk)
        except Exception:
            results = [parser.parse_sents([s])[0] if True else "" for s in chunk]
        for s, g in zip(chunk, results):
            parse_cache[s] = g or ""
    log.info("Parse done in %.1f s", time.time() - t0)

    # Apply rules and remember the expected polarity count.
    pending: List[Tuple[int, Optional[str], int]] = []
    fired = no_fire = parse_fail = 0
    for idx, row in df.iterrows():
        graph_str = parse_cache.get(row["Original_Sentence"], "")
        if not graph_str:
            parse_fail += 1
            pending.append((idx, None, 0))
            continue
        try:
            g = penman.decode(graph_str)
        except Exception:
            parse_fail += 1
            pending.append((idx, None, 0))
            continue
        rule_name = TAG_TO_RULE[row["Tag"]]
        try:
            rule_obj = get_rule(rule_name)
        except KeyError:
            pending.append((idx, None, 0))
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
            pending.append((idx, None, 0))
            continue
        try:
            expected_pol = count_polarity_neg(penman.decode(chosen))
        except Exception:
            expected_pol = 0
        fired += 1
        pending.append((idx, chosen, expected_pol))
    log.info("Rule applied: fired=%d no_fire=%d parse_fail=%d",
             fired, no_fire, parse_fail)

    # Direct T5 sampling with k=num_samples per anchor.
    log.info("Loading T5 %s", args.gen_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5 = T5ForConditionalGeneration.from_pretrained(args.gen_model).to(device)
    t5.eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    max_in_len = t5.config.task_specific_params["translation_amr_to_text"]["max_in_len"]
    max_out_len = t5.config.task_specific_params["translation_amr_to_text"]["max_out_len"]

    work = [(idx, amr, exp) for idx, amr, exp in pending if amr is not None]
    log.info("Sampled-gen on %d items × %d samples each (T=%.2f, p=%.2f)",
             len(work), args.num_samples, args.temperature, args.top_p)
    samples: Dict[int, List[str]] = {}
    expected_polarity: Dict[int, int] = {}
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for chunk in batched(work, args.gen_batch_size):
            idxs = [i for i, _, _ in chunk]
            for i, _, exp in chunk:
                expected_polarity[i] = exp
            graph_inputs = []
            for _, amr, _ in chunk:
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
            for i, idx in enumerate(idxs):
                samples[idx] = texts[i * args.num_samples : (i + 1) * args.num_samples]
            n_done += len(chunk)
            if n_done % 500 < args.gen_batch_size:
                log.info("  %d / %d done (%.1f s)", n_done, len(work), time.time() - t0)
    log.info("Sampled gen done in %.1f s", time.time() - t0)

    # Silence the parser's verbose per-graph INFO spam (90MB log otherwise).
    logging.getLogger("penman").setLevel(logging.WARNING)
    logging.getLogger("amrlib").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    # Re-parse each sample, check polarity-parity vs expected.
    log.info("Parsing candidates for verifier filter…")
    flat_candidates: List[Tuple[int, int, str]] = []  # (row_idx, sample_k, text)
    for idx, sample_list in samples.items():
        for k, t in enumerate(sample_list):
            if t and t.strip():
                flat_candidates.append((idx, k, t.strip()))
    log.info("  total candidates to verify: %d", len(flat_candidates))
    verify_results: Dict[Tuple[int, int], bool] = {}
    t0 = time.time()
    n_verified = 0
    # Intermediate save in case of mid-run crash.
    interim_path = args.output_csv.with_suffix(".interim.csv")
    for chunk in batched(flat_candidates, args.parse_batch_size):
        texts_chunk = [t for _, _, t in chunk]
        try:
            parsed = parser.parse_sents(texts_chunk)
        except Exception:
            parsed = [""] * len(texts_chunk)
        for (idx, k, t), p in zip(chunk, parsed):
            if not p:
                verify_results[(idx, k)] = False
                continue
            try:
                got = count_polarity_neg(penman.decode(p))
            except Exception:
                verify_results[(idx, k)] = False
                continue
            exp = expected_polarity.get(idx, 0)
            # parity match AND no complete loss / drift
            ok = (got % 2) == (exp % 2)
            if exp > 0 and got == 0:
                ok = False
            if abs(got - exp) >= 2:
                ok = False
            verify_results[(idx, k)] = ok
        n_verified += len(chunk)
        if n_verified % 2000 < args.parse_batch_size:
            elapsed = time.time() - t0
            rate = n_verified / max(1, elapsed)
            eta = (len(flat_candidates) - n_verified) / max(1, rate)
            log.info("  verified %d / %d  (%.0f cand/s, ETA %.0f s)",
                     n_verified, len(flat_candidates), rate, eta)
    log.info("Verifier filter done in %.1f s", time.time() - t0)

    # Emit kept candidates.
    out_rows = []
    n_pass = 0
    n_drop = 0
    for idx, row in df.iterrows():
        if idx not in samples:
            continue
        for k, t in enumerate(samples[idx]):
            t = (t or "").strip()
            if not t:
                continue
            if not verify_results.get((idx, k), False):
                n_drop += 1
                continue
            n_pass += 1
            out_rows.append({
                "Origin": f"{row['Origin']}_v11_s{k}",
                "Original_Sentence": row["Original_Sentence"],
                "Generated_Sentence": t,
                "BLEU_Score": row.get("BLEU_Score", 0),
                "Label": int(row["Label"]),
                "Tag": row["Tag"],
                "logic_words": row.get("logic_words", ""),
            })
    out_df = pd.DataFrame(out_rows)
    log.info("Verifier filter: %d pass / %d drop (%.1f%% kept)",
             n_pass, n_drop, 100 * n_pass / max(1, n_pass + n_drop))
    log.info("v11 rows: %d", len(out_df))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf8")
    log.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
