"""v13 — paraphrase the v6 positive sentence2 column with a frontier LLM
to recover the surface diversity lost by v4 T5's polarity-cleaning
fine-tune.

Hypothesis: an instruction-tuned LLM can produce K=2 paraphrases of
each polarity-correct v6 positive that vary lexically (recovering
diversity) while preserving the logical content (unlike the sampled v4
T5 in v9). This directly attacks the DIVERSITY_FINAL.md trade-off.

Generic across LLMs — accepts `--model` for the HF id. Tested on
Llama 3.1 8B, Qwen 3 8B, Gemma 4 E4B/31B, Llama 3.3 70B.

Output schema: same as v6 list (Origin, Original_Sentence,
Generated_Sentence, BLEU_Score, Label, Tag, logic_words), but
sentence2 is the LLM paraphrase. Each input row is expanded into
K rows (one per paraphrase). Negative samples (Label=0) are passed
through unchanged.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

warnings.filterwarnings("ignore")
log = logging.getLogger("v13_llm_paraphrase")


PARAPHRASE_PROMPT = """You are a careful paraphraser. Rewrite the sentence below using DIFFERENT WORDING but keeping EXACTLY THE SAME MEANING.

CRITICAL RULES:
- Keep every subject, object, and named entity (e.g. "the bald eagle", "Bob", "the mouse") unchanged.
- Keep every negation: if the original has "not", "isn't", "doesn't", your paraphrase must keep the SAME number of negations.
- Keep every "if ... then" as a conditional (you may use "when", "whenever", "in case", or "provided that").
- Keep every "and" / "or" structure.
- Do NOT add new entities, do NOT change the topic, do NOT make up details.
- Only vary the surface form (word order, vocabulary, voice).

Sentence: {sentence}

Paraphrase (output only the paraphrased sentence, nothing else):"""


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v6.csv"),
    )
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True,
                    help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--num-paraphrases", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label-filter", type=int, default=None,
                    help="if set (e.g. 1), only paraphrase rows with that label")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading input %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)
    log.info("Rows: %d", len(df))

    log.info("Loading model %s", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Filter to rows we'll paraphrase
    if args.label_filter is not None:
        to_paraphrase_mask = df["Label"] == args.label_filter
    else:
        to_paraphrase_mask = pd.Series([True] * len(df), index=df.index)
    log.info("Paraphrasing %d rows × %d paraphrases each",
             to_paraphrase_mask.sum(), args.num_paraphrases)

    # Build prompts (each row × num_paraphrases sampled variants)
    targets = df[to_paraphrase_mask]
    prompts = []
    indices = []  # (df_idx, k)
    for idx, row in targets.iterrows():
        text = str(row["Generated_Sentence"]).strip()
        if not text:
            continue
        for k in range(args.num_paraphrases):
            messages = [{"role": "user",
                         "content": PARAPHRASE_PROMPT.format(sentence=text)}]
            prompt_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt_str)
            indices.append((idx, k))

    log.info("Total generations: %d", len(prompts))

    # Batched generation
    paraphrases = {}
    t0 = time.time()
    for chunk_idx, chunk_start in enumerate(range(0, len(prompts), args.batch_size)):
        chunk_prompts = prompts[chunk_start : chunk_start + args.batch_size]
        chunk_indices = indices[chunk_start : chunk_start + args.batch_size]
        enc = tok(
            chunk_prompts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for (idx, k), text in zip(chunk_indices, decoded):
            t = (text or "").strip()
            # Strip leading role markers from chat templates
            for prefix in ("assistant\n\n", "assistant\n", "assistant:",
                           "model\n\n", "model\n",
                           "Paraphrase:", "Output:", "Here", "Sure", "Of course"):
                if t.startswith(prefix):
                    t = t[len(prefix):].lstrip(":\n ").strip()
            # Strip trailing chat artifacts
            t = t.split("\n\n")[0].strip()
            t = t.strip('"').strip("'")
            paraphrases[(idx, k)] = t
        if chunk_idx % 50 == 0 and chunk_idx > 0:
            done = chunk_start + len(chunk_prompts)
            log.info("  %d / %d done (%.1f s, %.2f / s)",
                     done, len(prompts), time.time() - t0,
                     done / (time.time() - t0))

    log.info("Generation done in %.1f s", time.time() - t0)

    # Emit new dataset: each paraphrase becomes a new row
    out_rows = []
    paraphrased_count = 0
    passthrough_count = 0
    for idx, row in df.iterrows():
        if to_paraphrase_mask[idx]:
            for k in range(args.num_paraphrases):
                p = paraphrases.get((idx, k), "").strip()
                if not p or len(p) < 4:
                    continue
                # Discard if the paraphrase failed (model echoed something useless)
                if p.lower().startswith(("here is", "sure,", "i ", "as ", "the paraphrase")):
                    # try to strip preamble
                    if ":" in p:
                        p = p.split(":", 1)[1].strip()
                if not p or len(p) < 4:
                    continue
                out_rows.append({
                    "Origin": f"{row['Origin']}_v13_p{k}",
                    "Original_Sentence": row["Original_Sentence"],
                    "Generated_Sentence": p,
                    "BLEU_Score": row.get("BLEU_Score", 0),
                    "Label": int(row["Label"]),
                    "Tag": row["Tag"],
                    "logic_words": row.get("logic_words", ""),
                })
                paraphrased_count += 1
        else:
            # Pass-through (Label=0 rows when --label-filter=1)
            out_rows.append({
                "Origin": f"{row['Origin']}_v13_orig",
                "Original_Sentence": row["Original_Sentence"],
                "Generated_Sentence": row["Generated_Sentence"],
                "BLEU_Score": row.get("BLEU_Score", 0),
                "Label": int(row["Label"]),
                "Tag": row["Tag"],
                "logic_words": row.get("logic_words", ""),
            })
            passthrough_count += 1

    out_df = pd.DataFrame(out_rows)
    log.info("v13 rows: %d (paraphrased=%d, passthrough=%d)",
             len(out_df), paraphrased_count, passthrough_count)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf8")
    log.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
