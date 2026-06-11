"""Zero-shot AGIEval LSAT evaluation for ReClor-fine-tuned checkpoints.

Loads a DeBERTa(-v2) multiple-choice checkpoint and evaluates it on the
AGIEval v1_1 LSAT subsets (lsat-ar / lsat-lr / lsat-rc, 5-option MCQA)
without any LSAT-specific training — pure transfer from ReClor.

Input format mirrors utils_multiple_choice's ReClor processor:
text_a = passage, text_b = question + " " + option.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("agieval_lsat")

LABEL_TO_IDX = {c: i for i, c in enumerate("ABCDE")}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_option(opt: str) -> str:
    opt = opt.strip()
    if len(opt) >= 3 and opt[0] == "(" and opt[2] == ")":
        opt = opt[3:].strip()
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-type", choices=["deberta", "debertav2"], default="deberta")
    ap.add_argument("--data-dir", type=Path, default=Path("/tmp/AGIEval/data/v1_1"))
    ap.add_argument("--subsets", nargs="+", default=["lsat-ar", "lsat-lr", "lsat-rc"])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    if args.model_type == "debertav2":
        from transformers import DebertaV2Tokenizer, DebertaV2ForMultipleChoice
        tok = DebertaV2Tokenizer.from_pretrained(args.model_path)
        model = DebertaV2ForMultipleChoice.from_pretrained(args.model_path)
    else:
        from transformers import DebertaTokenizer
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "BERT"))
        from deberta_multiple_choice import DebertaForMultipleChoice
        tok = DebertaTokenizer.from_pretrained(args.model_path)
        model = DebertaForMultipleChoice.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    results = {}
    for subset in args.subsets:
        rows = load_jsonl(args.data_dir / f"{subset}.jsonl")
        n_correct = n_total = 0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts_a, texts_b, labels, n_opts = [], [], [], []
            for r in batch:
                passage = (r.get("passage") or "").strip()
                question = (r.get("question") or "").strip()
                options = [clean_option(o) for o in r["options"]]
                lbl = LABEL_TO_IDX.get(str(r.get("label", "")).strip(), None)
                if lbl is None or lbl >= len(options):
                    continue
                for opt in options:
                    texts_a.append(passage)
                    texts_b.append(question + " " + opt)
                labels.append(lbl)
                n_opts.append(len(options))
            if not labels:
                continue
            enc = tok(
                texts_a, texts_b, padding="max_length", truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            # All LSAT rows have 5 options; reshape (B, 5, L)
            k = n_opts[0]
            input_ids = enc["input_ids"].view(len(labels), k, -1).to(device)
            attention_mask = enc["attention_mask"].view(len(labels), k, -1).to(device)
            token_type_ids = enc.get("token_type_ids")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids.view(len(labels), k, -1).to(device)
            with torch.no_grad():
                logits = model(**kwargs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            for p, l in zip(preds, labels):
                n_correct += int(p == l)
                n_total += 1
        acc = n_correct / n_total if n_total else 0.0
        results[subset] = {"acc": round(acc * 100, 2), "n": n_total}
        log.info("%s: %.2f%% (%d examples)", subset, acc * 100, n_total)

    macro = sum(r["acc"] for r in results.values()) / len(results)
    results["macro_avg"] = round(macro, 2)
    log.info("macro avg: %.2f%%", macro)

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2))
        log.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
