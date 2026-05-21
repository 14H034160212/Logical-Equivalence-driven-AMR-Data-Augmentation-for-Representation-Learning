"""Small-scale GRPO experiment to validate the verifier-backed reward pipeline.

Trains Qwen2.5-0.5B-Instruct (~1B params, fits on one A100) on a small
subset of contrastive data, using the AMR-LDA verifier as the reward.

This is a PROOF-OF-CONCEPT run — not a full paper-grade experiment. The
goal is to demonstrate that:
  1. The verifier-backed reward signal trains effectively
  2. The reward improves over the course of training
  3. The trained policy outperforms the base model on logical equivalence

Full-scale runs (Qwen2.5-7B, 8×A100, 12-24h) are documented in
extensions/rl/README.md.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=6 \\
        /data/qbao775/miniconda3/envs/qwen3-rl/bin/python \\
            -m extensions.rl.train_grpo_small
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")
log = logging.getLogger("grpo_small")


REWRITE_PROMPT_TEMPLATE = """Apply the {rule} law of logical equivalence to rewrite this sentence. Output only the rewritten sentence, nothing else.

Sentence: {sentence}
Rewrite:"""


def load_contrastive_data(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-data", type=Path,
                    default=Path("extensions/pilot_study/pararule_contrastive.jsonl"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("extensions/rl/ckpts/grpo_small"))
    ap.add_argument(
        "--amrlib-model-dir",
        default="amrlib/data/model_parse_xfm_bart_large-v0_1_0",
    )
    ap.add_argument("--num-generations", type=int, default=4,
                    help="rollouts per prompt for GRPO")
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train-samples", type=int, default=64)
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--max-prompt-length", type=int, default=128)
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="skip the V2 LLM-judge component of the reward")
    args = ap.parse_args()

    from extensions.auto_verifier.amr_verifier import AmrlibParser
    from extensions.auto_verifier.types import Label
    from extensions.auto_verifier.amr_verifier import AMRVerifier

    # ---- Set up data ----
    log.info("Loading contrastive data from %s", args.train_data)
    rows = load_contrastive_data(args.train_data)
    # Filter to records where positive != negative (real contrastive)
    rows = [r for r in rows if r.get("positive") and r.get("negative")
            and r["positive"].strip().lower() != r["negative"].strip().lower()]
    log.info("Loaded %d non-collision records", len(rows))
    rows = rows[: args.max_train_samples]
    log.info("Training on %d examples", len(rows))

    # Build HF dataset
    from datasets import Dataset

    def make_prompt(rec):
        return REWRITE_PROMPT_TEMPLATE.format(
            rule=rec["rule"], sentence=rec["anchor"]
        )

    dataset = Dataset.from_list([{
        "prompt": make_prompt(r),
        "rule": r["rule"],
        "input_sentence": r["anchor"],
        "gold": r["positive"],
    } for r in rows])

    # ---- Set up verifier reward ----
    log.info("Loading AMR parser for verifier...")
    parser = AmrlibParser(args.amrlib_model_dir)
    amr_verifier = AMRVerifier(parser=parser, threshold=0.55)

    def verifier_reward(prompts, completions, **kwargs):
        """GRPO-compatible reward function.

        Args:
            prompts: list of prompt strings
            completions: list of (possibly chat-format) completion strings
        Returns:
            list of float rewards (one per prompt)
        """
        rules = kwargs.get("rule", [None] * len(prompts))
        inputs = kwargs.get("input_sentence", [None] * len(prompts))
        rewards = []
        for prompt, completion, rule, input_sent in zip(prompts, completions, rules, inputs):
            # completion may be a string or a list of dicts (chat format)
            if isinstance(completion, list):
                comp_text = "".join(
                    str(m.get("content", "")) if isinstance(m, dict) else str(m)
                    for m in completion
                )
            else:
                comp_text = str(completion)
            comp_text = comp_text.strip()
            # Limit length
            comp_text = comp_text.split("\n")[0][:200]
            try:
                v1 = amr_verifier.verify(input_sent, comp_text, rule)
                score = 1.0 if v1.label == Label.EQUIVALENT else 0.0
            except Exception as e:
                log.warning("Verifier error: %s", e)
                score = 0.0
            rewards.append(score)
        return rewards

    # ---- Set up policy ----
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Initializing GRPOTrainer...")
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        logging_steps=2,
        save_steps=1000,
        report_to="none",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=verifier_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info("Starting GRPO training (small-scale POC)...")
    log.info("  model: %s", args.model)
    log.info("  examples: %d, epochs: %d, generations/prompt: %d",
             len(rows), args.epochs, args.num_generations)
    trainer.train()

    log.info("Training done. Saving final model to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
