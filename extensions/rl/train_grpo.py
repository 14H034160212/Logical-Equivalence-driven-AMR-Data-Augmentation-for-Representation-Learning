"""GRPO-style RL training loop for logical-equivalence rewrite, scaffolded
against `trl.GRPOTrainer`.

This file is a **scaffold** — it documents the integration of the
auto-verifier reward into a standard GRPO loop. It is NOT a one-button-run
production script; running it requires:

  - 8 × A100 (or similar) GPU resources for ~12-24 hours on Qwen-2.5-7B
  - `pip install trl>=0.7 datasets accelerate peft`
  - The AMR parser + AMR-to-text generator loaded (~2GB RAM each)
  - OPENAI_API_KEY for the V2 LLM-judge component of the reward

Usage (after preparing the env):
    PYTHONPATH=. python -m extensions.rl.train_grpo \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --train-data extensions/pilot_study/contrastive_dataset.jsonl \\
        --output-dir extensions/rl/ckpts/grpo-v1

Reward signal
-------------
Each rollout produces a (prompt, completion, rule) triple. The reward is
the AMR+UMR verifier consensus score:

    reward = 0.5 * V1.amr_structural_equivalence
           + 0.3 * V2.llm_judge_equivalence (if API key configured)
           + 0.2 * self_consistency_check_passed
           + 0.1 * gold_match (if gold rewrite known)

The default config is tuned against the pilot study; for new domains, sweep
the weights via small-scale runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("train_grpo")


REWRITE_PROMPT_TEMPLATE = """You are an expert in formal logic. Rewrite the following sentence using the {rule} law of logical equivalence. Output only the rewritten sentence.

Sentence: {sentence}
Rewrite:"""


def load_train_data(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset_from_jsonl(path: Path):
    """Build a HuggingFace dataset whose `prompt` is the rewrite request and
    `rule` / `gold` are tags used by the reward function.

    Returns a `datasets.Dataset` instance.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise RuntimeError("pip install datasets")
    rows = load_train_data(path)
    examples = []
    for r in rows:
        # Use the anchor + rule pair as the training prompt.
        examples.append({
            "prompt": REWRITE_PROMPT_TEMPLATE.format(
                rule=r["rule"], sentence=r["anchor"]
            ),
            "rule": r["rule"],
            "input_sentence": r["anchor"],
            "gold": r.get("positive"),  # the AMR-LDA positive as gold target
        })
    return Dataset.from_list(examples)


def make_reward_fn(amrlib_model_dir: str, llm_judge: str = "gpt-4o-mini"):
    """Build the verifier-backed reward function."""
    from extensions.auto_verifier.amr_verifier import AmrlibParser
    from extensions.rl.verifier_reward import VerifierReward

    parser = AmrlibParser(amrlib_model_dir)
    reward_fn = VerifierReward(parser, llm_judge_model=llm_judge)

    def batched_reward(samples, **kwargs):
        """TRL-compatible signature: receives a list of (prompt, completion)
        and returns a list of floats."""
        rewards = []
        for sample, completion in zip(samples, kwargs.get("completions", [])):
            rule = sample.get("rule", "contraposition")
            input_sent = sample.get("input_sentence", "")
            gold = sample.get("gold")
            r = reward_fn(input_sent, completion, rule, gold=gold)
            rewards.append(r.total)
        return rewards

    return batched_reward


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--amrlib-model-dir",
        default="amrlib/data/model_parse_xfm_bart_large-v0_1_0",
    )
    ap.add_argument("--llm-judge", default="gpt-4o-mini")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="set up everything but don't actually start training",
    )
    args = ap.parse_args()

    # --- Set up data ---
    log.info("Loading training data from %s", args.train_data)
    dataset = build_dataset_from_jsonl(args.train_data)
    log.info("Loaded %d examples", len(dataset))

    # --- Set up reward function ---
    log.info("Initializing verifier reward function...")
    reward_fn = make_reward_fn(args.amrlib_model_dir, llm_judge=args.llm_judge)

    if args.dry_run:
        log.info("--dry-run: scaffolding complete. Skipping actual training.")
        # Sanity-check: invoke reward on the first example
        example = dataset[0]
        log.info("Sample reward (using gold as completion):")
        r = reward_fn(
            [example], completions=[example.get("gold", "")]
        )
        log.info("  reward=%.3f", r[0])
        return

    # --- Set up policy model ---
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "pip install 'trl>=0.7.0' accelerate peft transformers"
        )

    log.info("Loading policy model %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        args=config,
    )
    log.info("Starting GRPO training...")
    trainer.train()
    log.info("Training complete. Checkpoints in %s", args.output_dir)


if __name__ == "__main__":
    main()
