"""Production-scale GRPO training on Qwen2.5-7B-Instruct using 2 A100 GPUs.

Trains the policy on PARARULE-Plus contrastive data with the AMR+UMR
verifier as reward. Uses bf16 + gradient checkpointing to fit Qwen-7B
on a single 80GB A100; data parallelism across 2 GPUs gives ~2× speedup.

Wallclock target: ~30-60 min for a meaningful demonstration run
(172 examples × 3 epochs × 8 generations per prompt).

Usage:
    cd /path/to/repo
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \\
        CUDA_VISIBLE_DEVICES=6,7 \\
        /data/qbao775/miniconda3/envs/qwen3-rl/bin/accelerate launch \\
            --num_processes 2 --mixed_precision bf16 \\
            extensions/rl/train_grpo_7b.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
log = logging.getLogger("grpo_7b")


REWRITE_PROMPT_TEMPLATE = """Apply the {rule} law of logical equivalence to rewrite this sentence. Output only the rewritten sentence, nothing else.

Sentence: {sentence}
Rewrite:"""


def load_contrastive_data(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [
        r for r in rows
        if r.get("positive") and r.get("negative")
        and r["positive"].strip().lower() != r["negative"].strip().lower()
    ]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--use-lora", action="store_true", default=True,
                    help="Use PEFT LoRA adapter (default: True, makes big models tractable)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--train-data", type=Path,
                    default=Path("extensions/pilot_study/pararule_contrastive.jsonl"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("extensions/rl/ckpts/grpo_7b"))
    ap.add_argument(
        "--amrlib-model-dir",
        default="amrlib/data/model_parse_xfm_bart_large-v0_1_0",
    )
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-7)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train-samples", type=int, default=172)
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    from extensions.auto_verifier.amr_verifier import AmrlibParser, AMRVerifier
    from extensions.auto_verifier.types import Label

    log.info("Loading contrastive data from %s", args.train_data)
    rows = load_contrastive_data(args.train_data)
    log.info("Loaded %d non-collision records", len(rows))
    rows = rows[: args.max_train_samples]
    log.info("Training on %d examples × %d epochs × %d generations/prompt",
             len(rows), args.epochs, args.num_generations)

    from datasets import Dataset
    dataset = Dataset.from_list([{
        "prompt": REWRITE_PROMPT_TEMPLATE.format(
            rule=r["rule"], sentence=r["anchor"]
        ),
        "rule": r["rule"],
        "input_sentence": r["anchor"],
        "gold": r["positive"],
    } for r in rows])

    log.info("Loading AMR parser for verifier (this is expensive once)...")
    parser = AmrlibParser(args.amrlib_model_dir)
    amr_verifier = AMRVerifier(parser=parser, threshold=0.55)

    def verifier_reward(prompts, completions, **kwargs):
        rules = kwargs.get("rule", [None] * len(prompts))
        inputs = kwargs.get("input_sentence", [None] * len(prompts))
        rewards = []
        for completion, rule, input_sent in zip(completions, rules, inputs):
            if isinstance(completion, list):
                comp_text = "".join(
                    str(m.get("content", "")) if isinstance(m, dict) else str(m)
                    for m in completion
                )
            else:
                comp_text = str(completion)
            comp_text = comp_text.strip().split("\n")[0][:200]
            try:
                v = amr_verifier.verify(input_sent, comp_text, rule)
                score = 1.0 if v.label == Label.EQUIVALENT else 0.0
            except Exception:
                score = 0.0
            rewards.append(score)
        return rewards

    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer

    log.info("Initializing GRPOTrainer with %s on %d GPUs (bf16, ckpt)...",
             args.model, int(os.environ.get("WORLD_SIZE", "1")))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        logging_steps=2,
        save_steps=10_000,
        report_to="none",
        bf16=True,
    )

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        log.info("Using LoRA: r=%d alpha=%d targets=q/k/v/o_proj",
                 args.lora_rank, args.lora_alpha)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=verifier_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    log.info("Starting GRPO training...")
    trainer.train()
    log.info("Training done. Saving to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
