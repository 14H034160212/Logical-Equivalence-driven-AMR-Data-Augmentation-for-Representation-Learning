# RL Verifier Integration

This directory plugs the AMR+UMR auto-verifier into a GRPO-style RL loop so a
policy LLM can be trained to produce logically-equivalent rewrites.

## What's here

- [verifier_reward.py](verifier_reward.py) — composable reward function
  (V1 AMR-struct + V2 LLM-judge + self-consistency + gold-match bonus)
- [train_grpo.py](train_grpo.py) — scaffolded GRPO training loop targeting
  `trl.GRPOTrainer` on Qwen-2.5-7B-Instruct (or similar)

## Why a verifier-backed reward is right for this task

| Property | LLM-as-judge alone | AMR+UMR verifier (this) |
|---|---|---|
| Deterministic | ✗ | **✓** |
| Process supervision (per step) | hard | **✓** (each AMR transformation) |
| Reward hacking risk | high | **low** (rules are fixed) |
| Cost per call | $ | free after parser cost |
| Coverage of rule space | full | **89.5%** (V1 EQ rate) |

## Configuration

`RewardConfig` exposes 4 weights:

```python
RewardConfig(
    weight_v1_amr      = 0.5,  # AMR-struct verifier (deterministic)
    weight_v2_llm      = 0.3,  # LLM-as-judge (handles surface fluency)
    weight_self_check  = 0.2,  # T5wtense polarity parity check
    gold_match_bonus   = 0.1,  # exact-match bonus to encourage convergence
)
```

These defaults are calibrated against the pilot study (run6); for new
domains, do a small grid sweep.

## How to run (when you have the GPU budget)

```bash
# 1. Build contrastive training data from any English sentence corpus
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
  python -m extensions.pilot_study.build_contrastive_dataset \
    --input-sentences /path/to/corpus.txt \
    --output extensions/rl/data/contrastive.jsonl

# 2. Dry-run: verify reward function works without starting training
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
  OPENAI_API_KEY=... \
  python -m extensions.rl.train_grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-data extensions/rl/data/contrastive.jsonl \
    --output-dir extensions/rl/ckpts/test \
    --dry-run

# 3. Real run (8 x A100, ~12-24h)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
  OPENAI_API_KEY=... \
  python -m extensions.rl.train_grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-data extensions/rl/data/contrastive.jsonl \
    --output-dir extensions/rl/ckpts/grpo-v1 \
    --batch-size 8 \
    --learning-rate 1e-6 \
    --epochs 3
```

## What gets logged

Per rollout: `(prompt, completion, v1_score, v2_score, self_check_score, total_reward)`.
This is enough to diagnose:
- Reward hacking (policy outputs that game one verifier but not others)
- Self-check trigger patterns (which rules suffer T5wtense generator drift)
- Convergence dynamics (when EQ rate stabilizes per rule)

## Open evaluation points (post-training)

After GRPO training, evaluate the resulting policy on:
1. **Held-out pilot sentences**: re-run `pilot_study/run_llm_baseline.py` with
   the trained model as a `--models policy_v1` entry; compare to gpt-4o etc.
2. **Robustness benchmarks**: ReClor-plus, LogiQA-plus, Reversal-Curse test
3. **OOD generalization**: PARARULE-Plus, R-GSM, FOLIO

The RL-trained policy should outperform LLM-as-rewriter on logical
equivalence rate while matching surface fluency (the V2 weight is what
encourages fluency).
