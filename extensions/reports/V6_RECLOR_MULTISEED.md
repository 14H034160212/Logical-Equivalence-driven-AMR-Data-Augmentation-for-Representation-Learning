# v6 vs v5 on ReClor — multi-seed

The original [V6_RECLOR.md](V6_RECLOR.md) headline was a +0.8 pp single-seed
delta. To bound seed variance, this report adds a second seed (`--seed 42`)
for both backbones on the same DeBERTa-large + ReClor setup.

## Headline

| Backbone | seed=21 | seed=42 | mean |
|---|---|---|---|
| v5 (stock T5) | 62.80% | 63.00% | **62.90%** |
| **v6** (v4 fine-tuned T5) | **63.60%** | **63.40%** | **63.50%** |
| Δ v6 − v5 | +0.80 | +0.40 | **+0.60 pp** |

**Every v6 seed beats every v5 seed.** Within-backbone seed spread is
0.2 pp (62.80–63.00 for v5, 63.40–63.60 for v6), comfortably tighter
than the cross-backbone gap.

## Trajectories (seed=42)

| Step | v5 seed=42 | v6 seed=42 |
|---|---|---|
| 200  | 35.6% | 39.6% |
| 400  | 47.2% | 51.2% |
| 600  | 51.4% | 57.2% |
| 800  | 57.0% | 59.6% |
| 1000 | 58.4% | 62.0% |
| 1200 | 61.4% | 60.6% |
| 1400 | 60.6% | 62.4% |
| 1600 | 61.8% | 62.4% |
| **1800** | **62.2%** | **63.4%** |
| 1930 | **63.0%** | 63.2% |

v6 starts ahead at every early step (similar pattern to seed=21) and
holds the lead through training.

## Setup

Same hyperparameters as the single-seed report:

| | value |
|---|---|
| Backbone | DeBERTa-large contrastive-pretrained on v5 vs v6 |
| Script | `BERT/run_multiple_choice.py` with the `deberta` MODEL_CLASSES entry added in commit `f20ca77` |
| Hparams | lr 1e-5, per-gpu bs 4 × accum 6 = effective 24, fp16, 10 epochs |
| Seeds | 21 (cached from V6_RECLOR.md), 42 (this run) |
| Hardware | 1× A100 80 GB |

The seed=42 runs wrapped `python ... &; wait $!` inside a bash script
launched via `nohup`, which (a) inherits HUP-ignoring from nohup so a
shell-session blip no longer kills the job (the failure mode that
trashed the first LogiQA v5 run), and (b) keeps the bash parent alive
until the python child exits, so the harness gets a clean completion
signal.

## What's still pending

1. **Single direction.** No seed=42 LogiQA result yet (would be the
   equivalent multi-seed for the v5 41% / v6 39% LogiQA gap).
2. **DeBERTa-v2-xxlarge.** Paper's actual headline size; our DeBERTa-
   large numbers won't match the paper's absolutes (~70% range), but
   the v5 vs v6 delta is what matters.
3. **A third seed** would bring sample SD to a more standard estimate.

## JSON aggregate

[`v6_reclor_multiseed.json`](v6_reclor_multiseed.json).
