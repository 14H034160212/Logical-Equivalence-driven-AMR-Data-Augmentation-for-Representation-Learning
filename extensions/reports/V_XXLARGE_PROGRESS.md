# DeBERTa-v2-xxlarge scale — progress + caveats (in flight)

Scaling the v5/v6 contrastive→downstream comparison from DeBERTa-large
(400M) to DeBERTa-v2-xxlarge (1.5B), the paper's headline regime.

## v6 xxlarge contrastive — DONE

Standard recipe (lr 3e-6, no warmup) got STUCK at chance (~50% binary
eval) for 7000 steps — the well-known DeBERTa-v2-xxlarge fine-tuning
instability. Fixed with **warmup 1000 + lr 1e-6 + 6 epochs**:

  eval trajectory: 76.7% (ep1) → 98.2% → 98.6% → 98.5% → 98.6% → 98.79% (final)

Saved: BERT/Transformers/deberta-v2-xxlarge-our-model-v6/ (6.3 GB).

## v6 xxlarge ReClor — DONE

Needed gradient checkpointing (added env-var switch GRADIENT_CHECKPOINTING=1
to run_multiple_choice.py) + bs2×accum48 to fit under cluster contention
(other jobs holding ~36 GB on the shared GPU).

  best dev_acc = 64.8% @ 10 epochs (still climbing at the end)

## The comparison caveat

| backbone | recipe | ReClor dev_acc |
|---|---|---|
| paper v5 xxlarge (existing ckpt) | lr 3e-6, no warmup, 10 ep, merged data | 78.8% |
| paper v5 xxlarge merged seed-21 | + MERIt-style merge | 80.2% |
| **my v6 xxlarge** | **lr 1e-6, warmup 1000, 6 ep** | **64.8%** |

**These are NOT comparable** — different contrastive recipes. The
paper's v5 reached 78.8% with lr 3e-6 + merged augmentation; my v6 used
the warmup/low-lr recipe forced by the chance-stuck issue, plus the
plain (non-merged) contrastive data.

A valid v5-vs-v6 xxlarge delta requires training v5 xxlarge with the
**same** recipe as v6 (lr 1e-6, warmup 1000, 6 ep, plain data). That
run is the next step.

## Status

- [x] v6 xxlarge contrastive (98.79%)
- [x] v6 xxlarge ReClor (64.8%, recipe-mismatched vs paper v5)
- [ ] v5 xxlarge contrastive (matched recipe) — IN PROGRESS
- [ ] v5 xxlarge ReClor (matched recipe)
- [ ] valid v5-vs-v6 xxlarge ReClor delta

The DeBERTa-large results (v6 +0.6 pp ReClor, -2.0 pp LogiQA, both
seed-robust) remain the primary, clean finding; xxlarge is a
robustness check still in progress.
