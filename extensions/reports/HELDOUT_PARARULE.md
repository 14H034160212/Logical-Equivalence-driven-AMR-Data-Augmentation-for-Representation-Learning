# Held-out PARARULE-Plus generalization test

The v4 T5 improvements documented in [T5_FT_RECOVERY.md](T5_FT_RECOVERY.md)
were measured on the 49-sentence pilot (test_sentences.json), which
overlapped the fine-tune training distribution. This run measures the
same self-check pass-rate on a held-out shard.

## Setup

| | value |
|---|---|
| Shard | 60 sentences extracted from `PARARULE-Plus-main/Depth5/PARARULE_Plus_Depth5_shuffled_test.jsonl` |
| Hold-out criterion | string-match exclusion against `pararule_contrastive.jsonl`, `pararule_sentences.jsonl`, `contrastive_dataset_smoketest.jsonl` |
| Rules tested | contraposition, double_negation, implication, de_morgan, commutative |
| Items × rules | 60 × 5 = 300 (rule-did-not-fire on 157, generator-tested on 143) |
| Same code path | `extensions.pilot_study.generate_amr_lda` with self-check |

The 60 input sentences are simple Animal/D5 PARARULE entries like _"The
bald eagle is boring."_, _"The dinosaur attacks the rabbit."_, _"The
rabbit is furry."_ — domain-shifted vs the 49-sentence pilot
(human/object/event mix).

## Headline

| | Stock T5 | **v4 T5** | Δ |
|---|---|---|---|
| `ok` on generator-tested items | 101 / 143 | **103 / 143** | +2 |
| Self-check pass rate | 70.6% | **72.0%** | **+1.4 pp** |
| Recovered (stock fail → v4 ok) | — | **16** | |
| Regressed (stock ok → v4 fail) | — | 14 | |
| Net | — | **+2** | |

The held-out delta is **smaller and more brittle** than in-distribution
(+5.5 pp on the full 49-sentence pilot). 16 recoveries vs 14 regressions
means the per-item churn is high — the fine-tune does change the model's
output distribution on unseen data, but it's not uniformly an improvement.

## By-rule (held-out only)

| Rule | Stock | v4 | Δ | In-distribution Δ |
|---|---|---|---|---|
| commutative | 9 / 9 | 7 / 9 | **−2** | unchanged |
| contraposition | 22 / 37 | 23 / 37 | +1 | +4 |
| **double_negation** | 49 / 60 | **55 / 60** | **+6** | +0 (anchor golds) |
| implication | 21 / 37 | 18 / 37 | **−3** | +1 |
| de_morgan | (no fires on either) | (no fires) | — | +1 |

Reading:

- **double_negation generalises well.** The anchor golds added in v4 to
  close in-distribution regressions also lift held-out pass rate by
  6 items — likely because they teach the decoder to preserve the
  `:polarity -` pattern in general, not just on the S005/S042 specific
  cases.
- **implication regresses on held-out (−3).** The hand-derived golds
  for S004/S026 may have overfit the decoder to specific impl phrasings
  ("X does not Y, or Z") that don't transfer to D5-style simple
  propositions.
- **commutative is a small regression (−2).** The held-out shard has
  only 9 commutative-firing items, so this is within noise.

## Caveats

1. **Generator-tested rate is balanced.** 143/300 in both — the rule
   firing pattern is identical for stock and v4 (the AMR parser and
   logic rules are deterministic and the v4 fine-tune doesn't change
   them; only the surface text generation changes).
2. **Single shard.** PARARULE-Plus Depth5 simple propositions are easier
   than the pilot's mix; numbers may not transfer to longer/harder
   PARARULE shards or to ReClor's natural-language reasoning.
3. **No LLM-judge.** Self-check pass rate measures `:polarity` parity
   preservation, not semantic equivalence. The recovered/regressed
   counts are upper bounds on actual quality change.

JSON: [`heldout_pararule_summary.json`](heldout_pararule_summary.json).

## Reproducing

```bash
# 1. Build held-out shard (60 sentences from Depth5 test)
/data/qbao775/miniconda3/envs/leamr/bin/python -c "
# see extensions/pilot_study/heldout_pararule_depth5.json builder snippet
# in this report's git commit; just rerun once if you need to refresh
pass
"

# 2. Stock T5 baseline
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
    -m extensions.pilot_study.generate_amr_lda \
        --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
        --gen-model amrlib/data/model_generate_t5wtense-v0_1_0 \
        --test-sentences extensions/pilot_study/heldout_pararule_depth5.json \
        --out extensions/pilot_study/results/ft_t5_recovery/heldout_stock.jsonl

# 3. v4 T5
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
    -m extensions.pilot_study.generate_amr_lda \
        --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
        --gen-model extensions/pilot_study/ft_t5wtense_v4 \
        --test-sentences extensions/pilot_study/heldout_pararule_depth5.json \
        --out extensions/pilot_study/results/ft_t5_recovery/heldout_v4.jsonl
```
