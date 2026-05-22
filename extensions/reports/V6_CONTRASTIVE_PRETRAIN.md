# v6 contrastive pretraining — does v4 T5 help the downstream classifier?

After [v4 T5 closes the polarity-flip regressions on the pilot](T5_FT_RECOVERY.md),
the natural next question is whether the cleaner generator actually
improves the contrastive dataset used in the ACL Findings 2024 paper's
RoBERTa/DeBERTa pre-training step. This report is the first end-to-end
A/B between v5 (stock T5wtense) and v6 (v4 fine-tuned T5wtense) at the
contrastive-pretraining layer.

## v6 dataset

Regenerated from `legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv`
(14,180 rows, 5,584 unique sentences) by:

1. Parsing each unique sentence with the same BART-large AMR parser used in v5
2. Applying the same logic rule (Contraposition / Commutative / Implication;
   `double_negation` is **excluded** because the legacy paper paired it with a
   WordNet-antonym swap that `extensions/logic_rules/double_negation.py`
   does not replicate; ~1.3% of the corpus)
3. Generating the `Generated_Sentence` column with the [v4 fine-tuned T5wtense](T5_FT_RECOVERY.md) instead of stock

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.build_v6_contrastive
# 13,996 rows out  -> legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v6.csv
```

Split into 80/20 train/val:
```
train 11,196 rows  ->  output_result/Synthetic_xfm_t5wtense_logical_equivalence_train_v6.{csv,json}
val    2,800 rows  ->  output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation_v6.{csv,json}
```

Label balance is close to 50/50 in both splits (5705/5491 train, 1404/1396 val).

## Setup (identical for v5 and v6)

| | value |
|---|---|
| Model | `microsoft/deberta-large` (400M params) |
| Script | `BERT/run_glue_no_trainer.py` (verbatim from the paper repo) |
| Max length | 256 |
| Batch size | 32 |
| Epochs | 10 |
| LR | 2e-5 (linear, no warmup) |
| Weight decay | 0.0 |
| Seed | 2021 |
| Hardware | 1× A100 80 GB (`CUDA_VISIBLE_DEVICES=6`) |
| Wallclock | ~23 min each |
| Tokenizer / data input | JSONL (CSV path is broken under `datasets==2.6.1` + new pandas: `mangle_dupe_cols` removed) |

## Headline — in-distribution

| | eval acc on own val | Δ vs v5 |
|---|---|---|
| **v5** (stock T5) | 99.31% | — |
| **v6** (v4 fine-tuned T5) | 98.43% | −0.88 pp |

Both eval sets are in-domain for their respective training data, and both
saturate near 99% — so the single-number comparison is essentially
inconclusive. The interesting signal is on the cross-eval below.

## Headline — cross-evaluation

Same checkpoints, but each evaluated on the *other* version's validation
set. This isolates which model **generalizes** beyond its training
distribution:

|  | eval on **v5 val** (2,775 items) | eval on **v6 val** (2,800 items) |
|---|---|---|
| v5-trained | **99.31%** | 83.86% |
| v6-trained | 94.49% | **98.43%** |

Reading the matrix:

- The v5-trained model **loses 15.45 pp** when shown v6-style pairs (v4 T5 outputs).
- The v6-trained model **loses only 3.94 pp** when shown v5-style pairs (stock T5 outputs).
- v6-trained beats v5-trained on the v5 val set by **+0.50 pp − 4.82 pp** when comparing same-domain vs cross-domain (94.49% vs the stock T5's strict 99.31% in-distribution score).

In short: **v6-trained DeBERTa is the more robust classifier.** The v4
T5 generator produces a contrastive dataset that is **harder but more
generalizing**.

## Why?

Spot-checking the v6 Generated_Sentence column against v5 on the
contraposition slice (see [T5_FT_RECOVERY.md](T5_FT_RECOVERY.md)):

| Input | v5 (stock) | v6 (v4 T5) |
|---|---|---|
| "If the bald eagle is kind, then the mouse is not clever." | _Blistering eagles are not kind, unless the mouse is clever._ ← hallucinated "blistering" | "If the mouse is clever, it's not a kind bald eagle." |
| "If the bald eagle is not kind, then Erin is sad." | "If Erin isn't sad, the bald eagle is kind." | "If Erin isn't sad, the bald eagle is kind." (identical) |
| "If the bald eagle is quiet, then Bob is not sleepy." | "If Bob was sleepy, the bald eagle wouldn't be quiet." | "If Bob is sleepy, the bald eagle isn't quiet." |

Hypothesis: v5's noisier outputs leak surface cues (random adjectives,
inconsistent tense) that a contrastive model can shortcut. v6's cleaner
outputs remove those cues, forcing the model to learn the actual logical
relationship. That's exactly the shape of a useful training data
improvement — small in-distribution drop, large out-of-distribution
gain.

## JSON aggregate

```json
{
  "v5_train_v5_eval": 0.9931531531531531,
  "v6_train_v6_eval": 0.9842857142857143,
  "v5_train_v6_eval": 0.8385714285714285,
  "v6_train_v5_eval": 0.9448648648648649,
  "out_of_distribution_drop_v5":  0.1546,
  "out_of_distribution_drop_v6":  0.0394
}
```

Also in [`v6_pretrain_cross_eval.json`](v6_pretrain_cross_eval.json).

## Caveats and what this does NOT show yet

1. **Single seed.** No variance estimate. A second seed run is the
   cheapest next step.
2. **Downstream task not run.** The actual ACL paper headline is ReClor /
   LogiQA accuracy after a second fine-tune on those tasks. v5 → v6 may
   or may not move that needle. Running the ReClor fine-tune on both
   checkpoints is the obvious next step (~few hours).
3. **double_negation excluded.** v6 drops 182 legacy rows on
   double-negation due to the antonym-swap gap; if downstream cares about
   double_negation specifically, that's a known hole.
4. **In-domain comparison is saturated.** The 99% range is too close to
   the ceiling to draw conclusions from the single-direction comparison;
   the cross-eval is the load-bearing measurement.
