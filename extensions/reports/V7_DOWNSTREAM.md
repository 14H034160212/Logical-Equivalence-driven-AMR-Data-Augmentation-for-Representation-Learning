# v7 — rule-fix dataset, downstream is unchanged

After [the De Morgan-aware contraposition fix](RULEFIX_DEMORGAN.md) lifted
the pilot self-check pass rate from 78.9% to 82.2%, the natural question is
whether the cleaner AMR transformation also helps the downstream classifier.
This report runs the same v5/v6 pipeline a third time on a v7 dataset built
with the rule fix in place.

## v7 dataset

Re-runs [`extensions.pilot_study.build_v6_contrastive`](../pilot_study/build_v6_contrastive.py)
with the [`extensions.logic_rules.contraposition`](../logic_rules/contraposition.py)
fix from commit `3dc22ec` already in the repo. Same legacy 14,180-row
input, same v4 fine-tuned T5wtense for generation. Output:
`legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v7.csv` →
13,996 rows (same as v6; `double_negation` still excluded).

Split: 11,196 train / 2,800 val, label balance 51:49.

## Contrastive pretrain

Same hyperparameters as v5/v6 (DeBERTa-large, lr 2e-5, bs 32, 10 epochs,
seed 2021).

| | train -> val (in-distribution) |
|---|---|
| v5 | 99.31% |
| v6 | 98.43% |
| **v7** | 98.43% |

v7 hits the same in-distribution eval accuracy as v6.

## Cross-eval (V5 / V6 / V7 backbones × V5 / V6 / V7 val sets)

| Train ↓ / Eval → | v5 val | v6 val | v7 val |
|---|---|---|---|
| v5 | 99.31% | 83.86% | 83.86% |
| v6 | 94.49% | 98.43% | **98.89%** |
| **v7** | **94.49%** | **98.89%** | 98.43% |

v6 and v7 are functionally interchangeable: v6 evaluated on v7's val and v7
evaluated on v6's val are both 98.89%, virtually unchanged from each model's
own val. The rule fix changes the AMR's polarity count to match what T5
already generated; the contrastive pairs themselves are practically
identical.

## Downstream (single seed = 21)

|  | ReClor | LogiQA |
|---|---|---|
| v5 | 62.80% | 41.01% |
| v6 | 63.60% | **39.17%** |
| **v7** | **63.60%** | **39.17%** |

ReClor: v7 best dev_acc at step 1600 = **63.6%**, identical to v6 at the
same step.  Trajectory comparison (per-eval, every 200 steps):

| Step | v5 | v6 | v7 |
|---|---|---|---|
| 200  | 31.6% | 38.6% | 38.6% |
| 400  | 48.2% | 55.2% | 55.2% |
| 600  | 53.0% | 54.8% | 54.8% |
| 800  | 55.4% | 58.0% | 58.0% |
| 1000 | 58.4% | 58.8% | 58.8% |
| 1200 | 59.8% | 62.2% | 62.2% |
| 1400 | 60.8% | 61.8% | 61.8% |
| **1600** | **62.8%** | **63.6%** | **63.6%** |
| 1800 | 62.8% | 62.8% | 62.8% |
| 1930 | 62.8% | 63.4% | 63.4% |

v6 and v7 are byte-for-byte identical at every step on ReClor.

LogiQA: same story — best dev_acc at step 2400 = **39.17%**, exactly
matching v6's 39.17% at step 2400, with identical per-step values
throughout.

## Interpretation

The v4 T5wtense decoder already produced surface text in De Morgan
distributed form ("not A or not B") regardless of whether the rule-applied
AMR had outer or distributed negation. So:

- The rule fix makes the **AMR's polarity count** match the **T5 output's
  polarity count**, which un-blocks the polarity-parity self-check
  (pilot pass rate 78.9% → 82.2%).
- It does **not** change the actual `sentence2` text in the contrastive
  pairs — those still come from the same v4 T5 outputs.
- So contrastive pretraining sees the same supervised signal in v6 and
  v7, and the downstream task sees the same DeBERTa initialisation.

The rule fix is a **self-check** win, not a **downstream** win at this
scale.

## What might surface a v7 > v6 delta downstream

1. **Larger backbone** (deberta-v2-xxlarge in the paper's headline regime)
   — not run; ~3 hours of GPU per checkpoint.
2. **Reimplementing double_negation with the antonym swap** — that's
   actually different *training pairs*, not the same pairs with a
   different AMR. Currently the strongest hypothesis for the v6 LogiQA
   regression (−1.8 pp) and a candidate for a v8 dataset.
3. **Multi-seed averaging** — the v5/v6/v7 numbers are single-seed; ±2 pp
   variance is plausible.

## JSON

- [`v7_summary.json`](v7_summary.json) — aggregated cross-eval and downstream numbers
