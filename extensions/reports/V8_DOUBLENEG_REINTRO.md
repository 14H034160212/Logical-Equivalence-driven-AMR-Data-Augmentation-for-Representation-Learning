# v8 — re-introducing double_negation doesn't close the LogiQA reverse

[V6_LOGIQA.md](V6_LOGIQA.md) flagged that v6's −1.8 pp LogiQA regression
vs v5 might be caused by v6 dropping 182 rows of legacy
`double_negation` training pairs (1.3% of corpus). v7's null result
([V7_DOWNSTREAM.md](V7_DOWNSTREAM.md)) ruled out the De Morgan rule
fix as the explanation. v8 tests the double_negation-drop hypothesis
directly by re-introducing those 182 rows.

## Dataset

`legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v8.csv` =
`v7.csv` + the 182 legacy `Double negation law` rows (with stock T5
outputs and the WordNet antonym swap the legacy paper used).

Splits:
- train: 11,342 rows (51:49 balance)
- val: 2,836 rows

## Contrastive pretrain

Same hparams as v5/v6/v7 (DeBERTa-large, lr 2e-5, bs 32, 10 epochs,
seed 2021).

| Backbone | train -> val |
|---|---|
| v5 | 99.31% |
| v6 | 98.43% |
| v7 | 98.43% |
| **v8** | **98.45%** |

v8 ~ v6: the added 182 rows don't change the contrastive task difficulty.

## Downstream (seed = 21)

|  | ReClor | LogiQA |
|---|---|---|
| v5 | 62.80% | **41.01%** |
| v6 | 63.60% | 39.17% |
| v7 | 63.60% | 39.17% |
| **v8** | 63.00% | **38.71%** |

**The reintroduction did not close the LogiQA reverse.** v8 LogiQA is
actually 0.5 pp **below** v6, putting it −2.3 pp below v5. On ReClor,
v8 sits between v5 and v6.

## What this rules out

The working hypothesis from V6_LOGIQA.md was:

> v6 excludes the 182 legacy double_negation rows because
> extensions/logic_rules/double_negation.py omits the antonym-swap step.
> Closing that parity gap is the natural next step.

v8 closes that parity gap by importing the legacy rows verbatim (so the
antonym-swap-derived sentence2 IS in the training data). LogiQA still
regresses. So the LogiQA gap is **not** a missing-rule problem.

## Remaining candidates for the LogiQA regression

1. **v4 T5 outputs are systematically less varied than stock T5**:
   the cleaner outputs may strip surface-form diversity that helps
   LogiQA's multi-hop deduction. Cross-eval (V6_CONTRASTIVE_PRETRAIN.md)
   already showed v6-trained DeBERTa is *more* robust to OOD pairs, so
   "less varied" is consistent.
2. **Effective corpus size**: v6/v7/v8 are 13,996 / 13,996 / 14,178
   rows; v5 is 14,180. Drop is < 0.02% — unlikely.
3. **Distribution of pair "hardness"**: v6/v7's positives are
   structurally tighter (T5 generates the De Morgan distributed form
   that exactly matches the parser's expected polarity count); easier
   pairs may train a less discriminative classifier than v5's noisier
   stock outputs.
4. **LogiQA-specific evaluation noise**: single seed; the LogiQA dev
   set is 651 items, so ±2 pp is one-seed variance. Multi-seed run is
   the cheapest way to confirm.

## JSON

[`v8_summary.json`](v8_summary.json) — aggregated numbers across v5/v6/v7/v8.
