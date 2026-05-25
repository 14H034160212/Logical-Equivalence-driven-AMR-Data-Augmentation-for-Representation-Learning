# v10 — naive concat(v5, v6) loses on both tasks

After [DIVERSITY_ROOT_CAUSE.md](DIVERSITY_ROOT_CAUSE.md) identified
surface diversity as the v6 LogiQA reverse driver, the obvious
mitigation is "combine v5 diversity with v6 polarity preservation by
mixing the two corpora". This report runs that.

## v10 dataset

```
v10 = v5_list (14,180 rows) ++ v6_list (13,996 rows) = 28,176 rows
```

Both lists carry the same (Original_Sentence, Tag, Label) cross-product
but with different Generated_Sentence — v5 uses stock T5, v6 uses v4
fine-tuned T5. Concatenation doubles each (anchor, label) class with
two surface variants per anchor.

Split: 22,540 train / 5,636 val (80/20).

## Results

Contrastive eval (in-distribution):

| | eval acc |
|---|---|
| v5 | 99.31% |
| v6 | 98.43% |
| v8 | 98.45% |
| **v10** | **98.23%** |

v10 is slightly LOWER than v6, consistent with the model getting
contradictory surface forms for the same (anchor, label).

Downstream (seed=21, single seed):

|  | ReClor | LogiQA |
|---|---|---|
| v5 (stock) | 62.80% | **41.01%** |
| v6 (v4 T5) | **63.60%** | 39.17% |
| v8 (+ legacy dn) | 63.00% | 38.71% |
| **v10 (v5 + v6)** | **62.40%** ⬇ | **38.10%** ⬇ |

**v10 loses on both tasks.** ReClor: −0.4 vs v5, −1.2 vs v6. LogiQA:
−2.9 vs v5, −1.1 vs v6. Worst result on both.

## Why mix doesn't help

For the same `(anchor=X, label=1)` class the head now sees:

- A v5 positive: _"Blistering eagles are not kind, unless the mouse is clever."_
- A v6 positive: _"If the mouse is clever, it's not a kind bald eagle."_

These disagree on surface form (different vocabulary, different
syntactic frames), so the contrastive head's decision boundary is
averaged across two regions of input space. Less signal per direction,
not more.

In the cross-eval matrix from
[V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md), v5 and v6
classifiers disagree by 15 pp on each other's val sets (83.86% vs
99.31%). v10 is forced to satisfy both regions simultaneously — and
ends up worse at both than either alone.

## What actually might work

1. **Sampled / temperature-decoded v4 T5** (un-run). Generate multiple
   v6-style positives per anchor with sampling: each is structurally
   correct but surface-diverse. Adds diversity *within* the v6
   distribution rather than mixing two distributions.
2. **Larger backbone** (DeBERTa-v2-xxlarge, the paper's headline
   regime; un-run). May have capacity to handle both surface forms as
   consistent.
3. **Diversity-aware sampling at training time**: re-weight examples
   by surface novelty rather than putting both surfaces in the
   training set.

JSON: [`v10_summary.json`](v10_summary.json).
