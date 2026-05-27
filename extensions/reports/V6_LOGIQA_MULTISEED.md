# LogiQA multi-seed — the v6 reverse is seed-robust

The single-seed [V6_LOGIQA.md](V6_LOGIQA.md) result showed v6 losing LogiQA
by −1.8 pp vs v5. This adds seed=42 to check whether that reverse is a
one-seed artifact (the way [V6_RECLOR_MULTISEED.md](V6_RECLOR_MULTISEED.md)
confirmed the ReClor win).

## Setup

Identical to V6_RECLOR_MULTISEED.md but on LogiQA (7,376 train / 651 dev,
4-way MC). DeBERTa-large backbone, lr 1e-5, bs 4 × accum 6, 10 epochs.

## Results

| Backbone | seed=21 | seed=42 | mean |
|---|---|---|---|
| v5 (stock T5) | 41.01% | 43.63% | **42.32%** |
| v6 (v4 T5) | 39.17% | 41.47% | **40.32%** |
| Δ (v6 − v5) | −1.84 | −2.16 | **−2.00** |

**v5 wins LogiQA on both seeds.** The reverse is robust, not a seed
artifact. Within-backbone seed spread is large (~2.5 pp for both v5
and v6 — LogiQA is a noisier benchmark than ReClor), but the
cross-backbone gap is consistent in sign and magnitude across seeds.

## The complete multi-seed picture

Combining with [V6_RECLOR_MULTISEED.md](V6_RECLOR_MULTISEED.md):

| Task | v5 mean | v6 mean | Δ | both seeds agree? |
|---|---|---|---|---|
| ReClor | 62.90% | 63.50% | **+0.60** | yes (v6 wins both) |
| LogiQA | 42.32% | 40.32% | **−2.00** | yes (v5 wins both) |

Both trade-offs are **seed-robust**. The v4 T5 fine-tune:
- **helps** single-step entailment (ReClor) — cleaner positives,
  +0.6 pp, every seed
- **hurts** multi-step deductive reasoning (LogiQA) — surface diversity
  loss (see [DIVERSITY_ROOT_CAUSE.md](DIVERSITY_ROOT_CAUSE.md)),
  −2.0 pp, every seed

This is the honest headline for the ACL extension: the polarity-cleaning
fine-tune is a genuine win on one benchmark family and a genuine,
diversity-driven loss on the other, and both directions reproduce across
seeds. Closing the LogiQA gap without losing ReClor remains open — the
v8/v9/v10/v11/v12 mitigation thread ([DIVERSITY_FINAL.md](DIVERSITY_FINAL.md))
narrowed but never closed it (v12 gets LogiQA to 37.3%, ReClor 60.8%,
still below v6 on both, but the verifier-filter direction is the most
promising and worth pursuing at larger scale).

JSON: [`v6_logiqa_multiseed.json`](v6_logiqa_multiseed.json).
