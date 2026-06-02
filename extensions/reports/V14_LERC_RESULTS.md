# v14 — Logic-Equivalent Rule Composition (LeRC): downstream results

The novel algorithm proposed in the [Status section](index.md#proposed-novel-direction--logic-equivalent-rule-composition-lerc):
compose the 14 logical-equivalence rules in `extensions/logic_rules/` as
an algebra of equivalence-preserving operators to produce K
structurally-distinct but logically-equivalent paraphrases per anchor,
**without sampling and without a verifier filter** — correctness is
guaranteed by construction (composition of provably
equivalence-preserving rules).

This was meant to attack the diversity-vs-polarity trade-off identified
in [DIVERSITY_FINAL.md](DIVERSITY_FINAL.md) at the logic layer instead
of the surface layer.

## Setup

- **Composition templates** (5 paths per anchor):
  1. `[contraposition]`
  2. `[implication]`
  3. `[commutative]`
  4. `[contraposition, commutative]`
  5. `[implication, commutative]`
- **Generator**: v4 fine-tuned T5wtense (beam decoding, deterministic).
- **Dataset**: 22,280 rows (15,170 label=1 / 7,110 label=0). Each
  anchor expands into up to 5 positives plus its original negative.
- **No filter**: composition-of-equivalence guarantees logical
  correctness; no polarity check, no F1 threshold.

## Contrastive pretrain (DeBERTa-large, same hparams as v5/v6)

| Backbone | Final eval_acc |
|---|---|
| v5 | 99.31% |
| v6 | 98.43% |
| v11 (sampled + polarity filter) | 99.81% |
| v12 (sampled + AMR-struct F1) | 99.58% |
| **v14 (LeRC, NEW)** | **99.80%** |

v14's contrastive eval is the highest of all backbones (tied with v11).
The composition-template structure is highly learnable: the head fits
the corpus near-perfectly. **W&B run**:
<https://wandb.ai/qbao775/amr-lda-extensions/runs/11p97wfg>.

Trajectory (eval every epoch on v14 val):

```
ep1: 99.28%   ep2: 99.55%   ep3: 99.69%   ep4: 99.62%   ep5: 99.48%
ep6: 99.64%   ep7: 99.80%   ep8: 99.80%   ep9: 99.75%   ep10: 99.80%
```

## Downstream (seed=21, single seed for now)

| Backbone | ReClor dev_acc | LogiQA dev_acc | W&B |
|---|---|---|---|
| v5 (stock) | 62.8 | **41.0** | — |
| v6 (v4 T5, beam) | **63.6** | 40.3¹ | — |
| v12 (sampled + V1 filter) | 60.8 | 37.3 | — |
| **v14 (LeRC NEW)** | **61.2** | **37.3** | [reclor](https://wandb.ai/qbao775/amr-lda-extensions/runs/375ehyob) · [logiqa](https://wandb.ai/qbao775/amr-lda-extensions/runs/3fd7tmcr) |

¹ v5 / v6 LogiQA mean is over two seeds (42.3 / 40.3); see [V6_LOGIQA_MULTISEED.md](V6_LOGIQA_MULTISEED.md).

**v14 ties v12 on both downstream tasks.** It's the best of all
mitigation attempts (v8 / v10 / v9 / v11 / v12 / v14), but still loses
to **both** v5 (LogiQA winner) and v6 (ReClor winner).

## Why LeRC doesn't break the trade-off

LeRC produces **logic-layer diversity** — 5 structurally-distinct AMRs
of the same logical content. But each AMR is fed to the SAME (v4 T5,
beam) decoder, which is the polarity-cleaned but surface-narrowed
generator that originally caused the diversity drop.

The composition templates are themselves few (5) and repetitive
(every anchor gets the same 5 transformations). The contrastive head
fits this regular structure near-perfectly (99.80% in-distribution),
but the regularity hurts OOD transfer — the model learns the template
shapes rather than the logical content.

Even if LeRC delivered 50 distinct compositions per anchor, the v4 T5
would still render each into a small region of surface space. The
bottleneck is the **decoder**, not the rule space.

## What this confirms

The five mitigation attempts (v8, v9, v10, v11, v12, v14) consistently
fail to recover v5's LogiQA edge while keeping v6's ReClor edge,
across:

- corpus expansion (v8, v10)
- decoder sampling (v9)
- decoder sampling + polarity verifier (v11)
- decoder sampling + AMR-struct verifier (v12)
- **logic-layer rule composition (v14)**

The trade-off is structurally coupled to the v4 T5 decoder. Future
work that might break it lives *outside* dataset-level operations:

1. **RL co-training** of the v4 T5 generator against a richer verifier
   (extensions/rl/ scaffolding already in place).
2. **Generator replacement** with a frontier LLM (v13_llama in
   progress; v13_qwen3 / v13_gemma4 / v13_llama70 planned).
3. **Multi-decoder ensemble** — generate from stock-T5 *and* v4-T5,
   keep both surfaces per anchor, with a curriculum that interleaves.

## JSON aggregate

- [`v14_summary.json`](v14_summary.json)
- Builder source: [`extensions/pilot_study/build_v14_lerc.py`](https://github.com/14H034160212/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/blob/main/extensions/pilot_study/build_v14_lerc.py)
