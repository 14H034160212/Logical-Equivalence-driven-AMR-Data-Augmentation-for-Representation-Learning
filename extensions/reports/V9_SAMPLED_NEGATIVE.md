# v9 — sampled v4 T5 decoding makes things WORSE on both tasks

After [V10_MIX_NEGATIVE.md](V10_MIX_NEGATIVE.md) showed naive concat
fails, this report tests the other mitigation suggested in
[DIVERSITY_ROOT_CAUSE.md](DIVERSITY_ROOT_CAUSE.md): generate multiple
surface variants per anchor via sampled (not beam) decoding from the
same v4 T5 model.

## Setup

[`extensions/pilot_study/build_v9_sampled.py`](../pilot_study/build_v9_sampled.py)
bypasses amrlib's beam-search wrapper and calls
`T5ForConditionalGeneration.generate(...)` directly with:

  `do_sample=True, top_p=0.9, temperature=1.0, num_return_sequences=2`

Same parse + rule-application pipeline as v6/v7; same v4 fine-tuned
T5wtense; only the decoder is changed.

## v9 dataset

|  | value |
|---|---|
| Rows | 27,992 |
| Unique Generated_Sentence | 26,352 (**94%** unique, vs v6 ~85%) |
| Avg unique surfaces per anchor | 4.80 |
| Same rule coverage as v6 | Implication / Commutative / Contraposition |

Diversity is restored — the surface variation is materially higher
than v6.

## Results

Contrastive eval (in-distribution):

| | eval acc |
|---|---|
| v5 | 99.31% |
| v6 | 98.43% |
| v8 | 98.45% |
| v10 | 98.23% |
| **v9** | **97.95%** ← lowest of all |

Downstream (seed=21):

|  | ReClor | LogiQA |
|---|---|---|
| v5 (stock) | 62.80% | **41.01%** |
| v6 (v4 T5 beam) | **63.60%** | 39.17% |
| v7 (rulefix) | 63.60% | 39.17% |
| v8 (+ legacy dn) | 63.00% | 38.71% |
| v10 (v5+v6 mix) | 62.40% | 38.10% |
| **v9 (v4 T5 sampled)** | **59.60%** ⬇⬇ | **29.34%** ⬇⬇⬇ |

**v9 is the WORST on both tasks**, by a wide margin on LogiQA (random
baseline ≈ 25%).

## Why sampled v9 fails

Sampling at temperature=1.0 with top_p=0.9 produces diverse surface
forms, but the same temperature also lets the model occasionally drop
a polarity edge or pick a wrong subject — exactly the failure modes
the v4 fine-tune was designed to avoid (see
[T5_FT_RECOVERY.md](T5_FT_RECOVERY.md)). A spot-check on the
contraposition slice:

| Input | v6 (beam) | v9 sample 0 | v9 sample 1 |
|---|---|---|---|
| "If the bald eagle is kind, then the mouse is not clever." | "If the mouse is clever, it's not a kind bald eagle." | "The bald eagle would not have been kind with a clever mouse." | "If the mouse was clever, it wouldn't be a kind bald eagle." |
| "If the bald eagle is not kind, then Erin is sad." (label=0) | _(stock-style negation flip)_ | "The mouse was not clever unless it was a kind bald eagle." | _(same as sample 0 — sampling collision)_ |

The label=0 negative example for the second row is the SAME for both
samples — sampling collisions happen frequently when the model is
confident, deflating the apparent diversity.

More importantly, some samples are subtly wrong logically (~10% based
on a quick read of 30 contraposition pairs). For ReClor's
single-step entailment this introduces moderate noise; for LogiQA's
multi-step deductive chains the noise compounds and the classifier
basically learns chance.

## Both diversity mitigations failed

| Mitigation | ReClor delta vs v6 | LogiQA delta vs v6 |
|---|---|---|
| v8 (re-introduce legacy dn rows) | −0.6 pp | −0.5 pp |
| v10 (concat v5 + v6) | −1.2 pp | −1.1 pp |
| **v9 (sampled v4 T5)** | **−4.0 pp** | **−9.8 pp** |

The diversity root cause from
[DIVERSITY_ROOT_CAUSE.md](DIVERSITY_ROOT_CAUSE.md) is real, but the
two dataset-level interventions that should fix it both regressed.
This means the trade-off is structural: the v4 T5's polarity-cleaning
is precisely what makes its output less diverse, and you can't
recover the diversity with a noise-tolerant decoding strategy because
the underlying decoder's beam-search safety margin IS its
polarity-preservation.

## What might still work (un-run)

1. **Smaller temperature with rejection filtering**: T=0.7, top_p=0.9,
   `num_return_sequences=4`, then keep only samples that re-parse to
   the expected polarity count. Higher cost, higher quality. Worth a
   v11.
2. **Larger backbone** (DeBERTa-v2-xxlarge): may have capacity to
   absorb the v4-T5 tighter pairs without losing the diversity signal
   downstream. The paper's headline regime — not run here.
3. **Generator-discriminator co-training**: use the AMR verifier
   ([../auto_verifier/](../auto_verifier/)) to filter v9 samples by
   verified equivalence — only keep diverse samples that ALSO pass V1
   equivalence check.

## JSON

[`v9_summary.json`](v9_summary.json)
