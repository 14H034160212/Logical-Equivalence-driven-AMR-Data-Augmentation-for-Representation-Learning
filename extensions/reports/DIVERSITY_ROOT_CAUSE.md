# Root cause of the v6 LogiQA reverse — surface diversity drop

After [V8_DOUBLENEG_REINTRO.md](V8_DOUBLENEG_REINTRO.md) ruled out the
missing-`double_negation` hypothesis, what actually explains why v6 (v4
fine-tuned T5) wins on ReClor (+0.8 pp) but loses on LogiQA (−1.8 pp) vs
v5 (stock T5)?

[`extensions/pilot_study/analyze_diversity.py`](../pilot_study/analyze_diversity.py)
measures n-gram diversity and near-duplicate rates on the
positive (`label=1`) `sentence2` of each contrastive corpus.

## Headline

| Metric (on label=1 sentence2) | v5 stock | v6 v4 T5 | Δ | v7 rulefix | v8 +dn |
|---|---|---|---|---|---|
| n_pos | 6,980 | 6,887 | −1.3% | 6,887 | 6,979 |
| avg length (tokens) | 9.12 | 9.47 | +3.8% | 9.47 | 9.40 |
| **distinct-1** (unique unigrams / total) | **0.0040** | **0.0029** | **−28%** | 0.0029 | 0.0029 |
| **distinct-2** (unique bigrams / total) | **0.0576** | **0.0437** | **−24%** | 0.0437 | 0.0436 |
| **distinct-3** (unique trigrams / total) | **0.2180** | **0.1803** | **−17%** | 0.1803 | 0.1797 |
| **near-dup rate (jaccard ≥ 0.7)** | **6.9%** | **10.85%** | **+57%** | 10.85% | 11.15% |
| near-dup rate (jaccard ≥ 0.8) | 1.75% | 2.65% | +51% | 2.65% | 2.45% |
| mean pair jaccard (anchor ↔ pos) | 0.462 | 0.489 | +6% | 0.489 | 0.488 |

(near-dup rate is over a 4,000-row subsample for tractability.)

## Reading the numbers

**Lexical diversity is sharply lower in v6**: 24-28% fewer unique
n-grams across unigrams, bigrams, and trigrams. The vocabulary is
shrinking and so is the bigram/trigram template space.

**Near-duplicate rate is ~57% higher in v6**: ~10.9% of positive
sentence2 rows have a Jaccard ≥ 0.7 with some earlier row, vs ~6.9% in
v5. The corpus is more repetitive.

**Positives are closer to their anchors in v6**: mean anchor↔positive
Jaccard rises from 0.462 to 0.489. The transformation produces text
that overlaps more with the input — less paraphrastic distance.

**v7 is byte-identical to v6**: the De Morgan rule fix changes the
rule-applied AMR's polarity count to match what T5 already produced,
but doesn't change the surface text. Same diversity profile, same
downstream numbers (V7_DOWNSTREAM.md).

**v8 ≈ v6**: adding the 182 legacy double_negation rows tweaks near-dup
slightly in opposite directions at different thresholds (the antonym-swap
templates are themselves repetitive, so 0.7-near-dup goes UP a touch),
but the broad shape is unchanged. Consistent with v8 not closing the
LogiQA reverse (V8_DOUBLENEG_REINTRO.md).

## Why this explains the ReClor / LogiQA split

| Property of the downstream task | What it rewards |
|---|---|
| **ReClor** — single-step entailment from a paragraph + question | tighter, more "canonical" positive pairs → less ambiguous training signal → **v6 wins** |
| **LogiQA** — multi-step deductive chains over short premises | richer surface forms → better generalization to unseen wordings in deductive chains → **v5 wins** |

The same property (cleaner, less varied) is helpful for one task and
harmful for the other. v4 T5's polarity-preservation gain costs surface
diversity that LogiQA leverages.

## Smoking-gun example (representative)

Looking at the contraposition slice from
[V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md) and
[T5_FT_RECOVERY.md](T5_FT_RECOVERY.md):

| Input | v5 (stock) | v6 (v4 T5) |
|---|---|---|
| "If the bald eagle is kind, then the mouse is not clever." | _Blistering eagles are not kind, unless the mouse is clever._ ← hallucinated "blistering" | "If the mouse is clever, it's not a kind bald eagle." |

v5's output adds the random adjective "blistering" — clear noise, but
the kind of noise that **forces the contrastive head to ignore surface
form and focus on logical structure**. v6 is clean but predictable.

## Implications

1. The v4 T5 fine-tune trades surface diversity for polarity
   preservation. On tasks where polarity is what matters (ReClor), this
   helps. On tasks where surface variation matters (LogiQA), it hurts.
2. The optimal generator may not be "as clean as possible" — sampling
   or temperature-decoding the v4 T5 might recover the diversity v5 has
   while keeping the polarity gain.
3. This finding is corpus-level; the contrastive head is paying
   attention to surface forms more than we assumed, which is consistent
   with the V6_CONTRASTIVE_PRETRAIN.md cross-eval result (v6-trained
   model is MORE robust OOD, suggesting v6 forces real
   structure-learning — yet **on LogiQA** the loss of surface diversity
   appears to dominate).

## Mitigation candidates (un-run)

- v9: v4 T5 with `--num_return_sequences 3 --do_sample True` and
  sample selection — recover diversity while keeping the polarity
  preservation gains.
- v10: mix v5 and v6 positives 50/50 — combine surface diversity (v5)
  with polarity preservation (v6).
- Larger backbone (DeBERTa-v2-xxlarge) may smooth over the diversity
  drop entirely.

JSON: [`diversity_v5_v6_v7_v8.json`](diversity_v5_v6_v7_v8.json).
