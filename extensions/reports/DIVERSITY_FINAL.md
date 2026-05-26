# Diversity vs polarity — final summary across v5–v11

Unified summary of the v6→v11 exploration thread investigating whether
the LogiQA regression introduced by v4 T5 fine-tuning can be closed by
recovering surface diversity.

## Background

The story so far:
- [T5_FT_RECOVERY.md](T5_FT_RECOVERY.md): v4 fine-tuned T5wtense lifts
  pilot polarity-preservation pass rate 68.9% → 78.9%.
- [RULEFIX_DEMORGAN.md](RULEFIX_DEMORGAN.md): De Morgan-aware
  contraposition rule fix lifts pilot 78.9% → 82.2%.
- [V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md) +
  [V6_RECLOR.md](V6_RECLOR.md): v6 contrastive backbone wins ReClor
  +0.8 pp vs v5.
- [V6_LOGIQA.md](V6_LOGIQA.md): v6 LOSES LogiQA by −1.8 pp.
- [V6_RECLOR_MULTISEED.md](V6_RECLOR_MULTISEED.md): v6 wins ReClor
  +0.6 pp mean across 2 seeds, every seed beats v5.
- [DIVERSITY_ROOT_CAUSE.md](DIVERSITY_ROOT_CAUSE.md): v4 T5 outputs
  have **−28% unigram diversity, +57% near-duplicate rate, +6% mean
  pair Jaccard** vs stock T5. Surface diversity is the root cause of
  the LogiQA reverse.

This summary covers the **four mitigations** tried to close the LogiQA
reverse: v8, v10, v9, v11.

## All numbers — DeBERTa-large contrastive pretrain → downstream

| Backbone | Description | Contrastive eval | ReClor (seed=21) | LogiQA (seed=21) |
|---|---|---|---|---|
| v5 | stock T5 beam | 99.31% | 62.80% | **41.01%** |
| v6 | v4 fine-tuned T5 beam | 98.43% | **63.60%** | 39.17% |
| v7 | v6 + De Morgan rule fix | 98.43% | 63.60% | 39.17% |
| v8 | v6 + 182 legacy `double_negation` rows | 98.45% | 63.00% | 38.71% |
| v10 | concat(v5, v6) — 28k rows | 98.23% | 62.40% | 38.10% |
| v9 | v4 T5 sampled (T=1.0, top_p=0.9, k=2) | 97.95% | 59.60% | 29.34% |
| v11 | v9 + polarity-parity verifier filter | **99.81%**¹ | 59.80% | 32.26% |

¹ v11 contrastive pretrain was killed at epoch ~4 of 10 — the 99.81%
is the best eval reached, higher than any other backbone reached at
any point.

## Deltas vs v6 (the cleanest baseline)

| | ReClor Δ | LogiQA Δ | net | verdict |
|---|---|---|---|---|
| v7 | 0 | 0 | 0 | (rule fix doesn't change pair content) |
| v8 (+ legacy dn) | −0.6 | −0.5 | −1.1 | fails |
| v10 (concat) | −1.2 | −1.1 | −2.3 | fails |
| v9 (sampled) | −4.0 | −9.8 | −13.8 | fails badly |
| **v11 (sampled + verifier)** | **−3.8** | **−6.9** | **−10.7** | **fails, but better than v9** |

The verifier in v11 helps vs v9 (+0.2 pp ReClor, +2.9 pp LogiQA) —
filtering polarity-bad samples is real signal — but the gap to v6 still
opens. No corpus-level mitigation recovers v5's LogiQA edge.

## Why none of the mitigations work

| Mitigation | What it adds | Why it fails |
|---|---|---|
| v8 (legacy dn re-add) | 182 high-quality double-neg pairs with antonym-swap | LogiQA test isn't bottlenecked on the missing `double_negation` rule — broader diversity matters |
| v10 (concat v5+v6) | v5 surface forms back in the corpus | Same (anchor, label) class gets two contradictory surface forms; contrastive head averages out into weaker decision boundary |
| v9 (sampled v4) | per-anchor sampled diversity | T=1.0 sampling drops or shifts the polarity ~10% of the time; LogiQA chains amplify the noise |
| v11 (sampled + verifier) | sampling + filter for polarity correctness | Polarity verifier passes ~93% of samples but misses semantic errors that don't change polarity count (wrong subject, scope, predicate) |

## The structural take

v4 T5's polarity preservation and surface diversity are **coupled**. The
fine-tune tightened the decoder's beam distribution exactly on the
words/positions where polarity lives. The "safety margin" that gives v6
its polarity edge IS what shrinks its surface diversity. You can't
sample for diversity without sampling into the same neighborhood that v4
was trained to AVOID.

So the apparent ReClor / LogiQA trade-off **is a property of this T5
checkpoint and this verifier**, not a fundamental property of AMR-LDA.
Future directions that might break the trade-off:

1. **Semantic verifier beyond polarity** — use V1 AMR-struct similarity
   (already implemented in [`extensions/auto_verifier/amr_verifier.py`](../auto_verifier/amr_verifier.py))
   as the filter instead of polarity-parity. Higher computational cost
   per sample but catches scope/predicate errors that polarity-parity
   misses.
2. **Source-side augmentation** — paraphrase the anchor sentences BEFORE
   parsing (back-translation, T5-paraphrase, GPT rewrite). v4 T5 applied
   to diverse-anchor distribution would produce diverse outputs naturally
   without sampling noise.
3. **Larger backbone** — DeBERTa-v2-xxlarge (1.5B; paper's headline
   regime; un-run). The diversity drop may matter less at 4× the model
   size.
4. **Co-trained generator + verifier** — RL loop where v4 T5 is fine-
   tuned against the V1 verifier reward, jointly maximising polarity
   preservation AND structural diversity. See
   [`extensions/rl/train_grpo_7b.py`](../rl/train_grpo_7b.py) — already
   wired for this in the small-scale POC; needs to be scaled.

## What stands as the ACL Findings 2024 extension headline

- **Methods**: v4 T5 polarity fine-tune + De Morgan-aware contraposition
  rule fix (commits `c0ffd80` + `3dc22ec`).
- **Pilot polarity-preservation pass rate**: 68.9% → **82.2%** (full
  49-sentence pilot, +13.3 pp; contraposition 8/15 → **15/15** perfect).
- **Held-out PARARULE-Plus Depth5**: 70.6% → 73.4% (+2.8 pp).
- **ReClor downstream**: 62.8% → **63.5%** mean across 2 seeds (+0.6 pp,
  every v6 seed beats every v5 seed).
- **LogiQA downstream**: 41.0% → **39.2%** (−1.8 pp; honest counterexample
  — surface diversity drop documented in DIVERSITY_ROOT_CAUSE.md;
  closing this gap is open future work).

Honest framing: the v4 T5 + rule-fix approach is a clear win on the
single-step entailment benchmark (ReClor), trades off on the multi-hop
deductive benchmark (LogiQA) because polarity-cleaning shrinks surface
diversity. Four corpus-level mitigations explored, none recovered the
LogiQA edge without losing ReClor.
