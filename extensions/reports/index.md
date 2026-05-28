# AMR-LDA Extension Research

Extension work on top of **Bao et al. (ACL Findings 2024)** — *Abstract
Meaning Representation-Based Logic-Driven Data Augmentation for Logical
Reasoning*. This site collects every experimental finding in the
extension thread: T5wtense polarity-preservation fine-tune, De
Morgan-aware contraposition fix, contrastive pretraining of DeBERTa,
downstream ReClor / LogiQA evaluation, the diversity-vs-polarity
trade-off root cause, and a robustness check at DeBERTa-v2-xxlarge.

---

## Status (collaborator update)

**Code repo:** <https://github.com/14H034160212/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning>
**Base paper:** Bao et al. ACL Findings 2024 — <https://aclanthology.org/2024.findings-acl.353/>

### What's been built

1. **T5wtense polarity-preservation fine-tune (v1 → v4).** Closes the
   original generator's habit of dropping `:polarity -` edges when
   rendering rule-modified AMRs. Four iterations of gold-augmentation
   training, each landing a tighter eval_loss and higher self-check
   pass rate. v4 is the current production checkpoint.
2. **De Morgan-aware contraposition rule fix.** Found that the rule
   library's `contraposition.apply_positive` did not distribute negation
   over conjunctive antecedents, which silently broke S008/S028/S045 in
   the pilot. Patched `extensions/logic_rules/base.py` with a recursive
   `negate_with_demorgan` helper. Pilot contraposition pass rate jumped
   from 8/15 to **15/15**.
3. **End-to-end downstream evaluation.** Re-generated the legacy
   contrastive corpus with v4 T5 (v6 dataset, 14k rows), pretrained
   DeBERTa-large contrastive backbones for both v5 (stock) and v6 (v4 T5),
   and fine-tuned each on ReClor and LogiQA across two seeds.
4. **Root-cause analysis of the LogiQA reverse.** v6 wins ReClor but
   loses LogiQA, reproducibly. Built `analyze_diversity.py` and showed
   the v4 fine-tune has −28% unigram diversity and +57% near-duplicate
   rate vs stock. Diversity vs polarity is a structural trade-off.
5. **Four mitigation attempts** (v8 legacy-`double_negation` re-add,
   v10 v5+v6 concat, v9 sampled v4 T5, v11 polarity-verifier filter,
   v12 V1 AMR-struct-verifier filter). **All fail** to recover the
   LogiQA edge without sacrificing the ReClor edge — documented
   honestly as a negative result.
6. **xxlarge robustness check.** Matched-recipe v5/v6 at
   DeBERTa-v2-xxlarge. Direction agrees with DeBERTa-large (v6 > v5 on
   ReClor) but v5's training collapsed late, so magnitude is fragile.

### Headline numbers (DeBERTa-large, multi-seed)

| Task | v5 (stock T5) | v6 (v4 T5) | Δ | Note |
|---|---|---|---|---|
| **ReClor** dev_acc (mean of 2 seeds) | 62.9% | **63.5%** | **+0.6 pp** | every v6 seed beats every v5 seed |
| **LogiQA** dev_acc (mean of 2 seeds) | **42.3%** | 40.3% | −2.0 pp | every v5 seed beats every v6 seed |
| Pilot self-check pass rate | 68.9% | **82.2%** | +13.3 pp | with the De Morgan rule fix |
| Held-out PARARULE-Plus Depth5 | 70.6% | **73.4%** | +2.8 pp | 60 sentences not in v4 training |

### Diversity root cause (DeBERTa-large contrastive corpus)

| Metric on positive sentence2 | v5 stock | v6 v4 T5 | Δ |
|---|---|---|---|
| distinct-1 (unique unigrams / total) | 0.0040 | 0.0029 | −28% |
| distinct-3 (unique trigrams / total) | 0.2180 | 0.1803 | −17% |
| near-duplicate rate (Jaccard ≥ 0.7) | 6.9% | 10.9% | +57% |

### Honest framing

- **What works.** v4 T5 polarity preservation + De Morgan rule fix +
  downstream contrastive pretraining beats the stock pipeline on
  single-step entailment (ReClor) by a small but seed-robust margin.
- **What doesn't.** The same backbone loses 2 pp on multi-step deductive
  reasoning (LogiQA), because polarity-cleaning shrinks surface
  diversity that LogiQA reasoning chains rely on.
- **What we ruled out.** Four corpus-level mitigations (legacy data
  re-add, mixing v5+v6, sampled decoding, sampled+verifier filtering)
  all fail. The diversity-vs-polarity trade-off is structurally coupled
  in the seq2seq generator.

### Open questions / next steps (un-run)

1. **Semantic verifier beyond polarity-parity** — V12 used AMR triple-F1
   at threshold 0.85 and still lost; needs richer SMATCH-like scoring or
   a learned semantic verifier.
2. **Source-side paraphrase augmentation** — vary the anchor sentence
   distribution before parsing, instead of varying outputs.
3. **Generator-verifier RL co-training** — scaffolding already in
   `extensions/rl/`; small-scale GRPO POC confirmed feasibility.
4. **xxlarge with stable v5 training** — multi-seed runs to rule out
   the v5 late-training collapse.
5. **LLM-judge audit of v6 outputs** — blocked on API key.

---

## Headline numbers (DeBERTa-large, single-direction unless noted)

### Polarity-preservation in the AMR-LDA pipeline

| Generator | Pilot self-check pass rate |
|---|---|
| Stock T5wtense (paper baseline) | 68.9% |
| v4 fine-tuned T5wtense | 78.9% |
| **v4 + De Morgan rule fix** | **82.2%** |

Held-out PARARULE-Plus Depth5: stock 70.6% → v4+rulefix 73.4%.
Contraposition specifically: **8/15 → 15/15 perfect** on the pilot.

### Downstream — multi-seed (seed=21, 42)

| Task | v5 (stock T5) | v6 (v4 T5) | Δ |
|---|---|---|---|
| **ReClor** dev_acc (mean of 2 seeds) | 62.9% | **63.5%** | **+0.6 pp** |
| **LogiQA** dev_acc (mean of 2 seeds) | **42.3%** | 40.3% | −2.0 pp |

Both deltas are **seed-robust** — every v6 seed beats every v5 seed on
ReClor; every v5 seed beats every v6 seed on LogiQA.

### Diversity vs polarity — the structural trade-off

| Metric (positive sentence2) | v5 stock | v6 v4 T5 |
|---|---|---|
| Distinct-1 unigrams | 0.0040 | 0.0029 (−28%) |
| Distinct-3 trigrams | 0.2180 | 0.1803 (−17%) |
| Near-dup rate (Jaccard ≥ 0.7) | 6.9% | 10.9% (+57%) |

v4 T5's polarity-cleaning trades surface diversity for cleaner
semantics. Four corpus-level mitigations (v8, v10, v9, v11, v12)
**all fail** to recover both edges — the trade-off is structurally
coupled.

## Quick reading order

1. [T5 fine-tune recovery (v1→v4)](T5_FT_RECOVERY.md) — how polarity preservation got built up
2. [De Morgan rule fix](RULEFIX_DEMORGAN.md) — closing the conjunctive-antecedent failure mode
3. [v6 contrastive pretrain](V6_CONTRASTIVE_PRETRAIN.md) — DeBERTa-large backbone + cross-eval matrix
4. [v6 ReClor multi-seed](V6_RECLOR_MULTISEED.md) — **the headline win** (+0.6 pp seed-robust)
5. [v6 LogiQA multi-seed](V6_LOGIQA_MULTISEED.md) — **the honest reverse** (−2.0 pp seed-robust)
6. [Diversity root cause](DIVERSITY_ROOT_CAUSE.md) — why LogiQA reverses
7. [Diversity final summary](DIVERSITY_FINAL.md) — unified v5..v12 mitigation table
8. [xxlarge delta](V_XXLARGE_DELTA.md) — paper-headline scale robustness check

## Figures

![v1→v4 T5 fine-tune trajectory](figures/fig1_t5_trajectory.png)
*Self-check pass rate on the 15-failure subset and the full 49-sentence
pilot, across v1→v4 fine-tunes. Each version adds a small targeted
gold dataset; v4 has the anchor-gold patch that closes all v3
regressions vs stock.*

![v5/v6 contrastive cross-eval](figures/fig2_v6_cross_eval.png)
*v5-trained DeBERTa loses 15.5 pp out-of-distribution on v6's val; v6-trained
loses only 3.9 pp on v5's val. v6 is the more robust classifier.*

![ReClor dev_acc trajectory](figures/fig3_reclor_trajectory.png)
*v6 ReClor leads at every evaluation step (single seed shown; multi-seed
mean still +0.6 pp).*

![Held-out PARARULE by-rule](figures/fig4_heldout_pararule.png)
*60-sentence PARARULE-Plus Depth5 shard (held out from v4 T5 training).
v4 wins on double_negation/contraposition/modal-strength but loses
on commutative/implication.*

## What's in this site

The left nav groups reports by topic. Every report is a single markdown
file in [`extensions/reports/`](https://github.com/14H034160212/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning/tree/main/extensions/reports)
of the repository, paired with a JSON aggregate so any number on this
site can be checked against the source data.

## License and citation

Original paper: Bao et al. ACL Findings 2024,
<https://aclanthology.org/2024.findings-acl.353/>. Extension code under
the same license as the upstream repository.
