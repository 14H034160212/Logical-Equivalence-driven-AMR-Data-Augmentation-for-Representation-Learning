# Pilot Study Reports

Published reports from the AMR-LDA extension pilot. Each report is generated
from the auto-verifier consensus pipeline ([../auto_verifier/](../auto_verifier/))
applied to the LLM-as-rewriter pilot outputs
([../pilot_study/results/](../pilot_study/results/), gitignored).

## Headline

| Report | Purpose |
|---|---|
| [THREE_WAY_v4.md](THREE_WAY_v4.md) | Latest 3-way comparison: AMR-LDA vs gpt-4o vs gpt-4o-mini under multi-verifier consensus |
| [TRAJECTORY_v2.md](TRAJECTORY_v2.md) | AMR-LDA improvement trajectory across 4 patch rounds (run3 → run6) |
| [SELF_CHECK.md](SELF_CHECK.md) | T5wtense self-consistency check breakdown — 15 known generator failures |
| [BEFORE_AFTER.md](BEFORE_AFTER.md) | run3 → run4: items recovered by `:condition`-style patches |
| [BEFORE_AFTER_UMR.md](BEFORE_AFTER_UMR.md) | run4 → run5: items recovered by 4 new UMR-style rules |
| [BEFORE_AFTER_SELFCHECK.md](BEFORE_AFTER_SELFCHECK.md) | run5 → run6: 3 items recovered by self-check + arg-pair patches |

## Headline numbers (run6)

| System | Coverage | Quality (EQ/decided) | Overall |
|---|---|---|---|
| **amr_lda** | 68.9% | **43.8%** | 7.8% |
| gpt-4o | 100% | 80.9% | 61.1% |
| gpt-4o-mini | 100% | 78.3% | 60.0% |

AMR-LDA quality went **8.6% → 43.8%** (5× improvement) across 4 patch rounds.
The remaining gap to LLMs is driven mostly by T5wtense generator noise, not
by rule-logic errors (V1 AMR-struct verifier says 89% of AMR-LDA outputs are
structurally correct).

## Underlying data

JSON aggregates ship alongside the markdown:

- [pilot_summary.json](pilot_summary.json) — per-rule equivalence rates per LLM
- [run6_summary_by_verifier.json](run6_summary_by_verifier.json) — V1 vs V2 EQ rates
- [run6_summary_by_system.json](run6_summary_by_system.json) — consensus per system

## Integration guides

- [MULTI_LLM_SETUP.md](MULTI_LLM_SETUP.md) — how to add Claude / DeepSeek /
  Llama-3-70B baselines (code already supports them; needs API keys + a
  small re-run)

## RL training

- [GRPO_RESULTS.md](GRPO_RESULTS.md) — small-scale GRPO experiment
  (Qwen2.5-0.5B, 16 examples, 113 sec on one A100). Reward improved
  **43.75% → 62.5%** over 1 epoch, validating the verifier-backed reward
  pipeline end-to-end.
- [GRPO_3B_RESULTS.md](GRPO_3B_RESULTS.md) — production-scale GRPO with
  Qwen2.5-3B-Instruct + LoRA on 2× A100. 64 examples × 3 epochs × 4
  generations/prompt; **reward 37.5% → 93.75%** in 13 minutes.

## T5wtense fine-tune

- [T5_FINETUNE_RESULTS.md](T5_FINETUNE_RESULTS.md) — domain-adapting the
  AMR-to-text generator to preserve `:polarity -` edges. 389 silver pairs
  harvested from the pilot, 3 epochs, eval_loss **0.278 → 0.240**.
  JSON: [ft_t5wtense_report.json](ft_t5wtense_report.json).

## v6 contrastive + ReClor + held-out (full downstream story)

- [V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md) — v4 T5 →
  v6 contrastive corpus → DeBERTa-large. In-distribution v6 98.4% vs
  v5 99.3%, but cross-eval shows v6-trained generalises +10pp better.
- [V6_RECLOR.md](V6_RECLOR.md) — ReClor downstream fine-tune on top
  of the two backbones. v6 **wins +0.8 pp dev_acc (63.6% vs 62.8%)**.
- [HELDOUT_PARARULE.md](HELDOUT_PARARULE.md) — 60-sentence held-out
  PARARULE-Plus Depth5 shard, self-check pass rate **70.6% → 72.0%**.

## Paper figures

Auto-generated from the JSON aggregates by
[`extensions/pilot_study/make_paper_figures.py`](../pilot_study/make_paper_figures.py).
Saved to [figures/](figures/):

- ![fig1](figures/fig1_t5_trajectory.png) — T5wtense fine-tune trajectory v1→v4
- ![fig2](figures/fig2_v6_cross_eval.png) — v5/v6 contrastive cross-eval heatmap
- ![fig3](figures/fig3_reclor_trajectory.png) — ReClor dev-acc trajectory
- ![fig4](figures/fig4_heldout_pararule.png) — held-out PARARULE by-rule pass rate
- [T5_FT_RECOVERY.md](T5_FT_RECOVERY.md) — A/B of the AMR-LDA pipeline
  on the 15 known polarity-flips with stock vs fine-tuned T5wtense across
  v1/v2/v3/v4. Final pass rate **34.8% → 73.9%** (subset),
  **68.9% → 78.9%** (full 49-sentence pilot, +9 net recoveries).
  JSONs: [v1](t5_ft_recovery_summary.json), [v2](t5_ft_recovery_v2_summary.json),
  [v3](t5_ft_recovery_v3_summary.json), [v4](t5_ft_recovery_v4_summary.json),
  [full-pilot v3](pilot_full_v3_summary.json).
- [T5_FT_CONSENSUS.md](T5_FT_CONSENSUS.md) — same v4 pilot output run
  back through the AMR + SMATCH consensus verifier; AMR-LDA EQ-rate
  75.4% → 76.5%. V2 (LLM-judge) skipped pending API key.
  JSON: [v4_consensus_summary.json](v4_consensus_summary.json).

## AMR → UMR converter

- [UMR_CONVERTER.md](UMR_CONVERTER.md) — Post et al. 2024 reproduction.
  Rule-based F1 43.7% (aspect), 27.2% (modal). **Hybrid (rule + neural classifier)
  achieves gold-conditional accuracy 77.9% on aspect** (vs rule-only ~36%).
- [DOC_LEVEL_UMR.md](DOC_LEVEL_UMR.md) — document-level annotation derivation.
  Modal F1 32% (P=70%), coref F1 5%, temporal baseline = 0 (granularity issue
  documented; next-step work).
- JSON aggregates: [umr_validation_report.json](umr_validation_report.json),
  [aspect_nn_report.json](aspect_nn_report.json),
  [hybrid_aspect_report.json](hybrid_aspect_report.json),
  [document_umr_report.json](document_umr_report.json)
