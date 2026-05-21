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
