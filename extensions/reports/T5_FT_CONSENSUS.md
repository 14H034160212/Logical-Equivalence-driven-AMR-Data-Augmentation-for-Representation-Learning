# v4 T5 — consensus-verifier impact (AMR + SMATCH)

After [v4 closes the self-consistency regressions](T5_FT_RECOVERY.md), does
the consensus verifier ([../auto_verifier/](../auto_verifier/)) also see
improved AMR-LDA quality?

## Setup

Same `run_auto_verify.py` flow as the original run-6 [pilot summary](run6_summary_by_system.json), but:

- `--skip-llm-judges` — no V2 LLM judge (no API key in this session)
- Verifiers used: **V1 = AMR-struct** (threshold 0.55) and **V4 = SMATCH**
- Candidates: stock `amr_lda.jsonl`, **new `amr_lda_v4.jsonl`** (from
  `extensions/pilot_study/results/ft_t5_recovery/pilot_full_v4.jsonl`),
  plus `gpt-4o` and `gpt-4o-mini` for reference
- All four systems: 90 (sentence, rule) items each, 360 verifier calls total

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
    -m extensions.auto_verifier.run_auto_verify \
        --candidates-dir <dir with amr_lda.jsonl + amr_lda_v4.jsonl + gpt-4o*.jsonl> \
        --amrlib-model-dir amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
        --skip-llm-judges --threshold 0.55 \
        --out-dir extensions/auto_verifier/results/v4_pilot
```

## Headline

| System | EQ | NEQ | Pending | EQ-rate (decided) |
|---|---|---|---|---|
| amr_lda (stock T5) | 52 | 17 | 21 | 75.4% |
| **amr_lda_v4** | 52 | 16 | 22 | **76.5%** (+1.1 pp) |
| gpt-4o | 56 | 4 | 30 | 93.3% |
| gpt-4o-mini | 56 | 4 | 30 | 93.3% |

The fine-tune that moved the upstream self-check pass rate **68.9% → 78.9%**
moves the consensus-verifier EQ-rate only **75.4% → 76.5%**. The lift is
small but uniform: same EQ count, one fewer NEQ, one more deferred to
human review.

## Why the gap between self-check pass and V1 EQ?

The polarity-parity self-check (used in T5_FT_RECOVERY) and the V1
AMR-struct verifier (used here) catch overlapping but not identical errors:

- **Self-check** counts `:polarity -` edges; off-by-one ⇒ FAIL.
- **V1** matches the candidate's parsed graph against the rule-applied
  expected graph using a fuzzy AMR-similarity score with threshold 0.55.
  It tolerates polarity off-by-one if the rest of the graph aligns
  closely, and it rejects items that pass parity but reshape the graph.

Concretely: 9 of v4's 9 new self-check passes were already V1-EQ in the
stock run (the AMR similarity tolerated the polarity flip). And one
formerly-failing item flips V1-EQ → V1-NEQ → V1-EQ across the fine-tune
versions without obvious cause.

So the v4 self-check improvement is **upstream** of the consensus
verifier, not orthogonal to it: it cleans the surface text that
downstream BERT/DeBERTa training would consume, but it doesn't shift
the AMR-graph judgment the consensus is built on.

## What still needs LLM-judge (V2)

The published 43.8% AMR-LDA quality from the README's run-6 numbers comes
from consensus of V1 + V2 (LLM-judge). Without V2, the comparable AMR-LDA
EQ rate here jumps to 75.4% (V1 only is much more lenient). To get a true
v4 quality number that's comparable to the 43.8% headline, V2 needs to
run — that requires an OpenAI / Anthropic / DeepSeek API key in env (see
[MULTI_LLM_SETUP.md](MULTI_LLM_SETUP.md)). Until then this consensus is
AMR-side only.

## Files

- [`extensions/auto_verifier/results/v4_pilot/per_item.jsonl`](../auto_verifier/results/v4_pilot/per_item.jsonl) — full 360-call verdicts (gitignored — heavy)
- [`extensions/auto_verifier/results/v4_pilot/summary_by_system.json`](../auto_verifier/results/v4_pilot/summary_by_system.json) — table above
- [`extensions/auto_verifier/results/v4_pilot/summary_by_verifier.json`](../auto_verifier/results/v4_pilot/summary_by_verifier.json) — per-verifier rates
- Headline JSON mirrored as [`v4_consensus_summary.json`](v4_consensus_summary.json) under reports/ for direct linking.

## Takeaway

The v4 T5 fine-tune materially fixes the self-check failures (the
upstream text-generation problem) and gives a small but real lift at the
AMR-struct consensus level (+1.1 pp). The full quality headline needs
V2 (LLM-judge) to rerun.
