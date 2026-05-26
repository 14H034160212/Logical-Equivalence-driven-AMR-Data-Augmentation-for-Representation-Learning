# v11 — verifier-filtered sampled diversity also fails

The strongest remaining mitigation candidate after the v8/v10/v9 negative
results in [DIVERSITY_ROOT_CAUSE.md](DIVERSITY_ROOT_CAUSE.md) → [V8_DOUBLENEG_REINTRO.md](V8_DOUBLENEG_REINTRO.md)
→ [V10_MIX_NEGATIVE.md](V10_MIX_NEGATIVE.md) → [V9_SAMPLED_NEGATIVE.md](V9_SAMPLED_NEGATIVE.md):
combine v4 T5 sampling (recovers surface diversity) with the AMR polarity-
parity self-check (filters out semantically wrong samples). If
"diversity without noise" is the right framing, v11 should win.

## Setup

`extensions/pilot_study/build_v11_verified.py`:

1. Same anchor parsing + rule application as v6/v9.
2. Each modified AMR → 4 sampled outputs via direct T5 `do_sample=True,
   top_p=0.9, temperature=0.8, num_return_sequences=4`.
3. Each sampled candidate is parsed back to AMR and kept only if the
   `:polarity -` count matches the expected count from the modified
   AMR — the same polarity-parity check used in
   `generate_amr_lda.py::_self_consistency_check`.

```
candidates generated:  55,984 (13,996 modified AMRs × 4 samples)
candidates passed:     52,018
filter pass rate:      92.9%
```

## Headline numbers

Contrastive pretrain eval (in-distribution; trajectory from tensorboard
since the run was killed at epoch ~4 of 10):

| | eval acc |
|---|---|
| v5 (10 ep) | 99.31% |
| v6 (10 ep) | 98.43% |
| v9 (10 ep) | 97.95% |
| v10 (10 ep) | 98.23% |
| **v11 (~4 ep, killed)** | **99.81%** |

v11 has the *highest* contrastive eval despite being undertrained —
verifier-filtered sampled pairs are easier to fit, structurally cleaner
than v6.

Downstream (seed=21, single seed):

|  | ReClor | LogiQA |
|---|---|---|
| v5 (stock) | 62.80% | **41.01%** |
| v6 (v4 T5 beam) | **63.60%** | 39.17% |
| v9 (v4 T5 sampled) | 59.60% | 29.34% |
| **v11 (v4 sampled + verifier)** | **59.80%** | **32.26%** |

v11 vs v9: verifier helps (+3 pp on LogiQA, +0.2 on ReClor) — the
9% rejected samples were indeed harmful.

v11 vs v6: still loses on both (−3.8 pp ReClor, −6.9 pp LogiQA). The
verifier catches polarity errors but misses **semantic** errors —
wrong subject/scope/predicate that don't change the polarity count.

## Why polarity-parity isn't enough

Sample failure modes the polarity verifier *misses*:

| Sample | What's wrong | Polarity parity |
|---|---|---|
| "The mouse was not clever unless it was a kind bald eagle." | wrong scope of `unless` (no longer the intended contrapositive) | matches (1=1) |
| "A bald eagle isn't kind to a clever mouse." | wrong predicate frame (`kind-to`, semantic drift) | matches (1=1) |
| "The mouse is not clever; the eagle is kind." | conjunction instead of conditional (truth conditions changed) | matches (1=1) |

These passed the 92.9% filter because they have one `:polarity -` edge,
matching the expected count. But they don't represent the actual logical
equivalence the rule was supposed to capture.

A meaningful verifier here would need to check the FULL graph structure
(node types, edge labels, scope) — i.e., something closer to a SMATCH
score or a learned semantic verifier — not just polarity-parity counting.

## Status of all diversity mitigations

| | ReClor Δ vs v6 | LogiQA Δ vs v6 | Verdict |
|---|---|---|---|
| v8 (legacy double_negation re-add) | −0.6 | −0.5 | fails |
| v10 (v5 + v6 concat) | −1.2 | −1.1 | fails |
| v9 (sampled v4 T5) | −4.0 | −9.8 | fails |
| **v11 (sampled v4 + polarity verifier)** | **−3.8** | **−6.9** | **fails** |

None of the four corpus-level mitigations recover the v5 LogiQA edge
while keeping v6's ReClor edge. v6 / v7 (the De Morgan-aware
contraposition rule fix) remain the best known checkpoints.

## See also

[`DIVERSITY_FINAL.md`](DIVERSITY_FINAL.md) — unified summary across all
v5–v11 backbones with the conclusion that v4 T5's polarity preservation
and surface diversity are structurally coupled.
JSON: [`v11_summary.json`](v11_summary.json).
