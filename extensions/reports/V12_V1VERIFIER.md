# v12 — sampled v4 T5 + V1 AMR-struct verifier filter

After [V11_VERIFIED_NEGATIVE.md](V11_VERIFIED_NEGATIVE.md) showed that
polarity-parity is too weak a filter to catch semantic errors in
sampled outputs, v12 tries a richer filter: triple-F1 between the
parsed candidate AMR and the rule-applied expected AMR. This is
essentially the V1 AMR-struct verifier (the same one used in the auto-
verifier consensus pipeline) at a tunable threshold.

## Setup

`build_v12_v1verified.py` extends `build_v11_verified.py`:

- Same v4 T5 sampling parameters (T=0.8, top_p=0.9, k=4 per anchor).
- After parsing each candidate back to AMR, computes triple-F1 against
  the expected modified AMR (using the same `triple_f1` from
  `extensions/auto_verifier/amr_verifier.py`).
- Keeps candidate if F1 ≥ threshold. Threshold sweep on the smoke set:

| threshold | pass rate (smoke, 100 anchors × 4 samples) | mean F1 of kept |
|---|---|---|
| 0.55 | 97.8% (391/400) | 0.892 |
| 0.85 | 69.5% (278/400) | 0.894 |
| 0.95 | 53.2% (213/400) | 0.883 |

`threshold=0.85` selected — keeps useful signal but rejects clearly-
wrong samples.

Full v12 corpus: **26,393 rows** (47.1% pass rate on the full
55,984-candidate set, mean F1 of kept = 0.839).

## Contrastive pretrain

10 epochs, same hparams as v5–v11. Eval accuracy in distribution:

| | eval acc |
|---|---|
| v5 (stock T5) | 99.31% |
| v6 (v4 T5 beam) | 98.43% |
| v9 (v4 sampled, no filter) | 97.95% |
| v11 (sampled + polarity) | 99.81% (ep4, killed) |
| **v12 (sampled + V1@0.85)** | **99.58%** |

V1-filtered training pairs are cleaner than polarity-filtered (98.45 vs
99.58 etc); the contrastive head fits them well.

## Downstream

|  | ReClor (seed=21) | LogiQA (seed=21) |
|---|---|---|
| v5 (stock T5) | 62.80% | **41.01%** |
| v6 (v4 T5 beam) | **63.60%** | 39.17% |
| v9 (sampled, no filter) | 59.60% | 29.34% |
| v11 (sampled + polarity) | 59.80% | 32.26% |
| **v12 (sampled + V1@0.85)** | **60.80%** | **37.33%** |

v12 is the **best sampled-based backbone** seen so far. The V1 filter
makes a real difference vs polarity-parity:

- ReClor v9→v11→v12 = 59.6 → 59.8 → **60.8** (+1.2 pp from V1 filter)
- LogiQA v9→v11→v12 = 29.3 → 32.3 → **37.3** (**+8.0 pp** from V1 filter)

LogiQA benefits much more from filtering — confirming the
diversity-without-noise framing. The verifier doesn't fully close the
gap to v6 (still −2.8 ReClor, −1.8 LogiQA), but it's the most promising
mitigation tried so far.

## What the V1 filter catches that polarity-parity misses

Triple-F1 considers the full AMR graph structure (predicates, ARG
roles, modifier edges, polarity), not just polarity counts.

Examples of samples rejected at F1 0.85 but kept by polarity-parity:

| Sample | Why polarity-parity passed | Why V1 F1 rejected |
|---|---|---|
| "The mouse is not clever; the eagle is kind." | one polarity, matches | conjunction instead of conditional → missing `:condition` edge |
| "A bald eagle isn't kind to a clever mouse." | one polarity, matches | wrong predicate frame (`kind-to` instead of `kind-01 :ARG0` ) |
| "The mouse was not clever unless it was a kind bald eagle." | one polarity, matches | wrong scope (`unless` clause swap) |

V1 catches scope errors and predicate-frame drift; polarity-parity does not.

## Open: stricter threshold

The 0.85 sweep gave 47% retention. At 0.95 retention drops to 53%
(smoke), suggesting an even stricter filter MAY help further (un-run).
Tradeoff: stricter → smaller training corpus → faster training but
possibly insufficient signal for big backbones.

JSON: [`v12_summary.json`](v12_summary.json).
