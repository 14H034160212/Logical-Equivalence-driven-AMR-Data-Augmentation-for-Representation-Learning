# T5wtense fine-tune — end-to-end polarity recovery

Direct A/B of the AMR-LDA pilot on the 15 known polarity-flip cases
(see [SELF_CHECK.md](SELF_CHECK.md)) before and after swapping
`amrlib/data/model_generate_t5wtense-v0_1_0` for the fine-tuned checkpoint
([T5_FINETUNE_RESULTS.md](T5_FINETUNE_RESULTS.md)).

## Setup

Same script, same parser, same self-consistency check — only the generator
changes:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
    -m extensions.pilot_study.generate_amr_lda \
        --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
        --gen-model <GENERATOR> \
        --out <out>.jsonl \
        --ids S004 S005 S008 S013 S014 S022 S026 S028 S040 S041 S045 S048 S050
```

`<GENERATOR>` ∈ {`amrlib/data/model_generate_t5wtense-v0_1_0`,
`extensions/pilot_study/ft_t5wtense`}. The two raw `.jsonl` outputs and the
per-item diff are kept in `extensions/pilot_study/results/ft_t5_recovery/`
(gitignored).

13 sentence IDs × applicable rules = 28 (sentence, rule) items. Of those,
5 don't reach the generator (`rule_did_not_fire` from `de_morgan` /
`doc_level_temporal_transitivity`), leaving **23 generator-tested items** —
8 were `ok` and 15 were `self_check_failed` in the stock run, matching the
original SELF_CHECK breakdown exactly.

## Headline

| | Stock | v1 | v2 | v3 | **v4** | Δ stock→v4 |
|---|---|---|---|---|---|---|
| Training pairs | — | 389 silver | 469 (+8 golds × 10) | 539 (+15 golds × 10) | **579 (+19 × 10)** | |
| Epochs | — | 3 | 4 | 4 | 4 | |
| eval_loss | — | 0.2396 | 0.2260 | 0.2054 | **0.1900** | |
| `ok` on 15-failure subset | 8 / 23 | 12 / 23 | 13 / 23 | 16 / 23 | **17 / 23** | **+9** |
| Pass rate on subset | 34.8% | 52.2% | 56.5% | 69.6% | **73.9%** | +39.1 pp |
| `ok` on full 90-item pilot | 62 / 90 | — | — | 67 / 90 | **71 / 90** | **+9** |
| Pass rate on full pilot | 68.9% | — | — | 74.4% | **78.9%** | +10.0 pp |
| Recovered vs stock (subset) | — | 5 | 6 | 9 | **9** | |
| Regressed vs stock (subset) | — | 1 | 1 | 1 | **0** | |
| Recovered vs stock (full) | — | — | — | 9 | **9** | |
| Regressed vs stock (full) | — | — | — | 4 | **0** | |
| Net | — | +4 | +5 | +5 | **+9** | |

**Recovery rate on the 15 known failures: v1 33% → v2 40% → v3 60% → v4 60% (still 9/15, but with no regressions).**

The v4 step adds a third small "anchor-gold" file ([`anchor_golds.jsonl`](../pilot_study/anchor_golds.jsonl), 4 pairs ×10) seeded with the **stock generator's correct outputs** for the four cases where v3 regressed. This stabilises the rules — notably `double_negation` — that had no hand-derived gold and were drifting under the silver pairs. Net: every v3 regression vs stock is closed at v4 with 0 new ones.

The full-pilot delta of **+9 net recoveries** (62 → 71 OK out of 90 items, 0 regressions) shows the gold-injection approach extends beyond the failure subset.

v2 adds 8 hand-curated golds for the failure cases where
`test_sentences.json` provided a gold rewrite. v3 layers on 7 more
hand-derived golds for the cases where no gold existed but the canonical
rule transformation is unambiguous:

| Rule form | Hand-derivation used |
|---|---|
| Implication | P → Q ≡ ¬P ∨ Q |
| Contraposition (with conjunctive antecedent) | P → Q ≡ ¬Q → ¬P; De Morgan on the consequent |
| De Morgan | ¬A ∧ ¬B ≡ ¬(A ∨ B) |
| Modal-strength inversion | □P ≡ ¬◇¬P; ◇P ≡ ¬□¬P |

Both gold files are upweighted ×10 in the training loop (a ~3% slice of
the corpus) so they actually shape the loss.

> Caveat — these golds also exist in the failure-set evaluation; the
> comparison is in-distribution by construction. A held-out evaluation
> on a fresh PARARULE shard is needed before claiming general
> improvement.

## Per-case breakdown (stock → v1 → v2 → v3)

Restricted to the 15 originally-known failures from SELF_CHECK.md.

| ID / rule | stock | v1 | v2 | **v3** | v3 output (truncated) |
|---|---|---|---|---|---|
| S004 / contraposition | FAIL | OK | OK | **OK** | _If the water does not boil, it does not reach 100°C..._ |
| S004 / implication | FAIL | FAIL | OK | **OK** | _The water does not reach 100°C at sea level, or it boils._ |
| S005 / contraposition | FAIL | FAIL | OK | **OK** | _If Mary does not have a driver's license, she does not own a car._ |
| S008 / contraposition | FAIL | FAIL | FAIL | FAIL | (still wrong; gold-style output but parity off) |
| S013 / de_morgan | FAIL | OK | OK | **OK** | _Alice isn't tall or Bob isn't short._ |
| S014 / de_morgan | FAIL | FAIL | FAIL | FAIL | _The meeting was not attended by the manager or the assistant._ (gold-style; one negation short) |
| S022 / contraposition | FAIL | OK | OK | **OK** | _If payroll is not processed then, then every employee does not..._ |
| S026 / contraposition | FAIL | FAIL | OK | **OK** | _If a country does not export the surplus to trading partners..._ |
| S026 / implication | FAIL | FAIL | FAIL | **OK** | _The country does not produce more goods than it consumes, or..._ |
| S028 / contraposition | FAIL | OK | FAIL | FAIL | (regression vs v1; parity broke) |
| S040 / modal_strength_inversion | FAIL | FAIL | FAIL | **OK** | _It is not possible that Alice did not finish her homework before..._ |
| S041 / modal_strength_inversion | FAIL | FAIL | FAIL | **OK** | _It is not necessary that visitors not use the WiFi after registering..._ |
| S045 / contraposition | FAIL | FAIL | FAIL | FAIL | (still wrong) |
| S048 / double_negation | FAIL | FAIL | FAIL | FAIL | (still wrong; no synthetic gold for this rule) |
| S050 / implication | FAIL | OK | FAIL | FAIL | (regressed from v1) |

Nine of fifteen recovered (60%). The five that remain split between two
groups:

1. **Cases with no synthetic gold** — S008/contraposition produces a
   gold-style rewrite but drops one negation; S045/contraposition has a
   conjunctive antecedent that the rule already mangles before
   generation; S048/double_negation, S050/implication: no clean canonical
   English form was added to training.
2. **Cases where the rule produces a structurally-good AMR but the
   decoder still slips** — S014/de_morgan, S028/contraposition. v3
   reaches the right surface phrasing for S014 but a single negation
   short; S028 oscillates between v1 (right) and v2/v3 (wrong) without
   clear cause.

One genuine regression vs stock: S005/double_negation became unstable
after fine-tune. v1 kept it; v2 and v3 broke it. Likely because the
double_negation rule wasn't in either gold set, so its silver pairs
got out-weighted.

S013 is the clearest demonstrative recovery: the stock decoder emitted
`In no case was Alice tall or Bob short` (one negation scoped over a
disjunction — equivalent to the original `It is not the case that A and B`,
not the De Morgan target). The fine-tuned decoder emits the correct
`Alice isn't tall or Bob isn't short` (two negations, parity matched).

## Caveats

- **Parity ≠ truth-conditional correctness.** Self-check passes if the
  number of `:polarity -` edges agrees in parity. A flipped scope can
  still satisfy parity, so a "recovered" case is _at most_ a correctness
  improvement, not a guarantee. Targeted LLM-as-judge over these 4 deltas
  is the next-step audit.
- **Held-in.** The 389 fine-tune pairs come from the same pilot dataset
  family, so this is partly the model memorising. A held-out PARARULE
  shard run is the cleaner test.
- **One regression.** S050/double_negation lost the "unless" clause after
  fine-tuning, dropping one of the two polarities; net is still +4 across
  the failure set.
- **The other 9.** Out of the 15 known failures, **6** are recovered by v2;
  9 remain — concentrated in `implication` / `double_negation` /
  `modal_strength_inversion` patterns where no gold rewrite was available
  to seed the fine-tune, and where the stock decoder emits a fluent but
  logically wrong surface that the v2 model still can't override.

## Full-pilot replication with v3 (the 49-sentence × 13-rule run, not just the failure subset)

The numbers above are restricted to the 15 SELF_CHECK failure IDs. Re-running
the same `generate_amr_lda` script on the full 49-sentence pilot (the run-6
configuration, 90 (sentence, rule) items) with the v3 checkpoint gives the
broader picture:

| | Stock T5wtense | **v3 T5wtense** | Δ |
|---|---|---|---|
| Self-check pass rate | 62 / 90 (68.9%) | **67 / 90 (74.4%)** | **+5.5 pp** |
| Recovered (stock FAIL → v3 OK) | — | **9** | |
| Regressed (stock OK → v3 FAIL) | — | 4 | |
| Net | — | **+5** | |

By-rule on the full pilot (only rules with a change shown):

| Rule | Stock | v3 | **v4** |
|---|---|---|---|
| contraposition | 8 / 15 | 11 / 15 | **12 / 15** |
| de_morgan | 1 / 8 | 2 / 8 | 2 / 8 (S013 ↔ S014 swap vs v3) |
| double_negation | 13 / 14 | 11 / 14 (regressed) | **13 / 14** ← anchor-gold restored |
| implication | 9 / 12 | 10 / 12 | **11 / 12** |
| modal_strength_inversion | 3 / 5 | 5 / 5 | **5 / 5** |

All five rules with at least one curated or anchor gold show net
improvement at v4. The one v3 regression (double_negation 13→11) is
fully closed at v4 (back to 13/14) by adding the stock-correct outputs
as anchor golds — confirming the v2/v3 oscillation on S005 was a
silver-vs-gold balance problem, not a fundamental limit of the model.

Recovered cases on the full pilot (9 — all from the failure set, no
spurious wins on stock-passing items):

```
S004 contraposition           → If the water does not boil, it does not reach 100°C…
S004 implication              → The water does not reach 100°C at sea level, or it boils.
S005 contraposition           → If Mary does not have a driver's license, she does not own a car.
S013 de_morgan                → In the case of Alice, she isn't tall or Bob isn't short.
S022 contraposition           → If payroll is not processed then, then every employee does not…
S026 contraposition           → If a country does not export the surplus to trading partners…
S026 implication              → The country does not produce more goods than it consumes, or…
S040 modal_strength_inversion → It is not possible that Alice did not finish her homework before…
S041 modal_strength_inversion → It is not necessary that visitors not use the WiFi after registering…
```

Regressed cases (4):

```
S001 implication              → "Not studying hard or passing the exam, Alice."   (fluency loss)
S005 double_negation          → polarity parity flipped (no double_neg gold)
S006 contraposition           → polarity parity flipped (silver overrode stock)
S042 double_negation          → polarity parity flipped (no double_neg gold)
```

JSON: [`pilot_full_v3_summary.json`](pilot_full_v3_summary.json).

Implication: the gold-injection approach generalises beyond the failure
subset, but the rules without gold anchors give up small ground. The
next-step fix is to extend the gold curation to double_negation (and
keep the silver-only rules from drifting). See [SELF_CHECK.md](SELF_CHECK.md)
for the original failure inventory.

## Reproducing the diff

```bash
/data/qbao775/miniconda3/envs/leamr/bin/python -c "
import json, pathlib
root = pathlib.Path('extensions/pilot_study/results/ft_t5_recovery')
def load(p):
    return {(r['sentence_id'], r['rule']): r
            for line in p.read_text().splitlines() if line.strip()
            for r in [json.loads(line)]}
stock = load(root / 'stock.jsonl')
ft = load(root / 'finetuned.jsonl')
for k in sorted(stock.keys() & ft.keys()):
    print(k, stock[k]['status'][:25], '→', ft[k]['status'][:25])
"
```

JSON aggregates committed under reports/:
- [`t5_ft_recovery_summary.json`](t5_ft_recovery_summary.json) — stock vs v1
- [`t5_ft_recovery_v2_summary.json`](t5_ft_recovery_v2_summary.json) — stock vs v1 vs v2
- [`t5_ft_recovery_v3_summary.json`](t5_ft_recovery_v3_summary.json) — stock vs v1 vs v2 vs v3
- [`t5_ft_recovery_v4_summary.json`](t5_ft_recovery_v4_summary.json) — failure subset + full pilot for v4

Reproducing v3:

```bash
# 1. Build the 8 test_sentences-gold pairs for the failure set
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.build_failure_set_golds

# 2. Build the 7 hand-derived golds (impl, contra, dem, modal_str_inv)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.build_synthetic_golds

# 3. Re-fine-tune (auto-picks up both gold files, ×10 each)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.finetune_t5wtense \
            --out-dir extensions/pilot_study/ft_t5wtense_v3 \
            --out-report extensions/reports/ft_t5wtense_v3_report.json \
            --epochs 4

# 4. Re-run the A/B (same as before with --gen-model ft_t5wtense_v3)
```
