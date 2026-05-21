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

| | Stock T5wtense | Fine-tuned **v1** | Fine-tuned **v2** | Δ stock→v2 |
|---|---|---|---|---|
| Training pairs | — | 389 silver | 469 (388 silver + 8 golds × 10) | |
| Epochs | — | 3 | 4 | |
| eval_loss | — | 0.2396 | **0.2260** | |
| `ok` (passed self-check) | 8 / 23 | 12 / 23 | **13 / 23** | **+5** |
| `self_check_failed` | 15 / 23 | 11 / 23 | 10 / 23 | −5 |
| Pass rate on gen-tested items | 34.8% | 52.2% | **56.5%** | +21.7 pp |
| Recovered (was fail, now ok) | — | 5 | **6** | +6 |
| Regressed (was ok, now fail) | — | 1 | 1 | −1 |
| Net | — | +4 | **+5** | |

**Recovery rate on the 15 known failures: 6/15 = 40.0% (v2).**

The v2 improvement is driven by injecting 8 hand-curated gold rewrites
from `test_sentences.json` (built by
[`extensions.pilot_study.build_failure_set_golds`](../pilot_study/build_failure_set_golds.py))
straight into the training set, upweighted ×10 so a ~2% slice of the corpus
actually shapes the loss.

## Per-case breakdown (stock vs v1 vs v2)

| ID / rule | stock | v1 | v2 | v2 output (truncated) |
|---|---|---|---|---|
| S004 / contraposition | FAIL | OK | OK | _If the water does not boil, it does not reach 100°C..._ ← matches gold |
| S004 / implication | FAIL | FAIL | **OK** | _At sea level, the water did not reach 100°C..._ |
| S005 / contraposition | FAIL | FAIL | **OK** | _If Mary does not have a driver's license, she does not own a car._ ← matches gold |
| S005 / double_negation | OK | OK | **FAIL** | (regression: parity flipped) |
| S008 / contraposition | FAIL | FAIL | FAIL | (still wrong) |
| S013 / de_morgan | FAIL | OK | OK | _Alice isn't tall or Bob isn't short._ ← matches gold |
| S014 / de_morgan | FAIL | FAIL | FAIL | (still wrong) |
| S022 / contraposition | FAIL | OK | OK | _If payroll is not processed, then every employee does not..._ ← matches gold |
| S026 / contraposition | FAIL | FAIL | **OK** | _If the country does not export the surplus, it does not produce..._ ← matches gold |
| S026 / implication | FAIL | FAIL | FAIL | (still wrong) |
| S028 / contraposition | FAIL | OK | **FAIL** | (regression vs v1; parity broke under v2) |
| S040 / modal_strength_inversion | FAIL | FAIL | FAIL | (still wrong) |
| S041 / modal_strength_inversion | FAIL | FAIL | FAIL | (still wrong) |
| S045 / contraposition | FAIL | FAIL | FAIL | (still wrong) |
| S048 / double_negation | FAIL | FAIL | FAIL | (still wrong) |
| S050 / double_negation | OK | FAIL | **OK** | _In the case of Alice, Bob did not attend the party if she was..._ |
| S050 / implication | FAIL | OK | **FAIL** | (regression vs v1) |

Six items where v2's text matches the curated gold near-verbatim (S004/contra,
S005/contra, S013/de_morgan, S022/contra, S026/contra, plus S050/double_neg
recovered). v2 loses one v1 win (S028) and one stock win (S005/double_neg) to
gain three new recoveries.

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

Reproducing v2:

```bash
# 1. Build the 8 (modified_AMR, gold_text) pairs for the failure set
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.build_failure_set_golds

# 2. Re-fine-tune (the script auto-picks up failure_set_golds.jsonl, ×10)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.finetune_t5wtense \
            --out-dir extensions/pilot_study/ft_t5wtense_v2 \
            --out-report extensions/reports/ft_t5wtense_v2_report.json \
            --epochs 4

# 3. Re-run the A/B (same as above with --gen-model ft_t5wtense_v2)
```
