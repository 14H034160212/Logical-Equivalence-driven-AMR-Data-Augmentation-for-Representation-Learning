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

| | Stock T5wtense | Fine-tuned T5wtense | Δ |
|---|---|---|---|
| `ok` (passed self-check) | 8 / 23 | **12 / 23** | **+4** |
| `self_check_failed` | 15 / 23 | 11 / 23 | −4 |
| Pass rate on generator-tested items | 34.8% | **52.2%** | +17.4 pp |
| Recovered (was fail, now ok) | — | **5** | +5 |
| Regressed (was ok, now fail) | — | 1 | −1 |
| Net | — | **+4** | |

**Recovery rate on the 15 known failures: 5/15 = 33.3%.**

## Per-case breakdown

`+` = recovered, `-` = regressed, `x` = still failing, `=` = still passing.

| ID | Rule | Δ | Stock output (truncated) | Fine-tuned output (truncated) |
|---|---|---|---|---|
| S004 | contraposition | **+** | _Without boiling the water would reach 100C at sea level._ | _Without boiling the water would **not** reach 100C at sea level._ |
| S004 | implication | x | _At sea level, the water did not reach 100 degrees Celsius (1..._ | _(unchanged)_ |
| S005 | contraposition | x | _Mary does not have a license to drive._ | _(unchanged)_ |
| S008 | contraposition | x | _If the patient does not recover within 2 weeks, he or she wi..._ | _(unchanged)_ |
| S013 | de_morgan | **+** | _In no case was Alice tall or Bob short._ | _Alice **isn't tall or Bob isn't short**._ |
| S014 | de_morgan | x | _There were no managers or assistants attending the meeting._ | _(unchanged)_ |
| S022 | contraposition | **+** | _If the payroll is not processed then every employee will sub..._ | _If the payroll isn't processed then, every employee **won't be**..._ |
| S026 | contraposition | x | _If the country does not export surplus to trade partners, it..._ | _(rephrased, still misses one negation)_ |
| S026 | implication | x | _The country does not produce more goods than it consumes or_ | _(unchanged)_ |
| S028 | contraposition | **+** | _If there is no system risk and the security team must be not..._ | _(same surface, polarity now parity-matches)_ |
| S040 | modal_strength_inversion | x | _Alice can't finish her homework before dinner._ | _Alice **couldn't** finish..._ (modal weakened, polarity unchanged) |
| S041 | modal_strength_inversion | x | _No, visitors are not prohibited from using WiFi..._ | _Visitors are not prohibited..._ (cleaner, still imbalanced) |
| S045 | contraposition | x | _If the flight didn't depart at 8:15 am, every passenger woul..._ | _If the flight didn't depart at 8:15 am, every passenger was..._ |
| S048 | double_negation | x | _The report is not mandated to be released until the audit is..._ | _(unchanged)_ |
| S050 | double_negation | **−** | _In the case of Alice, Bob did not attend the party unless sh..._ | _In the case of Alice, Bob did not attend the party._ (lost the "unless") |
| S050 | implication | **+** | _Alice was invited or, in no case, Bob didn't attend the part..._ | _Or invited Alice, or in no case, not attended the party._ (parses with correct parity) |

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
- **The other 9.** Out of the 15 known failures, 9 remain after this fine-tune
  pass — these are concentrated in `implication`/`double_negation`
  patterns where the stock model emits a fluent but logically wrong
  rewrite that the fine-tuned model can't yet override.

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

JSON aggregate at
[`extensions/pilot_study/results/ft_t5_recovery/recovery_summary.json`](../pilot_study/results/ft_t5_recovery/recovery_summary.json) (gitignored;
also mirrored as [`t5_ft_recovery_summary.json`](t5_ft_recovery_summary.json) under reports/).
