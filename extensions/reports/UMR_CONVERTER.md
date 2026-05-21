# AMR → UMR Converter — Validation Report

Rule-based reproduction of [Post et al. 2024](https://aclanthology.org/2024.dmr-1.15/)
*"Accelerating UMR Adoption: Neuro-Symbolic Conversion from AMR-to-UMR with
Low Supervision"* (DMR 2024).

Validated on UMR 2.0 English ([umr4nlp/umr-data](https://github.com/umr4nlp/umr-data),
596 sentences with gold aspect or modal-strength annotations out of 31,147 total).

## Results (rule-only, no neural component)

| Attribute | Precision | Recall | F1 |
|---|---|---|---|
| `:aspect` (state/activity/performance/habitual/process/inceptive) | 56.1% | 35.8% | **43.7%** |
| `:modal-strength` (full-aff/partial-aff/full-neg/...) | 22.8% | 33.8% | **27.2%** |

These are reasonable numbers for a pure-rule approach. The original Post et
al. paper achieves higher F1 by adding a small neural classifier for the
non-deterministic cases — that's a hook our converter exposes but doesn't yet
implement.

## What the converter does

For each predicate node in an input AMR graph, the converter assigns:

1. **`:aspect`** — based on a curated dictionary of PropBank framesets:
   - **state**: copular framesets (`have-XX-91`), cognitive/emotive verbs
     (`know-01`, `love-01`, `fear-01`), existential / spatial / possessive
   - **performance**: telic events (`say-01`, `arrive-01`, `kill-01`, `marry-01`,
     `decide-01`, ~80 frames total)
   - **activity**: atelic ongoing (`walk-01`, `read-01`, `think-01`,
     weather verbs, ~30 frames)
   - **habitual**: triggered by `:freq` or `:every` markers
   - **inceptive**: `start-01`, `begin-01`, `launch-01`
   - **process**: `grow-01`, `change-01`, `develop-01`, `rise-01`
2. **`:modal-strength`** — applied to root-level events:
   - `full-affirmative` is the default for affirmed propositions
   - `full-negative` when `:polarity-` is present
   - explicit overrides for `obligate-01`, `possible-01`, `permit-01`, etc.
3. **`:animacy`** — based on a small dictionary of person/animal/object concepts

## Top confusions (where the rules struggle)

| Gold → Predicted | Count |
|---|---|
| process → performance | 19 |
| activity → performance | 18 |
| performance → state | 17 |
| state → performance | 13 |
| habitual → performance | 12 |
| partial-affirmative → full-affirmative | 7 |
| neutral-affirmative → full-affirmative | 5 |

Most aspect errors are minor category-shift errors between related labels.
The "performance vs state" boundary is the biggest source of confusion — UMR
sometimes labels what looks like a telic event (e.g., `die-01`) as state
("being dead") rather than performance.

## Future work (planned)

- **Add neural classifier** for non-deterministic cases (the Post et al. 2024
  approach): fine-tune a small BERT on (AMR-context → aspect-label) pairs
  from UMR 2.0 to handle the gray zone between performance/process/activity.
- **Document-level temporal** annotation (`:before` / `:after` / `:overlap`
  between events) — currently we extract only sentence-level info.
- **Document-level modal dependencies** (the speaker-author modal subgraph).
- **Animacy classifier** — use it as input to argument-role disambiguation
  (which is what Post et al. focus on).

## How to reproduce

```bash
# 1. Clone UMR 2.0 corpus (~30s)
bash extensions/umr/download_umr_data.sh

# 2. Run validation (~1 min for 1000 sentences)
PYTHONPATH=. python -m extensions.umr.validate_converter \
    --umr-root extensions/umr/umr-data \
    --max-sentences 1000 \
    --out-json extensions/umr/validation_report.json
```

The converter is in [extensions/umr/converter.py](../umr/converter.py) and the
validator in [extensions/umr/validate_converter.py](../umr/validate_converter.py).
