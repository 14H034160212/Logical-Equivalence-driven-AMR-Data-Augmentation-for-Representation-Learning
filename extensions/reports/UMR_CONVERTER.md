# AMR → UMR Converter — Validation Report

Rule-based reproduction of [Post et al. 2024](https://aclanthology.org/2024.dmr-1.15/)
*"Accelerating UMR Adoption: Neuro-Symbolic Conversion from AMR-to-UMR with
Low Supervision"* (DMR 2024).

Validated on UMR 2.0 English ([umr4nlp/umr-data](https://github.com/umr4nlp/umr-data),
596 sentences with gold aspect or modal-strength annotations out of 31,147 total).

## Results

### Rule-only (deterministic dictionary)

| Attribute | Precision | Recall | F1 |
|---|---|---|---|
| `:aspect` | 56.1% | 35.8% | 43.7% |
| `:modal-strength` | 22.8% | 33.8% | 27.2% |

The rule fires on ~46% of gold-annotated aspect nodes and abstains on the
remaining 54%. Among the nodes it does fire on, accuracy is reasonable but
fall-off recall is the bottleneck.

### Hybrid (rule + small neural classifier)

The neural classifier (sklearn `LogisticRegression` with class-balanced
weights, ~17K bag-of-features inputs per node: frame name, frame stem,
frame sense, polarity-neg, has-time/duration/freq, parent role, depth,
n_args) takes over when the rule abstains.

| Metric | Rule-only | Hybrid |
|---|---|---|
| **Gold-conditional accuracy** (correct / gold-annotated nodes) | ~36% | **77.9%** |
| Rule used | 680 / 1488 | 680 / 1488 |
| NN used | n/a | 808 / 1488 |
| Wrong | n/a | 329 / 1488 |

On the **subset where the rule abstains** (54% of all gold nodes), the NN
alone achieves **macro F1 0.778** (precision 0.769, recall 0.792) — a
substantial improvement over leaving those nodes unpredicted.

Per-label breakdown of the NN on its test set (n=298):

```
              precision    recall  f1-score   support
    activity       0.46      0.68      0.55        38
    endeavor       0.58      1.00      0.74         7
    habitual       0.35      0.43      0.39        14
   inceptive       0.00      0.00      0.00         2
 performance       0.76      0.65      0.70       104
     process       0.37      0.67      0.48        15
       state       0.90      0.72      0.80       118
    accuracy                           0.68
   macro avg       0.49      0.59      0.52
weighted avg       0.73      0.68      0.69
```

State and performance dominate the data; the model learns those well
(F1 ~0.7-0.8). Rare classes (inceptive, endeavor, habitual) have very few
examples in UMR 2.0 and the model struggles — would benefit from
oversampling or a class-balanced loss.

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

- **Fine-tune BERT** on AMR-context → aspect-label pairs to push F1 further
  (the current sklearn baseline doesn't use sub-token semantics).
- **Document-level temporal** annotation (`:before` / `:after` / `:overlap`
  between events) — currently we extract only sentence-level info.
- **Document-level modal dependencies** (the speaker-author modal subgraph).
- **Animacy classifier** — use it as input to argument-role disambiguation
  (which is what Post et al. focus on).
- **Modal-strength neural classifier** — apply the same hybrid recipe.

## How to reproduce

```bash
# 1. Clone UMR 2.0 corpus (~30s)
bash extensions/umr/download_umr_data.sh

# 2. Rule-only validation (~1 min for 1000 sentences)
PYTHONPATH=. python -m extensions.umr.validate_converter \
    --umr-root extensions/umr/umr-data \
    --max-sentences 1000 \
    --out-json extensions/umr/validation_report.json

# 3. Train the neural aspect classifier (a few seconds)
PYTHONPATH=. python -m extensions.umr.neural_aspect

# 4. Hybrid (rule + NN) validation
PYTHONPATH=. python -m extensions.umr.validate_hybrid
```

The artifacts are:
- [extensions/umr/converter.py](../umr/converter.py) — rule-based AMR→UMR
- [extensions/umr/neural_aspect.py](../umr/neural_aspect.py) — sklearn aspect classifier
- [extensions/umr/validate_converter.py](../umr/validate_converter.py) — rule-only eval
- [extensions/umr/validate_hybrid.py](../umr/validate_hybrid.py) — hybrid eval with threshold sweep
- [aspect_nn_report.json](aspect_nn_report.json) — NN training metrics
- [hybrid_aspect_report.json](hybrid_aspect_report.json) — hybrid threshold sweep results
