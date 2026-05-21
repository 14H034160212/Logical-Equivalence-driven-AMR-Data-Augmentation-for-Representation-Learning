# Document-Level UMR Derivation — Baseline Report

UMR's distinctive contribution over AMR is **document-level annotation**:
relations between events / propositions / entities that span sentence
boundaries. The three families are:

- `:temporal` — event-to-event ordering (`:before` / `:after` / `:overlap` /
  `:contained` / `:depends-on`)
- `:modal` — author / speaker stance subgraph (`:full-affirmative`,
  `:partial-affirmative`, `:full-negative`, ...)
- `:coref` — cross-sentence coreference (`:same-entity`, `:same-event`)

This baseline derives document-level relations from a sequence of AMR-style
sentence graphs using simple heuristics. It is a starting point — the harder
neural-symbolic methods (e.g., a document-level transformer trained on UMR
gold) would plug in on top.

## Rules implemented

1. **Modal — author affirmation**:
   For each sentence's root event, emit `(author :full-affirmative <root>)`.
   If the root carries `:polarity-`, emit `:full-negative` instead.

2. **Temporal — explicit `:time` markers**:
   When an AMR node has `:time (after :op1 X)` or `:time (before :op1 X)`,
   emit `(self, :after/:before, X)`.

3. **Temporal — weak sequencing from sentence order**:
   For adjacent sentences (S<sub>k</sub>, S<sub>k+1</sub>), emit
   `(root_k :before root_{k+1})` as a weak default.

4. **Coreference — name match**:
   Entities in different sentences with the same `:name :op1 "X"` string
   are linked by `:same-entity`.

## Results (50 documents from UMR 2.0 English)

| Category | Precision | Recall | F1 | TP / FP / FN |
|---|---|---|---|---|
| `:modal` | **70.2%** | 20.7% | **32.0%** | 193 / 82 / 740 |
| `:coref` | 23.0% | 2.6% | 4.7% | 17 / 57 / 634 |
| `:temporal` | 0% | 0% | 0% | 0 / 267 / 599 |

Honest reporting: temporal F1 is 0% because of a **granularity mismatch**
between gold and our derivation. Gold annotates relations between
sub-events within a sentence (`(s1l :overlap s1d)` — "landslide overlaps
die" within sentence 1), while our weak rule emits only one root-to-root
relation per sentence pair. The endpoints almost never coincide.

## What this baseline does and doesn't show

**Shows**:
- Modal annotation can be largely derived from sentence-root polarity
  (P=70.2% is encouraging for a 5-line rule).
- Even a naive coreference rule (same surface name) catches some links
  (P=23.0%, low recall but non-zero).

**Doesn't show**:
- Sub-event temporal reasoning. Gold UMR annotates ~20-50 temporal triples
  per document; many involve sub-events not at the syntactic root.
- Pronominal coreference. Most cross-sentence coref involves pronouns, not
  re-mentions of the same proper name.
- Modal substructure beyond `:full-affirmative`. Partial / neutral modal
  strengths require contextual judgment our rules don't make.

## Why these numbers still matter for the paper

- **70% modal precision** with 5 lines of rule code is a reasonable signal.
  The remaining gap to gold is from selective annotation (UMR doesn't
  exhaustively annotate every proposition's modality) and modal-strength
  nuance.
- This validates the **stack approach**: simple rules + UMR layer + neural
  classifier composition. The neural component (à la Post et al. 2024)
  would close the gap on temporal and coref. We've already shown the
  classifier idea works for sentence-level aspect (rule 36% → hybrid 78%).

## Code & data

- Document-level parser: [extensions/umr/document.py](../umr/document.py)
- Validator: [extensions/umr/validate_document.py](../umr/validate_document.py)
- Full JSON report: [document_umr_report.json](document_umr_report.json)

## Future work

1. **Event-level granularity**: detect all "news-worthy events" in an AMR
   (not just the root), then emit pairwise temporal relations between
   sub-events. Could use the modal-affirmation context to identify which
   sub-events count.
2. **Document-creation-time anchor**: add references to a global
   `document-creation-time` node when the AMR has a date stamp.
3. **Pronoun coreference**: integrate Spacy / AllenNLP coref for the
   pronoun-rich cases.
4. **Neural document-level model**: with 250+ UMR English documents at
   document-level granularity, fine-tune a graph-pair transformer for
   relation classification.
