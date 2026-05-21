# Human Evaluation Rubric — Spot-check Protocol for Auto-Verifier Disagreement

This rubric is **only** for the items flagged by the auto-verifier pipeline (`extensions/auto_verifier/`) as needing human review. The auto-verifier already labels the majority of items via multi-verifier consensus (V1: AMR structural, V2: GPT-4-as-judge, V3: Claude-as-judge, V4: SMATCH similarity). Humans only adjudicate the **disagreement subset** — typically 20-30% of items.

This is a substantial change from the original 5-annotators × 4,500-item plan. Rationale:

1. **AMR-based verification is exactly the verifier we need for the RL paper anyway** — using it here is methodologically consistent.
2. **Anti-circularity** is preserved because LLM-judge verifiers (V2, V3) are independent of our AMR pipeline and the report breaks down equivalence rates per-verifier.
3. **Cost reduction**: ~1,750 items × 20% disagreement = ~350 items × 1-2 humans = a few days of work, not weeks.

## Setup

- **Annotators**: 2 senior annotators (one of whom can be the paper's first author for adjudication). One annotator suffices for clear cases; a second is needed only for items where the first is uncertain.
- **Items**: Output of `extensions/auto_verifier/run_auto_verify.py` → `review_queue.jsonl`.
- **Blinding**: Each item displays only the input, candidate, rule, and the per-verifier verdicts (so the annotator can see what disagreed). System identity (AMR-LDA vs which LLM) is blinded.
- **Calibration set**: A small (~50-item) random sample where humans label *every* item — used to validate that auto-verifier rates correlate with human rates (target Pearson r > 0.85).

## Three Evaluation Axes

For each (input sentence, rule, rewritten sentence) tuple, annotators score on three independent axes.

### Axis 1 — Logical Equivalence (Primary)

**Question**: Does the rewritten sentence have the same truth conditions as the original under the specified rule?

| Score | Label | Definition |
|---|---|---|
| 2 | EQUIVALENT | Identical truth conditions in every possible model. No information added, no information lost, no modal/aspectual shift. |
| 1 | NEAR_EQUIVALENT | Truth conditions match in typical cases but with a minor caveat (e.g., a hedge added, a slightly different connotation in edge cases). Note the caveat in free-text comment. |
| 0 | NOT_EQUIVALENT | Truth conditions differ in at least one model. Common causes: modal strength change, quantifier scope shift, added/removed exclusive-or, content drift, missing rule application. |
| -1 | UNGRAMMATICAL_OR_UNINTELLIGIBLE | Cannot judge equivalence because the output is not a well-formed sentence or makes no sense. |

**Tie-breaker convention**: If unsure between 2 and 1, prefer 1 (be strict). If unsure between 1 and 0, prefer 0.

### Axis 2 — Fluency

**Question**: Is the rewritten sentence natural and grammatically well-formed English?

| Score | Label | Definition |
|---|---|---|
| 3 | NATIVE_FLUENT | Reads as natural native English. No grammatical errors, no awkward phrasing. |
| 2 | ACCEPTABLE | Minor grammatical slip or slight awkwardness, but easily understood. |
| 1 | AWKWARD | Understandable but clearly machine-generated; tortured syntax or unnatural collocation. |
| 0 | UNGRAMMATICAL | Contains clear grammatical errors that impede comprehension. |

### Axis 3 — Information Preservation

**Question**: Is all information from the original sentence preserved (and no information added)?

| Score | Label | Definition |
|---|---|---|
| 2 | EXACT | All entities, relations, modifiers, and connectives from the original are accounted for in the output. No extra information. |
| 1 | PARTIAL | One minor element changed or omitted (e.g., a non-truth-conditional adverb dropped), or one minor addition (e.g., an extra hedge "perhaps"). |
| 0 | DRIFT | Significant content added or removed beyond what the rule requires. |

## Annotation Worksheet

For each item, annotators fill in:

```
item_id:              <e.g., S001_contraposition_gpt-4o>
input_sentence:       <verbatim>
rule:                 <one of 13 rules>
output_sentence:      <verbatim>
logical_equivalence:  <2 | 1 | 0 | -1>
fluency:              <3 | 2 | 1 | 0>
info_preservation:    <2 | 1 | 0>
failure_mode_tag:     <optional, choose from list below or write own>
comment:              <free text, ≤ 200 chars>
```

### Standard Failure Mode Tags (multi-select)

Use these tags to enable automatic aggregation of failure types:

- `LEX_ANTONYM_SUB` — Lexical antonym substituted for negation (e.g., "fails" for "does not pass")
- `MODAL_DOWNGRADE` — Modal strength weakened (e.g., "must" → "should")
- `MODAL_UPGRADE` — Modal strength strengthened (e.g., "may" → "must")
- `QUANTIFIER_SCOPE_CHANGE` — Quantifier scope reordered (∀∃ ↔ ∃∀)
- `XOR_INTRODUCED` — Exclusive-or added where original is inclusive ("but not both")
- `XOR_REMOVED` — Exclusive-or stripped where original is exclusive
- `NEGATION_MISCOUNT` — Number of negations wrong (e.g., dropped one of two)
- `RULE_NOT_APPLIED` — Output is essentially a paraphrase, the rule was not applied at all
- `RULE_MISAPPLIED` — Wrong rule applied (e.g., asked for contraposition, got converse)
- `EXTRA_HEDGE` — Hedge added ("perhaps", "likely", "appears to") that is not in original
- `TEMPORAL_DRIFT` — Temporal relation changed (before/after/during)
- `ASPECT_DRIFT` — Aspect category changed (activity → performance, etc.)
- `INFO_LOSS` — Truth-conditional information removed
- `INFO_ADDITION` — Truth-conditional information added
- `PASSIVE_ACTIVE_OK` — Voice change is appropriate inverse-relation marker (not a failure)
- `REFUSED` — LLM refused the task or output a meta-comment
- `OTHER` — Specify in comment

## Calibration Set (Human-vs-Auto Correlation)

Before running the full pipeline, randomly sample 50 items spanning all 13 rules and all 7 systems. Have 2 annotators label ALL of these items (not just the disagreement subset). Then:

1. Compute **Cohen's κ** between the two human annotators on this set. Target κ ≥ 0.7 on the equivalence axis.
2. Compute **Pearson correlation** between human majority labels and the auto-verifier consensus labels. Target r ≥ 0.85.
3. If r < 0.85, the auto-verifier thresholds or LLM-judge prompts need tuning — adjust and re-run.

The calibration set's purpose is to give the paper a defensible claim: "On 50 calibration items where humans labeled every output, our auto-verifier's consensus label agrees with human majority X% of the time (Cohen's κ = Y between auto and human)." That number alone defuses the most likely reviewer objection.

## Adjudication Protocol for Disagreement Items

For each item in `review_queue.jsonl`:
- The first annotator gives a verdict on the equivalence axis (2/1/0/-1).
- If the verdict is **clear** (2 or 0), it overrides the auto-verifier and we move on.
- If the verdict is **uncertain** (1, or annotator notes ambiguity in comment), a second annotator independently labels it. If they agree → use that label. If they disagree → adjudication by the paper's first author.

This two-tier scheme avoids exhaustive double-annotation while preserving high quality on the truly hard cases.

## Anti-Bias Measures (still required)

1. **Hidden system identity**: The auto-verifier already strips system labels before passing items to humans.
2. **Filler items**: 10% of the calibration set should be obviously equivalent or obviously non-equivalent gold pairs.
3. **Randomized order**: Items in `review_queue.jsonl` are shuffled before being shown.

## Output Format

Final annotation file (one JSON object per item):

```json
{
  "item_id": "S001_contraposition_gpt-4o",
  "input": "If Alice studies hard, then she passes the exam.",
  "rule": "contraposition",
  "output": "If Alice fails the exam, then she does not study hard.",
  "system_blinded_id": "System_C",
  "annotations": [
    {"annotator": "A1", "equivalence": 0, "fluency": 3, "info": 1, "tags": ["LEX_ANTONYM_SUB"], "comment": "fails != does not pass"},
    {"annotator": "A2", "equivalence": 0, "fluency": 3, "info": 1, "tags": ["LEX_ANTONYM_SUB"], "comment": ""},
    {"annotator": "A3", "equivalence": 1, "fluency": 3, "info": 2, "tags": [], "comment": "pragmatically equivalent"},
    {"annotator": "A4", "equivalence": 0, "fluency": 3, "info": 1, "tags": ["LEX_ANTONYM_SUB"], "comment": ""},
    {"annotator": "A5", "equivalence": 0, "fluency": 3, "info": 1, "tags": ["LEX_ANTONYM_SUB"], "comment": ""}
  ],
  "majority": {"equivalence": 0, "fluency": 3, "info": 1, "tags": ["LEX_ANTONYM_SUB"]},
  "needs_adjudication": false
}
```

## Reporting

The paper reports three tables built from `summary_by_system.json` and `summary_by_verifier.json`:

### Table A — Per-system equivalence rates (under consensus)
| System | Equivalence rate | Items pending human review |
|---|---|---|
| AMR-LDA (rule) | ... | ... |
| UMR-LDA (rule) | ... | ... |
| GPT-4o (rewriter) | ... | ... |
| Claude Opus 4.7 (rewriter) | ... | ... |
| ... | ... | ... |

### Table B — Anti-circularity: per-system rates broken down by *which* verifier
| System | V1 (AMR) | V2 (GPT-4-judge) | V3 (Claude-judge) | V4 (SMATCH) | Consensus |
|---|---|---|---|---|---|
| AMR-LDA | ... | ... | ... | ... | ... |
| best LLM | ... | ... | ... | ... | ... |

The key claim: AMR-LDA wins even under V2 + V3 (LLM judges that are *not* derived from our AMR pipeline). This is the structural defense against "you used AMR to judge AMR".

### Table C — Failure mode distribution
Per-rule breakdown of failure modes (LEX_ANTONYM_SUB, MODAL_DOWNGRADE, etc.) using the disagreement-subset human annotations.

### Headline numbers
"Across 13 equivalence rules and 50 test sentences (~1,750 evaluation items), AMR-LDA achieves **X%** logical equivalence under multi-verifier consensus, UMR-LDA achieves **Y%**, while the best LLM-as-rewriter (Z) achieves **W%**. The gap holds under each individual verifier (V1-V4), with statistically significant differences (p < 0.01, McNemar's test) on the following rules: ..."
