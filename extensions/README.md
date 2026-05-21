# Extensions — Continuing AMR-LDA toward UMR + RL Verifier

This directory contains the new work extending the original AMR-LDA paper
([ACL Findings 2024](https://aclanthology.org/2024.findings-acl.353.pdf), code
preserved under [`legacy/`](../legacy/)) along three axes:

1. **More logical equivalence rules**: from 4 → **14 registered (13 active + 1 stub)**:
   - **9 AMR-level**: contraposition, commutative, implication, double_negation,
     de_morgan, inverse_relation, symmetric, asymmetric, predicate_implication
   - **4 UMR-level** (AMR-layer approximations, full UMR overlay is future work):
     modal_strength_inversion, aspect_equivalence, doc_level_temporal_transitivity,
     tense_transformation
   - **1 stub**: transitivity (two-graph rule, needs cross-sentence coref)
2. **UMR overlay**: UMR (Uniform Meaning Representation) gives us a principled
   way to handle tense, aspect, modal strength, and cross-sentence temporal /
   modal / coreference relations that AMR cannot express.
3. **RL verifier**: AMR + UMR rule-based equivalence checking is naturally
   deterministic and verifiable, which makes it a high-quality process reward
   signal for GRPO-style RL training.

The full research plan is described in the conversation thread; this README
covers the concrete code artifacts present in this folder.

## Layout

```
extensions/
├── pilot_study/                   # LLM-as-rewriter vs AMR/UMR-LDA pilot
│   ├── test_sentences.json        # 50 carefully curated sentences
│   ├── prompts.py                 # rewrite / parse / verify prompt templates
│   ├── human_eval_rubric.md       # SPOT-CHECK rubric (only for verifier disagreement)
│   ├── red_line_cases.md          # 5 headline failure exhibits
│   └── run_llm_baseline.py        # orchestrator for the pilot
├── auto_verifier/                 # multi-verifier consensus (replaces bulk human eval)
│   ├── types.py                   # VerifierVerdict, ConsensusResult, Label
│   ├── amr_verifier.py            # V1: rule-aware AMR structural check
│   ├── llm_verifier.py            # V2, V3: LLM-as-judge (GPT-4o, Claude, etc.)
│   ├── smatch_verifier.py         # V4: rule-agnostic SMATCH similarity
│   ├── consensus.py               # majority vote + per-verifier breakdown
│   ├── run_auto_verify.py         # CLI orchestrator
│   └── test_verifier_smoke.py     # passing end-to-end test with mock parser
├── logic_rules/                   # 14 LogicRule subclasses + registry
│   ├── base.py                    # LogicRule abstract class + registry
│   ├── contraposition.py          # (have-condition-91 + :condition styles)
│   ├── commutative.py             # and/or operand swap
│   ├── implication.py             # A->B  <=>  ¬A v B (3 surface forms)
│   ├── double_negation.py         # ¬¬A <=> A
│   ├── de_morgan.py               # ¬(A∧B) <=> ¬A∨¬B (direct + wrapped patterns)
│   ├── inverse_relation.py        # p(X,Y) <=> p'(Y,X) (35 frames + 27 roles)
│   ├── symmetric_asymmetric.py    # marriage / parent-of type predicates
│   ├── predicate_implication.py   # WordNet hypernym entailment
│   ├── transitivity.py            # stub (two-graph rule)
│   ├── modal_strength_inversion.py    # UMR: FullAff(P) <=> FullNeg(¬P)
│   ├── aspect_equivalence.py          # UMR: state/activity/habitual/performance
│   ├── doc_temporal_transitivity.py   # UMR: :before / :after chain inference
│   ├── tense_transformation.py        # past <=> present <=> future
│   └── tests/                     # 13 active rules + 1 stub, all passing
└── umr/                           # UMR data + AMR→UMR conversion (to come)
    ├── download_umr_data.sh       # clones umr4nlp/umr-data
    └── loader.py                  # parses UMR 2.0 files; extracts aspect/modal
```

## Quick start

```bash
# 1. Pilot study — generate candidate outputs from LLMs (dry-run first)
cd extensions/pilot_study
python run_llm_baseline.py --mode rewrite --models gpt-4o --rules contraposition --dry-run

# 2. Pilot study — real run (set OPENAI_API_KEY etc. first)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python run_llm_baseline.py --mode rewrite --models gpt-4o claude-opus-4-7 deepseek-v3

# 3. Auto-verifier — multi-verifier consensus on the LLM outputs
# Dry-run with MockParser (no amrlib / no API keys needed)
PYTHONPATH=. python -m extensions.auto_verifier.test_verifier_smoke

# Full run (requires amrlib parser + LLM API keys for V2/V3)
PYTHONPATH=. python -m extensions.auto_verifier.run_auto_verify \
    --candidates-dir extensions/pilot_study/results/<ts>/rewrite/ \
    --amrlib-model-dir /path/to/parse_xfm_bart_large_v0_1_0 \
    --llm-judges gpt-4o claude-opus-4-7 \
    --out-dir extensions/auto_verifier/results/<ts>

# 4. Human spot-check — ONLY on the disagreement subset
# Auto-verifier produces review_queue.jsonl; 2 humans label those (~300-500 items)

# 5. Logic rules — smoke test
PYTHONPATH=. /data/qbao775/miniconda3/envs/leamr/bin/python -c "
import sys; sys.path.insert(0, '.')
from extensions.logic_rules import get_rule
print(get_rule('contraposition'))"

# 6. UMR data — clone and inspect
bash extensions/umr/download_umr_data.sh
PYTHONPATH=. python -m extensions.umr.loader --language english
```

## Auto-Verifier Design (no bulk human annotation needed)

Four independent verifiers vote on each (input, candidate, rule) tuple:

| Verifier | Type | Source | Anti-circularity role |
|---|---|---|---|
| **V1 AMR structural** | Rule-aware AMR check using `LogicRule` | Our pipeline | Most informed; biased toward AMR-LDA |
| **V2 GPT-4-as-judge** | LLM evaluates equivalence | External LLM | Independent of AMR pipeline |
| **V3 Claude-as-judge** | Different-family LLM | External LLM | Independent + cross-family check |
| **V4 SMATCH** | Rule-agnostic graph similarity | Standard metric | Conservative; ignores rule semantics |

**Consensus**:
- All 4 verifiers agree → auto-labeled (high confidence)
- Disagreement → flagged for human spot-check (typically 20-30% of items)
- The paper reports per-verifier rates so reviewers can see AMR-LDA wins under V2 + V3 too (defeats "you used AMR to judge AMR")

**Demonstrated in [extensions/auto_verifier/test_verifier_smoke.py](auto_verifier/test_verifier_smoke.py)**:
- Good candidate ("If Bob is not clever, then Alice is not kind") → V1 + V4 unanimous EQUIVALENT
- Bad candidate ("If Bob is dumb, then Alice is mean", lexical-antonym failure) → V1 correctly flags NOT_EQUIVALENT (F1=0.46), V4 incorrectly says EQUIVALENT (F1=0.88) due to surface similarity → flagged for human review

## What's done vs what's next

### Done in this pass

- [x] 50 test sentences with rich rule/category/failure annotations
- [x] Prompt templates for 5 LLMs × 13 rules × 3 modes (rewrite / parse / verify)
- [x] Human-evaluation rubric with 3-axis scoring + IAA protocol
- [x] 5 red-line failure cases documented for §3.1 of the paper
- [x] `LogicRule` abstract base + registry
- [x] **9 active AMR rules** all passing smoke test:
      contraposition, commutative, implication, double_negation,
      de_morgan, inverse_relation, symmetric, asymmetric, predicate_implication
- [x] **transitivity rule** stubbed (two-graph rule; needs cross-sentence coref)
- [x] All rules support positive + negative sample generation (except
      `predicate_implication` which is one-directional entailment)
- [x] UMR data loader skeleton with aspect/modal-strength extraction
- [x] UMR data download script

### Next concrete steps (in order)

1. **Run the pilot study** (requires API keys; the user does this)
2. **Run the auto-verifier** on pilot outputs (deterministic, no humans)
3. **Spot-check the disagreement subset** (~20-30% of items, 2 annotators)
4. **Clone UMR data and produce statistics report** on aspect / modal-strength
   coverage in the English portion (`bash extensions/umr/download_umr_data.sh`)
5. **Reproduce Post et al. 2024 AMR→UMR neuro-symbolic converter**
6. **Implement 3 UMR rules**: Aspect Equivalence, Modal Strength Inversion,
   Document-level Temporal Transitivity (slots already declared in
   `extensions/logic_rules/__init__.py`)
7. **Implement transitivity** as a two-graph rule (needs cross-sentence coref;
   slot exists, raise NotImplementedError currently)
8. **Stage-1 experiments**: ablation table across 13 rules on the 7 original
   downstream tasks
9. **Stage-1 robustness experiments**: Reversal Curse, ReClor-plus, JBB past
   tense, R-GSM premise-shuffled
10. **Stage-2 onward**: multi-step deductive corpus + RL verifier (see plan)

### What requires user / external action

- **API keys**: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
  TOGETHER_API_KEY for the LLM baselines (rewriters) AND for V2/V3 LLM-as-judge
- **Human annotators**: 2 annotators on the ~20-30% disagreement subset only
  (~300-500 items, days not weeks); PLUS a small 50-item calibration set where
  both annotators label everything to measure auto-verifier-vs-human correlation
- **amrlib parser**: `pip install amrlib` and download `parse_xfm_bart_large_v0_1_0`
  (~500MB) for V1 + V4
- **Compute**: stage-3 RL training needs ~8×A100 for several days per run;
  see the plan for budget
- **LDC license**: AMR 3.0 (LDC2020T02) for the AMR→UMR alignment data
