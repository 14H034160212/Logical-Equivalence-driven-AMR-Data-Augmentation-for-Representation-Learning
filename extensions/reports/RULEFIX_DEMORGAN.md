# De Morgan-aware contraposition closes the conjunctive-antecedent gap

After [v4 T5 fine-tune](T5_FT_RECOVERY.md) left 6 of 15 known
polarity-flips unrecovered, an inspection showed that 3 of the 6 were
**not generator errors** — the v4 T5 produced correct natural English,
but the rule-applied AMR had fewer `:polarity -` edges than the
distributed-De Morgan form T5 was generating, so the self-consistency
check falsely flagged them.

## Root cause

For `P → Q` where `P` is a conjunction `A ∧ B`:

  P → Q ≡ ¬Q → ¬P
        ≡ ¬Q → ¬(A ∧ B)
        ≡ ¬Q → (¬A ∨ ¬B)    [De Morgan]

The old [`extensions/logic_rules/contraposition.py`](../logic_rules/contraposition.py)
toggled `:polarity -` on the `and` node directly, producing a rule-applied
AMR with **outer** negation. The T5wtense decoder almost always renders
this in the **distributed** form ("not A or not B" with two polarities),
which made the polarity-count parity check reject the (correct) output.

Three cases affected: S008 (medication+instructions), S028 (vulnerability
+patch), S045 (passenger+crew). One related case was already broken in
v4: S014 (de_morgan, antecedent already negated and conjoined).

## Fix

[`extensions/logic_rules/base.py`](../logic_rules/base.py) gains
`negate_with_demorgan(g, node)`:

```python
def negate_with_demorgan(g, node):
    concept = find_instance_target(g, node)
    if concept == "and":
        replace_instance(g, node, "or")
        for s, role, t in list(g.triples):
            if s == node and role.startswith(":op"):
                negate_with_demorgan(g, t)
    elif concept == "or":
        replace_instance(g, node, "and")
        for s, role, t in list(g.triples):
            if s == node and role.startswith(":op"):
                negate_with_demorgan(g, t)
    else:
        toggle_polarity_neg(g, node)
```

`contraposition.apply_positive` now calls `negate_with_demorgan` instead
of `toggle_polarity_neg` for both the antecedent and consequent (both
condition styles). Atomic predicates fall through to the same
toggle_polarity_neg behaviour, so simple cases are unchanged.

## Result — full 49-sentence pilot

| | stock T5 | v4 T5 | **v4 T5 + rule fix** | Δ stock→rulefix |
|---|---|---|---|---|
| Total ok | 62/90 | 71/90 | **74/90** | +12 items |
| Pass rate | 68.9% | 78.9% | **82.2%** | **+13.3 pp** |

Recovered vs v4: 3 (S008, S028, S045). Regressed vs v4: 0.

**Contraposition specifically: 8/15 stock → 12/15 v4 → 15/15 rulefix (perfect).**

## Result — held-out PARARULE-Plus Depth5 (60-sentence shard)

| | stock T5 | v4 T5 | **v4 T5 + rule fix** | Δ stock→rulefix |
|---|---|---|---|---|
| Pass rate | 70.6% (101/143) | 72.0% (103/143) | **73.4%** (105/143) | **+2.8 pp** |
| contraposition | 22/37 | 23/37 | **25/37** | +3 items |
| double_negation | 49/60 | 55/60 | 55/60 | unchanged |
| implication | 21/37 | 18/37 | 18/37 | (still regressed) |
| commutative | 9/9 | 7/9 | 7/9 | (unchanged regression) |

The rule fix lifts held-out contraposition by **+3 items**, confirming
this is a general rule-logic fix, not pilot-specific.

## Reproducing

```bash
# 1. Rule fix is in commit 3dc22ec — just check out main.

# 2. Re-run pilot self-check with v4 T5 + the fixed rule
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.generate_amr_lda \
            --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
            --gen-model extensions/pilot_study/ft_t5wtense_v4 \
            --out extensions/pilot_study/results/ft_t5_recovery/pilot_full_v4_rulefix.jsonl

# 3. And the held-out shard
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.generate_amr_lda \
            --parse-model amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
            --gen-model extensions/pilot_study/ft_t5wtense_v4 \
            --test-sentences extensions/pilot_study/heldout_pararule_depth5.json \
            --out extensions/pilot_study/results/ft_t5_recovery/heldout_v4_rulefix.jsonl
```

JSON aggregates: [rulefix_pilot_summary.json](rulefix_pilot_summary.json),
[rulefix_heldout_summary.json](rulefix_heldout_summary.json).
