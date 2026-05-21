"""Smoke tests for all registered LogicRule subclasses.

Run:
    PYTHONPATH=. /data/qbao775/miniconda3/envs/leamr/bin/python \\
        -m extensions.logic_rules.tests.test_all_rules
"""

from __future__ import annotations

import sys

import penman

from extensions.logic_rules import get_rule, rule_names


# (rule_name, input_penman, expect_positive_graph, expect_negative_graph)
CASES = [
    (
        "contraposition",
        "(h / have-condition-91 "
        ":ARG1 (c / clever-01 :ARG1 (p1 / person :name (n1 / name :op1 \"Bob\"))) "
        ":ARG2 (k / kind-01 :ARG1 (p2 / person :name (n2 / name :op1 \"Alice\"))))",
        True, True,
    ),
    (
        "commutative",
        "(a / and "
        ":op1 (k / kind-01 :ARG1 (p1 / person :name (n1 / name :op1 \"Alice\"))) "
        ":op2 (c / clever-01 :ARG1 (p2 / person :name (n2 / name :op1 \"Bob\"))))",
        True, True,
    ),
    (
        "implication",
        "(h / have-condition-91 "
        ":ARG1 (c / clever-01 :ARG1 (p1 / person)) "
        ":ARG2 (k / kind-01 :ARG1 (p2 / person)))",
        True, True,
    ),
    (
        "double_negation",
        "(k / kind-01 :ARG1 (p / person :name (n / name :op1 \"Alice\")))",
        True, True,
    ),
    (
        "de_morgan",
        "(a / and :polarity - "
        ":op1 (t / tall-01 :ARG1 (p1 / person :name (n1 / name :op1 \"Alice\"))) "
        ":op2 (s / short-01 :ARG1 (p2 / person :name (n2 / name :op1 \"Bob\"))))",
        True, True,
    ),
    (
        "inverse_relation",
        "(b / buy-01 "
        ":ARG0 (p1 / person :name (n1 / name :op1 \"Alice\")) "
        ":ARG1 (c / car) "
        ":ARG2 (p2 / person :name (n2 / name :op1 \"Bob\")))",
        True, True,
    ),
    (
        "symmetric",
        "(m / marry-01 "
        ":ARG0 (p1 / person :name (n1 / name :op1 \"Alice\")) "
        ":ARG1 (p2 / person :name (n2 / name :op1 \"Bob\")))",
        True, True,  # has negative via polarity toggle on op1
    ),
    (
        "asymmetric",
        "(p / parent-01 "
        ":ARG0 (a / person :name (n1 / name :op1 \"Alice\")) "
        ":ARG1 (b / person :name (n2 / name :op1 \"Bob\")))",
        True, True,
    ),
    (
        "predicate_implication",
        "(d / bark-01 :ARG0 (x / dog))",
        True, False,
    ),
    (
        "modal_strength_inversion",
        "(o / obligate-01 "
        ":ARG0 (p / person :name (n / name :op1 \"Alice\")) "
        ":ARG2 (f / finish-01 :ARG0 p :ARG1 (h / homework)))",
        True, True,
    ),
    (
        "aspect_equivalence",
        "(s / study-01 "
        ":ARG0 (p / person :name (n / name :op1 \"Alice\")) "
        ":duration (t / temporal))",
        True, True,
    ),
    (
        "doc_level_temporal_transitivity",
        "(t / take-off-01 "
        ":ARG0 (f / flight) "
        ":time (a / after :op1 (c / clear-01 :ARG1 (w / weather))))",
        True, True,
    ),
    (
        "tense_transformation",
        "(r / run-02 "
        ":ARG0 (p / person :name (n / name :op1 \"Alice\")) "
        ":ARG1 (m / marathon) "
        ":time (y / yesterday))",
        True, True,
    ),
]


def main():
    expected_rules = {
        # AMR-level
        "contraposition", "commutative", "implication", "double_negation",
        "de_morgan", "inverse_relation", "symmetric", "asymmetric",
        "predicate_implication", "transitivity",
        # UMR-level (AMR-layer approximations)
        "modal_strength_inversion", "aspect_equivalence",
        "doc_level_temporal_transitivity", "tense_transformation",
    }
    actual = set(rule_names())
    assert actual == expected_rules, (
        f"registered rules mismatch: missing {expected_rules - actual}, extra {actual - expected_rules}"
    )
    print(f"[OK] {len(actual)} rules registered: {sorted(actual)}")

    failed = []
    for name, penman_str, expect_pos, expect_neg in CASES:
        g = penman.decode(penman_str)
        rule = get_rule(name)
        results = rule.apply(g)
        if not results:
            failed.append((name, "no results"))
            print(f"[FAIL] {name}: rule did not fire")
            continue
        r0 = results[0]
        ok_pos = (r0.positive_graph is not None) == expect_pos
        ok_neg = (r0.negative_graph is not None) == expect_neg
        if ok_pos and ok_neg:
            print(f"[OK] {name}: pos={r0.positive_graph is not None} neg={r0.negative_graph is not None}")
        else:
            failed.append((name, f"pos={r0.positive_graph is not None} neg={r0.negative_graph is not None} (expected pos={expect_pos} neg={expect_neg})"))
            print(f"[FAIL] {name}: pos={r0.positive_graph is not None} neg={r0.negative_graph is not None}")

    # Transitivity is a stub; just check detect returns empty
    matches = get_rule("transitivity").detect(penman.decode("(k / kind-01 :ARG1 (p / person))"))
    if matches == []:
        print(f"[OK] transitivity: stub returns no matches (expected)")
    else:
        failed.append(("transitivity", f"expected 0 matches, got {len(matches)}"))

    print()
    if failed:
        print(f"FAILURES: {len(failed)}")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print(f"ALL {len(CASES)} ACTIVE RULES + 1 STUB PASS")


if __name__ == "__main__":
    main()
