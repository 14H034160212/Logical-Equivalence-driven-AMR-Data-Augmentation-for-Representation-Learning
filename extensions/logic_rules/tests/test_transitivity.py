"""Smoke test for the two-graph transitivity rule.

Run:
    PYTHONPATH=. /data/qbao775/miniconda3/envs/leamr/bin/python \\
        -m extensions.logic_rules.tests.test_transitivity
"""

import penman

from extensions.logic_rules import get_rule
from extensions.logic_rules.transitivity import TransitivityRule


def main():
    rule = get_rule("transitivity")
    assert isinstance(rule, TransitivityRule)

    # Sentence 1: "If Alice studies hard, then Bob is clever."
    # AMR: (be-clever :condition (study-hard))
    # Using simplified forms for clarity.
    g1 = penman.decode("""
        (c / clever-01
            :ARG1 (p1 / person :name (n1 / name :op1 "Bob"))
            :condition (s / study-01
                :ARG0 (p2 / person :name (n2 / name :op1 "Alice"))
                :ARG1-of (h / hard-02)))
    """)

    # Sentence 2: "If Bob is clever, then he wins."
    # AMR: (win :condition (be-clever))
    g2 = penman.decode("""
        (w / win-01
            :ARG0 (p3 / person :name (n3 / name :op1 "Bob"))
            :condition (c2 / clever-01
                :ARG1 p3))
    """)

    # The consequent of g1 ("Bob is clever") should match the antecedent of g2.
    match = rule.detect_two(g1, g2, match_threshold=0.2)
    print("MATCH:", match)
    assert match is not None, "transitivity didn't detect chain"
    print(f"  similarity: {match['similarity']:.3f}")

    # Apply
    merged = rule.apply_two_positive(g1, g2, match)
    assert merged is not None
    print()
    print("MERGED (Alice studies -> Bob wins):")
    print(penman.encode(merged))

    # Negative case: unrelated sentences
    g3 = penman.decode("""
        (r / rain-01
            :time (n / now))
    """)
    no_match = rule.detect_two(g1, g3, match_threshold=0.5)
    print()
    print("UNRELATED (should be None):", no_match)
    assert no_match is None, "transitivity should not fire on unrelated graphs"

    print()
    print("ALL TRANSITIVITY TESTS PASS")


if __name__ == "__main__":
    main()
