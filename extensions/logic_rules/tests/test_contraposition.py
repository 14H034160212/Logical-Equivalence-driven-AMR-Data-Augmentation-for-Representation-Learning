"""Smoke tests for the refactored ContrapositionRule.

Run:
    cd extensions
    python -m pytest logic_rules/tests/ -v
"""

import penman
import pytest

from extensions.logic_rules import get_rule


@pytest.fixture
def simple_implication_graph():
    """AMR for: 'If Alice is kind, then Bob is clever.'

    Constructed by hand to mirror the AMR-LDA paper's running example.
    """
    g = penman.decode(
        """
        (h / have-condition-91
           :ARG1 (c / clever-01
                    :ARG1 (p1 / person :name (n1 / name :op1 "Bob")))
           :ARG2 (k / kind-01
                    :ARG1 (p2 / person :name (n2 / name :op1 "Alice"))))
        """
    )
    return g


def test_detect_finds_one_match(simple_implication_graph):
    rule = get_rule("contraposition")
    matches = rule.detect(simple_implication_graph)
    assert len(matches) == 1
    assert matches[0].anchor == "h"


def test_apply_positive_produces_logically_equivalent_graph(simple_implication_graph):
    rule = get_rule("contraposition")
    results = rule.apply(simple_implication_graph)
    assert len(results) == 1
    pos = results[0].positive_graph
    assert pos is not None
    decoded = penman.decode(pos)
    # The two :polarity - triples should appear (negation introduced on both sides)
    polarity_triples = [t for t in decoded.triples if t[1] == ":polarity"]
    assert len(polarity_triples) == 2


def test_apply_negative_is_different_from_positive(simple_implication_graph):
    rule = get_rule("contraposition")
    results = rule.apply(simple_implication_graph)
    pos = results[0].positive_graph
    neg = results[0].negative_graph
    assert pos is not None and neg is not None
    assert pos != neg


def test_legacy_batch_api_returns_pos_neg_pairs(simple_implication_graph):
    rule = get_rule("contraposition")
    graph_str = penman.encode(simple_implication_graph)
    rets, labels, meta = rule.transform_batch(
        graphs=[graph_str],
        sentence_list=["If Alice is kind, then Bob is clever."],
        logic_word_list=["if,then"],
    )
    assert len(rets) >= 1
    assert 1 in labels  # positive sample present
    if 0 in labels:
        # if a negative was produced, it should also carry the sentence metadata
        assert len(meta) == len(rets)


def test_rule_is_registered():
    from extensions.logic_rules import rule_names

    assert "contraposition" in rule_names()


def test_no_match_on_unrelated_graph():
    rule = get_rule("contraposition")
    g = penman.decode("(s / smile-01 :ARG0 (p / person :name (n / name :op1 \"Alice\")))")
    assert rule.detect(g) == []
    assert rule.apply(g) == []
