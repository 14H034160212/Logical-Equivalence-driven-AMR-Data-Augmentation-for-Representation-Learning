"""Smoke test: run the full pipeline end-to-end with MockParser + no LLM judges.

We hand-craft a tiny fixture so the test runs without amrlib / API keys.

Run:
    PYTHONPATH=. python -m extensions.auto_verifier.test_verifier_smoke
"""

from __future__ import annotations

from .amr_verifier import AMRVerifier, MockParser
from .consensus import aggregate, summary_by_verifier
from .smatch_verifier import SmatchVerifier


def main():
    # --- fixture: AMR penman strings keyed by sentence text ---
    # The mock parser returns these when asked to parse the corresponding sentence.
    table = {
        # Original: "If Alice is kind, then Bob is clever."
        "If Alice is kind, then Bob is clever.": (
            "(h / have-condition-91 "
            ":ARG1 (c / clever-01 :ARG1 (p1 / person :name (n1 / name :op1 \"Bob\"))) "
            ":ARG2 (k / kind-01 :ARG1 (p2 / person :name (n2 / name :op1 \"Alice\"))))"
        ),
        # GOOD contraposition (AMR-LDA-style): "If Bob is not clever, then Alice is not kind."
        "If Bob is not clever, then Alice is not kind.": (
            "(h / have-condition-91 "
            ":ARG1 (k / kind-01 :polarity - :ARG1 (p2 / person :name (n2 / name :op1 \"Alice\"))) "
            ":ARG2 (c / clever-01 :polarity - :ARG1 (p1 / person :name (n1 / name :op1 \"Bob\"))))"
        ),
        # BAD output (LLM-style failure): "If Bob is dumb, then Alice is mean."
        # Wrong lexical antonyms instead of polarity flip
        "If Bob is dumb, then Alice is mean.": (
            "(h / have-condition-91 "
            ":ARG1 (m / mean-01 :ARG1 (p2 / person :name (n2 / name :op1 \"Alice\"))) "
            ":ARG2 (d / dumb-01 :ARG1 (p1 / person :name (n1 / name :op1 \"Bob\"))))"
        ),
    }
    parser = MockParser(table)

    # --- run the AMR + SMATCH verifiers on two candidates ---
    rule = "contraposition"
    input_sent = "If Alice is kind, then Bob is clever."
    good_cand = "If Bob is not clever, then Alice is not kind."
    bad_cand = "If Bob is dumb, then Alice is mean."

    amr_v = AMRVerifier(parser=parser, threshold=0.85)
    sm_v = SmatchVerifier(parser=parser)

    print("=" * 60)
    print("GOOD CANDIDATE (rule-correct contraposition)")
    print("=" * 60)
    v1 = amr_v.verify(input_sent, good_cand, rule)
    v2 = sm_v.verify(input_sent, good_cand, rule)
    print(f"V1 [AMR]    label={v1.label.value:20s}  conf={v1.confidence:.2f}  score={v1.score}")
    print(f"V2 [SMATCH] label={v2.label.value:20s}  conf={v2.confidence:.2f}  score={v2.score}")
    cr1 = aggregate("test::good", input_sent, good_cand, rule, [v1, v2])
    print(f"CONSENSUS: {cr1.majority_label.value}  unanimous={cr1.unanimity}  review_needed={cr1.needs_human_review}")

    print()
    print("=" * 60)
    print("BAD CANDIDATE (lexical-antonym failure)")
    print("=" * 60)
    v1 = amr_v.verify(input_sent, bad_cand, rule)
    v2 = sm_v.verify(input_sent, bad_cand, rule)
    print(f"V1 [AMR]    label={v1.label.value:20s}  conf={v1.confidence:.2f}  score={v1.score}")
    print(f"V2 [SMATCH] label={v2.label.value:20s}  conf={v2.confidence:.2f}  score={v2.score}")
    cr2 = aggregate("test::bad", input_sent, bad_cand, rule, [v1, v2])
    print(f"CONSENSUS: {cr2.majority_label.value}  unanimous={cr2.unanimity}  review_needed={cr2.needs_human_review}")

    # --- assertions ---
    from .types import Label

    assert cr1.majority_label == Label.EQUIVALENT, (
        f"Good candidate should be EQUIVALENT, got {cr1.majority_label.value}"
    )
    assert cr2.majority_label == Label.NOT_EQUIVALENT, (
        f"Bad candidate should be NOT_EQUIVALENT, got {cr2.majority_label.value}"
    )
    print()
    print("ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
