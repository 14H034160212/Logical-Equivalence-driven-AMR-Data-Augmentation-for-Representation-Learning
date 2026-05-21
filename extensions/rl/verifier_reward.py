"""Reward function for RL-trained logical-reasoning models, backed by the
AMR+UMR auto-verifier.

The verifier is naturally well-suited as an RL reward signal because:
  - It is DETERMINISTIC (same input → same score; no LLM-judge variance)
  - It returns STRUCTURED rationales (which rule fired, where the parity
    drift is) — useful for credit assignment in process supervision
  - It can score INTERMEDIATE reasoning steps (not just final answers)

Reward shape
------------
Per (input_sentence, candidate_sentence, rule_name) tuple:

    reward = w_amr * V1.equivalent(...) + w_llm * V2.equivalent(...)
           + w_selfcheck * 1_{passed_self_check}

Default weights are calibrated against the pilot study so that the
weighted-sum cleanly distinguishes:
  - Generator-noisy-but-logically-equivalent outputs (V1=1, V2=0)
  - Surface-natural-but-logically-wrong outputs    (V1=0, V2=1)
  - Both-pass / Both-fail extremes

Usage
-----
The class can be plugged into common RL frameworks (TRL, OpenRLHF, custom
GRPO loops). Two integration points:

    # 1. As a callable for synchronous reward computation
    reward_fn = VerifierReward(parser, llm_judge_model="gpt-4o-mini")
    r = reward_fn(input_sent, candidate_sent, rule_name="contraposition")

    # 2. As a batched scorer in a TRL-style trainer
    rewards = reward_fn.batch_score(prompts, completions, rules)

This module deliberately does NOT include the policy training loop itself
— that lives in user-chosen RL framework. We document the interface here
so the loop can drop us in as-is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger("verifier_reward")


@dataclass
class RewardConfig:
    """Default weights calibrated against the pilot study (run6)."""

    weight_v1_amr: float = 0.5
    weight_v2_llm: float = 0.3
    weight_self_check: float = 0.2
    # If V1 abstains (rule didn't fire on input), what reward should we give?
    # Setting this > 0 prevents the policy from being penalized for choosing
    # to abstain when it can't structurally verify.
    abstain_reward: float = 0.0
    # Mark a special TARGET — if the candidate output exactly matches a gold
    # rewrite, give a small bonus to encourage approaching the surface form.
    gold_match_bonus: float = 0.1


@dataclass
class RewardBreakdown:
    """For logging / introspection."""

    total: float
    v1_score: Optional[float]
    v2_score: Optional[float]
    self_check_score: Optional[float]
    gold_match: bool
    details: dict = field(default_factory=dict)


class VerifierReward:
    """Compose the auto-verifier components into an RL reward function."""

    def __init__(
        self,
        amr_parser,
        config: Optional[RewardConfig] = None,
        llm_judge_model: Optional[str] = "gpt-4o-mini",
        self_check_fn=None,
    ):
        from extensions.auto_verifier.amr_verifier import AMRVerifier
        from extensions.auto_verifier.llm_verifier import LLMVerifier
        from extensions.auto_verifier.types import Label

        self.cfg = config or RewardConfig()
        self.amr_verifier = AMRVerifier(parser=amr_parser, threshold=0.60)
        self.llm_verifier = None
        if llm_judge_model:
            try:
                self.llm_verifier = LLMVerifier(llm_judge_model)
            except Exception as e:
                log.warning("LLM judge init failed: %s", e)
        self.self_check_fn = self_check_fn
        self.Label = Label

    def __call__(
        self,
        input_sentence: str,
        candidate_sentence: str,
        rule_name: str,
        gold: Optional[str] = None,
    ) -> RewardBreakdown:
        v1 = self.amr_verifier.verify(input_sentence, candidate_sentence, rule_name)
        v1_score = 1.0 if v1.label == self.Label.EQUIVALENT else 0.0
        if v1.label == self.Label.PARSE_FAILED:
            v1_score = self.cfg.abstain_reward
        v2_score = None
        if self.llm_verifier:
            v2 = self.llm_verifier.verify(input_sentence, candidate_sentence, rule_name)
            v2_score = 1.0 if v2.label == self.Label.EQUIVALENT else 0.0
        sc_score = None
        if self.self_check_fn:
            try:
                sc_passed, _ = self.self_check_fn(input_sentence, candidate_sentence)
                sc_score = 1.0 if sc_passed else 0.0
            except Exception:
                sc_score = None
        gold_match = bool(gold and candidate_sentence.strip() == gold.strip())

        total = self.cfg.weight_v1_amr * v1_score
        if v2_score is not None:
            total += self.cfg.weight_v2_llm * v2_score
        if sc_score is not None:
            total += self.cfg.weight_self_check * sc_score
        if gold_match:
            total += self.cfg.gold_match_bonus
        return RewardBreakdown(
            total=total,
            v1_score=v1_score,
            v2_score=v2_score,
            self_check_score=sc_score,
            gold_match=gold_match,
            details={
                "v1_label": v1.label.value if hasattr(v1.label, "value") else str(v1.label),
                "v1_f1": v1.details.get("f1") if v1.details else None,
            },
        )

    def batch_score(
        self,
        prompts: List[str],
        completions: List[str],
        rules: List[str],
        golds: Optional[List[Optional[str]]] = None,
    ) -> List[RewardBreakdown]:
        if golds is None:
            golds = [None] * len(prompts)
        out: List[RewardBreakdown] = []
        for p, c, r, g in zip(prompts, completions, rules, golds):
            out.append(self(p, c, r, gold=g))
        return out
