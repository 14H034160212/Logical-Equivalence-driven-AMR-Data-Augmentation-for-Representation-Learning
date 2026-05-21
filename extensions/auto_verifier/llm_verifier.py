"""V2, V3: LLM-as-judge verifiers.

Asks a chat LLM whether two sentences are logically equivalent. Reuses the
prompt templates from extensions/pilot_study/prompts.py. We support multiple
LLM families so the consensus is not LLM-family-specific.
"""

from __future__ import annotations

import logging
from typing import Optional

from .types import Label, VerifierVerdict


log = logging.getLogger(__name__)


class LLMVerifier:
    """LLM-as-judge verifier. Pass model_name like 'gpt-4o' or 'claude-opus-4-7'."""

    def __init__(self, model_name: str, name_override: Optional[str] = None):
        # Late imports to avoid circular dependencies + keep optional installs.
        from ..pilot_study.prompts import MODELS, build_verify_prompt
        from ..pilot_study.run_llm_baseline import call_model

        if model_name not in MODELS:
            raise KeyError(f"Unknown model: {model_name}")

        self.cfg = MODELS[model_name]
        self._build = build_verify_prompt
        self._call = call_model
        self.NAME = name_override or f"llm_{model_name}"

    def verify(
        self, input_sentence: str, candidate_sentence: str, rule: str
    ) -> VerifierVerdict:
        # `rule` is included in the prompt context so the LLM knows what
        # transformation was supposedly applied — improves judgment quality.
        prompt = self._build(
            model=self.cfg.name,
            sentence_a=input_sentence,
            sentence_b=candidate_sentence,
        )
        # We append rule context to the user message of the last entry.
        if prompt and prompt[-1]["role"] == "user":
            prompt[-1]["content"] += f"\n\n(Context: candidate was generated using rule '{rule}'.)"

        try:
            response = self._call(prompt, self.cfg, dry_run=False).strip().upper()
        except Exception as e:
            return VerifierVerdict(
                verifier_name=self.NAME,
                label=Label.UNKNOWN,
                confidence=0.0,
                error=str(e),
            )

        # Normalize the response. The strict verifier prompt asks for one of
        # {EQUIVALENT, NOT_EQUIVALENT} but models may add extra tokens.
        if response.startswith("EQUIVALENT") or "EQUIVALENT" in response and "NOT" not in response:
            label = Label.EQUIVALENT
            confidence = 0.9
        elif "NOT_EQUIVALENT" in response or response.startswith("NOT"):
            label = Label.NOT_EQUIVALENT
            confidence = 0.9
        else:
            label = Label.UNKNOWN
            confidence = 0.3

        return VerifierVerdict(
            verifier_name=self.NAME,
            label=label,
            confidence=confidence,
            details={"raw_response": response[:200]},
        )
