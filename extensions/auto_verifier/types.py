"""Shared types for the auto-verifier pipeline.

A Verifier maps (input_sentence, candidate_sentence, rule_name) to a
VerifierVerdict. Multiple verifiers' verdicts are merged by ConsensusEngine
into a ConsensusResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Label(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    UNGRAMMATICAL = "UNGRAMMATICAL"
    PARSE_FAILED = "PARSE_FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class VerifierVerdict:
    """A single verifier's judgment on one (input, candidate, rule) tuple."""

    verifier_name: str
    label: Label
    confidence: float = 1.0  # in [0, 1]; 1.0 = high confidence
    score: Optional[float] = None  # optional continuous score (e.g., SMATCH F1)
    details: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ConsensusResult:
    """Aggregate of multiple verifier verdicts."""

    item_id: str
    input_sentence: str
    candidate_sentence: str
    rule: str
    verdicts: List[VerifierVerdict]
    majority_label: Label
    unanimity: bool  # True iff all verifiers agree on the majority label
    needs_human_review: bool  # True iff non-unanimous
    confidence: float  # aggregate confidence in majority_label
