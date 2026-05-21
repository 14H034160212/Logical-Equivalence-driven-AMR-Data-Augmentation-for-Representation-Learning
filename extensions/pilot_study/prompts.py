"""Prompt templates for the LLM-as-rewriter / LLM-as-parser / LLM-as-verifier pilot study.

Rule-specific templates for 13 logical equivalence rules (9 AMR-level + 3 UMR-level + tense)
across 5 LLMs (GPT-4, GPT-4o, Claude Opus 4.7, DeepSeek-V3, Llama-3-70B-Instruct).

Three modes:
- REWRITE: rewrite a sentence using a specified equivalence rule
- PARSE: produce an AMR-style penman graph from a sentence (for LLM-as-parser baseline)
- VERIFY: judge whether two sentences are logically equivalent

Usage:
    from prompts import build_rewrite_prompt
    msgs = build_rewrite_prompt(model="gpt-4o", rule="contraposition", sentence="If A, then B.")
"""

from dataclasses import dataclass
from typing import List, Dict


SYSTEM_PROMPT_BASE = (
    "You are an expert in formal logic and natural language semantics. "
    "Your task is to perform LOGICAL EQUIVALENCE TRANSFORMATIONS on natural language sentences. "
    "You MUST produce a sentence that has the IDENTICAL truth conditions as the input under "
    "classical propositional or first-order logic. Do NOT add hedges, do NOT change quantifier "
    "scope, do NOT substitute synonyms or antonyms, and do NOT introduce information not in the "
    "original sentence. Preserve modal strength and aspect exactly. "
    "Output ONLY the rewritten sentence, no explanation."
)


RULE_INSTRUCTIONS: Dict[str, str] = {
    "contraposition": (
        "Apply the CONTRAPOSITION law: (A -> B) is equivalent to (not B -> not A). "
        "Negate the consequent, negate the antecedent, and swap them. "
        "Use 'does not / is not / are not' for negation; do NOT use lexical antonyms "
        "(e.g., for 'pass' use 'does not pass', NOT 'fail')."
    ),
    "commutative": (
        "Apply the COMMUTATIVE law on conjunction (A and B = B and A) or disjunction (A or B = B or A). "
        "Swap the two conjuncts/disjuncts. Keep all other elements identical, including modifiers and tense."
    ),
    "implication": (
        "Apply the IMPLICATION transformation: (A -> B) is equivalent to (not A or B). "
        "Negate the antecedent, replace the implication with disjunction, keep the consequent unchanged. "
        "Use INCLUSIVE 'or', not exclusive — do NOT add 'but not both'."
    ),
    "double_negation": (
        "Apply DOUBLE NEGATION elimination or introduction: not(not A) = A. "
        "Remove two negations that scope over the same proposition, OR introduce two negations. "
        "Do NOT remove only one negation; do NOT change the modal strength."
    ),
    "de_morgan": (
        "Apply DE MORGAN's law: not(A and B) = (not A) or (not B); "
        "not(A or B) = (not A) and (not B). "
        "Distribute the outer negation across both operands and flip the connective. "
        "Use inclusive 'or'; do NOT introduce 'either ... or' with exclusive reading."
    ),
    "transitivity": (
        "Apply TRANSITIVITY: from (A -> B) and (B -> C), conclude (A -> C). "
        "Output the inferred sentence (A -> C). Do NOT include the intermediate predicate B "
        "in the output."
    ),
    "inverse_relation": (
        "Apply the INVERSE RELATION transformation: for a binary predicate p(X, Y) with an "
        "inverse p' (such as 'parent_of'/'child_of', 'owns'/'is_owned_by', 'sells'/'is_sold_by'), "
        "rewrite p(X, Y) as p'(Y, X). Preserve all temporal and modal information. "
        "If a passive voice form is the natural inverse (e.g., 'X discovered Y' -> 'Y was discovered by X'), "
        "use that. Do NOT switch between active and passive when an inverse predicate exists."
    ),
    "symmetric_asymmetric": (
        "Apply the SYMMETRIC RELATION transformation: for a symmetric predicate p (such as "
        "'is married to', 'is a sibling of'), p(X, Y) is equivalent to p(Y, X). "
        "Swap the arguments. If the predicate is ASYMMETRIC (such as 'is taller than', "
        "'is the parent of'), then p(X, Y) entails NOT p(Y, X), NOT p(Y, X) itself. "
        "Be explicit about which case applies."
    ),
    "predicate_implication": (
        "Apply PREDICATE IMPLICATION: if predicate p semantically entails predicate q "
        "(e.g., 'barks' entails 'makes a sound'; 'robin' entails 'bird'), rewrite p(X) as q(X). "
        "Use the IMMEDIATE hypernym, not a distant ancestor."
    ),
    "aspect_equivalence": (
        "Apply UMR ASPECT-LEVEL EQUIVALENCE: rewrite the sentence using a different surface form "
        "while preserving the UMR aspect category (state, activity, endeavor, performance, habitual, "
        "generic, process, inceptive). For example, 'Alice is studying' (activity) <=> 'Alice is "
        "engaged in studying' (activity, atelic continuous). Do NOT shift from one aspect category to "
        "another (e.g., do NOT turn an activity into a performance by adding telos)."
    ),
    "modal_strength_inversion": (
        "Apply UMR MODAL STRENGTH INVERSION: rewrite by inverting modal strength while preserving "
        "logical equivalence. UMR strengths are FullAff (must, will), PrtAff (may, might), NeutAff, "
        "NeutNeg, PrtNeg (need not), FullNeg (must not, cannot). "
        "Use the equivalence: FullAff(p) <=> FullNeg(not p); PrtAff(p) <=> PrtNeg(not p). "
        "Do NOT downgrade FullAff to PrtAff (e.g., do NOT rewrite 'must' as 'should')."
    ),
    "doc_level_temporal_transitivity": (
        "Apply UMR DOCUMENT-LEVEL TEMPORAL TRANSITIVITY: given event A :before event B and "
        "event B :before event C, infer A :before C. Output the inferred temporal claim as a "
        "single sentence."
    ),
    "tense_transformation": (
        "Apply TENSE TRANSFORMATION: rewrite the sentence in a different tense (past/present/future) "
        "while preserving the proposition. For UMR-aware versions, also preserve aspect: a past-tense "
        "performance must map to a past-tense performance, not to a past habitual."
    ),
}


FEW_SHOT_EXAMPLES: Dict[str, List[Dict[str, str]]] = {
    "contraposition": [
        {
            "input": "If it rains, the ground gets wet.",
            "output": "If the ground does not get wet, then it does not rain.",
        },
        {
            "input": "If you study, you will pass.",
            "output": "If you will not pass, then you do not study.",
        },
    ],
    "commutative": [
        {"input": "Alice is kind and Bob is clever.", "output": "Bob is clever and Alice is kind."},
        {
            "input": "Either the door is locked or the window is open.",
            "output": "Either the window is open or the door is locked.",
        },
    ],
    "implication": [
        {
            "input": "If you study, you will pass.",
            "output": "You do not study, or you will pass.",
        },
    ],
    "double_negation": [
        {
            "input": "It is not the case that Alice is not invited.",
            "output": "Alice is invited.",
        },
    ],
    "de_morgan": [
        {
            "input": "It is not the case that Alice is tall and Bob is short.",
            "output": "Alice is not tall or Bob is not short.",
        },
        {
            "input": "Neither the door is locked nor the window is open.",
            "output": "The door is not locked and the window is not open.",
        },
    ],
    "inverse_relation": [
        {
            "input": "Alice is the parent of Bob.",
            "output": "Bob is the child of Alice.",
        },
        {
            "input": "Marie Curie discovered radium.",
            "output": "Radium was discovered by Marie Curie.",
        },
    ],
    "symmetric_asymmetric": [
        {"input": "Alice is married to Bob.", "output": "Bob is married to Alice."},
    ],
    "predicate_implication": [
        {"input": "The dog barks.", "output": "The dog makes a sound."},
    ],
    "modal_strength_inversion": [
        {
            "input": "Alice must finish her homework.",
            "output": "It is not the case that Alice may skip her homework.",
        },
    ],
}


def build_rewrite_prompt(
    model: str, rule: str, sentence: str, use_few_shot: bool = True
) -> List[Dict[str, str]]:
    """Build a chat-format prompt for the LLM-as-rewriter task."""
    if rule not in RULE_INSTRUCTIONS:
        raise ValueError(f"Unknown rule: {rule}. Available: {list(RULE_INSTRUCTIONS.keys())}")

    system = SYSTEM_PROMPT_BASE
    user_parts: List[str] = [
        f"RULE: {rule}",
        f"INSTRUCTION: {RULE_INSTRUCTIONS[rule]}",
    ]
    if use_few_shot and rule in FEW_SHOT_EXAMPLES:
        user_parts.append("\nEXAMPLES:")
        for ex in FEW_SHOT_EXAMPLES[rule]:
            user_parts.append(f"Input: {ex['input']}")
            user_parts.append(f"Output: {ex['output']}")
        user_parts.append("")
    user_parts.append(f"Now transform this sentence:\nInput: {sentence}\nOutput:")
    user = "\n".join(user_parts)

    if model.startswith("claude"):
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


PARSE_SYSTEM_PROMPT = (
    "You are an expert in Abstract Meaning Representation (AMR). Given an English sentence, "
    "produce its AMR penman-format graph. Use PropBank framesets (e.g., 'work-01'), use ':ARG0', "
    "':ARG1', etc. for core arguments, ':polarity -' for negation, ':condition' for conditional, "
    "':time', ':location', ':mod' for adjuncts. Output ONLY the penman graph, no explanation."
)


def build_parse_prompt(model: str, sentence: str) -> List[Dict[str, str]]:
    """Build a chat-format prompt for the LLM-as-parser task (text -> AMR penman)."""
    user = (
        "Parse the following sentence into AMR penman format.\n\n"
        f"Sentence: {sentence}\n\nAMR:"
    )
    if model.startswith("claude"):
        return [{"role": "user", "content": f"{PARSE_SYSTEM_PROMPT}\n\n{user}"}]
    return [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


VERIFY_SYSTEM_PROMPT = (
    "You are an expert in formal logic. Judge whether two natural language sentences are "
    "LOGICALLY EQUIVALENT under classical propositional or first-order logic. Two sentences are "
    "logically equivalent if and only if they have the same truth value in every possible model. "
    "Be strict: differences in modal strength, quantifier scope, or pragmatic implicature that "
    "change truth conditions count as NOT equivalent. Output exactly one token: EQUIVALENT or "
    "NOT_EQUIVALENT."
)


def build_verify_prompt(model: str, sentence_a: str, sentence_b: str) -> List[Dict[str, str]]:
    """Build a chat-format prompt for the LLM-as-verifier task."""
    user = (
        f"Sentence A: {sentence_a}\n"
        f"Sentence B: {sentence_b}\n\n"
        "Are A and B logically equivalent? Answer EQUIVALENT or NOT_EQUIVALENT."
    )
    if model.startswith("claude"):
        return [{"role": "user", "content": f"{VERIFY_SYSTEM_PROMPT}\n\n{user}"}]
    return [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@dataclass
class ModelConfig:
    """Per-model client configuration."""

    name: str
    provider: str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 256


MODELS: Dict[str, ModelConfig] = {
    "gpt-4": ModelConfig("gpt-4", "openai", "gpt-4-0613"),
    "gpt-4o": ModelConfig("gpt-4o", "openai", "gpt-4o-2024-08-06"),
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", "openai", "gpt-4o-mini-2024-07-18"),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", "openai", "gpt-4-turbo-2024-04-09"),
    "claude-opus-4-7": ModelConfig("claude-opus-4-7", "anthropic", "claude-opus-4-7"),
    "deepseek-v3": ModelConfig("deepseek-v3", "deepseek", "deepseek-chat"),
    "llama-3-70b": ModelConfig(
        "llama-3-70b", "together", "meta-llama/Llama-3-70b-chat-hf"
    ),
}


if __name__ == "__main__":
    msgs = build_rewrite_prompt(
        model="gpt-4o",
        rule="contraposition",
        sentence="If Alice studies hard, then she passes the exam.",
    )
    for m in msgs:
        print(f"=== {m['role']} ===\n{m['content']}\n")
