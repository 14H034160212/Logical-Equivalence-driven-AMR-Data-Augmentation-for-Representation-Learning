"""Run the LLM-as-rewriter / parser / verifier pilot study across multiple providers.

Usage:
    # Run all 50 sentences x 5 models x rewrite mode (default):
    python run_llm_baseline.py --mode rewrite --models gpt-4o claude-opus-4-7 deepseek-v3

    # Just one model, one rule, dry-run (prints prompts, no API call):
    python run_llm_baseline.py --mode rewrite --models gpt-4o --rules contraposition --dry-run

    # Verifier mode (compare AMR-LDA outputs vs LLM judgments):
    python run_llm_baseline.py --mode verify --pairs amr_lda_outputs.json

Provider API keys are read from environment:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, TOGETHER_API_KEY

Outputs:
    results/<timestamp>/<mode>/<model>.jsonl   one record per (sentence_id, rule)

Design notes:
- This is a SKELETON. The API call functions raise NotImplementedError unless the relevant
  client library is installed and an API key is present. The user is expected to install
  `openai`, `anthropic`, etc. and fill in any project-specific routing.
- We use temperature=0.0 for determinism; results are still subject to provider drift
  across model versions.
- Each provider has its own rate limiter and retry policy stub; tune as needed for your
  account tier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    # Preferred: as a package member from the repo root
    from extensions.pilot_study.prompts import (
        MODELS,
        ModelConfig,
        build_parse_prompt,
        build_rewrite_prompt,
        build_verify_prompt,
    )
except ImportError:
    # Fallback: when run directly from extensions/pilot_study/
    from prompts import (
        MODELS,
        ModelConfig,
        build_parse_prompt,
        build_rewrite_prompt,
        build_verify_prompt,
    )


PILOT_DIR = Path(__file__).parent
SENTENCES_PATH = PILOT_DIR / "test_sentences.json"
RESULTS_DIR = PILOT_DIR / "results"


# ---------------------------------------------------------------------------
# Provider call dispatchers — stubs to be filled in by the user with their
# preferred client library + auth approach. The signature is fixed.
# ---------------------------------------------------------------------------


def call_openai(messages: List[Dict[str, str]], cfg: ModelConfig) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai>=1.0")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=cfg.model_id,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(messages: List[Dict[str, str]], cfg: ModelConfig) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")
    client = Anthropic(api_key=api_key)
    user_msg = messages[0]["content"] if messages else ""
    resp = client.messages.create(
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()


def call_deepseek(messages: List[Dict[str, str]], cfg: ModelConfig) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai>=1.0 (DeepSeek is OpenAI-compatible)")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model=cfg.model_id,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_together(messages: List[Dict[str, str]], cfg: ModelConfig) -> str:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    try:
        from together import Together
    except ImportError:
        raise RuntimeError("pip install together")
    client = Together(api_key=api_key)
    resp = client.chat.completions.create(
        model=cfg.model_id,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content.strip()


PROVIDER_DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "deepseek": call_deepseek,
    "together": call_together,
}


def call_model(messages: List[Dict[str, str]], cfg: ModelConfig, dry_run: bool = False) -> str:
    if dry_run:
        return "[DRY_RUN] would call " + cfg.name
    dispatcher = PROVIDER_DISPATCH.get(cfg.provider)
    if dispatcher is None:
        raise RuntimeError(f"Unknown provider: {cfg.provider}")
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            return dispatcher(messages, cfg)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"3 retries failed for {cfg.name}: {last_err}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sentences() -> List[dict]:
    with open(SENTENCES_PATH) as f:
        return json.load(f)["sentences"]


def filter_sentences(
    sentences: List[dict], rules: Optional[List[str]], ids: Optional[List[str]]
) -> Iterable[tuple]:
    """Yield (sentence_dict, rule_name) for each (sentence, applicable_rule) pair."""
    for s in sentences:
        if ids and s["id"] not in ids:
            continue
        for r in s["applicable_rules"]:
            if rules and r not in rules:
                continue
            yield s, r


# ---------------------------------------------------------------------------
# Per-mode runners
# ---------------------------------------------------------------------------


def run_rewrite(
    sentences: List[dict],
    model_name: str,
    rules: Optional[List[str]],
    ids: Optional[List[str]],
    out_path: Path,
    dry_run: bool,
) -> None:
    cfg = MODELS[model_name]
    n = 0
    with open(out_path, "w") as fout:
        for s, r in filter_sentences(sentences, rules, ids):
            prompt = build_rewrite_prompt(model=cfg.name, rule=r, sentence=s["text"])
            try:
                out = call_model(prompt, cfg, dry_run=dry_run)
            except Exception as e:
                out = f"[ERROR] {e}"
            record = {
                "sentence_id": s["id"],
                "input": s["text"],
                "rule": r,
                "model": cfg.name,
                "output": out,
                "gold": s.get("gold_outputs", {}).get(r),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"  {model_name}: {n} items written", file=sys.stderr)
    print(f"[{model_name}] wrote {n} records -> {out_path}")


def run_parse(
    sentences: List[dict],
    model_name: str,
    ids: Optional[List[str]],
    out_path: Path,
    dry_run: bool,
) -> None:
    cfg = MODELS[model_name]
    n = 0
    with open(out_path, "w") as fout:
        for s in sentences:
            if ids and s["id"] not in ids:
                continue
            prompt = build_parse_prompt(model=cfg.name, sentence=s["text"])
            try:
                out = call_model(prompt, cfg, dry_run=dry_run)
            except Exception as e:
                out = f"[ERROR] {e}"
            record = {
                "sentence_id": s["id"],
                "input": s["text"],
                "model": cfg.name,
                "amr_penman": out,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{model_name}] parsed {n} sentences -> {out_path}")


def run_verify(
    pairs_path: str,
    model_name: str,
    out_path: Path,
    dry_run: bool,
) -> None:
    cfg = MODELS[model_name]
    with open(pairs_path) as f:
        pairs = json.load(f)
    n = 0
    with open(out_path, "w") as fout:
        for p in pairs:
            prompt = build_verify_prompt(
                model=cfg.name, sentence_a=p["sentence_a"], sentence_b=p["sentence_b"]
            )
            try:
                out = call_model(prompt, cfg, dry_run=dry_run)
            except Exception as e:
                out = f"[ERROR] {e}"
            record = {
                **p,
                "model": cfg.name,
                "verdict": out,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"[{model_name}] verified {n} pairs -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rewrite", "parse", "verify"], default="rewrite")
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "claude-opus-4-7", "deepseek-v3"])
    ap.add_argument("--rules", nargs="*", default=None, help="filter to specific rules")
    ap.add_argument("--ids", nargs="*", default=None, help="filter to specific sentence ids")
    ap.add_argument("--pairs", default=None, help="for verify mode: JSON list of pairs")
    ap.add_argument("--dry-run", action="store_true", help="print prompts only, no API calls")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / ts / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing results to {out_dir}", file=sys.stderr)

    sentences = load_sentences()

    for m in args.models:
        if m not in MODELS:
            print(f"Skipping unknown model: {m}", file=sys.stderr)
            continue
        out_path = out_dir / f"{m}.jsonl"
        if args.mode == "rewrite":
            run_rewrite(sentences, m, args.rules, args.ids, out_path, args.dry_run)
        elif args.mode == "parse":
            run_parse(sentences, m, args.ids, out_path, args.dry_run)
        elif args.mode == "verify":
            if not args.pairs:
                raise SystemExit("verify mode requires --pairs <path>")
            run_verify(args.pairs, m, out_path, args.dry_run)


if __name__ == "__main__":
    main()
