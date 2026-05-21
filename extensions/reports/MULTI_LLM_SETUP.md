# Adding More LLM Baselines (Claude / DeepSeek / Llama-3-70B)

The pilot study currently has data for `gpt-4o` and `gpt-4o-mini` only
(because that's what the OpenAI key in the session covered). The
infrastructure already supports four more providers — you just need to
add the API keys and re-run the pipeline.

## What's already wired up

[extensions/pilot_study/prompts.py](../pilot_study/prompts.py) declares 7
models with provider dispatch:

```python
MODELS = {
    "gpt-4":           ModelConfig("gpt-4",           "openai",    "gpt-4-0613"),
    "gpt-4o":          ModelConfig("gpt-4o",          "openai",    "gpt-4o-2024-08-06"),
    "gpt-4o-mini":     ModelConfig("gpt-4o-mini",     "openai",    "gpt-4o-mini-2024-07-18"),
    "gpt-4-turbo":     ModelConfig("gpt-4-turbo",     "openai",    "gpt-4-turbo-2024-04-09"),
    "claude-opus-4-7": ModelConfig("claude-opus-4-7", "anthropic", "claude-opus-4-7"),
    "deepseek-v3":     ModelConfig("deepseek-v3",     "deepseek",  "deepseek-chat"),
    "llama-3-70b":     ModelConfig("llama-3-70b",     "together",  "meta-llama/Llama-3-70b-chat-hf"),
}
```

The provider dispatchers in
[run_llm_baseline.py](../pilot_study/run_llm_baseline.py) are:

| Provider | Client lib | Env var |
|---|---|---|
| `openai` | `openai>=1.0` | `OPENAI_API_KEY` |
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| `deepseek` | `openai>=1.0` (OpenAI-compatible base URL) | `DEEPSEEK_API_KEY` |
| `together` | `together` | `TOGETHER_API_KEY` |

## Step-by-step: adding Claude

```bash
# 1. Install the Anthropic Python SDK
pip install anthropic

# 2. Set your key (NEVER commit it — .env / .gitignore already covers this)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the pilot (rewrite mode)
cd /path/to/repo
PYTHONPATH=. python -m extensions.pilot_study.run_llm_baseline \
    --mode rewrite \
    --models claude-opus-4-7 \
    --out-dir extensions/pilot_study/results/claude_run

# 4. Add Claude as V3 LLM-judge in the auto-verifier
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
  ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  OPENAI_API_KEY=$OPENAI_API_KEY \
  python -m extensions.auto_verifier.run_auto_verify \
    --candidates-dir extensions/pilot_study/results/combined/rewrite/ \
    --amrlib-model-dir amrlib/data/model_parse_xfm_bart_large-v0_1_0 \
    --llm-judges gpt-4o-mini claude-opus-4-7 \
    --out-dir extensions/auto_verifier/results/run7_with_claude
```

Repeat for DeepSeek and Together, swapping the model name and env var.

## Why this matters for the paper

Adding Claude as a V3 LLM-judge closes the **anti-circularity** argument
that's the foundation of the multi-verifier consensus design:

> When we evaluate AMR-LDA outputs with V2 (GPT-4o-mini judge) only, a
> reviewer might claim we're using an OpenAI judge to score OpenAI-family
> outputs — same-family bias. Adding V3 (Claude judge, different family)
> defeats this attack: if AMR-LDA still wins under both V2 AND V3, the
> family-bias hypothesis is rejected.

The current pilot only has V2 (gpt-4o-mini). The paper's anti-circularity
section ([../pilot_study/red_line_cases.md](../pilot_study/red_line_cases.md))
is incomplete without V3.

## Cost estimate (full pilot rerun)

| Model | API cost per pilot run (~500 calls) |
|---|---|
| gpt-4o | ~$5 |
| gpt-4o-mini | ~$0.50 (already done) |
| gpt-4 | ~$30 |
| gpt-4-turbo | ~$10 |
| claude-opus-4-7 | ~$15 |
| deepseek-v3 | ~$0.30 |
| llama-3-70b (Together) | ~$1 |

Total for adding three families (Claude + DeepSeek + Llama): ~$17.

## Diff to expect

Once all 7 LLMs have results:

1. Per-system table grows from 3 systems to 8 (amr_lda + 7 LLMs)
2. V3 column appears in `summary_by_verifier.json`
3. Anti-circularity table shows AMR-LDA winning under V1, V2, AND V3
4. Per-family analysis: OpenAI vs Anthropic vs DeepSeek-family
   judgement patterns

## Re-running summary scripts

After adding more LLMs, regenerate reports with the existing one-shot:

```bash
bash extensions/pilot_study/generate_final_report.sh
```

The script picks up whatever .jsonl files are in
`extensions/pilot_study/results/combined/rewrite/` so adding new systems
"just works" once their outputs are saved.
