# T5wtense fine-tune for polarity preservation

The pilot study self-consistency check (see [SELF_CHECK.md](SELF_CHECK.md))
found that the stock `model_generate_t5wtense-v0_1_0` checkpoint drops the
`:polarity -` edge in ~19% of generations from logical-equivalence rewrites
(15 of 77 records in run6). Concretely, an AMR meaning "it is not the case
that Alice may skip her homework" was generated as "Alice can't finish her
homework" — the negation moved from the modal to the predicate.

This run fine-tunes T5wtense on pilot-harvested (AMR, text) pairs to bias
the decoder toward keeping `:polarity -` where the AMR says so.

## Setup

| | value |
|---|---|
| Base model | `amrlib/data/model_generate_t5wtense-v0_1_0` (T5-base, 220M params) |
| Tokenizer | `t5-base` (the checkpoint dir ships no tokenizer files) |
| Training pairs | 389 unique (modified_AMR, target_text) from pilot contrastive datasets |
| Split | 330 train / 59 eval (85/15) |
| Optimizer | AdamW, lr=3e-5, weight_decay=0.01, linear warmup 10% |
| Batch size | 8 |
| Max input / target length | 512 / 64 tokens |
| Epochs | 3 |
| Hardware | 1× A100 80 GB (GPU 0) |
| Wallclock | ~50 sec total |
| Seed | 42 |

Training pairs are pulled from:

1. `extensions/reports/contrastive_pilot_smoketest.jsonl`
2. `extensions/pilot_study/results/combined/rewrite/amr_lda.jsonl`
3. `extensions/pilot_study/pararule_contrastive.jsonl`

paired as `(amr_positive → positive)`. Where a hand-curated gold rewrite
exists in `extensions/pilot_study/test_sentences.json` for the same
`(sentence_id, rule)` key, the gold supersedes the silver LLM output.

## Result

```
epoch 1/3  train_loss=0.5600  eval_loss=0.2775
epoch 2/3  train_loss=0.3376  eval_loss=0.2417
epoch 3/3  train_loss=0.2965  eval_loss=0.2396
```

Both losses fall monotonically; eval keeps decreasing through epoch 3
(no overfit signal in this regime). Saved to
[`extensions/pilot_study/ft_t5wtense/`](../pilot_study/ft_t5wtense/) with the
copied `amrlib_meta.json` so amrlib can load the checkpoint directly.

## Reproducing

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.finetune_t5wtense
```

Output: `extensions/pilot_study/ft_t5wtense/` + report at
[`extensions/reports/ft_t5wtense_report.json`](ft_t5wtense_report.json).

## Notes / open items

- Pairs are silver-standard (verifier-accepted LLM outputs) where no human
  gold exists. Targeted human gold for the 15 known-failure cases would
  give the strongest signal but the silver-only signal is enough to bend
  the loss curve.
- The downstream self-check pass-rate measurement (re-run AMR-LDA
  generation with this checkpoint, count how many of the 15 known
  polarity flips are recovered) is the next step.
- A larger ablation should sweep epochs (more may help) and run the
  contrastive dev set through the new vs. stock generator to bound the
  hit rate change end-to-end.
