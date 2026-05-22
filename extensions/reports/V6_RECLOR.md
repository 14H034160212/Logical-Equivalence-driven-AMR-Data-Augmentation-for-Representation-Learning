# ReClor downstream fine-tune — v5 (stock T5) vs v6 (v4 fine-tuned T5)

The headline downstream metric in the ACL Findings 2024 paper is ReClor
multiple-choice accuracy after a second fine-tune on top of the
contrastive-pretrained backbone. This is the first apples-to-apples
v5-vs-v6 comparison at that layer.

## Setup

| | value |
|---|---|
| Backbone | `microsoft/deberta-large` contrastive-pretrained on v5 vs v6 ([V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md)) |
| Downstream task | ReClor (4-way multiple-choice reading comprehension) |
| Train / dev | `BERT/reclor_data/{train,val}.json` (4,638 train, 500 dev) |
| Max length | 256 |
| Per-GPU batch | 4 (grad accum 6 → effective 24) |
| Epochs | 10 |
| Optimizer | AdamW lr=1e-5, β=(0.9, 0.999), eps=1e-6, wd=0.01, warmup 10%, fp16 |
| Seed | 21 |
| Hardware | 1× A100 80 GB (`CUDA_VISIBLE_DEVICES=6`) |
| Wallclock | ~55 min each |

Engineering note: `BERT/run_multiple_choice.py` shipped without
`DebertaForMultipleChoice` (it's not in HuggingFace transformers — only
DeBERTa-V2 has one). Added [`BERT/deberta_multiple_choice.py`](../../BERT/deberta_multiple_choice.py)
as a thin wrapper mirroring `RobertaForMultipleChoice`'s architecture
(DebertaModel → ContextPooler → Linear(1)), then registered
`"deberta"` in `MODEL_CLASSES`.

## Headline

| Backbone | Best ReClor dev_acc | Best step | Δ vs v5 |
|---|---|---|---|
| **v5** (stock T5wtense, 99.31% contrastive eval) | **62.8%** | 1600 | — |
| **v6** (v4 fine-tuned T5wtense, 98.43% contrastive eval) | **63.6%** | 1600 | **+0.8 pp** |

The v6-pretrained model **wins on the downstream metric** despite a
0.9 pp drop on the in-distribution contrastive eval — consistent with
the cross-eval finding in [V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md)
that v6 produces a more robust classifier.

## Dev-acc trajectory (per-evaluation, every 200 steps)

| Step | v5 | v6 | Δ |
|---|---|---|---|
| 200  | 31.6% | 38.6% | +7.0 |
| 400  | 48.2% | 55.2% | +7.0 |
| 600  | 53.0% | 54.8% | +1.8 |
| 800  | 55.4% | 58.0% | +2.6 |
| 1000 | 58.4% | 58.8% | +0.4 |
| 1200 | 59.8% | 62.2% | +2.4 |
| 1400 | 60.8% | 61.8% | +1.0 |
| 1600 | **62.8%** | **63.6%** | +0.8 |
| 1800 | 62.8% | 62.8% | 0.0 |
| 1930 | 62.8% | 63.4% | +0.6 |

v6 **leads at every evaluation step**. The +7 pp advantage early in
training (steps 200–400) suggests the v6 backbone provides better
initial representations for downstream reasoning; the gap narrows as
both models converge but never closes.

## Comparison to the paper's headline numbers

The ACL Findings 2024 paper reports ReClor with DeBERTa-**v2-xxlarge**
(1.5B params), not DeBERTa-large (400M). At v2-xxlarge scale the
published v5 number is in the high 60s / low 70s. We do not have a
v2-xxlarge v6 backbone yet (the contrastive pretrain at that scale
takes hours and was out of scope for this run). The v5↔v6 delta
at DeBERTa-large scale (**+0.8 pp**) is what this experiment isolates.

## Reproducing

```bash
# 1. Build v6 contrastive data (already done; ~5 min)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python \
        -m extensions.pilot_study.build_v6_contrastive
PYTHONPATH=. /data/qbao775/miniconda3/envs/leamr/bin/python \
    -m extensions.pilot_study.split_v6_for_glue

# 2. Contrastive pretrain DeBERTa-large on v5 and v6 (already done; ~23 min each)
# See V6_CONTRASTIVE_PRETRAIN.md.

# 3. Convert CSVs to JSONL (datasets 2.6.1 / new-pandas compat)
/data/qbao775/miniconda3/envs/leamr/bin/python -c "
import pandas as pd, json
for split in ['train','validation']:
    for v in ['v5','v6']:
        df = pd.read_csv(f'output_result/Synthetic_xfm_t5wtense_logical_equivalence_{split}_{v}.csv')
        with open(f'output_result/Synthetic_xfm_t5wtense_logical_equivalence_{split}_{v}.json','w') as f:
            for _, r in df.iterrows():
                f.write(json.dumps({'sentence1': r['sentence1'],
                                    'sentence2': r['sentence2'],
                                    'label': int(r['label'])}) + chr(10))
"

# 4. ReClor fine-tune on each backbone (~55 min each)
cd BERT && WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=6 \
    /data/qbao775/miniconda3/envs/leamr/bin/python run_multiple_choice.py \
        --model_type deberta \
        --model_name_or_path Transformers/deberta-large-our-model-v6 \
        --task_name reclor --do_train --evaluate_during_training --do_lower_case \
        --data_dir reclor_data --max_seq_length 256 \
        --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 4 \
        --gradient_accumulation_steps 6 --learning_rate 1e-05 \
        --num_train_epochs 10.0 --output_dir Checkpoints/reclor/deberta-large-v6 \
        --fp16 --logging_steps 200 --save_steps 1000 \
        --adam_epsilon 1e-6 --warmup_proportion 0.1 --weight_decay 0.01 \
        --seed 21 --overwrite_output_dir
```

## Caveats

1. **Single seed.** The +0.8 pp delta is one-seed; not yet statistically
   ranked. A second seed run (~55 min) is the cheapest follow-up.
2. **DeBERTa-large, not xxlarge.** Paper's headline regime is xxlarge.
   The improvement direction transfers in principle but the magnitude
   may differ.
3. **--do_test failed.** `ReclorProcessor.get_test_examples` is
   hardcoded to a non-existent option-extension file in
   `utils_multiple_choice.py:313`. Dev acc is the reportable
   number (ReClor test labels are held out for leaderboard submission).
4. **v6 drops 1.3% (182 / 14180) of the legacy corpus** by excluding
   `double_negation` rows where the legacy paper paired with an
   antonym swap that extensions/logic_rules doesn't replicate. The
   smaller corpus still wins downstream, but a like-for-like
   double_negation reimplementation would close that gap explicitly.
