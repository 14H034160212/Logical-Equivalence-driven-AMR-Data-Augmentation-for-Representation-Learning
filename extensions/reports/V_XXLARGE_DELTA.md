# v5 vs v6 at DeBERTa-v2-xxlarge — recipe-matched ReClor

After the deberta-large multi-seed result confirmed v6 wins ReClor by
+0.6 pp ([V6_RECLOR_MULTISEED.md](V6_RECLOR_MULTISEED.md)), this report
checks whether the win holds at the paper's headline regime
(DeBERTa-v2-xxlarge, 1.5B).

## Matched recipe (forced by xxlarge instability)

Standard xxlarge contrastive recipe (lr 3e-6, no warmup) got stuck at
50% binary chance for thousands of steps. Fixed for v6 with:
- lr 1e-6
- 1000 warmup steps
- 6 epochs (instead of 10)
- plain v6 / v5 train file (not the paper's Text2Text-augmented "merged")

v5 was retrained xxlarge with the **same** recipe so the comparison is
valid. Downstream ReClor used `--per_gpu_train_batch_size 2`,
`gradient_accumulation_steps 48`, `GRADIENT_CHECKPOINTING=1` (the
custom env-var switch in `BERT/run_multiple_choice.py`) to fit under
cluster GPU contention.

## Contrastive pretrain

| | final eval | trajectory |
|---|---|---|
| v5 matched | 99.21% | 88.2 → 98.4 → 99.2 → 99.2 → 98.9 → 99.2 |
| v6 matched | 98.79% | 76.7 → 98.2 → 98.6 → 98.5 → 98.6 → 98.8 |

Both contrastive pretrains converge cleanly with warmup. v5 sits 0.4 pp
above v6, consistent with the deberta-large pattern (v5 99.31 vs v6 98.43).

## Downstream ReClor (seed=21)

| | best dev_acc | best step | final step (480) |
|---|---|---|---|
| v5 matched | 45.2% | 100 (epoch ~2) | **24.4%** (collapsed) |
| **v6 matched** | **64.8%** | 480 (epoch ~10) | 64.8% |
| **delta (best)** | **+19.6 pp** | | |
| paper v5 (mismatched recipe) | 78.8% | 480 | — |

v6 matched ReClor trajectory (every 50 steps):

```
50:  44.2%   100: 47.8%   150: 50.2%   200: 53.0%   250: 57.4%
300: 58.0%   350: 60.4%   400: 60.8%   450: 64.0%   480: 64.8%
```

v5 matched ReClor trajectory:

```
50:  35.6%   100: 45.2%   150: 37.6%   200: 39.4%   250: 37.4%
300: 35.0%   350: 42.2%   400: 27.0%   450: 26.0%   480: 24.4%
```

The v5 matched run **degenerates after step 100** — peaks at 45.2%,
then drops below chance (25% for 4-way) by the end. v6 climbs steadily
to 64.8% and stabilises there.

## Honest reading

- **Direction agrees with deberta-large**: v6 > v5 on ReClor.
- **Magnitude is not reliable** at xxlarge because v5's late-training
  collapse suggests the matched recipe (lr 1e-6) is too weak or too
  unstable for v5's noisier surface forms to train cleanly. With a
  different LR / warmup schedule for v5, the gap might be smaller.
- The paper's v5 xxlarge ckpt reaches 78.8% with the paper's recipe
  (lr 3e-6, no warmup, merged data). That recipe didn't work on our
  v5/v6 plain data here (xxlarge got stuck at chance), so the matched
  comparison is the cleanest available.

## What stands

The primary, clean, seed-robust finding remains DeBERTa-large:
- **ReClor**: v6 62.9 → 63.5 (+0.6 pp, both seeds agree)
- **LogiQA**: v5 42.3 → 40.3 (−2.0 pp, both seeds agree)

The xxlarge robustness check confirms the **direction** on ReClor
(v6 > v5) but produces unreliable magnitudes because of the v5
training collapse. v6 xxlarge contrastive trains cleanly with warmup;
v5 xxlarge contrastive trains cleanly too (final 99.21%) but its
ReClor head fails to converge under the same downstream hparams.

## What could close the open question (un-run)

- Retrain v5 xxlarge ReClor with multiple seeds (the single-seed
  collapse may be unlucky).
- Sweep ReClor LR (e.g., 5e-6, 2e-5) for v5 matched.
- Use the paper's full "merged" recipe (lr 3e-6, no warmup, merged
  data) — but that requires solving the chance-stuck issue at xxlarge
  contrastive for plain data, which we couldn't.
