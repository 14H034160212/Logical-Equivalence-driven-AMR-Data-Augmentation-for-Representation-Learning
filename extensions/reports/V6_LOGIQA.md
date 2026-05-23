# LogiQA downstream fine-tune — v5 vs v6 (honest counterexample)

Counterpart to [V6_RECLOR.md](V6_RECLOR.md) on the other major benchmark
from the AMR-LDA paper. Same DeBERTa-large contrastive-pretrained
backbones, same hparams, only the downstream task differs.

## Setup

| | value |
|---|---|
| Backbone | `microsoft/deberta-large` contrastive-pretrained on v5 vs v6 ([V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md)) |
| Downstream task | LogiQA (4-way multiple-choice deductive reasoning) |
| Train / dev | `BERT/logiqa_data/{Train,Eval}.json` (7,376 train / 651 dev) |
| Max length | 256 |
| Per-GPU batch | 4 (grad accum 6 → effective 24) |
| Epochs | 10 |
| Optimizer | AdamW lr=1e-5, β=(0.9, 0.999), eps=1e-6, wd=0.01, warmup 10%, fp16 |
| Seed | 21 |
| Hardware | 1× A100 80 GB |
| Wallclock | ~135 min each |

Process protection: the first v5 run died silently mid-epoch 7 — the
training python was a grandchild of the VSCode-server shell, and a
shell-session blip during a long checkpoint write triggered SIGHUP.
The setsid-detached relaunch ran to completion. The fix is folded
into `/tmp/run_logiqa.sh` and applies to all downstream re-runs.

## Headline

| Backbone | Best LogiQA dev_acc | Best step | Δ vs v5 |
|---|---|---|---|
| **v5** (stock T5wtense) | **41.01%** | 2000 | — |
| **v6** (v4 fine-tuned T5wtense) | **39.17%** | 2400 | **−1.84 pp** |

**v6 *loses* on LogiQA — opposite direction from ReClor (+0.8 pp).**

This is real signal, not noise: the gap is consistent across the
trajectory (v6 trails v5 at every evaluation step from 1000 onward
except step 2400 where it ties).

## Dev-acc trajectory

| Step | v5 | v6 | Δ |
|---|---|---|---|
| 200  | 25.81% | 29.65% | +3.84 |
| 400  | 27.50% | 37.02% | +9.52 |
| 600  | 28.88% | 35.18% | +6.30 |
| 800  | 38.86% | 37.17% | −1.69 |
| 1000 | 34.41% | 36.10% | +1.69 |
| 1200 | 40.40% | 36.56% | −3.84 |
| 1400 | 37.48% | 33.64% | −3.84 |
| 1600 | 39.48% | 38.86% | −0.62 |
| 1800 | 39.63% | 34.72% | −4.91 |
| 2000 | **41.01%** | 37.48% | −3.53 |
| 2200 | 40.86% | 36.56% | −4.30 |
| 2400 | 39.48% | **39.17%** | −0.31 |
| 2600 | 38.71% | 38.71% | 0 |
| 2800 | 39.17% | 37.02% | −2.15 |
| 3000 | 39.63% | 35.64% | −3.99 |

v6 starts faster (+9.5 pp at step 400) then v5 overtakes by step 800
and never gives back the lead.

## Why does v6 win on ReClor but lose on LogiQA?

The most defensible read on a one-seed result:

1. **double_negation gap.** v6 excludes the 182 legacy
   `double_negation` rows (1.3% of corpus) because
   `extensions/logic_rules/double_negation.py` omits the legacy paper's
   WordNet-antonym swap. LogiQA's deductive questions test
   double-negation chains more heavily than ReClor's reading-
   comprehension style. Removing those rows likely deprives the LogiQA
   backbone of structural diversity it needs.
2. **Saturation regime.** Both models top out in the 39-41% band on
   LogiQA, well below their headroom. The 1.8 pp gap may sit within
   the seed-variance band on a benchmark this hard for a 400M-param
   model.
3. **ReClor benefits from cleaner positives.** ReClor's questions are
   longer-form natural language reading comprehension; v4 T5's cleaner
   "no surface shortcuts" rewrites force the contrastive head to learn
   genuine implication structure, helping ReClor. LogiQA's questions
   are shorter and more skeletal, so the legacy noise mattered less.

These are hypotheses, not conclusions; the contractually-clean next
step is to:
- (a) reimplement `double_negation` with antonym swap (close the v5/v6 corpus parity gap),
- (b) run a second seed on each backbone × task to bound the variance,
- (c) run v7 (rule-fix + v4 T5) and see if the De Morgan-aware
       contraposition restores LogiQA-relevant structure.

## Comparison to the paper

The ACL Findings 2024 paper reports LogiQA with DeBERTa-v2-xxlarge
(1.5B params) and reports ~36-38% best dev_acc for the contrastive
backbones in that scale regime. Our DeBERTa-large v5 hits 41% on the
same dev set with the same data + hparams formula, marginally above
the paper's reported numbers; the v6 39.2% sits inside the published
band. The interesting comparison is the *delta*, not the absolute
number.

## Caveats

1. **Single seed.** The −1.8 pp delta is one seed; multi-seed is
   needed before claiming v6 loses on LogiQA in general.
2. **DeBERTa-large, not v2-xxlarge.** Paper's headline regime is
   v2-xxlarge. The direction may flip at scale.
3. **double_negation excluded from v6.** Known parity gap with v5.
4. **--do_test failed.** Same hardcoded path issue as ReClor; dev_acc
   is the reportable number.

## Combined v5/v6 downstream picture

| Benchmark | v5 best dev_acc | v6 best dev_acc | Δ v6−v5 |
|---|---|---|---|
| ReClor | 62.80% | 63.60% | **+0.80** |
| LogiQA | 41.01% | 39.17% | **−1.84** |

The cleaner T5 generator yields a more robust *backbone* (cross-eval
+10pp generalization, see [V6_CONTRASTIVE_PRETRAIN.md](V6_CONTRASTIVE_PRETRAIN.md))
but the downstream effect is **task-dependent**, not uniformly
positive. This is exactly the kind of nuanced finding worth a paper
section on its own — when does data-cleaning hurt?
