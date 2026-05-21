# Logical-Equivalence-driven AMR Data Augmentation for Representation Learning

Code, data, and extension work for **"Abstract Meaning Representation-Based Logic-Driven Data Augmentation for Logical Reasoning"** (Bao et al., [ACL Findings 2024](https://aclanthology.org/2024.findings-acl.353.pdf)) — plus ongoing extension work toward UMR-grounded rules and reinforcement-learning verifiers.

## What's in here

```
.
├── extensions/           # Active research: rule extension, UMR overlay, RL verifier
├── legacy/               # Original ACL 2024 paper code (preserved as-is)
│   ├── amr_lda/          # AMR-LDA augmentation, synthetic dataset generation
│   ├── lreasoner/        # LReasoner baseline preprocessing
│   ├── reclor_preproc/   # ReClor if-then preprocessing scripts
│   └── data/             # Original dataset CSVs (Synthetic_xfm_t5wtense_*.csv)
├── BERT/                 # Training & evaluation pipeline (RoBERTa, DeBERTa, GLUE tasks)
│   ├── scripts/          # Per-task training shell scripts
│   ├── reclor_data/      # ReClor train/dev/test
│   ├── logiqa_data/      # LogiQA train/dev/test
│   ├── run_multiple_choice.py / run_glue.py / fine_tune.py
│   ├── npy_ensemble.py / load_then_predict.py
│   ├── Checkpoints/      # (gitignored; trained model weights — 187GB local)
│   ├── Transformers/     # (gitignored; downloaded base models — 125GB local)
│   └── wandb/            # (gitignored; training logs — 4.7GB local)
├── scripts/              # AMR text↔graph conversion (GSII, SPRING, XFM parsers; T5 generators)
├── amrlib/               # Vendored AMR parsing library (used by legacy code)
├── apex/                 # NVIDIA Apex for mixed-precision training
├── PARARULE-Plus-main/   # PARARULE-Plus dataset (Depth2-5 multi-step deductive reasoning)
├── paraphrased_pararule/ # Paraphrased version of PARARULE
├── extracted_data/       # Snapshots from reasoning datasets (ReClor, LogiQA, Cosmos, etc.)
├── Max_Tegmark/          # Tegmark dataset slices
├── Text2Text_Transformer/# T5 / T5wtense AMR-to-text models
├── tests/                # Original paper tests
├── output_result/        # Pre-built synthetic train/val splits for v3/v4/v5 of the paper
├── summaries/            # Paper figure / table source
├── configs/              # AMR parser configs
├── docs/                 # Public docs (mkdocs / readthedocs)
├── docs_internal/        # Working notes, idea PDFs, running notes (not published)
│   ├── state_of_the_art.pdf
│   ├── logic_transformations.pdf
│   ├── script_running_notes.txt
│   └── test_script.docx
├── requirements*.txt     # Pinned and latest-tested dependency sets
├── setup.py, MANIFEST.in, mkdocs.yml, .readthedocs.yml
└── README.md             # this file
```

## Two threads

### Thread 1 — Original AMR-LDA paper (frozen)
Code under [`legacy/`](legacy/) reproduces the ACL Findings 2024 paper. It generates AMR-driven logical-equivalence augmentations from PARARULE / synthetic templates and trains discriminative LLMs (RoBERTa-Large, DeBERTa-Large, DeBERTaV2-XXLarge) with contrastive learning on the produced positive/negative pairs.

Quick pointers:
- [`legacy/amr_lda/logical_equivalence_functions.py`](legacy/amr_lda/logical_equivalence_functions.py) — core rule implementations (contraposition, commutative, implication, double-negation)
- [`legacy/amr_lda/logical_equivalence_synthetic_dataset.py`](legacy/amr_lda/logical_equivalence_synthetic_dataset.py) — synthetic sentence generation
- [`legacy/lreasoner/LReasoner_*.py`](legacy/lreasoner/) — LReasoner baseline replication
- [`legacy/reclor_preproc/`](legacy/reclor_preproc/) — ReClor if-then sentence extraction
- [`legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list.csv`](legacy/data/) — full synthetic dataset (2 MB)
- [`BERT/scripts/`](BERT/scripts/) — training shell scripts for all paper experiments
- [`docs_internal/script_running_notes.txt`](docs_internal/script_running_notes.txt) — full reproduction commands

> Legacy scripts use relative paths (e.g. `./output_result/...`). Run them from the project root, e.g. `python legacy/amr_lda/logical_equivalence_synthetic_dataset.py`.

### Thread 2 — Active extensions (in development)
Work under [`extensions/`](extensions/) extends the original paper along three axes:

1. **More rules**: from 4 → **14 registered (13 active + 1 stub)**. See [`extensions/logic_rules/`](extensions/logic_rules/).
2. **UMR overlay**: aspect, modal strength, document-level temporal transitivity (AMR-layer approximations; full UMR is future work). See [`extensions/umr/`](extensions/umr/).
3. **Multi-verifier consensus**: AMR-struct + LLM-as-judge + SMATCH — replaces bulk human annotation. See [`extensions/auto_verifier/`](extensions/auto_verifier/).
4. **T5wtense self-consistency**: re-parse-and-check polarity parity to catch generator errors. See [`extensions/pilot_study/generate_amr_lda.py`](extensions/pilot_study/generate_amr_lda.py).

Plus a pilot study to defend against the "why not just use an LLM?" reviewer attack — see [`extensions/pilot_study/`](extensions/pilot_study/).

See [`extensions/README.md`](extensions/README.md) and [`extensions/SUMMARY.md`](extensions/SUMMARY.md) for the active research plan and project state.

#### Latest pilot results

Headlines from the most recent run (run6 — 4 patch rounds, 50 sentences × ~5 rules):

| System | Coverage | Quality (EQ/decided) |
|---|---|---|
| **amr_lda** (our pipeline) | 68.9% | **43.8%** |
| gpt-4o | 100% | 80.9% |
| gpt-4o-mini | 100% | 78.3% |

AMR-LDA quality improved **8.6% → 43.8%** (5×) across 4 patch rounds. See:

- 📊 **[Three-way comparison](extensions/reports/THREE_WAY_v4.md)** — system × rule × verifier breakdown
- 📈 **[Improvement trajectory](extensions/reports/TRAJECTORY_v2.md)** — per-run gains across all patches
- 🔬 **[Self-check analysis](extensions/reports/SELF_CHECK.md)** — 15 detected generator polarity-flips
- 🧬 **[AMR → UMR converter](extensions/reports/UMR_CONVERTER.md)** — Post et al. 2024 reproduction. Rule-only F1 43.7%; **rule + DistilBERT gold-accuracy 80.7%** on aspect (DistilBERT macro F1 0.63 vs sklearn LR 0.52)
- 🗂️ **[Document-level UMR](extensions/reports/DOC_LEVEL_UMR.md)** — cross-sentence temporal/modal/coref derivation baseline (modal P=70%)
- 🤖 **[RL verifier integration](extensions/rl/README.md)** + **[GRPO results](extensions/reports/GRPO_RESULTS.md)** — Qwen2.5-0.5B trained end-to-end, reward 43.75% → 62.5% in 113 seconds on one A100
- 🏋️ **[GRPO @ 3B + LoRA](extensions/reports/GRPO_3B_RESULTS.md)** — Qwen2.5-3B-Instruct with PEFT/LoRA on 2× A100; reward **37.5% → 93.75%** over 3 epochs (13 min)
- 🛠 **[T5wtense fine-tune](extensions/reports/T5_FINETUNE_RESULTS.md)** — 389 (AMR, text) pairs from pilot; eval_loss 0.278 → 0.240 over 3 epochs (~50 s on one A100), targeting the 19% polarity-drop failure mode
- 🎯 **[T5 fine-tune end-to-end recovery](extensions/reports/T5_FT_RECOVERY.md)** — pilot A/B on the 15 known polarity-flips: pass rate **34.8% → 69.6%** (v3, 9/15 recovered; gold + hand-derived rule canonical-form retrain)
- 🔌 **[Multi-LLM setup guide](extensions/reports/MULTI_LLM_SETUP.md)** — adding Claude / DeepSeek / Llama-3-70B baselines
- 🔧 **[Reports index](extensions/reports/README.md)** — all generated reports + JSON aggregates
- 📝 **[Red-line failure cases](extensions/pilot_study/red_line_cases.md)** — 5 paper-ready exhibits
- 🎯 **[Human-eval rubric](extensions/pilot_study/human_eval_rubric.md)** — spot-check protocol for verifier disagreement

## Setup

```bash
# Python env (the original repo uses Python 3.8; extensions tested on the leamr env)
conda activate leamr   # or whichever env has penman, amrlib, transformers, torch

pip install -r requirements.txt
# For latest tested-against versions:
# pip install -r requirements_latest.txt
```

Optional but recommended for `extensions/`:
- `pip install openai anthropic together`  (LLM baselines / judges)
- `pip install smatch`                       (real SMATCH for V4)
- `pip install pytest`                       (if you want to run extensions/logic_rules/tests/)

## Citation

```bibtex
@inproceedings{bao-etal-2024-abstract,
    title = "Abstract Meaning Representation-Based Logic-Driven Data Augmentation for Logical Reasoning",
    author = "Bao, Qiming and Peng, Alex Yuxuan and Deng, Zhenyun and Zhong, Wanjun and Gendron, Ga{\"e}l and Pistotti, Timothy and Tan, Ne{\c{s}}et and Young, Nathan and Chen, Yang and Zhu, Yonghua and Denny, Paul and Witbrock, Michael and Liu, Jiamou",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.353",
}
```

## License

See [LICENSE](LICENSE). Original synthetic datasets and PARARULE-Plus retain their own licenses (CC-BY equivalent).
