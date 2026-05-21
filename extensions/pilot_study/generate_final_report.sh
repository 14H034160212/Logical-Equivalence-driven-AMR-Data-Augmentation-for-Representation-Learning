#!/usr/bin/env bash
# One-shot final report generation.
# Run this once the run5 verifier task completes (per_item.jsonl exists and has 270 lines).

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root
ROOT="$PWD"
PY="/data/qbao775/miniconda3/envs/leamr/bin/python"

if [ ! -f extensions/auto_verifier/results/run5/per_item.jsonl ]; then
  echo "ERROR: run5 results not found"
  exit 1
fi

echo "[1/4] Re-aggregating run5 (dropping broken SMATCH, converting AMR no-fire to abstain)..."
PYTHONPATH="$ROOT" $PY -m extensions.auto_verifier.reaggregate \
  --in-dir extensions/auto_verifier/results/run5 \
  --out-dir extensions/auto_verifier/results/run5_clean \
  --drop-verifiers smatch_struct

echo
echo "[2/4] Three-way analysis (gpt-4o, gpt-4o-mini, amr_lda with full 13 active rules)..."
PYTHONPATH="$ROOT" $PY -m extensions.pilot_study.three_way_analysis \
  --rewrite-dir extensions/pilot_study/results/combined/rewrite/ \
  --verifier-items extensions/auto_verifier/results/run5_clean/per_item.jsonl \
  --out-md extensions/pilot_study/results/combined/THREE_WAY_v3.md

echo
echo "[3/4] Trajectory comparison across all 3 runs..."
PYTHONPATH="$ROOT" $PY -m extensions.pilot_study.run_comparison \
  --runs \
    run3_before_patches:extensions/auto_verifier/results/run3_clean \
    run4_after_basic_patches:extensions/auto_verifier/results/run4_clean \
    run5_after_umr_rules:extensions/auto_verifier/results/run5_clean \
  --out-md extensions/pilot_study/results/combined/TRAJECTORY.md

echo
echo "[4/4] Before/after diff (run4 -> run5; UMR rules contribution)..."
PYTHONPATH="$ROOT" $PY -m extensions.pilot_study.before_after_diff \
  --before extensions/auto_verifier/results/run4_clean/per_item.jsonl \
  --after extensions/auto_verifier/results/run5_clean/per_item.jsonl \
  --out-md extensions/pilot_study/results/combined/BEFORE_AFTER_UMR.md

# Self-consistency check breakdown (only meaningful for run6+)
if [ -f extensions/auto_verifier/results/run6_clean/per_item.jsonl ]; then
  echo
  echo "[5/5] Self-check breakdown..."
  PYTHONPATH="$ROOT" $PY -m extensions.pilot_study.self_check_analysis \
    --rewrite extensions/pilot_study/results/combined/rewrite/amr_lda.jsonl \
    --verifier-items extensions/auto_verifier/results/run6_clean/per_item.jsonl \
    --out-md extensions/pilot_study/results/combined/SELF_CHECK.md
fi

echo
echo "Done. Reports written to extensions/pilot_study/results/combined/"
echo "  - THREE_WAY_v3.md       — system-level coverage / quality / overall"
echo "  - TRAJECTORY.md         — per-run improvement over 3 patches"
echo "  - BEFORE_AFTER_UMR.md   — items the UMR rules recovered"
echo "  - SELF_CHECK.md         — quality split by self-consistency status"
