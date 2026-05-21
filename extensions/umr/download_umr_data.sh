#!/usr/bin/env bash
# Download the umr4nlp/umr-data repository (UMR 2.0 corpus).
# Per the readme, English has ~580 docs / 31K sentences / 297K words.
#
# Usage:
#   bash extensions/umr/download_umr_data.sh
#
# The repository is large (multilingual); we shallow-clone and you can prune
# unused languages afterwards.

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d umr-data ]; then
  echo "Cloning umr4nlp/umr-data (shallow)..."
  git clone --depth 1 https://github.com/umr4nlp/umr-data.git
else
  echo "umr-data already present, pulling latest..."
  cd umr-data && git pull --ff-only && cd ..
fi

echo
echo "UMR languages available:"
ls umr-data/ | grep -E '^(english|chinese|czech|latin|arapaho|kukama|navajo|sanapana)$'

echo
echo "Run: python -m extensions.umr.loader --language english"
