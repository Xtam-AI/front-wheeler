#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for mc in 100 200 400 800; do
    python src/exp3_transformer.py --schedule fw --tau 0.1 --seed $seed \
      --max-cooldown $mc --tag "gov${mc}"
  done
done
echo governor sweep done
