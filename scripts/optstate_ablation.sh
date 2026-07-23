#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for os in reset reset-warmup episodic; do
    python src/exp3_transformer.py --schedule fw --tau 0.02 --seed $seed \
      --opt-state "$os" --tag "os_${os//-/}"
    python src/exp3_transformer.py --schedule periodic --period 10 --seed $seed \
      --opt-state "$os" --tag "p10os_${os//-/}"
  done
done
echo optstate ablation done
