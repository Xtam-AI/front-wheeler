#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for m in ep dfa rmhebb; do
    python src/exp1_supermask.py --method $m --seed $seed --epochs 30 --depth 4 --tag d4
    python src/exp1_supermask.py --dataset cifar10 --method $m --seed $seed --epochs 40 --depth 4 --tag d4
  done
done
echo depth4 done
