#!/usr/bin/env bash
# CIFAR-10 replication: same experiments, harder dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="0 1 2"

echo "=== Exp 1 on CIFAR-10 ==="
for seed in $SEEDS; do
  for method in random readout dense ep dfa rmhebb; do
    python src/exp1_supermask.py --dataset cifar10 --method "$method" \
      --seed "$seed" --epochs 40
  done
done

echo "=== Exp 2 on CIFAR-10 ==="
for seed in $SEEDS; do
  for schedule in full front periodic fw; do
    python src/exp2_frontwheeler.py --dataset cifar10 --schedule "$schedule" \
      --seed "$seed" --epochs 30
  done
  for period in 2 5 20 50; do
    python src/exp2_frontwheeler.py --dataset cifar10 --schedule periodic \
      --seed "$seed" --period "$period" --epochs 30 --tag "p$period"
  done
  for tau in 0.005 0.05; do
    python src/exp2_frontwheeler.py --dataset cifar10 --schedule fw \
      --seed "$seed" --tau "$tau" --epochs 30 --tag "tau${tau//./}"
  done
done

echo "CIFAR-10 runs complete."
