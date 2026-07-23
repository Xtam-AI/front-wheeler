#!/usr/bin/env bash
# Sweeps for the cost-accuracy tradeoff (exp2) and sparsity (exp1).
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="0 1 2"

echo "=== Exp 2 sweep: periodic budgets ==="
for seed in $SEEDS; do
  for period in 2 5 20 50; do
    python src/exp2_frontwheeler.py --schedule periodic --seed "$seed" \
      --period "$period" --tag "p$period"
  done
done

echo "=== Exp 2 sweep: fw thresholds ==="
for seed in $SEEDS; do
  for tau in 0.005 0.01 0.05; do
    python src/exp2_frontwheeler.py --schedule fw --seed "$seed" \
      --tau "$tau" --tag "tau${tau//./}"
  done
done

echo "=== Exp 1 sweep: sparsity (ep + dfa) ==="
for seed in $SEEDS; do
  for sp in 0.3 0.7 0.9; do
    python src/exp1_supermask.py --method ep --seed "$seed" \
      --sparsity "$sp" --epochs 30 --tag "sp${sp//./}"
    python src/exp1_supermask.py --method dfa --seed "$seed" \
      --sparsity "$sp" --epochs 30 --tag "sp${sp//./}"
  done
done

echo "Sweeps complete."
