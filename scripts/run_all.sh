#!/usr/bin/env bash
# Run both experiments across seeds. Usage: bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="0 1 2"

echo "=== Experiment 1: supermask wiring rules ==="
for seed in $SEEDS; do
  for method in random readout dense ep dfa rmhebb; do
    echo "--- exp1 method=$method seed=$seed"
    python src/exp1_supermask.py --method "$method" --seed "$seed" --epochs 30
  done
done

echo "=== Experiment 2: front wheeler ==="
for seed in $SEEDS; do
  for schedule in full front periodic fw; do
    echo "--- exp2 schedule=$schedule seed=$seed"
    python src/exp2_frontwheeler.py --schedule "$schedule" --seed "$seed" --epochs 20
  done
done

echo "All runs complete."
