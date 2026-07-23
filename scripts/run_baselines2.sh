#!/usr/bin/env bash
# LP-FT and BCD discriminating baselines, both datasets.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="0 1 2"

run() { # dataset epochs
  local ds=$1 ep=$2
  for seed in $SEEDS; do
    for sf in 0.5 0.8 0.9 0.95 0.96; do
      python src/exp2_frontwheeler.py --dataset "$ds" --schedule lpft \
        --seed "$seed" --epochs "$ep" --switch-frac "$sf" --tag "sf${sf//./}"
    done
    python src/exp2_frontwheeler.py --dataset "$ds" --schedule bcd \
      --seed "$seed" --epochs "$ep"
  done
}

echo "=== MNIST ==="
run mnist 20
echo "=== CIFAR-10 ==="
run cifar10 30
echo "Baselines complete."
