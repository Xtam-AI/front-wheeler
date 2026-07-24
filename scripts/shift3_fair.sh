#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
S="--steps 10000 --shift-at 5400 --shift-corpus warpeace --eval-every 200"
for seed in 0 1 2; do
  python src/exp3_transformer.py --schedule fw --tau 0.02 $S --seed $seed --tag shift3
  python src/exp3_transformer.py --schedule pburst --pburst-period 833 $S --seed $seed --tag shift3
done
echo shift3 done
