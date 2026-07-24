#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
S="--steps 10000 --shift-at 5000 --eval-every 250"
for seed in 0 1 2; do
  python src/exp3_transformer.py --schedule full  $S --seed $seed --tag shift
  python src/exp3_transformer.py --schedule front $S --seed $seed --tag shift
  python src/exp3_transformer.py --schedule fw --tau 0.02 $S --seed $seed --tag shift
  python src/exp3_transformer.py --schedule ftlp --switch-frac 0.06 $S --seed $seed --tag shift
  python src/exp3_transformer.py --schedule pburst --pburst-period 833 $S --seed $seed --tag shift
  python src/exp3_transformer.py --schedule lpft --switch-frac 0.9 $S --seed $seed --tag shift
done
echo shift experiment done
