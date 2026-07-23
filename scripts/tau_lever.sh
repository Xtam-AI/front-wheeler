#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for tau in 0.1 0.2 0.5 1.0; do
    t="${tau//./}"
    python src/exp2_frontwheeler.py --dataset cifar10 --schedule fw --seed $seed \
      --epochs 30 --tau $tau --tag "lever${t}"
    python src/exp2_frontwheeler.py --dataset cifar10 --schedule fw --seed $seed \
      --epochs 30 --tau $tau --max-cooldown 50 --tag "lever${t}nb"
    python src/exp3_transformer.py --schedule fw --seed $seed \
      --tau $tau --tag "lever${t}"
    python src/exp3_transformer.py --schedule fw --seed $seed \
      --tau $tau --max-cooldown 50 --tag "lever${t}nb"
  done
done
echo tau lever sweep done
