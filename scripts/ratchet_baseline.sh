#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for rb in 0.1 0.3; do
    t="${rb//./}"
    python src/exp2_frontwheeler.py --dataset cifar10 --schedule ratchet \
      --seed $seed --epochs 30 --ratchet-by $rb --tag "rb${t}"
    python src/exp2_frontwheeler.py --schedule ratchet \
      --seed $seed --epochs 20 --ratchet-by $rb --tag "rb${t}"
    python src/exp3_transformer.py --schedule ratchet --seed $seed \
      --ratchet-by $rb --tag "rb${t}"
  done
done
echo ratchet baseline done
