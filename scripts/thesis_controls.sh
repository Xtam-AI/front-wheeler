#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  # reverse LP-FT (deep first): 6% and 20% deep-step budgets
  python src/exp3_transformer.py --schedule ftlp --switch-frac 0.06 --seed $seed --tag s006
  python src/exp3_transformer.py --schedule ftlp --switch-frac 0.20 --seed $seed --tag s02
  # open-loop periodic bursts (no trigger): matched budgets
  python src/exp3_transformer.py --schedule pburst --pburst-period 833 --seed $seed --tag p833
  python src/exp3_transformer.py --schedule pburst --pburst-period 250 --seed $seed --tag p250
  python src/exp3_transformer.py --schedule pburst --pburst-period 833 --opt-state reset-warmup --seed $seed --tag p833rw
  # same controls on CIFAR MLP
  python src/exp2_frontwheeler.py --dataset cifar10 --schedule ftlp --switch-frac 0.045 --seed $seed --epochs 30 --tag s0045
  python src/exp2_frontwheeler.py --dataset cifar10 --schedule pburst --pburst-period 1100 --seed $seed --epochs 30 --tag p1100
done
echo thesis controls done
