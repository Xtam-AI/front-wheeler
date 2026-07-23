#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  python src/exp3_transformer.py --schedule full  --seed $seed
  python src/exp3_transformer.py --schedule front --seed $seed
  for p in 5 10 20 50; do
    python src/exp3_transformer.py --schedule periodic --period $p --seed $seed --tag "p$p"
  done
  for sf in 0.5 0.8 0.9 0.95; do
    python src/exp3_transformer.py --schedule lpft --switch-frac $sf --seed $seed --tag "sf${sf//./}"
  done
  for tau in 0.005 0.02 0.05; do
    python src/exp3_transformer.py --schedule fw --tau $tau --seed $seed --tag "tau${tau//./}"
  done
  python src/exp3_transformer.py --schedule bcd --seed $seed
done
echo exp3 done
