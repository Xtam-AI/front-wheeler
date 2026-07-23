#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for schedule in full front periodic fw; do
    python src/exp2_frontwheeler.py --schedule "$schedule" --seed "$seed" --epochs 20
  done
done
bash scripts/run_sweeps.sh
