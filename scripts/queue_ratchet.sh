#!/usr/bin/env bash
while pgrep -f "optstate_ablatio[n]" >/dev/null || pgrep -f "tau_leve[r]" >/dev/null || pgrep -f "queue_optstat[e]" >/dev/null || pgrep -f "queue_leve[r]" >/dev/null; do sleep 120; done
export MAMBA_ROOT_PREFIX=~/.local/share/mamba
~/.local/bin/micromamba run -n front-wheeler bash scripts/ratchet_baseline.sh > run_ratchet.log 2>&1
