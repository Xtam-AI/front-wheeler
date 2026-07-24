#!/usr/bin/env bash
while pgrep -f "tau_leve[r]" >/dev/null || pgrep -f "ratchet_baselin[e]" >/dev/null || pgrep -f "queue_leve[r]" >/dev/null || pgrep -f "queue_ratche[t]" >/dev/null; do sleep 120; done
export MAMBA_ROOT_PREFIX=~/.local/share/mamba
~/.local/bin/micromamba run -n front-wheeler bash scripts/governor_sweep.sh > run_governor.log 2>&1
