#!/usr/bin/env bash
while pgrep -f "run_exp3.s[h]" >/dev/null || pgrep -f "optstate_ablatio[n]" >/dev/null || pgrep -f "queue_optstat[e]" >/dev/null; do sleep 120; done
export MAMBA_ROOT_PREFIX=~/.local/share/mamba
~/.local/bin/micromamba run -n front-wheeler bash scripts/tau_lever.sh > run_lever.log 2>&1
