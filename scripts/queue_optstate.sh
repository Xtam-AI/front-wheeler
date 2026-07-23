#!/usr/bin/env bash
while pgrep -f "run_exp3.s[h]" >/dev/null; do sleep 60; done
export MAMBA_ROOT_PREFIX=~/.local/share/mamba
~/.local/bin/micromamba run -n front-wheeler bash scripts/optstate_ablation.sh > run_optstate.log 2>&1
