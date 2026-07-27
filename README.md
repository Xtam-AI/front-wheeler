# Front Wheeler: Training with Full Backpropagation Only When It Pays

**XTAM Research Labs** · research@xtam.ai

Front wheeler (**FW**) is a training schedule that runs cheap steps by
default — updating only the last block and output head, with the trunk
forwarded gradient-free — and fires **bursts** of full backpropagation only
when a loss plateau signals the cheap subspace is exhausted, with exponential
backoff when a burst itself proves unproductive. On MLPs (MNIST, CIFAR-10)
and a 10.8M-parameter character-level transformer, FW at a few percent
deep-step budget beats or matches every fixed schedule we tested at matched
backward compute (ceding only the stationary mid-budget band, to schedules
whose required timing is catastrophic elsewhere); a single backoff
parameter acts as a budget dial along the compute–quality Pareto curve; and
under two mid-training distribution shifts FW leads or ties the cheap
field, with a one-line shift-aware governor reset making it the best cheap
schedule under the harshest condition tested.

Paper source is in `paper/` (arXiv preprint in preparation). Every number in
the paper is reproducible from the raw per-run CSVs in `results/` and the
exact commands in `scripts/`.

## Repository layout

```
paper/      LaTeX source (main.tex + sections/), figures; the appendix
            results tables are auto-generated (see scripts/report.py)
src/        experiments
  exp2_frontwheeler.py   FW on MLPs (MNIST / CIFAR-10), all schedules
  exp3_transformer.py    FW on a char-LM transformer, incl. distribution
                         shifts and the shift-aware governor reset
  exp1_supermask.py      origins study: supermask search with local rules
  paper_figures.py       regenerates all paper figures from results/
  resource_check.py      wall-clock / VRAM validation of the cost model
scripts/    exact run commands for every experiment in the paper, plus:
  report.py              canonical results tables from raw CSVs
                         (--latex regenerates the paper's appendix tables)
  verify_paper.py        checks every number in the paper's tables
                         against the raw CSVs (54 cells)
results/    raw per-run CSV logs (all seeds) and generated figures
notes/      historical planning notes (superseded by the paper)
```

## Quick start

Requires Python ≥ 3.10, PyTorch ≥ 2.0, torchvision, numpy, pandas,
matplotlib. Datasets (MNIST, CIFAR-10, tiny Shakespeare) download
automatically on first use into `data/`.

```bash
# FW on the transformer
# (schedules: full, front, periodic, lpft, ftlp, pburst, bcd, ratchet, fw)
python src/exp3_transformer.py --schedule fw --tau 0.02 --seed 0

# distribution shift with the shift-aware governor reset
python src/exp3_transformer.py --schedule fw --governor-reset \
  --shift-at 5400 --shift-corpus code --steps 10000 --seed 0

# verify the paper's numbers / rebuild the results tables
python scripts/verify_paper.py
python scripts/report.py

# FW on MLPs
python src/exp2_frontwheeler.py --dataset cifar10 --schedule fw --seed 0

# full experiment suites, exactly as reported in the paper
bash scripts/run_exp3.sh
bash scripts/run_all.sh
```

The complete experimental suite (three testbeds, all ablations and sweeps,
three seeds) consumes under ten GPU-hours on a single workstation GPU.

## Citing

See `CITATION.cff`, or cite the paper (preprint reference will be added on
arXiv posting). Licensed MIT.

This work was AI-assisted (Anthropic's Claude), under the direction and
review of the author; see the paper's acknowledgments.
