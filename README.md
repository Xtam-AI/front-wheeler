# front-wheeler

Two toy experiments toward an arXiv note (target: Sept 17).

## Exp 1 — learning is wiring, never updating
Frozen random MLP on MNIST; only a binary mask over weights is learned.
Compare mask-search rules: `ep` (gradient/edge-popup upper bound), `dfa`
(local, no weight transport), `rmhebb` (fully local three-factor Hebbian).
Controls: `random` mask, `readout` (train final layer only), `dense` (full BP).

```
python src/exp1_supermask.py --method ep --seed 0
```

## Exp 2 — front wheeler
Cheap steps update only the last hidden layer + output projection ("bounce");
full BP fires in bursts only when a loss-EMA plateau is detected. Goal:
match full-BP accuracy with a small fraction of full-BP steps.
Schedules: `full`, `front`, `periodic`, `fw`.

```
python src/exp2_frontwheeler.py --schedule fw --seed 0
```

## Setup (aiserver)
Project lives at `~/projects/front-wheeler`, micromamba env `front-wheeler`
(python 3.12, torch 2.13 cu130). Run everything via:

```
micromamba run -n front-wheeler bash scripts/run_all.sh
```

Results land in `results/exp1/*.csv`, `results/exp2/*.csv`.
