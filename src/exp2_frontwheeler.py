"""Experiment 2: front wheeler — minimize full backprop steps.

Ordinary MLP on MNIST. Most steps are cheap "front" steps: forward the whole
net without building a graph, then update ONLY the last hidden layer and the
output projection (bounce between the final projection and the hidden layer
right before it). While front steps keep improving the loss, keep bouncing.
When improvement over a window drops below a threshold, pay for a burst of
full-BP steps, then go back to bouncing.

Schedules compared (same net, same data order per seed):
  full     full backprop every step (baseline).
  front    front steps only, never full BP (lower bound; = deep readout).
  periodic full BP every N-th step, front otherwise (fixed-budget control).
  fw       adaptive front wheeler: plateau-triggered full-BP bursts.

Cost metric: fraction of steps that ran full BP, plus estimated backward FLOPs
(a front step back-propagates through 2 of the D weight matrices only).
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CSVLogger, INPUT_DIMS, RESULTS_DIR, data_loaders, evaluate,
                    get_device, set_seed)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
        return x

    def forward_split(self, x):
        """Run trunk without a graph, return detached features for the front."""
        with torch.no_grad():
            for i, lin in enumerate(self.linears[:-2]):
                x = F.relu(lin(x))
        x = x.detach()
        h = F.relu(self.linears[-2](x))
        return self.linears[-1](h)

    def forward_partial(self, x, n_frozen):
        """Ratchet baseline: first n_frozen matrices frozen (no_grad)."""
        with torch.no_grad():
            for lin in self.linears[:n_frozen]:
                x = F.relu(lin(x))
        x = x.detach()
        for i in range(n_frozen, len(self.linears)):
            x = self.linears[i](x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
        return x


def backward_flops(dims, mode, n_frozen=None):
    """Rough backward cost in MACs per sample (~2 GEMMs per weight matrix:
    activation-grad + weight-grad). trunk-only still pays the activation-grad
    GEMM through the head matrices — BCD can't truncate depth."""
    mats = list(zip(dims[:-1], dims[1:]))
    if mode == "front":
        return sum(2 * a * b for a, b in mats[-2:])
    if mode == "full":
        return sum(2 * a * b for a, b in mats)
    if mode == "trunk":
        return sum(2 * a * b for a, b in mats[:-2]) + sum(a * b for a, b in mats[-2:])
    if mode == "partial":
        return sum(2 * a * b for a, b in mats[n_frozen:])
    raise ValueError(mode)


def run(args):
    set_seed(args.seed)
    device = get_device()
    train_loader, test_loader = data_loaders(args.dataset, args.batch_size)
    dims = [INPUT_DIMS[args.dataset]] + [args.hidden] * args.depth + [10]
    model = MLP(dims).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    subdir = "exp2" if args.dataset == "mnist" else f"exp2_{args.dataset}"
    name = f"exp2_{args.schedule}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    log = CSVLogger(os.path.join(RESULTS_DIR, subdir, name + ".csv"),
                    ["epoch", "step", "full_steps", "bp_flops", "train_loss_ema",
                     "test_acc", "test_loss", "wall_s"])
    print(f"[{name}] dims={dims} lr={args.lr} device={device}")

    full_steps = 0
    bp_flops = 0
    step = 0
    ema = None                 # EMA of training loss
    ema_at_check = None        # EMA value at the last plateau check
    burst_left = 0             # remaining forced full-BP steps
    fw_state = "front"
    ema_at_burst_start = None
    cooldown = args.check_every  # front steps required before next plateau check
    steps_since_burst = 0

    head_params = list(model.linears[-2].parameters()) + list(model.linears[-1].parameters())

    def do_step(x, y, mode, n_frozen=None):
        nonlocal full_steps, bp_flops, ema
        if mode == "front":
            logits = model.forward_split(x)
        elif mode == "partial":
            logits = model.forward_partial(x, n_frozen)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        if mode == "trunk":  # BCD: complementary block, head frozen
            for p in head_params:
                p.grad = None
        # front-trained heads send rare huge error kicks into the stale trunk;
        # unclipped + momentum this dead-ReLUs the net at period >= 10
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        full_steps += int(mode == "full" or (mode == "partial" and n_frozen == 0))
        bp_flops += backward_flops(dims, mode, n_frozen) * x.shape[0]
        l = loss.item()
        ema = l if ema is None else 0.98 * ema + 0.02 * l
        return l

    total_steps = args.epochs * len(train_loader)
    switch_step = int(args.switch_frac * total_steps)
    burst_mode = "trunk" if args.schedule == "bcd" else "full"

    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.schedule == "full":
                mode = "full"
            elif args.schedule == "front":
                mode = "front"
            elif args.schedule == "periodic":
                mode = "full" if step % args.period == 0 else "front"
            elif args.schedule == "lpft":
                # linear-probe-then-finetune: one switch, front -> full
                mode = "front" if step < switch_step else "full"
            elif args.schedule == "ratchet":
                # progressive bottom-up freezing (AutoFreeze/Egeria-style
                # fixed ratchet): all-but-front frozen by ratchet_by * T
                n_freezable = len(model.linears) - 2
                frac = min(1.0, step / max(1, int(args.ratchet_by * total_steps)))
                mode, nf = "partial", int(n_freezable * frac)
            else:  # fw / bcd: adaptive with burst backoff
                if burst_left > 0:
                    mode = burst_mode
                    burst_left -= 1
                    if burst_left == 0:
                        # if the burst itself didn't help, the plateau is
                        # global, not the front's fault -> back off
                        burst_impr = (ema_at_burst_start - ema) / max(abs(ema_at_burst_start), 1e-8)
                        if burst_impr < args.tau:
                            cooldown = min(cooldown * 2, args.max_cooldown)
                        else:
                            cooldown = args.check_every
                        fw_state = "front"
                        ema_at_check = ema
                        steps_since_burst = 0
                else:
                    mode = "front"
                    steps_since_burst += 1
                    if steps_since_burst >= cooldown and steps_since_burst % args.check_every == 0:
                        if ema_at_check is not None:
                            rel_impr = (ema_at_check - ema) / max(abs(ema_at_check), 1e-8)
                            if rel_impr < args.tau:
                                burst_left = args.burst
                                fw_state = "burst"
                                ema_at_burst_start = ema
                                mode = burst_mode
                                burst_left -= 1
                        ema_at_check = ema
            do_step(x, y, mode, nf if args.schedule == "ratchet" else None)
            step += 1
        acc, tloss = evaluate(model, test_loader, device)
        log.log(epoch=epoch, step=step, full_steps=full_steps, bp_flops=bp_flops,
                train_loss_ema=round(ema, 5), test_acc=acc, test_loss=round(tloss, 4))
        print(f"  epoch {epoch}: acc={acc:.4f} full={full_steps}/{step} "
              f"({100 * full_steps / step:.1f}%) state={fw_state}")
    log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True,
                   choices=["full", "front", "periodic", "fw", "lpft", "bcd",
                            "ratchet"])
    p.add_argument("--ratchet-by", type=float, default=0.3,
                   help="ratchet: fraction of training by which all-but-front is frozen")
    p.add_argument("--switch-frac", type=float, default=0.9,
                   help="lpft: fraction of training spent in the probe phase")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4, help="number of hidden layers")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--period", type=int, default=10, help="periodic: full BP every N steps")
    p.add_argument("--check-every", type=int, default=50, help="fw: plateau check interval")
    p.add_argument("--tau", type=float, default=0.02, help="fw: relative improvement threshold")
    p.add_argument("--burst", type=int, default=50, help="fw: full-BP steps per burst")
    p.add_argument("--max-cooldown", type=int, default=2000,
                   help="fw: cap on post-unproductive-burst backoff (front steps)")
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--tag", default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
