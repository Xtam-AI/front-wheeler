"""Experiment 1: learning = wiring, never updating.

Frozen random MLP on MNIST. The only thing that changes during learning is a
binary mask over the frozen weights (which connections exist). We compare
mask-search rules:

  ep      edge-popup: gradient search over mask scores via backprop +
          straight-through estimator (the supermask upper bound — proves the
          target subnetwork exists, but smuggles BP back in).
  dfa     local rule, no weight transport: the layer's error signal is a fixed
          random projection of the output error (direct feedback alignment),
          combined with the layer's own pre/post activity. Score update is
          three-factor: broadcast error x pre x frozen weight.
  rmhebb  fully local three-factor Hebbian: eligibility = pre x post x frozen
          weight, modulated by a global scalar reward (correct/incorrect of a
          sampled action) minus a running baseline. REINFORCE-flavored.

Controls:
  random  fixed random mask at the same sparsity (no learning at all).
  readout backprop trains ONLY the final linear layer on frozen random
          features (shows masks do more than a linear readout).
  dense   ordinary BP training of all weights (no mask) — upper bound.
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CSVLogger, INPUT_DIMS, RESULTS_DIR, data_loaders, evaluate,
                    get_device, set_seed)


class GetSubnet(torch.autograd.Function):
    """Top-k mask over (already-abs) scores, straight-through gradient.

    Callers pass scores.abs() so autograd supplies the d|s|/ds = sign(s)
    factor — dropping it flips the update for negative-score synapses.
    """

    @staticmethod
    def forward(ctx, scores, k):
        out = torch.zeros_like(scores)
        _, idx = scores.flatten().sort(descending=True)
        out.flatten()[idx[: int(k * scores.numel())]] = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class MaskedLinear(nn.Module):
    def __init__(self, in_f, out_f, sparsity=0.5, init="signed_constant"):
        super().__init__()
        self.k = 1.0 - sparsity  # fraction of weights KEPT
        w = torch.empty(out_f, in_f)
        nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
        if init == "signed_constant":
            std = w.std()
            w = w.sign() * std
        self.register_buffer("weight", w)  # frozen forever
        self.scores = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def mask(self):
        return GetSubnet.apply(self.scores.abs(), self.k)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask())


class MaskedMLP(nn.Module):
    def __init__(self, dims, sparsity=0.5, init="signed_constant"):
        super().__init__()
        self.layers = nn.ModuleList(
            [MaskedLinear(dims[i], dims[i + 1], sparsity, init) for i in range(len(dims) - 1)]
        )

    def forward(self, x, record=False):
        acts = [x]  # post-activation inputs to each layer
        pre = []    # pre-activations z of each layer
        for i, layer in enumerate(self.layers):
            z = layer(x)
            pre.append(z)
            x = F.relu(z) if i < len(self.layers) - 1 else z
            acts.append(x)
        if record:
            return x, acts, pre
        return x


class DenseMLP(nn.Module):
    def __init__(self, dims, train_readout_only=False):
        super().__init__()
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)
        if train_readout_only:
            for p in self.net[:-1].parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- training loops

def train_ep(model, loaders, device, args, log):
    """Gradient mask search (edge-popup)."""
    train_loader, test_loader = loaders
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    step = 0
    for ep in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        sched.step()
        acc, tloss = evaluate(model, test_loader, device)
        log.log(epoch=ep, step=step, test_acc=acc, test_loss=round(tloss, 4))
        print(f"  epoch {ep}: test_acc={acc:.4f}")


def train_dense(model, loaders, device, args, log):
    train_ep(model, loaders, device, args, log)  # same loop, different model


@torch.no_grad()
def train_dfa(model, loaders, device, args, log):
    """Local mask search: DFA error broadcast x presynaptic activity x frozen weight.

    No backprop anywhere: each layer's 'delta' is a fixed random projection of
    the output error, gated by its own ReLU derivative. The score update for
    synapse (i<-j) uses only quantities available at that synapse:
    delta_i, activity_j, frozen weight w_ij.
    """
    train_loader, test_loader = loaders
    n_layers = len(model.layers)
    n_out = model.layers[-1].weight.shape[0]
    B = [torch.randn(model.layers[i].weight.shape[0], n_out, device=device)
         / math.sqrt(n_out) for i in range(n_layers - 1)]
    step = 0
    for ep in range(args.epochs):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * ep / args.epochs))
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, acts, pre = model(x, record=True)
            e = F.softmax(logits, 1) - F.one_hot(y, n_out).float()  # (bs, out)
            bs = x.shape[0]
            for i, layer in enumerate(model.layers):
                if i == n_layers - 1:
                    delta = e
                else:
                    delta = (e @ B[i].t()) * (pre[i] > 0).float()
                grad_w = delta.t() @ acts[i] / bs                     # (out_i, in_i)
                # straight-through: dL/ds = dL/dw_eff * w * sign(s)
                layer.scores -= lr * grad_w * layer.weight * layer.scores.sign()
            step += 1
        acc, tloss = evaluate(model, test_loader, device)
        log.log(epoch=ep, step=step, test_acc=acc, test_loss=round(tloss, 4))
        print(f"  epoch {ep}: test_acc={acc:.4f}")


@torch.no_grad()
def train_rmhebb(model, loaders, device, args, log):
    """Fully local three-factor rule: reward-modulated Hebbian mask search.

    Hidden layers see ONE global scalar per sample: reward - running baseline
    (reward = correctness of a sampled action, or negative per-sample loss).
    Eligibility at synapse (i<-j) is batch-centered post_i * pre_j * w_ij.
    The readout layer uses the exact local delta rule (its target is locally
    available); hidden layers get no error vector, only the scalar.
    """
    train_loader, test_loader = loaders
    n_out = model.layers[-1].weight.shape[0]
    baseline = 0.1 if args.reward == "action" else -math.log(1.0 / n_out)
    step = 0
    for ep in range(args.epochs):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * ep / args.epochs))
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, acts, pre = model(x, record=True)
            probs = F.softmax(logits / args.temp, 1)
            if args.reward == "action":
                action = torch.multinomial(probs, 1).squeeze(1)
                r = (action == y).float()                             # (bs,)
                m = r - baseline
                baseline = 0.99 * baseline + 0.01 * r.mean().item()
            else:  # reward = negative per-sample loss (still one global scalar/sample)
                loss_i = F.cross_entropy(logits, y, reduction="none")  # (bs,)
                m = baseline - loss_i                                  # higher = better
                baseline = 0.99 * baseline + 0.01 * loss_i.mean().item()
            bs = x.shape[0]
            for i, layer in enumerate(model.layers):
                if i == len(model.layers) - 1:
                    # readout's target is locally available: exact delta rule,
                    # not reward-modulated
                    delta = probs - F.one_hot(y, n_out).float()
                    elig = -(delta.t() @ acts[i]) / bs
                else:
                    # covariance rule: batch-centered activity, else ReLU's
                    # all-positive post*pre self-amplifies and logits blow up
                    post = F.relu(pre[i])
                    post = post - post.mean(0)
                    a = acts[i] - acts[i].mean(0)
                    elig = (post * m.unsqueeze(1)).t() @ a / bs
                elig = elig * layer.weight * layer.scores.sign()
                # RMS-normalize so lr sets the score step scale directly
                layer.scores += lr * elig / (elig.pow(2).mean().sqrt() + 1e-12)
            step += 1
        acc, tloss = evaluate(model, test_loader, device)
        log.log(epoch=ep, step=step, test_acc=acc, test_loss=round(tloss, 4))
        print(f"  epoch {ep}: test_acc={acc:.4f}, baseline={baseline:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["ep", "dfa", "rmhebb", "random", "readout", "dense"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    p.add_argument("--init", default="signed_constant", choices=["signed_constant", "kaiming"])
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--reward", default="loss", choices=["action", "loss"],
                   help="rmhebb modulator: sampled-action correctness or -loss")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--tag", default="")
    args = p.parse_args()

    default_lr = {"ep": 0.1, "dfa": 0.05, "rmhebb": 0.005, "random": 0.0,
                  "readout": 0.1, "dense": 0.1}
    if args.lr is None:
        args.lr = default_lr[args.method]

    set_seed(args.seed)
    device = get_device()
    loaders = data_loaders(args.dataset, args.batch_size)
    dims = [INPUT_DIMS[args.dataset]] + [args.hidden] * args.depth + [10]

    subdir = "exp1" if args.dataset == "mnist" else f"exp1_{args.dataset}"
    name = f"exp1_{args.method}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    log = CSVLogger(os.path.join(RESULTS_DIR, subdir, name + ".csv"),
                    ["epoch", "step", "test_acc", "test_loss", "wall_s"])
    print(f"[{name}] dims={dims} sparsity={args.sparsity} lr={args.lr} device={device}")

    if args.method in ("ep", "dfa", "rmhebb", "random"):
        model = MaskedMLP(dims, args.sparsity, args.init).to(device)
        if args.method == "ep":
            train_ep(model, loaders, device, args, log)
        elif args.method == "dfa":
            train_dfa(model, loaders, device, args, log)
        elif args.method == "rmhebb":
            train_rmhebb(model, loaders, device, args, log)
        else:  # random mask, no training — just evaluate
            acc, tloss = evaluate(model, loaders[1], device)
            log.log(epoch=0, step=0, test_acc=acc, test_loss=round(tloss, 4))
            print(f"  random mask: test_acc={acc:.4f}")
    else:
        model = DenseMLP(dims, train_readout_only=(args.method == "readout")).to(device)
        train_dense(model, loaders, device, args, log)
    log.close()


if __name__ == "__main__":
    main()
