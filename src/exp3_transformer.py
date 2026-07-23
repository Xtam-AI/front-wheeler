"""Experiment 3: front wheeler on a transformer (char-LM, tiny Shakespeare).

Same question as exp2, real architecture: front steps update only the last
transformer block + final LayerNorm + LM head (trunk forwarded under
no_grad); full-BP bursts fire on loss-EMA plateau with backoff. The LM head
is deliberately UNTIED from the token embedding so front steps cannot leak
trunk updates through weight tying.
"""
import argparse
import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CSVLogger, DATA_DIR, RESULTS_DIR, get_device, set_seed

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d, h, ctx, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(drop))
        self.register_buffer("mask", torch.triu(
            torch.full((ctx, ctx), float("-inf")), diagonal=1))

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T],
                         need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab, d=384, h=6, layers=6, ctx=256, drop=0.2):
        super().__init__()
        self.ctx = ctx
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(ctx, d)
        self.emb_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, h, ctx, drop) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)  # untied on purpose

    def trunk(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.emb_drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks[:-1]:
            x = b(x)
        return x

    def front(self, x):
        return self.head(self.ln_f(self.blocks[-1](x)))

    def forward(self, idx):
        return self.front(self.trunk(idx))

    def forward_split(self, idx):
        with torch.no_grad():
            x = self.trunk(idx)
        return self.front(x.detach())


# ---------------------------------------------------------------- data

def load_data(device):
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n].to(device), data[n:].to(device), len(chars)


def get_batch(data, bs, ctx, gen):
    ix = torch.randint(len(data) - ctx - 1, (bs,), generator=gen,
                       device=data.device)
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_loss(model, vdata, bs, ctx, iters=20):
    model.eval()
    g = torch.Generator(device=vdata.device.type).manual_seed(1234)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(vdata, bs, ctx, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / iters


# ---------------------------------------------------------------- cost model

def bwd_macs_per_token(d, vocab, layers, mode):
    """Weight-GEMM backward MACs per token (2 GEMMs per weight matrix);
    attention-score terms omitted uniformly across modes."""
    block = 12 * d * d          # qkv 3d^2 + proj d^2 + mlp 8d^2
    head = d * vocab
    if mode == "front":
        return 2 * (block + head)
    if mode == "full":
        return 2 * (layers * block + head)
    if mode == "trunk":         # activation grads still traverse the front
        return 2 * (layers - 1) * block + (block + head)
    raise ValueError(mode)


# ---------------------------------------------------------------- training

def run(args):
    set_seed(args.seed)
    device = get_device()
    train, val, vocab = load_data(device)
    model = CharGPT(vocab, args.dmodel, args.heads, args.layers,
                    args.ctx, args.dropout).to(device)
    front_params = (list(model.blocks[-1].parameters())
                    + list(model.ln_f.parameters())
                    + list(model.head.parameters()))
    front_ids = {id(p) for p in front_params}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min((t + 1) / args.warmup, 1.0)
        * 0.5 * (1.0 + math.cos(math.pi * t / args.steps)))
    gen = torch.Generator(device=device.type).manual_seed(args.seed)

    name = f"exp3_{args.schedule}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    log = CSVLogger(os.path.join(RESULTS_DIR, "exp3_char", name + ".csv"),
                    ["step", "full_steps", "bp_flops", "train_loss_ema",
                     "val_loss", "wall_s"])
    print(f"[{name}] vocab={vocab} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"device={device}")

    switch_step = int(args.switch_frac * args.steps)
    burst_mode = "trunk" if args.schedule == "bcd" else "full"
    full_steps = 0
    bp_flops = 0
    ema = None
    ema_at_check = None
    ema_at_burst_start = None
    burst_left = 0
    cooldown = args.check_every
    steps_since_burst = 0

    for step in range(args.steps):
        if args.schedule == "full":
            mode = "full"
        elif args.schedule == "front":
            mode = "front"
        elif args.schedule == "periodic":
            mode = "full" if step % args.period == 0 else "front"
        elif args.schedule == "lpft":
            mode = "front" if step < switch_step else "full"
        else:  # fw / bcd
            if burst_left > 0:
                mode = burst_mode
                burst_left -= 1
                if burst_left == 0:
                    impr = (ema_at_burst_start - ema) / max(abs(ema_at_burst_start), 1e-8)
                    cooldown = (min(cooldown * 2, args.max_cooldown)
                                if impr < args.tau else args.check_every)
                    ema_at_check = ema
                    steps_since_burst = 0
            else:
                mode = "front"
                steps_since_burst += 1
                if (steps_since_burst >= cooldown
                        and steps_since_burst % args.check_every == 0):
                    if ema_at_check is not None:
                        rel = (ema_at_check - ema) / max(abs(ema_at_check), 1e-8)
                        if rel < args.tau:
                            burst_left = args.burst - 1
                            ema_at_burst_start = ema
                            mode = burst_mode
                    ema_at_check = ema

        x, y = get_batch(train, args.batch_size, args.ctx, gen)
        logits = model(x) if mode != "front" else model.forward_split(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        if mode == "trunk":
            for p in model.parameters():
                if id(p) in front_ids:
                    p.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        full_steps += int(mode != "front")
        bp_flops += (bwd_macs_per_token(args.dmodel, vocab, args.layers, mode)
                     * args.batch_size * args.ctx)
        l = loss.item()
        ema = l if ema is None else 0.98 * ema + 0.02 * l

        if (step + 1) % args.eval_every == 0:
            vl = val_loss(model, val, args.batch_size, args.ctx)
            log.log(step=step + 1, full_steps=full_steps, bp_flops=bp_flops,
                    train_loss_ema=round(ema, 5), val_loss=round(vl, 5))
            if (step + 1) % (4 * args.eval_every) == 0:
                print(f"  step {step+1}: val={vl:.4f} "
                      f"full={full_steps}/{step+1} "
                      f"({100*full_steps/(step+1):.1f}%)")
    log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True,
                   choices=["full", "front", "periodic", "fw", "lpft", "bcd"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dmodel", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--period", type=int, default=10)
    p.add_argument("--switch-frac", type=float, default=0.9)
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--burst", type=int, default=50)
    p.add_argument("--max-cooldown", type=int, default=2000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
