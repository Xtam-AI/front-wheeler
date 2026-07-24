"""One run per schedule on the transformer, measuring real resources:
wall clock, peak VRAM, and per-mode step times. Validates (or refutes) the
backward-FLOPs proxy used in the paper."""
import time

import torch
import torch.nn.functional as F

from common import get_device, set_seed
from exp3_transformer import CharGPT, bwd_macs_per_token, get_batch, load_data

STEPS = 2000
BATCH, CTX = 64, 256
D, HEADS, LAYERS = 384, 6, 6


def schedule_modes(name, steps):
    """Yield (mode, n_frozen) per step. fw uses its recorded ~6% cadence:
    burst of 50 every ~800 steps after an initial front phase."""
    for t in range(steps):
        if name == "full":
            yield "full", None
        elif name == "front":
            yield "front", None
        elif name == "fw":  # representative recorded cadence, ~6% deep
            yield ("full", None) if (t % 800) < 50 and t > 400 else ("front", None)
        elif name.startswith("ratchet"):
            by = 0.1 if name.endswith("01") else 0.3
            frac = min(1.0, t / (by * steps))
            yield "ratchet", int((LAYERS - 1) * frac)


def run(name):
    set_seed(0)
    device = get_device()
    train, _, vocab = load_data(device)
    model = CharGPT(vocab, D, HEADS, LAYERS, CTX, 0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    gen = torch.Generator(device=device.type).manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    mode_times = {}
    flops = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for mode, nf in schedule_modes(name, STEPS):
        ts = time.perf_counter()
        x, y = get_batch(train, BATCH, CTX, gen)
        if mode == "front":
            logits = model.forward_split(x)
        elif mode == "ratchet":
            logits = model.forward_ratchet(x, nf)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        key = mode if mode != "ratchet" else f"ratchet(nf={nf})"
        mode_times.setdefault(mode, []).append(time.perf_counter() - ts)
        flops += bwd_macs_per_token(D, vocab, LAYERS, mode, nf) * BATCH * CTX
    wall = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    per_mode = {m: f"{1e3*sum(v)/len(v):.1f}ms x{len(v)}" for m, v in mode_times.items()}
    print(f"{name:10s} wall={wall:6.1f}s  peakVRAM={peak_gb:5.2f}GB  "
          f"bwdGF={flops/1e9:9.0f}  {per_mode}")
    return wall, peak_gb, flops


if __name__ == "__main__":
    print(f"{STEPS} steps, batch {BATCH}, ctx {CTX}")
    results = {n: run(n) for n in ["full", "front", "fw", "ratchet01", "ratchet03"]}
    wf, _, ff = results["full"]
    print("\nrelative to full:  (wall, flops-predicted)")
    for n, (w, _, f) in results.items():
        print(f"  {n:10s} wall {w/wf:4.2f}  flops {f/ff:4.2f}")
