"""Publication figures -> paper/figures/*.pdf (vector).

Palette: validated categorical subset (CVD-safe order); controls in neutral
grays so method colors carry identity.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")

BLUE, AQUA, RED, VIOLET = "#2a78d6", "#1baf7a", "#e34948", "#4a3aa7"
GRAY_D, GRAY_M, GRAY_L = "#3a3a38", "#8a8985", "#c3c2bd"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linewidth": 0.5,
    "lines.linewidth": 1.6, "pdf.fonttype": 42,
})


def load(subdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, subdir, "*.csv"))):
        m = re.match(r"exp\d_(\w+?)_s(\d+)(?:_(.*))?\.csv", os.path.basename(p))
        if not m:
            continue
        df = pd.read_csv(p)
        df["variant"] = m.group(1) + (f"_{m.group(3)}" if m.group(3) else "")
        df["seed"] = int(m.group(2))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def curve(df, variant):
    g = df[df.variant == variant].groupby("epoch")["test_acc"]
    return g.mean(), g.std()


# ---------------------------------------------------------------- Fig 1: exp1
def fig_exp1():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    specs = [("ep", BLUE, "EP (gradient)", 5), ("dfa", AQUA, "DFA (local)", -7),
             ("rmhebb", VIOLET, "RM-Hebb (scalar)", 0)]
    for ax, sub, title in [(axes[0], "exp1", "MNIST"),
                           (axes[1], "exp1_cifar10", "CIFAR-10")]:
        df = load(sub)
        for v, c, lab, dy in specs:
            mu, sd = curve(df, v)
            ax.plot(mu.index, mu.values, color=c)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
            ax.annotate(lab, (mu.index[-1], mu.values[-1]), xytext=(3, dy),
                        textcoords="offset points", color=c, fontsize=7,
                        va="center")
        for v, ls, lab, dy in [("dense", "-", "dense BP", 2),
                               ("readout", "--", "readout", -8),
                               ("random", ":", "random mask", 2)]:
            mu, _ = curve(df, v)
            y = mu.values[-1]
            ax.axhline(y, color=GRAY_M, ls=ls, lw=1.0, zorder=0)
            ax.annotate(lab, (0.02, y), xycoords=("axes fraction", "data"),
                        xytext=(0, dy), textcoords="offset points",
                        color=GRAY_D, fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xlim(left=0)
        ax.margins(x=0.16)
    axes[0].set_ylabel("test accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp1_locality.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: pareto
BUDGET = {"periodic_p50": .02, "periodic_p20": .05, "periodic": .10,
          "periodic_p5": .20, "periodic_p2": .50,
          "lpft_sf096": .04, "lpft_sf095": .05, "lpft_sf09": .10,
          "lpft_sf08": .20, "lpft_sf05": .50}


def finals(df):
    f = df.loc[df.groupby(["variant", "seed"])["epoch"].idxmax()].copy()
    f["full_frac"] = f["full_steps"] / f["step"]
    return f.groupby("variant").agg(acc=("test_acc", "mean"),
                                    sd=("test_acc", "std"),
                                    frac=("full_frac", "mean"))


def fig_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, sub, title in [(axes[0], "exp2", "MNIST"),
                           (axes[1], "exp2_cifar10", "CIFAR-10")]:
        s = finals(load(sub))
        for fam, color, lab, (li, dx, dy, ha) in [
                ("periodic", RED, "periodic-N", (0, -4, -8, "left")),
                ("lpft", AQUA, "LP-FT", (2, 0, 8, "center"))]:
            pts = sorted([(BUDGET[v], r.acc, r.sd) for v, r in s.iterrows()
                          if v in BUDGET and v.startswith(fam)])
            x, y, e = zip(*pts)
            ax.errorbar(x, y, yerr=e, color=color, marker="o", ms=3.5,
                        capsize=2, elinewidth=0.8)
            ax.annotate(lab, (x[li], y[li]), xytext=(dx, dy),
                        textcoords="offset points", color=color, fontsize=7,
                        ha=ha)
        fws = s[s.index.str.startswith("fw")]
        ax.errorbar(fws.frac, fws.acc, yerr=fws.sd, color=BLUE, marker="D",
                    ms=4.5, ls="none", capsize=2, elinewidth=0.8)
        ax.annotate("FW (adaptive)", (fws.frac.mean(), fws.acc.max()),
                    xytext=(0, 7), textcoords="offset points", color=BLUE,
                    fontsize=7, ha="center", fontweight="bold")
        b = s.loc["bcd"]
        ax.errorbar([b.frac], [b.acc], yerr=[b.sd], color=VIOLET, marker="s",
                    ms=4, ls="none", capsize=2, elinewidth=0.8)
        ax.annotate("BCD", (b.frac, b.acc), xytext=(9, -8),
                    textcoords="offset points", color=VIOLET, fontsize=7)
        for v, ls, lab in [("full", "--", "full BP (100%)"),
                           ("front", ":", "front only (0%)")]:
            r = s.loc[v]
            ax.axhline(r.acc, color=GRAY_M, ls=ls, lw=1.0, zorder=0)
            ax.annotate(lab, (0.98, r.acc), xycoords=("axes fraction", "data"),
                        xytext=(0, 2), textcoords="offset points",
                        color=GRAY_D, fontsize=6.5, ha="right")
        ax.set_xscale("log")
        ax.set_xticks([.02, .05, .1, .2, .5])
        ax.set_xticklabels(["2%", "5%", "10%", "20%", "50%"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("full-backprop budget (fraction of steps, log)")
    axes[0].set_ylabel("final test accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp2_pareto.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: sparsity
def fig_sparsity():
    df = load("exp1")
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    kept = {"sp03": .7, "": .5, "sp07": .3, "sp09": .1}
    for base, color, lab in [("ep", BLUE, "EP"), ("dfa", AQUA, "DFA")]:
        pts = []
        for tag, k in kept.items():
            v = base if tag == "" else f"{base}_{tag}"
            g = df[df.variant == v]
            f = g.loc[g.groupby("seed")["epoch"].idxmax()]["test_acc"]
            pts.append((k, f.mean(), f.std()))
        pts.sort()
        x, y, e = zip(*pts)
        ax.errorbar(x, y, yerr=e, color=color, marker="o", ms=3.5, capsize=2,
                    elinewidth=0.8)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(5, 0), va="center",
                    textcoords="offset points", color=color, fontsize=7)
    ax.margins(x=0.15)
    g = df[df.variant == "readout"]
    y = g.loc[g.groupby("seed")["epoch"].idxmax()]["test_acc"].mean()
    ax.axhline(y, color=GRAY_M, ls="--", lw=1.0, zorder=0)
    ax.annotate("readout", (0.97, y), xycoords=("axes fraction", "data"),
                xytext=(0, 2), textcoords="offset points", color=GRAY_D,
                fontsize=6.5, ha="right")
    ax.set_xlabel("fraction of connections kept")
    ax.set_ylabel("final test accuracy")
    ax.set_xticks([.1, .3, .5, .7])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp1_sparsity.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_exp1()
    fig_pareto()
    fig_sparsity()
    print("wrote", sorted(os.listdir(OUT)))
