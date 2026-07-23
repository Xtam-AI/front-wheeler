"""Aggregate results CSVs into summary tables and figures (all datasets)."""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def load_runs(subdir):
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, subdir, "*.csv"))):
        m = re.match(r"exp\d_(\w+?)_s(\d+)(?:_(.*))?\.csv", os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        df["variant"] = m.group(1) + (f"_{m.group(3)}" if m.group(3) else "")
        df["seed"] = int(m.group(2))
        runs.append(df)
    return pd.concat(runs, ignore_index=True) if runs else None


def summarize_exp1(subdir):
    df = load_runs(subdir)
    if df is None:
        return
    final = df.loc[df.groupby(["variant", "seed"])["epoch"].idxmax()]
    summary = final.groupby("variant")["test_acc"].agg(["mean", "std", "count"])
    summary.to_csv(os.path.join(RESULTS_DIR, f"{subdir}_summary.csv"))
    print(f"== {subdir}: final test accuracy (mean over seeds) ==")
    print(summary.round(4).to_string(), "\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, g in df.groupby("variant"):
        curve = g.groupby("epoch")["test_acc"].mean()
        ax.plot(curve.index, curve.values, label=variant)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_title(f"{subdir}: mask-search rules on a frozen random MLP")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{subdir}_curves.png"), dpi=150)
    plt.close(fig)


def summarize_exp2(subdir):
    df = load_runs(subdir)
    if df is None:
        return
    final = df.loc[df.groupby(["variant", "seed"])["epoch"].idxmax()].copy()
    final["full_frac"] = final["full_steps"] / final["step"]
    summary = final.groupby("variant").agg(
        acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
        full_frac=("full_frac", "mean"), bp_gflops=("bp_flops", lambda s: s.mean() / 1e9))
    summary.to_csv(os.path.join(RESULTS_DIR, f"{subdir}_summary.csv"))
    print(f"== {subdir}: final accuracy vs full-BP budget ==")
    print(summary.round(4).to_string(), "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for variant, g in df.groupby("variant"):
        curve = g.groupby("epoch")["test_acc"].mean()
        axes[0].plot(curve.index, curve.values, label=variant)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("test accuracy")
    axes[0].set_title("accuracy vs epoch"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for variant, g in final.groupby("variant"):
        axes[1].scatter(g["bp_flops"] / 1e9, g["test_acc"], label=variant, s=40)
    axes[1].set_xlabel("backward GFLOPs/sample-equivalent (total, est.)")
    axes[1].set_ylabel("final test accuracy")
    axes[1].set_title("accuracy vs backward compute"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{subdir}_tradeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for d in sorted(os.listdir(RESULTS_DIR)):
        if not os.path.isdir(os.path.join(RESULTS_DIR, d)):
            continue
        if d.startswith("exp1"):
            summarize_exp1(d)
        elif d.startswith("exp2"):
            summarize_exp2(d)
    print(f"Figures in {FIG_DIR}")
