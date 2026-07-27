"""Canonical results report, generated mechanically from raw CSVs.
One table per (condition, metric). No rankings, no derived quantities.
Row label = schedule(all params). n = seeds aggregated."""
import glob, os, re, sys
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def finals(subdir, tag_filter, metric, step_col, full_steps_expected=None):
    rows = []
    for p in sorted(glob.glob(os.path.join(R, subdir, "*.csv"))):
        m = re.match(r"exp\d_(\w+?)_s(\d+)(?:_(.*))?\.csv", os.path.basename(p))
        if not m: continue
        sched, seed, tag = m.group(1), int(m.group(2)), m.group(3) or ""
        if not tag_filter(sched, tag): continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not len(df): continue
        last = df.iloc[-1]
        if full_steps_expected and last[step_col] != full_steps_expected: continue
        rows.append(dict(sched=sched, tag=tag, seed=seed, metric=last[metric],
                         deep=last.full_steps / last[step_col],
                         gf=last.bp_flops))
    return pd.DataFrame(rows)

NAME = {
    ("full",""): "FULL", ("front",""): "FRONT", ("bcd",""): "BCD",
    ("fw",""): "FW(t=.02,k=2000)", ("fw","tau0005"): "FW(t=.005,k=2000)",
    ("fw","tau001"): "FW(t=.01,k=2000)", ("fw","tau005"): "FW(t=.05,k=2000)",
    ("periodic",""): "PERIODIC-10", ("periodic","p2"): "PERIODIC-2",
    ("periodic","p5"): "PERIODIC-5", ("periodic","p20"): "PERIODIC-20",
    ("periodic","p50"): "PERIODIC-50",
    ("lpft","sf05"): "LP-FT(.5)", ("lpft","sf08"): "LP-FT(.8)",
    ("lpft","sf09"): "LP-FT(.9)", ("lpft","sf095"): "LP-FT(.95)",
    ("lpft","sf096"): "LP-FT(.96)",
    ("ratchet","rb01"): "RATCHET(.1)", ("ratchet","rb03"): "RATCHET(.3)",
    ("ftlp","s006"): "FTLP(.06)", ("ftlp","s02"): "FTLP(.2)",
    ("pburst","p833"): "PBURST(833)", ("pburst","p250"): "PBURST(250)",
    ("pburst","p833rw"): "PBURST(833)+resetwarmup",
    ("fw","gov100"): "FW(t=.1,k=100)", ("fw","gov200"): "FW(t=.1,k=200)",
    ("fw","gov400"): "FW(t=.1,k=400)", ("fw","gov800"): "FW(t=.1,k=800)",
    ("fw","lever01"): "FW(t=.1,k=2000)", ("fw","lever01nb"): "FW(t=.1,k=50)",
    # bare names appear only in shift-context tables (params fixed per protocol)
    ("ftlp",""): "FTLP(.06)", ("pburst",""): "PBURST(833)",
    ("lpft",""): "LP-FT(.9)",
}

def emit(title, d, metric_name, full_gf=None):
    if not len(d): return
    print(f"\n== {title} | metric: {metric_name} ==")
    g = d.groupby(["sched", "tag"])
    out = []
    for (sched, tag), grp in g:
        name = NAME.get((sched, tag))
        if name is None: continue
        row = dict(schedule=name,
                   mean=round(grp.metric.mean(), 4),
                   std=round(grp.metric.std(), 4) if len(grp) > 1 else None,
                   n=len(grp),
                   deep_steps_pct=round(100 * grp.deep.mean(), 1))
        if full_gf:
            row["bwd_compute_pct"] = round(100 * grp.gf.mean() / full_gf, 1)
        out.append(row)
    print(pd.DataFrame(out).sort_values("schedule").to_string(index=False))

# stationary
for sub, metric, title in [("exp2", "test_acc", "MLP MNIST stationary"),
                            ("exp2_cifar10", "test_acc", "MLP CIFAR-10 stationary")]:
    d = finals(sub, lambda s, t: not any(k in t for k in
               ["lever", "rb", "smoke", "verify", "s0045", "p1100"]), metric, "step")
    emit(title, d, metric + " (higher better)")

d = finals("exp3_char", lambda s, t: t in ("", "tau0005", "tau001", "tau005",
           "p2", "p5", "p20", "p50", "sf05", "sf08", "sf09", "sf095",
           "rb01", "rb03", "s006", "s02", "p833", "p250", "p833rw",
           "gov100", "gov200", "gov400", "gov800", "lever01", "lever01nb"),
           "val_loss", "step")
fg = d[(d.sched == "full") & (d.tag == "")].gf.mean()
emit("Transformer stationary", d, "val_loss (lower better)", full_gf=fg)

# shifts
for tag, title, suffix in [
        ("shift2", "Transformer shift MILD (prose->prose, at 5000)", ""),
        ("shift3", "Transformer shift MILD fair-clock (at 5400)", ""),
        ("hshift", "Transformer shift HARSH (prose->code, at 5400)", ""),
        ("hshiftgr", "Transformer shift HARSH, governor-reset arm",
         "+govreset")]:
    d = finals("exp3_char", lambda s, t, tag=tag: t == tag, "val_loss", "step",
               full_steps_expected=10000)
    if not len(d): continue
    d["tag"] = ""
    if suffix:
        d = d.copy()
    emit(title, d, "val_loss (lower better)")
    if suffix:
        print(f"  (row above is FW(t=.02,k=2000){suffix})")
print("\nNote: resource metrics (wall clock, peak VRAM) exist only in the")
print("single-run study src/resource_check.py and are reported separately.")
