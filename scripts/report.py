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
    ("fw",""): "FW(t=.02,k=2000)", ("fw","tau002"): "FW(t=.02,k=2000)",
    ("fw","tau0005"): "FW(t=.005,k=2000)",
    ("fw","tau001"): "FW(t=.01,k=2000)", ("fw","tau005"): "FW(t=.05,k=2000)",
    ("periodic",""): "PERIODIC-10", ("periodic","p10"): "PERIODIC-10",
    ("periodic","p2"): "PERIODIC-2",
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


def latex_appendix(path):
    """Emit the complete-results appendix, mechanically, with stable labels."""
    L = []
    L.append("% AUTO-GENERATED by scripts/report.py --latex. Do not edit by hand.")
    L.append("\\section{Complete results tables}")
    L.append("\\label{app:tables}")
    L.append("All numbers are mean $\\pm$ std over seeds $\\{0,1,2\\}$ unless "
             "$n$ is noted; regenerate this appendix with "
             "\\texttt{python scripts/report.py --latex}.")

    def fmt(m, s, n, nd=4):
        s_txt = f" \\pm {s:.{nd}f}" if (s == s and s is not None) else ""
        note = f" \\,({n})" if n != 3 else ""
        return f"${m:.{nd}f}{s_txt}$" + note

    # --- MLP combined table
    rows = {}
    for sub, col in [("exp2", "MNIST"), ("exp2_cifar10", "CIFAR")]:
        d = finals(sub, lambda s, t: not any(k in t for k in
                   ["lever", "rb", "smoke", "verify", "s0045", "p1100"]),
                   "test_acc", "step")
        for (sched, tag), grp in d.groupby(["sched", "tag"]):
            name = NAME.get((sched, tag))
            if not name: continue
            rows.setdefault(name, {})[col] = (grp.metric.mean(), grp.metric.std(),
                                              len(grp), grp.deep.mean())
    L.append("\\begin{table}[h]\\centering\\small")
    L.append("\\caption{MLPs, stationary: all step-budget schedules, all "
             "budgets. Test accuracy; compare only within matched budget "
             "tiers. \\textsc{ratchet} is compared on the compute axis in "
             "the text (Section~\\ref{sec:exp2}).}")
    L.append("\\label{tab:exp2}")
    L.append("\\begin{tabular}{lccc}\\toprule")
    L.append("Schedule & Deep steps & MNIST & CIFAR-10 \\\\\\midrule")
    for name in sorted(rows):
        r = rows[name]
        deep = r.get("MNIST", r.get("CIFAR"))[3]
        mn = fmt(*r["MNIST"][:3]) if "MNIST" in r else "---"
        cf = fmt(*r["CIFAR"][:3]) if "CIFAR" in r else "---"
        L.append(f"{name} & {100*deep:.1f}\\% & {mn} & {cf} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # --- transformer stationary
    d = finals("exp3_char", lambda s, t: t in ("", "tau002", "tau0005",
               "tau005", "p5", "p10", "p20", "p50", "sf05", "sf08", "sf09",
               "sf095", "rb01", "rb03", "s006", "s02", "p833", "p250",
               "p833rw", "gov100", "gov200", "gov400", "gov800", "lever01",
               "lever01nb"), "val_loss", "step")
    fg = d[(d.sched == "full") & (d.tag == "")].gf.mean()
    L.append("\\begin{table}[h]\\centering\\small")
    L.append("\\caption{Transformer, stationary: complete population. "
             "Validation loss (lower is better); backward compute as \\% of "
             "\\textsc{full}'s; compare at matched backward compute.}")
    L.append("\\label{tab:exp3matrix}")
    L.append("\\begin{tabular}{lccc}\\toprule")
    L.append("Schedule & Deep steps & Bwd.\\ compute & Val.\\ loss \\\\\\midrule")
    for (sched, tag), grp in sorted(d.groupby(["sched", "tag"]),
                                    key=lambda kv: NAME.get(kv[0], "")):
        name = NAME.get((sched, tag))
        if not name: continue
        L.append(f"{name} & {100*grp.deep.mean():.1f}\\% & "
                 f"{100*grp.gf.mean()/fg:.1f}\\% & "
                 f"{fmt(grp.metric.mean(), grp.metric.std(), len(grp))} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    # --- shifts
    L.append("\\begin{table}[h]\\centering\\small")
    L.append("\\caption{Distribution shifts: post-shift validation loss. "
             "Mild = tiny Shakespeare $\\to$ War and Peace; harsh = $\\to$ C "
             "source code. In the mild column, \\fw{} and \\textsc{pburst} "
             "use the fair placement (switch at step 5{,}400, mid-gap for "
             "the clock) and the anchors and one-switch/deep-first rows the "
             "original placement (5{,}000); \\fw{} moves only ${\\sim}0.009$ "
             "between placements. The harsh column uses the fair placement "
             "throughout. `+govreset' is \\fw{} with the shift-aware "
             "governor reset.}")
    L.append("\\label{tab:shift}")
    L.append("\\begin{tabular}{lccc}\\toprule")
    L.append("Schedule & Deep steps & Mild & Harsh \\\\\\midrule")
    shift_rows = {}
    for tag, col in [("shift2", "mild"), ("shift3", "mild_fair"),
                     ("hshift", "harsh"), ("hshiftgr", "harsh_gr")]:
        d = finals("exp3_char", lambda s, t, tag=tag: t == tag, "val_loss",
                   "step", full_steps_expected=10000)
        for (sched, _), grp in d.groupby(["sched", "tag"]):
            shift_rows.setdefault(sched, {})[col] = (
                grp.metric.mean(), grp.metric.std(), len(grp), grp.deep.mean())
    order = ["full", "fw", "pburst", "ftlp", "lpft", "front"]
    disp = {"full": "\\textsc{full}", "fw": "\\fw{}($\\tau{=}.02$)",
            "pburst": "\\textsc{pburst}(833)", "ftlp": "\\textsc{ftlp}(.06)",
            "lpft": "\\lpft{}(.9)", "front": "\\textsc{front}"}
    for sched in order:
        r = shift_rows.get(sched, {})
        mild = r.get("mild_fair") or r.get("mild")
        harsh = r.get("harsh")
        deep = (harsh or mild)[3]
        L.append(f"{disp[sched]} & {100*deep:.1f}\\% & "
                 f"{fmt(*mild[:3]) if mild else '---'} & "
                 f"{fmt(*harsh[:3]) if harsh else '---'} \\\\")
    gr = shift_rows.get("fw", {}).get("harsh_gr")
    if gr:
        L.append(f"\\fw{{}}($\\tau{{=}}.02$)+govreset & {100*gr[3]:.1f}\\% & "
                 f"--- & {fmt(*gr[:3])} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path}")


if "--latex" in sys.argv:
    latex_appendix("paper/sections/appendix_tables.tex")
