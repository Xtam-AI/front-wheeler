"""Mechanically verify every number in the paper's main tables against raw CSVs."""
import glob, os, re
import pandas as pd

R = "results"

def agg(subdir, sched, tag, metric):
    vals = []
    for p in glob.glob(os.path.join(R, subdir, f"exp*_{sched}_s*.csv")):
        m = re.match(rf"exp\d_{sched}_s(\d+)(?:_(.*))?\.csv", os.path.basename(p))
        if not m or (m.group(2) or "") != tag: continue
        df = pd.read_csv(p)
        vals.append(df.iloc[-1][metric])
    s = pd.Series(vals)
    return s.mean(), (s.std() if len(s) > 1 else 0), len(s)

def check(label, paper_mean, paper_std, subdir, sched, tag, metric, scale=1, nd=3):
    m, sd, n = agg(subdir, sched, tag, metric)
    m, sd = round(m * scale, nd), round(sd * scale, nd)
    ok_m = abs(m - paper_mean) < 1.5 * 10**-nd
    ok_s = paper_std is None or abs(sd - paper_std) < 1.5 * 10**-nd
    status = "OK " if (ok_m and ok_s) else "MISMATCH"
    print(f"{status} {label:34s} paper={paper_mean}±{paper_std}  csv={m}±{sd} (n={n})")

print("--- Table: exp3 matrix (transformer stationary, val loss) ---")
for lbl, pm, ps, sc, tg in [
    ("FULL", 1.488, 0.008, "full", ""), ("FRONT", 1.836, 0.004, "front", ""),
    ("FW(.02,2000)", 1.773, 0.013, "fw", "tau002"),
    ("FW(.1,200)", 1.717, 0.009, "fw", "gov200"),
    ("FW(.1,100)", 1.668, 0.010, "fw", "gov100"),
    ("FW(.1,50)", 1.596, 0.009, "fw", "lever01nb"),
    ("BCD", 1.797, 0.010, "bcd", ""),
    ("LP-FT(.9)", 1.832, 0.004, "lpft", "sf09"),
    ("LP-FT(.8)", 1.823, 0.002, "lpft", "sf08"),
    ("LP-FT(.5)", 1.766, 0.007, "lpft", "sf05"),
    ("PERIODIC-50", 1.873, 0.031, "periodic", "p50"),
    ("PERIODIC-20", 1.910, 0.059, "periodic", "p20"),
    ("PERIODIC-10", 2.047, 0.127, "periodic", "p10"),
    ("PERIODIC-5", 2.404, 0.046, "periodic", "p5"),
    ("RATCHET(.1)", 2.271, 0.042, "ratchet", "rb01"),
    ("RATCHET(.3)", 1.696, 0.009, "ratchet", "rb03"),
    ("FTLP(.06)", 2.290, 0.035, "ftlp", "s006"),
    ("FTLP(.2)", 1.665, 0.007, "ftlp", "s02"),
    ("PBURST(833)", 1.814, 0.014, "pburst", "p833"),
    ("PBURST(250)", 1.747, 0.016, "pburst", "p250"),
]:
    check(lbl, pm, ps, "exp3_char", sc, tg, "val_loss")

print("--- Table: shift (post-shift val loss) ---")
for lbl, pm, ps, sc, tg in [
    ("FULL mild", 1.238, 0.004, "full", "shift2"),
    ("FW fair", 1.549, 0.013, "fw", "shift3"),
    ("PBURST fair", 1.561, 0.008, "pburst", "shift3"),
    ("FTLP mild", 1.605, 0.011, "ftlp", "shift2"),
    ("LP-FT mild", 1.608, 0.044, "lpft", "shift2"),
    ("FRONT mild", 1.611, 0.044, "front", "shift2"),
]:
    check(lbl, pm, ps, "exp3_char", sc, tg, "val_loss")

print("--- Table: optimizer state (val loss) ---")
for lbl, pm, ps, sc, tg in [
    ("FW persist", 1.773, 0.013, "fw", "tau002"),
    ("FW reset", 1.782, 0.013, "fw", "os_reset"),
    ("FW reset-warmup", 1.773, 0.013, "fw", "os_resetwarmup"),
    ("FW episodic", 2.132, 0.012, "fw", "os_episodic"),
    ("PER-10 persist", 2.047, 0.127, "periodic", "p10"),
    ("PER-10 reset", 2.333, 0.055, "periodic", "p10os_reset"),
    ("PER-10 reset-warmup", 1.818, 0.011, "periodic", "p10os_resetwarmup"),
    ("PER-10 episodic", 2.437, 0.008, "periodic", "p10os_episodic"),
]:
    check(lbl, pm, ps, "exp3_char", sc, tg, "val_loss")

print("--- Table: exp2 MLP (test acc, paper reports 1 decimal in %) ---")
for lbl, pm, sub, sc, tg in [
    ("MNIST FULL", 98.4, "exp2", "full", ""), ("MNIST FRONT", 94.9, "exp2", "front", ""),
    ("MNIST FW best-tau", 97.1, "exp2", "fw", "tau0005"),
    ("MNIST BCD", 97.2, "exp2", "bcd", ""),
    ("MNIST PER-20", 95.6, "exp2", "periodic", "p20"),
    ("MNIST LP-FT(.95)", 95.7, "exp2", "lpft", "sf095"),
    ("MNIST LP-FT(.96)", 95.1, "exp2", "lpft", "sf096"),
    ("CIFAR FULL", 55.0, "exp2_cifar10", "full", ""),
    ("CIFAR FRONT", 45.3, "exp2_cifar10", "front", ""),
    ("CIFAR FW best-tau", 51.6, "exp2_cifar10", "fw", "tau005"),
    ("CIFAR BCD", 50.5, "exp2_cifar10", "bcd", ""),
    ("CIFAR PER-20", 46.5, "exp2_cifar10", "periodic", "p20"),
    ("CIFAR LP-FT(.95)", 47.1, "exp2_cifar10", "lpft", "sf095"),
]:
    check(lbl, pm, None, sub, sc, tg, "test_acc", scale=100, nd=1)
