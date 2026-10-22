"""Build docs/RESULTS.md from the rolling-origin CSVs (baselines, env-only neural, sightings+month seeds)."""
import glob, os, re
import numpy as np, pandas as pd

OUT = "output"
base = pd.read_csv(f"{OUT}/rolling_all_models.csv")
rows = []
for f in sorted(glob.glob(f"{OUT}/rolling_nn_*_sight_month_s*.csv")):
    if f.endswith("_by_period.csv"):
        continue
    d = pd.read_csv(f)
    d["seed"] = int(re.search(r"_s(\d+)\.csv$", f).group(1))
    rows.append(d)
sm = pd.concat(rows) if rows else pd.DataFrame()
L = ["# Locust module: rolling-origin results (2019-2020, 45 origins per horizon)", "",
     "PR-AUC on known cells; chance level = prevalence. Higher is better. Persistence = decayed recent-sightings rule.", ""]
for tgt in ["any", "breeding"]:
    L += [f"## Target: {tgt}", "", "| Model | 28 d | 42 d | 56 d | Notes |", "|---|---|---|---|---|"]
    b = base[base.target == tgt]
    prev = b.groupby("horizon").prevalence.first()
    L.append(f"| Chance (prevalence) | {prev[28]:.3f} | {prev[42]:.3f} | {prev[56]:.3f} | |")
    for m, g in b.groupby("model", sort=False):
        g = g.set_index("horizon").pr_auc
        L.append(f"| {m} | {g[28]:.3f} | {g[42]:.3f} | {g[56]:.3f} | single run |")
    if len(sm):
        s = sm[sm.target == tgt]
        for m, g in s.groupby("model", sort=False):
            p = g.pivot_table(index="seed", columns="horizon", values="pr_auc")
            n = len(p)
            cells = [f"{p[h].mean():.3f}" + (f" ± {p[h].std(ddof=1):.3f}" if n > 1 else "") for h in (28, 42, 56)]
            L.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | mean ± sd over {n} seed(s) |")
    L.append("")
L += ["Notes: the three-seed runs use 'sight_month' inputs (6 anomaly channels + 3 recent-sightings channels + sin/cos month).",
      "Bootstrap intervals for single runs are in output/rolling_nn_*.csv (columns pr_auc_lo, pr_auc_hi); they overlap persistence."]
os.makedirs("docs", exist_ok=True)
open("docs/RESULTS.md", "w").write("\n".join(L))
print("\n".join(L))
