import os, sys, time
import numpy as np, pandas as pd
import config as C
from dataset import Samples
import baselines as B, metrics as Mx, rolling as R

target = sys.argv[1] if len(sys.argv) > 1 else C.PRIMARY_TARGET
S = Samples(target, include_all=True)
rows, per, allp = [], [], {}
for j, h in enumerate(C.HORIZONS):
    factories = {}
    for name, m in {**B.make_models(rf_trees=100), **B.make_sighting_models(target, rf_trees=100)}.items():
        factories[name] = (lambda nm=name: {**B.make_models(rf_trees=100), **B.make_sighting_models(target, rf_trees=100)}[nm])
    ev, preds = R.rolling_predictions(S, j, factories, group=6)
    Y, M = S.Y[ev], S.M[ev]
    half = np.array([f"{S.origin_date[k].year}{'H1' if S.origin_date[k].month <= 6 else 'H2'}" for k in ev])
    for name, P in preds.items():
        allp[f"{name}|{h}"] = P
        r = Mx.summary(P, Y, M, n_boot=500); r.update(model=name, horizon=h, target=target, n_origins=len(ev))
        r["lift"] = r["pr_auc"] / r["prevalence"]
        rows.append(r)
        for hp in sorted(set(half)):
            s = half == hp
            per.append(dict(model=name, horizon=h, period=hp, origins=int(s.sum()),
                            prevalence=Mx.prevalence(Y[s], M[s]), pr_auc=Mx.pr_auc(P[s], Y[s], M[s])))
        print(f"[{target} h={h}] {name:32s} PR-AUC {r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}] "
              f"prev {r['prevalence']:.3f} lift {r['lift']:.2f} rec@10% {r['recall@10%']:.3f} "
              f"tol-rec@10% {r['tol_recall@10%']:.3f} brier {r['brier']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(f"output/rolling_{target}.csv", index=False)
pd.DataFrame(per).to_csv(f"output/rolling_{target}_by_period.csv", index=False)
np.savez_compressed(f"output/rolling_preds_{target}.npz", **allp)
print("done", flush=True)
