"""Fit baselines on TRAIN and report VALIDATION metrics. Test is never touched here."""
import os, sys, time
import numpy as np, pandas as pd
import config as C
from dataset import Samples
import baselines as B
import metrics as Mx

target = sys.argv[1] if len(sys.argv) > 1 else C.PRIMARY_TARGET
S = Samples(target)
rows, preds = [], {}
for j, h in enumerate(C.HORIZONS):
    tr, va = S.select("train", j), S.select("val", j)
    Yv, Mv = S.Y[va], S.M[va]
    models = {**B.make_models(), **B.make_sighting_models(target)}
    for name, model in models.items():
        t0 = time.time()
        model.fit(S, tr)
        P = model.predict(S, va)
        preds[f"{name}|{h}"] = P
        r = Mx.summary(P, Yv, Mv, n_boot=500)
        r.update(model=name, horizon=h, target=target, seconds=round(time.time() - t0, 1))
        rows.append(r)
        print(f"[{target} h={h}] {name:26s} PR-AUC {r['pr_auc']:.3f} "
              f"[{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}] (prev {r['prevalence']:.3f}) "
              f"rec@10% {r['recall@10%']:.3f} brier {r['brier']:.3f}  {r['seconds']}s", flush=True)
df = pd.DataFrame(rows)
os.makedirs("output", exist_ok=True)
df.to_csv(f"output/baselines_val_{target}.csv", index=False)
np.savez_compressed(f"output/baseline_val_preds_{target}.npz", **preds)
