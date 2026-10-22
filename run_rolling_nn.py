"""Rolling-origin evaluation of the neural models (ConvLSTM 'M' and per-cell LSTM 'B3').

python3 run_rolling_nn.py [any|breeding] [convlstm|celllstm] [env|sight|sight_month] [seed]

One multi-head network is refitted at each refit cut-off. The cut-offs are the first origins of the
groups of six used for the 28-day horizon; an evaluation forecast at origin o uses the latest model
whose cut-off is <= o, so no model sees a label whose target window ended after the forecast origin.
Each refit warm-starts from the previous one (earlier labels only) and keeps the best epoch on the
last 20% of the training origins (with an embargo), never on evaluation data.
"""
import sys, time
import numpy as np, pandas as pd
import config as C
from dataset import Samples
import metrics as Mx, rolling as R
import nn_models as N

target = sys.argv[1] if len(sys.argv) > 1 else C.PRIMARY_TARGET
kind = sys.argv[2] if len(sys.argv) > 2 else "convlstm"
variant = sys.argv[3] if len(sys.argv) > 3 else "env"
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
name = {"convlstm": "M ConvLSTM", "celllstm": "B3 per-cell LSTM"}[kind] + ("" if variant == "env" else f" [{variant}]")
epochs_first, epochs_next = (30, 12)

S = Samples(target, include_all=True)
D = N.OriginData(S)
ev = {j: R.eval_indices(S, j) for j in range(len(C.HORIZONS))}
groups28 = R.make_groups(ev[0], 6)
cutoffs = [S.origin_date[g[0]] for g in groups28]
print(f"{name} target={target}: {len(cutoffs)} refits at {cutoffs[0]}..{cutoffs[-1]}", flush=True)

import os, pickle
tag = f"{target}_{kind}" + ("" if variant == "env" and seed == 0 else f"_{variant}_s{seed}")
ckpt = f"output/ckpt_{tag}.pkl"          # resume point: the container may restart and kill the run
preds = {j: np.zeros((len(ev[j]), C.N_ROWS, C.N_COLS), np.float32) for j in ev}
prev, t0, log, start = None, time.time(), [], 0
if os.path.exists(ckpt):
    st = pickle.load(open(ckpt, "rb"))
    preds, log, start = st["preds"], st["log"], st["next"]
    prev = N.NeuralRisk(kind, variant=variant, seed=seed)
    prev.net = prev._new(); prev.net.load_state_dict(st["state"]); prev.net.eval()
    print(f"resuming at refit {start + 1}", flush=True)
for gi, cut in enumerate(cutoffs):
    if gi < start:
        continue
    model = N.NeuralRisk(kind, epochs=epochs_first if prev is None else epochs_next, variant=variant, seed=seed)
    model.fit(D, cut, warm=prev)
    msg = (f"refit {gi + 1}/{len(cutoffs)} cutoff {cut}: {model.n_train_origins} train origins, "
           f"{model.epochs_run} epochs, best val PR-AUC {model.best_val:.3f}, {time.time() - t0:.0f}s elapsed")
    print(msg, flush=True); log.append(msg)
    nxt = cutoffs[gi + 1] if gi + 1 < len(cutoffs) else None
    for j in ev:
        sel = [i for i, k in enumerate(ev[j]) if S.origin_date[k] >= cut and (nxt is None or S.origin_date[k] < nxt)]
        if sel:
            preds[j][sel] = model.predict(D, ev[j][sel])
    prev = model
    pickle.dump(dict(preds=preds, log=log, next=gi + 1,
                     state={k: v.clone() for k, v in model.net.state_dict().items()}), open(ckpt, "wb"))

rows, per, allp = [], [], {}
for j, h in enumerate(C.HORIZONS):
    idx = ev[j]; Y, M = S.Y[idx], S.M[idx]; P = preds[j]
    half = np.array([f"{S.origin_date[k].year}{'H1' if S.origin_date[k].month <= 6 else 'H2'}" for k in idx])
    allp[f"{name}|{h}"] = P
    r = Mx.summary(P, Y, M, n_boot=500); r.update(model=name, horizon=h, target=target, n_origins=len(idx))
    r["lift"] = r["pr_auc"] / r["prevalence"]
    rows.append(r)
    for hp in sorted(set(half)):
        s = half == hp
        per.append(dict(model=name, horizon=h, period=hp, origins=int(s.sum()),
                        prevalence=Mx.prevalence(Y[s], M[s]), pr_auc=Mx.pr_auc(P[s], Y[s], M[s])))
    print(f"[{target} h={h}] {name:20s} PR-AUC {r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}] "
          f"prev {r['prevalence']:.3f} lift {r['lift']:.2f} rec@10% {r['recall@10%']:.3f} "
          f"tol-rec@10% {r['tol_recall@10%']:.3f} brier {r['brier']:.3f}", flush=True)
# (tag defined above)
pd.DataFrame(rows).to_csv(f"output/rolling_nn_{tag}.csv", index=False)
pd.DataFrame(per).to_csv(f"output/rolling_nn_{tag}_by_period.csv", index=False)
np.savez_compressed(f"output/rolling_nn_preds_{tag}.npz", **allp)
open(f"output/rolling_nn_{tag}.log", "w").write("\n".join(log))
print("done", flush=True)
