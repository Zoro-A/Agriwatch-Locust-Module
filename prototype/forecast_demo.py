"""AgriWatch locust prototype: load the trained ConvLSTM, forecast, and draw risk maps.

python3 prototype/forecast_demo.py [--model prototype/model/convlstm_any_sight_month.pt]
                                   [--origins 2020-07-11 ...] [--out prototype/dashboard.html]

For each forecast origin the script
  1. takes the latest 12 input steps (6 anomaly channels, recent sightings, month),
  2. runs the trained ConvLSTM (three heads: 28, 42, 56 days ahead),
  3. computes the persistence baseline from recent sightings only,
  4. shows both next to what was actually recorded (known cells only) when the outcome exists.
It writes one self-contained HTML dashboard (images embedded) and prints the numbers.
The model was trained only on labels that had ended by its cut-off date (see the model card in
the dashboard), so origins on or after that date are honest, out-of-sample forecasts.
"""
import argparse, base64, io, os, sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config as C
from dataset import Samples
import nn_models as N
import metrics as Mx
from baselines import Persistence

HNAMES = [f"{h} days ahead" for h in C.HORIZONS]
EXTENT = [C.LON_MIN if hasattr(C, "LON_MIN") else 68, C.LON_MAX if hasattr(C, "LON_MAX") else 74,
          C.LAT_MIN if hasattr(C, "LAT_MIN") else 24, C.LAT_MAX if hasattr(C, "LAT_MAX") else 31]


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = N.NeuralRisk(ck["kind"], variant=ck["variant"], seed=ck["seed"])
    m.net = m._new(); m.net.load_state_dict(ck["state"]); m.net.eval()
    return m, ck


def live_forecast(S, D, model, oi):
    """Run the trained network on the latest 12 steps of this origin: (3, H, W) risk maps."""
    k0 = next(iter(D.by_origin[oi].values()))
    x = torch.tensor(N.build_input(S, k0, model.variant))[None]
    with torch.no_grad():
        return torch.sigmoid(model.net(x))[0].numpy()


def replay_forecast(S, D, preds, oi):
    """Stored rolling-origin forecast for this origin (made by the model refitted at the time)."""
    import rolling as R
    out = []
    for j in range(3):
        ev = R.eval_indices(S, j)
        out.append(preds[f"M ConvLSTM [sight_month]|{C.HORIZONS[j]}"][int(np.where(ev == D.by_origin[oi][j])[0][0])])
    return np.stack(out)


def fig_for_origin(S, D, P, oi, target="any"):
    k0 = next(iter(D.by_origin[oi].values()))
    pers = Persistence(0 if target == "any" else 1).predict(S, [k0])[0]  # same map for every horizon
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 11.5), dpi=90)
    obs_cmap = ListedColormap(["#DAD5C3", "#1F5A44", "#E3A72F"])        # unknown, negative, positive
    stats = []
    for j in range(3):
        k = D.by_origin[oi].get(j)
        a = ax[0, j]; im = a.imshow(P[j], vmin=0, vmax=1, cmap="YlOrRd", extent=EXTENT, aspect="auto")
        a.set_title(f"ConvLSTM, {HNAMES[j]}", fontsize=11)
        a = ax[1, j]; a.imshow(pers, vmin=0, vmax=1, cmap="YlOrRd", extent=EXTENT, aspect="auto")
        a.set_title("Persistence (sightings only)", fontsize=11)
        a = ax[2, j]
        s = dict(h=C.HORIZONS[j], known=0, pos=0, pr_m=np.nan, pr_p=np.nan)
        if k is not None and S.M[k].sum() > 0:
            Y, M = S.Y[k], S.M[k]
            o = np.where(M == 0, 0, np.where(Y == 1, 2, 1))
            a.imshow(o, cmap=obs_cmap, vmin=0, vmax=2, extent=EXTENT, aspect="auto")
            s.update(known=int(M.sum()), pos=int((Y * M).sum()))
            if 0 < s["pos"] < s["known"]:
                s["pr_m"] = Mx.pr_auc(P[j][None], Y[None], M[None]); s["pr_p"] = Mx.pr_auc(pers[None], Y[None], M[None])
            a.set_title("Recorded outcome (known cells)", fontsize=11)
        else:
            a.text(0.5, 0.5, "outcome not yet recorded", ha="center", va="center", transform=a.transAxes, fontsize=12)
            a.set_xticks([]); a.set_yticks([])
        stats.append(s)
    for a in ax.ravel():
        a.tick_params(labelsize=8)
    fig.colorbar(im, ax=ax[:2, :], shrink=0.5, label="forecast risk (probability-like score)")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model", "convlstm_any_sight_month.pt"))
    ap.add_argument("--origins", nargs="*", default=None, help="ISO dates to REPLAY from the stored rolling forecasts")
    ap.add_argument("--replay", default=os.path.join(os.path.dirname(HERE), "output", "rolling_nn_preds_any_convlstm_sight_month_s1.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "dashboard.html"))
    a = ap.parse_args()
    model, ck = load_model(a.model)
    S = Samples("any", include_all=True); D = N.OriginData(S)
    cut = date.fromisoformat(ck["cutoff"])
    preds = np.load(a.replay) if os.path.exists(a.replay) else None
    default = ["2019-06-10", "2019-12-03", "2020-04-06", "2020-06-09", "2020-07-11", "2020-08-12"]
    replay = [date.fromisoformat(s) for s in (a.origins if a.origins is not None else default)] if preds is not None else []
    usable = [o for i, o in enumerate(S.origins) if i in D.by_origin]
    jobs = [(S.origins.index(o), "replay") for o in replay if o in usable] + [(S.origins.index(usable[-1]), "live")]
    if not jobs:
        sys.exit("no usable origins")
    blocks = []
    for oi, mode in jobs:
        P = replay_forecast(S, D, preds, oi) if mode == "replay" else live_forecast(S, D, model, oi)
        img, st = fig_for_origin(S, D, P, oi)
        rows = "".join(
            f"<tr><td>{s['h']} d</td><td>{s['known']}</td><td>{s['pos']}</td>"
            f"<td>{'' if np.isnan(s['pr_m']) else format(s['pr_m'], '.2f')}</td>"
            f"<td>{'' if np.isnan(s['pr_p']) else format(s['pr_p'], '.2f')}</td></tr>" for s in st)
        tag = ("replay of the rolling forecast, out-of-sample" if mode == "replay"
               else "live run of the final model on the newest data")
        blocks.append((f"{S.origins[oi].isoformat()} ({tag})", img, rows))
        print(S.origins[oi], [(s["h"], s["known"], s["pos"], None if np.isnan(s["pr_m"]) else round(s["pr_m"], 2),
                               None if np.isnan(s["pr_p"]) else round(s["pr_p"], 2)) for s in st])
    opts = "".join(f'<option value="{i}">{d}</option>' for i, (d, _, _) in enumerate(blocks))
    panels = "".join(
        f'<div class="p" id="p{i}" style="display:{"block" if i == 0 else "none"}"><img src="data:image/png;base64,{img}" alt="risk maps for forecast origin {d}">'
        f'<table><tr><th>Horizon</th><th>Known cells</th><th>Positive cells</th><th>PR-AUC ConvLSTM</th><th>PR-AUC persistence</th></tr>{rows}</table></div>'
        for i, (d, img, rows) in enumerate(blocks))
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>AgriWatch locust prototype</title>
<style>body{{font-family:system-ui,Arial,sans-serif;background:#F4F1E8;color:#14251D;margin:0;padding:24px;max-width:1100px;margin:auto}}
h1{{margin:0 0 4px}}.card{{background:#FFFDF7;border:1px solid #D9D3C0;border-radius:12px;padding:16px;margin:16px 0}}
img{{max-width:100%}}table{{border-collapse:collapse;margin-top:8px}}td,th{{border:1px solid #D9D3C0;padding:6px 12px;text-align:right}}th{{background:#E8E2CF}}
select{{font-size:16px;padding:4px}}.warn{{background:#10231C;color:#F4F1E8;border-radius:12px;padding:14px 18px}}</style>
<h1>AgriWatch: desert locust risk forecast (prototype)</h1>
<p>Cholistan–Tharparkar, 24–31°N, 68–74°E, 0.25° cells. Target: locust presence at any life stage in a 16-day window,
28, 42 and 56 days after the forecast origin.</p>
<div class="card"><label>Forecast origin: <select onchange="for(const e of document.querySelectorAll('.p'))e.style.display='none';document.getElementById('p'+this.value).style.display='block'">{opts}</select></label>
{panels}
<p>Outcome colours: <b style="color:#B97A00">amber</b> = locust recorded, <b style="color:#1F5A44">dark green</b> = survey found none, <b style="color:#6B6650">grey</b> = unknown (not scored).
Risk maps: darker red = higher forecast risk.</p></div>
<div class="card"><b>Model card.</b> {ck['name']}; seed {ck['seed']}; inputs: 6 anomaly channels + 3 recent-sightings channels + month (11 channels × 12 steps);
trained on labels that had ended by {ck['cutoff']}; validation PR-AUC at that refit {ck['best_val']:.3f}.
Replay origins show the forecast the rolling protocol made at that time, using only labels known then. The live origin runs the final exported model on the newest input window, which has no recorded positives yet.</div>
<div class="warn"><b>Read with care.</b> Rolling-origin evaluation of this model family did not clearly beat the simple persistence rule
(see the results table in docs/RESULTS.md). The maps are a research prototype, not an operational warning.
Per-origin PR-AUC values come from a single forecast and are indicative only. Grey cells in the outcome panel are unknown, not negative.</div></html>"""
    open(a.out, "w").write(html)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
