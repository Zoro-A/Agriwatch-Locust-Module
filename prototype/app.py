"""AgriWatch locust prototype: a small local web app to try the trained model.

python3 prototype/app.py [--port 8000] [--model prototype/model/convlstm_any_sight_month.pt]
then open http://127.0.0.1:8000

Pick a forecast origin (a 16-day step with a full 12-step input history), choose "live" (run the trained
network now) or "replay" (the stored rolling-origin forecast made at that time), and see the risk maps at
28, 42 and 56 days beside the persistence baseline and the recorded outcome. Standard library server only
(plus torch and matplotlib already used by the project). Endpoints:
  /                      the page            /map?origin=2020-04-06&mode=live   PNG
  /api/forecast?origin=2020-04-06           JSON: risk maps (3 x 28 x 24) for ConvLSTM and persistence
  /api/origins                              JSON: usable origins
"""
import argparse, base64, csv, glob, json, mimetypes, os, re, sys, threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import forecast_demo as F
import nn_models as N
import config as C
import metrics as Mx
import rolling as R
from dataset import Samples
from baselines import Persistence

ROOT = os.path.dirname(HERE)
DIST = os.path.join(HERE, "web", "dist")           # built React app (npm run build in prototype/web)
LOCK = threading.Lock()                      # one forecast at a time (torch + matplotlib are not thread-safe here)
STATE = {}


def setup(model_path, replay_path):
    model, ck = F.load_model(model_path)
    S = Samples("any", include_all=True)
    D = N.OriginData(S)
    STATE.update(model=model, ck=ck, S=S, D=D, cut=date.fromisoformat(ck["cutoff"]),
                 preds=np.load(replay_path) if os.path.exists(replay_path) else None,
                 usable=[o for i, o in enumerate(S.origins) if i in D.by_origin])
    STATE["replayable"] = _replayable(S, D) if STATE["preds"] is not None else set()


def _replayable(S, D):
    """Origins for which a stored rolling forecast exists at all three horizons."""
    evs = [{int(k) for k in R.eval_indices(S, j)} for j in range(3)]
    return {S.origins[oi] for oi, ks in D.by_origin.items() if all(ks.get(j) in evs[j] for j in range(3))}


def forecast(origin, mode="live"):
    """-> (P (3,H,W) ConvLSTM, persistence (H,W), used_mode). Replay falls back to live if no stored forecast."""
    S, D = STATE["S"], STATE["D"]
    oi = S.origins.index(origin)
    if mode == "replay" and STATE["preds"] is not None:
        try:
            return F.replay_forecast(S, D, STATE["preds"], oi), _persistence(S, D, oi), "replay"
        except (IndexError, KeyError):
            pass
    return F.live_forecast(S, D, STATE["model"], oi), _persistence(S, D, oi), "live"


def _persistence(S, D, oi):
    return Persistence(0).predict(S, [next(iter(D.by_origin[oi].values()))])[0]


def outcome_and_stats(S, D, P, pers, oi):
    """Recorded outcome per horizon (0 unknown, 1 none found, 2 locust) plus counts and per-origin PR-AUC."""
    out, stats = [], []
    for j in range(3):
        k = D.by_origin[oi].get(j)
        st = dict(h=C.HORIZONS[j], window=None, known=0, pos=0, pr_convlstm=None, pr_persistence=None)
        if k is not None:
            st["window"] = [S.tw_start[k].isoformat(), (S.tw_end[k] - timedelta(days=1)).isoformat()]
            Y, M = S.Y[k], S.M[k]
            out.append(np.where(M == 0, 0, np.where(Y == 1, 2, 1)).astype(int).tolist())
            st.update(known=int(M.sum()), pos=int((Y * M).sum()))
            if 0 < st["pos"] < st["known"]:
                st["pr_convlstm"] = float(Mx.pr_auc(P[j][None], Y[None], M[None]))
                st["pr_persistence"] = float(Mx.pr_auc(pers[None], Y[None], M[None]))
        else:
            out.append(None)
        stats.append(st)
    return out, stats


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, v in r.items():
            try:
                r[key] = float(v) if v not in ("", None) else None
            except ValueError:
                pass
    return rows


def results_payload():
    """Everything the Results / Data screens need, straight from the result CSVs (nothing recomputed)."""
    out = os.path.join(ROOT, "output")
    seeds = []
    for f in sorted(glob.glob(os.path.join(out, "rolling_nn_*_s[0-9].csv"))):
        m = re.search(r"_s(\d+)\.csv$", f)
        for r in _read_csv(f):
            r["seed"] = int(m.group(1)); seeds.append(r)
    by_period = _read_csv(os.path.join(out, "rolling_any_by_period.csv"))
    for f in sorted(glob.glob(os.path.join(out, "rolling_nn_any_*_sight_month_s[0-9]_by_period.csv"))):
        m = re.search(r"_s(\d+)_by_period\.csv$", f)
        for r in _read_csv(f):
            r["seed"] = int(m.group(1)); by_period.append(r)
    refits = []
    log = os.path.join(out, "rolling_nn_any_convlstm_sight_month_s1.log")
    if os.path.exists(log):
        for line in open(log, encoding="utf-8"):
            m = re.match(r"refit (\d+)/(\d+) cutoff (\S+): (\d+) train origins, (\d+) epochs, best val PR-AUC ([\d.]+)", line)
            if m:
                refits.append(dict(i=int(m[1]), of=int(m[2]), cutoff=m[3], train_origins=int(m[4]), epochs=int(m[5]), val_pr_auc=float(m[6])))
    return dict(models=_read_csv(os.path.join(out, "rolling_all_models.csv")), seeds=seeds, by_period=by_period,
                labels=_read_csv(os.path.join(out, "label_summary.csv")), refits=refits, coverage=coverage_payload())


def coverage_payload():
    """Per-cell counts over the training block (28-day samples): locust recorded vs surveyed with none found."""
    S = STATE["S"]
    k = np.where((S.block == "train") & (S.hor_idx == 0))[0]
    Y, M = S.Y[k].astype(int), S.M[k].astype(int)
    return dict(positive=(Y * M).sum(0).tolist(), negative=((1 - Y) * M).sum(0).tolist(), samples=int(len(k)))


def meta_payload():
    ck, S = STATE["ck"], STATE["S"]
    origins = [dict(date=str(o), replay=o in STATE["replayable"], in_training=o < STATE["cut"]) for o in STATE["usable"]]
    return dict(model=dict(name=ck["name"], seed=int(ck["seed"]), cutoff=ck["cutoff"], best_val=float(ck["best_val"]),
                           variant=ck["variant"], kind=ck["kind"]),
                grid=dict(rows=C.N_ROWS, cols=C.N_COLS, lat=[C.LAT_MIN, C.LAT_MAX], lon=[C.LON_MIN, C.LON_MAX], cell=C.CELL),
                horizons=list(C.HORIZONS), step_days=C.STEP_DAYS, target_len=C.TARGET_LEN, seq_len=C.SEQ_LEN,
                clim_years=[C.CLIM_YEARS[0], C.CLIM_YEARS[-1]], blocks={k: [str(a), str(b)] for k, (a, b) in C.BLOCKS.items()},
                n_steps=int(len(S.step_end)), origins=origins)


def serve_static(handler, path):
    """Serve the built React app; unknown non-file paths fall back to index.html (client-side routing)."""
    rel = os.path.normpath(path.lstrip("/")).replace("\\", "/")
    full = os.path.join(DIST, rel)
    if rel.startswith("..") or not os.path.isfile(full):
        full = os.path.join(DIST, "index.html")
    with open(full, "rb") as f:
        body = f.read()
    ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
    handler._send(200, body, ctype + ("; charset=utf-8" if ctype.startswith("text/") or "javascript" in ctype else ""))

CHANNEL_NAMES = ["ndvi", "wi", "sm", "temp", "rain", "modis_observed", "sight_any", "sight_breeding", "surveyed"]
ANOMALY = CHANNEL_NAMES[:5]


def _k0(oi):
    return next(iter(STATE["D"].by_origin[oi].values()))


def inputs_payload(origin):
    """The 12-step input window the network sees for this origin (anomalies, observed mask, sightings)."""
    S = STATE["S"]; k = _k0(S.origins.index(origin))
    x = np.concatenate([S.window(k), S.sight_window(k).astype(np.float32)], 1)       # (12, 9, H, W)
    steps = [str(e)[:10] for e in S.step_end[S.win_start[k]:S.win_stop[k]]]
    return dict(origin=str(origin), steps=steps, channels=CHANNEL_NAMES, data=np.round(x, 2).tolist())


def whatif(origin, shifts=None, n_steps=3, box=None, sightings=()):
    """Run the final network on the real input window and on an edited copy; return both risk maps.
    shifts: anomaly shift (in standard deviations) per channel, added to the last n_steps steps inside `box`.
    sightings: cells at which a recent sighting (and the survey that found it) is injected into the last step."""
    import torch
    S, model = STATE["S"], STATE["model"]
    oi = S.origins.index(origin)
    x = N.build_input(S, _k0(oi), model.variant).copy()                               # (12, 11, H, W)
    base = x.copy()
    r0, r1, c0, c1 = (box["r0"], box["r1"], box["c0"], box["c1"]) if box else (0, C.N_ROWS - 1, 0, C.N_COLS - 1)
    n = max(1, min(int(n_steps), C.SEQ_LEN))
    for name, d in (shifts or {}).items():
        if name in ANOMALY and d:
            ci = ANOMALY.index(name)
            x[-n:, ci, r0:r1 + 1, c0:c1 + 1] = np.clip(x[-n:, ci, r0:r1 + 1, c0:c1 + 1] + float(d), -C.Z_CLIP, C.Z_CLIP)
    for s_ in sightings:
        r, c = int(s_["r"]), int(s_["c"])
        x[-1, 6, r, c] = 1.0; x[-1, 8, r, c] = 1.0
        if s_.get("breeding"):
            x[-1, 7, r, c] = 1.0

    def run(a):
        with torch.no_grad():
            return torch.sigmoid(model.net(torch.tensor(a)[None]))[0].numpy()
    p0, p1 = run(base), run(x)
    d = p1 - p0
    per_h = []
    for j in range(3):
        r, c = np.unravel_index(np.abs(d[j]).argmax(), d[j].shape)
        per_h.append(dict(h=C.HORIZONS[j], mean_before=float(p0[j].mean()), mean_after=float(p1[j].mean()),
                          max_change=float(d[j][r, c]), at=[int(r), int(c)]))
    return dict(origin=str(origin), base=np.round(p0, 4).tolist(), edited=np.round(p1, 4).tolist(),
                delta=np.round(d, 4).tolist(), summary=per_h)


_BACKTEST = {}


def backtest_payload():
    """Rolling-origin forecasts of the exported seed, scored per origin and pooled, against persistence."""
    if _BACKTEST:
        return _BACKTEST
    S, preds = STATE["S"], STATE["preds"]
    out = dict(horizons=[], note=STATE["ck"]["name"])
    qs = list(range(5, 101, 5))
    for j, h in enumerate(C.HORIZONS):
        ev = R.eval_indices(S, j)
        P = preds[f"M ConvLSTM [sight_month]|{h}"]
        pers = Persistence(0).predict(S, list(ev))
        Y, M = S.Y[ev], S.M[ev]
        per = []
        for i, k in enumerate(ev):
            known, pos = int(M[i].sum()), int((Y[i] * M[i]).sum())
            ok = 0 < pos < known
            per.append(dict(origin=str(S.origin_date[k]), known=known, pos=pos,
                            pr_convlstm=float(Mx.pr_auc(P[i][None], Y[i][None], M[i][None])) if ok else None,
                            pr_persistence=float(Mx.pr_auc(pers[i][None], Y[i][None], M[i][None])) if ok else None))
        out["horizons"].append(dict(
            h=h, origins=per, prevalence=float(Mx.prevalence(Y, M)),
            pooled=dict(convlstm=float(Mx.pr_auc(P, Y, M)), persistence=float(Mx.pr_auc(pers, Y, M))),
            curve=dict(area=qs, convlstm=[Mx.recall_at_alert_area(P, Y, M, q) for q in qs],
                       persistence=[Mx.recall_at_alert_area(pers, Y, M, q) for q in qs])))
    _BACKTEST.update(out)
    return _BACKTEST



def parse_origin(q):
    o = date.fromisoformat(q.get("origin", [str(STATE["usable"][-1])])[0])
    if o not in STATE["usable"]:
        raise ValueError(f"{o} is not a usable origin (needs a full 12-step input history)")
    return o


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><title>AgriWatch locust prototype</title>
<style>body{font-family:system-ui,Arial,sans-serif;background:#F4F1E8;color:#14251D;max-width:1100px;margin:auto;padding:24px}
.card{background:#FFFDF7;border:1px solid #D9D3C0;border-radius:12px;padding:16px;margin:16px 0}select,button{font-size:16px;padding:6px 10px}
img{max-width:100%}.warn{background:#10231C;color:#F4F1E8;border-radius:12px;padding:14px 18px}</style>
<h1>AgriWatch: desert locust risk prototype</h1>
<div class="card"><form onsubmit="go();return false">
<label>Forecast origin <select id="o">%OPTS%</select></label>
<label>Mode <select id="m"><option value="live">live (run trained model now)</option><option value="replay">replay (stored rolling forecast)</option></select></label>
<button>Forecast</button></form><p id="note"></p><img id="img" alt="risk maps" src=""></div>
<div class="card"><b>Model.</b> %CARD%</div>
<div class="warn"><b>Read with care.</b> In the rolling evaluation the ConvLSTM did not beat the simple persistence rule.
Origins before the model cut-off were seen in training, so live maps there look better than they should; use replay for honest history.
Grey cells in the outcome row are unknown, not negative. Research prototype, not an operational warning.</div>
<script>
const cut="%CUT%";
function go(){const o=document.getElementById('o').value,m=document.getElementById('m').value;
document.getElementById('note').textContent=(m==='live'&&o<cut)?'Note: this origin is inside the training period (illustrative).':'';
document.getElementById('img').src='/map?origin='+o+'&mode='+m;}
go();</script></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/" and os.path.isfile(os.path.join(DIST, "index.html")):
                serve_static(self, "index.html")
            elif u.path == "/" or u.path == "/legacy":
                opts = "".join(f'<option value="{o}"{" selected" if o == STATE["usable"][-1] else ""}>{o}</option>' for o in STATE["usable"])
                ck = STATE["ck"]
                card = (f"{ck['name']}; seed {ck['seed']}; trained on labels that had ended by {ck['cutoff']}; "
                        f"validation PR-AUC at that refit {ck['best_val']:.3f}.")
                self._send(200, PAGE.replace("%OPTS%", opts).replace("%CARD%", card).replace("%CUT%", ck["cutoff"]), "text/html")
            elif u.path == "/api/origins":
                self._send(200, json.dumps([str(o) for o in STATE["usable"]]), "application/json")
            elif u.path == "/api/meta":
                self._send(200, json.dumps(meta_payload()), "application/json")
            elif u.path == "/api/inputs":
                self._send(200, json.dumps(inputs_payload(parse_origin(q))), "application/json")
            elif u.path == "/api/backtest":
                with LOCK:
                    self._send(200, json.dumps(backtest_payload()), "application/json")
            elif u.path == "/api/results":
                self._send(200, json.dumps(results_payload()), "application/json")
            elif u.path == "/api/forecast":
                o = parse_origin(q)
                with LOCK:
                    P, pers, used = forecast(o, q.get("mode", ["live"])[0])
                    outcome, stats = outcome_and_stats(STATE["S"], STATE["D"], P, pers, STATE["S"].origins.index(o))
                self._send(200, json.dumps(dict(origin=str(o), mode=used, horizons=[28, 42, 56], in_training=o < STATE["cut"],
                                                convlstm=np.round(P, 4).tolist(), persistence=np.round(pers, 4).tolist(),
                                                outcome=outcome, stats=stats)), "application/json")
            elif u.path == "/map":
                o = parse_origin(q)
                with LOCK:
                    P, _, _ = forecast(o, q.get("mode", ["live"])[0])
                    img, _ = F.fig_for_origin(STATE["S"], STATE["D"], P, STATE["S"].origins.index(o))
                self._send(200, base64.b64decode(img), "image/png")
            elif os.path.isfile(os.path.join(DIST, "index.html")) and not u.path.startswith("/api/"):
                serve_static(self, u.path)
            else:
                self._send(404, "not found", "text/plain")
        except ValueError as e:
            self._send(400, str(e), "text/plain")


def _post(self):
    u = urlparse(self.path)
    try:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if u.path == "/api/whatif":
            o = date.fromisoformat(body.get("origin", str(STATE["usable"][-1])))
            if o not in STATE["usable"]:
                raise ValueError(f"{o} is not a usable origin")
            with LOCK:
                res = whatif(o, body.get("shifts"), body.get("n_steps", 3), body.get("box"), body.get("sightings", []))
            self._send(200, json.dumps(res), "application/json")
        else:
            self._send(404, "not found", "text/plain")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        self._send(400, str(e), "text/plain")


H.do_POST = _post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.path.join(HERE, "model", "convlstm_any_sight_month.pt"))
    ap.add_argument("--replay", default=os.path.join(os.path.dirname(HERE), "output", "rolling_nn_preds_any_convlstm_sight_month_s1.npz"))
    a = ap.parse_args()
    setup(a.model, a.replay)
    print(f"AgriWatch prototype on http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
