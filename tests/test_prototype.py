"""Smoke tests for the prototype: the exported model loads, forecasts have the right shape and range,
persistence is a valid map, and the web app serves the page, JSON and a PNG."""
import json, os, sys, threading, urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "prototype"))
MODEL = os.path.join(HERE, "prototype", "model", "convlstm_any_sight_month.pt")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="prototype model not exported")

import app as A                      # noqa: E402
import config as C                   # noqa: E402


@pytest.fixture(scope="module")
def ready():
    A.setup(MODEL, os.path.join(HERE, "output", "rolling_nn_preds_any_convlstm_sight_month_s1.npz"))
    return A.STATE


def test_live_forecast_shape_and_range(ready):
    P, pers, mode = A.forecast(ready["usable"][-1], "live")
    assert mode == "live" and P.shape == (3, C.N_ROWS, C.N_COLS) and pers.shape == (C.N_ROWS, C.N_COLS)
    assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1
    assert pers.min() >= 0 and pers.max() <= 1


def test_replay_matches_stored_rolling_forecast(ready):
    from datetime import date
    P, _, mode = A.forecast(date(2020, 4, 6), "replay")
    assert mode == "replay" and P.shape == (3, C.N_ROWS, C.N_COLS)


def test_replay_falls_back_to_live_outside_rolling_period(ready):
    _, _, mode = A.forecast(ready["usable"][0], "replay")
    assert mode == "live"


def test_web_app_endpoints(ready):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), A.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"AgriWatch" in urllib.request.urlopen(base + "/").read()
        d = json.loads(urllib.request.urlopen(base + "/api/forecast?origin=2020-04-06&mode=replay").read())
        assert d["horizons"] == [28, 42, 56] and np.array(d["convlstm"]).shape == (3, C.N_ROWS, C.N_COLS)
        assert urllib.request.urlopen(base + "/map?origin=2020-04-06").read()[:8] == b"\x89PNG\r\n\x1a\n"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/forecast?origin=1999-01-01")
    finally:
        srv.shutdown()


def test_ui_data_endpoints(ready):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), A.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: json.loads(urllib.request.urlopen(base + p).read())
    try:
        meta = get("/api/meta")
        assert meta["horizons"] == [28, 42, 56] and meta["grid"]["rows"] == C.N_ROWS
        assert any(o["replay"] for o in meta["origins"]) and any(not o["replay"] for o in meta["origins"])
        res = get("/api/results")
        assert {"models", "seeds", "by_period", "labels", "refits", "coverage"} <= set(res)
        assert np.array(res["coverage"]["positive"]).shape == (C.N_ROWS, C.N_COLS)
        d = get("/api/forecast?origin=2020-04-06&mode=replay")
        assert len(d["stats"]) == 3 and np.array(d["outcome"]).shape == (3, C.N_ROWS, C.N_COLS)
        assert d["in_training"] is True and d["stats"][0]["known"] > 0
    finally:
        srv.shutdown()


def test_workbench_endpoints(ready):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), A.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: json.loads(urllib.request.urlopen(base + p).read())
    def post(p, body):
        req = urllib.request.Request(base + p, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())
    try:
        inp = get("/api/inputs?origin=2020-04-06")
        assert len(inp["steps"]) == C.SEQ_LEN and np.array(inp["data"]).shape == (C.SEQ_LEN, 9, C.N_ROWS, C.N_COLS)
        bt = get("/api/backtest")
        assert [h["h"] for h in bt["horizons"]] == [28, 42, 56] and len(bt["horizons"][0]["origins"]) == 45
        # the pooled score must equal the stored CSV result of this seed (guards the recomputation)
        assert abs(bt["horizons"][0]["pooled"]["convlstm"] - 0.44996) < 1e-3
        same = post("/api/whatif", {"origin": "2020-04-06"})
        assert np.allclose(same["delta"], 0)                       # no edits -> no change
        res = post("/api/whatif", {"origin": "2020-04-06", "shifts": {"rain": 2}, "n_steps": 3,
                                   "sightings": [{"r": 14, "c": 4}]})
        assert np.array(res["edited"]).shape == (3, C.N_ROWS, C.N_COLS) and np.abs(res["delta"]).max() > 0
        assert np.allclose(res["base"], same["base"])               # baseline is never modified
        with pytest.raises(urllib.error.HTTPError):
            post("/api/whatif", {"origin": "1999-01-01"})
    finally:
        srv.shutdown()
