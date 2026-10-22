# AgriWatch locust prototype

A research prototype that forecasts desert-locust presence risk on the 28×24 grid (24–31°N, 68–74°E)
28, 42 and 56 days ahead, and compares the ConvLSTM with the persistence rule.

## Try the model interactively (from the project root, `locust/`)
```
pip install -r requirements.txt
python3 prototype/app.py            # then open http://127.0.0.1:8000
```
Pick any forecast origin, choose *live* (runs the trained network now) or *replay* (the stored rolling forecast), and see
the 28/42/56-day risk maps beside persistence and the recorded outcome. JSON is at `/api/forecast?origin=2020-04-06`.

## Web workbench (React)
`python3 prototype/app.py` serves a React workbench from `prototype/web/dist` (the old single page stays at `/legacy`).
Tools in the left sidebar (keys 1-6):
1. **Forecast** - pick a date, see ConvLSTM / persistence / recorded-outcome maps, an alert planner (survey only the top q% of cells, how many recorded locust cells are captured), ranked cells and a per-cell inspector.
2. **Inputs** - the 12-step input window per channel (anomalies, observed mask, sightings) that the network receives.
3. **What-if lab** - shift the anomaly inputs or inject sightings, re-run the final network (`POST /api/whatif`) and see the change in risk. Sensitivity of the trained model, not causal evidence.
4. **Backtest** - every rolling-origin forecast of the exported seed scored per date, pooled PR-AUC and a recall-vs-surveyed-area curve; click a date to open it.
5. **Model comparison** - all baselines and neural models on the same yardstick (from the result CSVs).
6. **Labels & data** - three-valued label map for the selected window, survey coverage maps, class balance.
```
cd prototype/web
npm install            # once
npm run build          # writes prototype/web/dist (rebuild after editing src/)
npm run dev            # optional hot-reload server on :5173, proxies /api to python app.py on :8000
```
JSON used by the workbench: `/api/meta`, `/api/results`, `/api/forecast`, `/api/inputs?origin=`, `/api/backtest`, and `POST /api/whatif`
(`{origin, shifts:{rain,sm,ndvi,wi,temp}, n_steps, box:{r0,r1,c0,c1}, sightings:[{r,c}]}`).

## Static dashboard and tests
```
python3 prototype/forecast_demo.py            # writes prototype/dashboard.html (open in any browser)
python3 prototype/forecast_demo.py --origins 2020-04-06 2020-06-09   # choose replay origins
```
`prototype/dashboard.html` is already built and opens without Python.
`python3 -m pytest -q tests/test_prototype.py` smoke-tests the model, forecast shapes and the web app.

## What it does
1. **Replay origins**: shows the forecast the rolling-origin protocol made at that date (made only from labels
   known then), read from `output/rolling_nn_preds_any_convlstm_sight_month_s1.npz`.
2. **Live origin**: loads `prototype/model/convlstm_any_sight_month.pt`, takes the latest 12 input steps
   (6 anomaly channels, 3 recent-sightings channels, month) and runs the network.
3. Draws ConvLSTM maps, persistence maps and the recorded outcome (known cells only) for each horizon.

## Honest limits
- The ConvLSTM did not clearly beat persistence in the rolling evaluation (see `docs/RESULTS.md`).
- The newest input window has no recorded positives, so the live origin has nothing to compare with.
- Research prototype, not an operational warning system.

## Rebuild the model file after a new run
`python3 prototype/export_model.py output/ckpt_any_convlstm_sight_month_s1.pkl 1`
