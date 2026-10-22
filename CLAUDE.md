# AgriWatch locust module — project guide for Claude (run locally or in the cloud)

Read this file first. It tells you what the project is, how to run and test it, how the code and data fit
together, what the results are, and which mistakes to avoid. Last updated 22 Oct 2026.

## 1. What this is
- **AgriWatch** is a FAST NUCES Lahore final-year project (F26-061). Team: Mahhee Ibn Ahmar Bukhari (23L-0990, leads the
  **locust module**, i.e. this folder), Muhammad Faizan (23L-0567, wheat/crop module, not in this folder), Hamza Hussain
  (23L-0609). Supervisor: Ms. Umm e Amarrah.
- **This folder**: forecast desert-locust presence risk on a **28×24 grid of 0.25° cells** (24–31°N, 68–74°E: Cholistan,
  Tharparkar and a margin of Indian Rajasthan) at **28, 42 and 56 days ahead**, with a ConvLSTM, against a ladder of baselines.
- **Honesty rule (non-negotiable):** report results as they are. In the current results the ConvLSTM does **not** beat the simple
  persistence baseline (see section 7). Never tune on evaluation data, never cherry-pick a seed, never claim "first/novel model".
  The contribution is the open dataset/pipeline, the three-valued labels, and a leakage-safe rolling-origin evaluation.

## 2. Quick start
```
python3 -m venv .venv && source .venv/bin/activate        # Windows: python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                           # numpy pandas pytest scikit-learn lightgbm torch matplotlib
python -m pytest -q tests                                 # expect 63 passed (~15 s)
python3 prototype/app.py                                  # interactive prototype at http://127.0.0.1:8000
```
Python 3.11/3.12. CPU only is fine (the ConvLSTM has 265,539 parameters; one multi-refit run takes ~25 min on 2 cores).
Run commands from this folder (`locust/`): scripts import each other by bare module name.

## 3. Repository map
| Path | Purpose |
|---|---|
| `config.py` | **Every setting** (grid, horizons 28/42/56, targets, split blocks, climatology years 2001–2019, SEQ_LEN=12). Change decisions here, not in logic. |
| `locust_labels.py` | Grid, 16-day MODIS-style calendar, three-valued labels, target windows, block assignment |
| `build_labels.py` | Reads `data/*.csv` sightings/absences → `output/labels.npz`, `label_summary.csv` |
| `features.py`, `build_features.py` | Anomaly tensors vs 2001–2019 climatology → `output/features.npz` (X: T×6×28×24) |
| `sightings.py` | Recent-sightings channels (any / breeding / surveyed per step) → `output/sightings.npz` |
| `dataset.py` | `Samples` class: (origin, horizon) samples, `window(k)`, `sight_window(k)`, Y/M arrays, block tags |
| `baselines.py` | B0 prior, B1 logistic, B2 random forest, B2b LightGBM+neighbours, `Persistence`, S2/S2b with sightings |
| `nn_models.py` | `ConvLSTMNet`, `CellLSTMNet`, `OriginData`, `masked_bce`, `NeuralRisk` (fit/predict), input variants |
| `metrics.py` | PR-AUC, recall@top-X% area, tolerant recall, Brier, block bootstrap, paired differences |
| `rolling.py` | Rolling-origin protocol helpers (`eval_indices`, `make_groups`, `train_indices`) |
| `run_baselines.py` | Baselines fitted on TRAIN, scored on VAL (development only) |
| `run_rolling.py` | Rolling-origin evaluation of all baselines → `output/rolling_{target}.csv` |
| `run_rolling_nn.py` | Rolling-origin ConvLSTM / per-cell LSTM (checkpoint + resume) |
| `run_queue.sh` | Resumable queue of the sight_month seed runs (skips finished CSVs) |
| `make_summary.py` | Builds `docs/RESULTS.md` from the CSVs |
| `prototype/` | `app.py` (web app), `forecast_demo.py` (static dashboard), `export_model.py`, `model/*.pt`, `dashboard.html`, README |
| `gee/extract_inputs.js` | Earth Engine extraction script (already run; outputs are in `data/inputs`) |
| `queries.txt` | The BigQuery SQL used to pull the FAO catalogue |
| `tests/` | 63 pytest tests (labels, features, rolling protocol, metrics, sightings, neural models, prototype) |
| `docs/` | `RESULTS.md`, the revised report PDF |
| `output/`, `logs/` | Generated arrays, result CSVs, per-run logs |

## 4. Data (what each file is)
- **FAO Desert Locust catalogue** (BigQuery view `vw_locust_master`, 2004–2026, pulled via API with a 50,000-row cap, so it was
  aggregated by cell and date in SQL; see `queries.txt`):
  - `data/positives_box_celldate.csv` (10,364 rows, 2007-08-05..2025-07-05): one row per cell-date with a locust record.
    `n` = COUNT(*) of records, `area_ha` = SUM(area_treated_in_ha) (**treated** area, not infested area). The model uses only that
    a row exists.
  - `data/absences_box_celldate.csv` (56,795 rows, 2007-07-20..2026-09-28): survey-confirmed **NO LOCUST** cell-dates.
  - `data/fao_region_raw.csv` (ends 2015-02-05), `fao_positives_region.csv` (ends 2019-12-29) and the year-split `fao_*` files are
    earlier/raw pulls, used only for audit. Do not build on them.
- **Satellite/climate inputs** in `data/inputs/` (four 5-year files each): `locust_modis_*.csv` (MOD13Q1 v061 NDVI, water index from
  NIR b02 / mid-IR b07, valid fraction) and `locust_era5land_*.csv` (ERA5-Land 2 m temperature, rain, soil moisture). Anomalies are
  taken against a climatology fitted on **2001–2019 only** (never 2020).
- **Grid:** rows = latitude (row 0 = northernmost), columns = longitude (col 0 = westernmost). 7° / 0.25 = 28 rows, 6° / 0.25 = 24
  columns, so the grid is taller than wide. A cell is about 27.8 km tall and 24.7 km wide (a degree of longitude is ~98 km at 27.5°N).

## 5. Labels and inputs
- **Three-valued labels** per (cell, 16-day target window): **positive** if any FAO record lies in the cell/window; **negative** only
  if a survey recorded NO LOCUST and there is no positive; otherwise **unknown** (mask = 0). Unknown cells are ignored by both the
  loss and the metrics. About 12% of origin-cell samples have a known label. Training, 28 d, any stage: 2,630 positive / 19,616
  negative.
- **Targets:** `any` (ADULT/HOPPER/BAND/SWARM; primary; prevalence ≈ 0.37 among known cells) and `breeding` (HOPPER/BAND; ≈ 0.17).
  Only ~63 hopper/band positive cell-steps exist in 2008–2018, almost all in 2010.
- **Sample:** origin date + horizon h; input = last 12 steps (~6 months); target window = 16 days starting h days after the origin.
- **Input variants** (`nn_models.VARIANTS`): `env` = 6 channels (NDVI, water index, temperature, rain, soil-moisture anomalies + MODIS
  observed mask); `sight` = 9 (+ any-stage, breeding, surveyed sightings channels); `sight_month` = 11 (+ sin/cos month).
- **Blocks** (by target-window dates, straddling windows embargoed per horizon): train 2008-01-01..2019-12-31, val 2020-01-01..
  2020-06-30, test 2020-07-01..2020-12-31. They are for development only; the reported evaluation is rolling-origin.

## 6. Models and evaluation protocol
- Baselines: **B0** climatological prior, **B1** logistic, **B2** random forest, **B2b** LightGBM + neighbour cells, **P** persistence
  (exponentially decayed recent-sightings history of the cell plus a 3×3 neighbourhood share; no fitting), **S2/S2b** random forest /
  LightGBM + sightings.
- Neural: **B3** per-cell LSTM (5,219 params, one LSTM shared by all cells, no neighbours) and **M** ConvLSTM (2 ConvLSTM layers of
  32 and 64 filters, 3×3, three 1×1 horizon heads, 265,539 params for 6 channels). Masked, positive-weighted BCE over known cells;
  AdamW, early stopping on the last 20% of training origins (with embargo), warm-started from the previous refit.
- **Rolling-origin evaluation (the reported result):** 45 forecast origins per horizon in 2019–2020, scored in time order. Models are
  refitted every 6 origins (8 refits) using **only labels whose target windows had ended by that cut-off**; each origin is scored once.
  A forecast at origin o uses the latest model with cut-off ≤ o. `tests/test_nn_models.py::test_fit_uses_no_future_labels` guards this.
- Metrics: **PR-AUC** on known cells (primary; read it against prevalence = chance, "lift" = PR-AUC / prevalence), recall at top
  5/10/20% alert area, spatially tolerant recall, Brier, 500-sample block bootstrap intervals, paired bootstrap difference vs persistence.
- A **seed** is the random initialisation/shuffle seed of the network; seeds change results a lot here because positives are few.

## 7. Current results (any-stage target, rolling origin, PR-AUC; chance 0.371 / 0.367 / 0.368)
| Model | 28 d | 42 d | 56 d |
|---|---|---|---|
| Persistence (sightings only) | 0.526 | 0.470 | 0.430 |
| ConvLSTM, sightings + month, **3 seeds** (mean ± sd) | 0.445 ± 0.053 | 0.432 ± 0.060 | 0.428 ± 0.063 |
| …single seeds at 28 d | 0.495 / 0.450 / 0.389 | | |
| Per-cell LSTM, env only | 0.413 | 0.404 | 0.410 |
| LightGBM + sightings (S2b) | 0.418 | 0.344 | 0.338 |
| ConvLSTM, env only | 0.352 | 0.371 | 0.388 |
| Best env-only baseline (B0 prior) | 0.317 | 0.317 | 0.317 |
Conclusion to state plainly: **the ConvLSTM does not beat persistence** (it ties at 56 d); bootstrap intervals are wide (~0.3–0.65) and
overlap. The only apparent neural win (per-cell LSTM, breeding target, 56 d) is not claimable. Full tables: `docs/RESULTS.md`,
`output/rolling_all_models.csv`, `output/rolling_nn_*.csv`.
**Not run yet:** breeding-target ConvLSTM seeds, per-cell LSTM seeds 1–2 (run `./run_queue.sh`; it resumes from checkpoints).

## 8. Prototype (`prototype/`)
- `python3 prototype/app.py [--port 8000]`: pick an origin; **live** runs `prototype/model/convlstm_any_sight_month.pt` (seed 1, the
  median seed; last refit, cut-off 2020-10-15); **replay** shows the stored rolling forecast made at that time
  (`output/rolling_nn_preds_any_convlstm_sight_month_s1.npz`). Shows 28/42/56-day maps, persistence and recorded outcome.
  JSON at `/api/forecast?origin=YYYY-MM-DD&mode=live|replay`, `/api/origins`.
- Live maps for origins **before 2020-10-15 were seen in training** and look better than they should; use replay for honest history.
- `python3 prototype/forecast_demo.py` rebuilds the static `prototype/dashboard.html`. After a new run, re-export the model with
  `python3 prototype/export_model.py output/ckpt_<tag>.pkl <seed>` (the checkpoint is deleted after packaging; rerun the model to get one).
- It is a research prototype, not an operational warning system.

## 9. Common tasks
```
python build_labels.py && python build_features.py && python sightings.py      # rebuild arrays from data/ (sightings.py writes output/sightings.npz when run as a script)
python run_rolling.py any                                                      # baselines, rolling origin (also: breeding)
python run_rolling_nn.py any convlstm sight_month 0                            # args: target kind variant seed
./run_queue.sh                                                                 # remaining seed runs, resumable
python make_summary.py                                                         # refresh docs/RESULTS.md
python -m pytest -q tests
```
Outputs: `output/rolling_nn_{target}_{kind}[_{variant}_s{seed}].csv|.log|_by_period.csv`, `rolling_nn_preds_*.npz`; checkpoint
`output/ckpt_{tag}.pkl` lets a killed run resume at the next refit.

## 10. Gotchas and rules for editing
- **Never leak the future.** Anything fitted (climatology, scalers, models, thresholds) may use only data/labels available at the cut-off.
  Keep the no-future-label tests green. Hyperparameters are fixed in advance; do not tune on rolling results.
- Keep `config.py` the single source of truth. Add new experiments as new variants/flags, not by editing old results.
- Add a variant or model → add a test in `tests/`; keep the suite green (63 tests).
- Cloud/sandbox note: long runs can be killed by container restarts. Use the checkpoint/resume path and relaunch `run_queue.sh`.
  Do not use `pkill -f <script name>` from a tool call (it kills your own shell); use `pkill -x` or the PID.
- Metrics are on **known cells only**; "unknown" is not "negative". Do not add random negatives without making it an explicit,
  reported ablation (N2/N3 negative-sampling ablations are planned, not done).
- Do not describe `area_ha` as infested area (it is treated area). Do not call the 2020 blocks an untouched test set: the rolling
  period overlaps val/test, so the rolling results are the reported evaluation.
- Any "first of its kind" or "novel architecture" claim needs a proper literature sweep first; candidate extra models (ConvGRU, small
  U-Net, persistence-residual ConvLSTM) should be chosen **before** looking at results.

## 11. Context the user may ask about
- **Why ConvLSTM:** it processes a sequence of maps with convolutions inside the recurrent cell, so it learns space (neighbouring cells)
  and time (the last 6 months) together (Shi et al. 2015). The per-cell LSTM is the ablation that removes the spatial part.
- **Literature story (report Ch. 3):** drivers (Klein, Kimathi 2020) → soil-moisture thresholds (Ellenburg 2021) → pseudo-absences for
  presence-only data (Barbet-Massin; Yusuf 2021) → tabular ML (LocustLens 2024) → **base paper Yusuf et al. 2024** (FAO records +
  environment series + images; 3D CNN, recurrent conv nets, fine-tuned Prithvi; F1 81.53%, Africa) → ConvLSTM (Shi 2015).
- **Report/deck:** the revised report PDF is `docs/F26-061_AgriWatch_Report_Revised.pdf` (submitted version is not to be resubmitted).
  Known stale text in the report: the constraints bullet says the FAO archive request is pending (resolved: the open catalogue returned
  2004–2026), and NFR-03 says five seeds (three were run). The 17-slide presentation is a Slides artifact in the user's Claude account.
- **Open flags:** crop-module feature-summary interpretation needs Faizan's confirmation; supervisor agreement on the target/split
  change is pending; report Ch. 5 needs a v4 update if the ConvLSTM results are to go in.
