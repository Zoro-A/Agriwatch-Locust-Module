# Locust module: rolling-origin results (2019-2020, 45 origins per horizon)

PR-AUC on known cells; chance level = prevalence. Higher is better. Persistence = decayed recent-sightings rule.

## Target: any

| Model | 28 d | 42 d | 56 d | Notes |
|---|---|---|---|---|
| Chance (prevalence) | 0.371 | 0.367 | 0.368 | |
| B0 prior | 0.317 | 0.317 | 0.317 | single run |
| B1 logistic | 0.303 | 0.293 | 0.299 | single run |
| B2 random forest | 0.280 | 0.276 | 0.269 | single run |
| B2b LightGBM+neighbours | 0.290 | 0.279 | 0.278 | single run |
| P persistence (sightings only) | 0.526 | 0.470 | 0.430 | single run |
| S2 random forest + sightings | 0.403 | 0.342 | 0.314 | single run |
| S2b LightGBM + sightings | 0.418 | 0.344 | 0.338 | single run |
| M ConvLSTM | 0.352 | 0.371 | 0.388 | single run |
| B3 per-cell LSTM | 0.413 | 0.404 | 0.410 | single run |
| B3 per-cell LSTM [sight_month] | 0.430 | 0.391 | 0.389 | mean ± sd over 1 seed(s) |
| M ConvLSTM [sight_month] | 0.445 ± 0.053 | 0.432 ± 0.060 | 0.428 ± 0.063 | mean ± sd over 3 seed(s) |

## Target: breeding

| Model | 28 d | 42 d | 56 d | Notes |
|---|---|---|---|---|
| Chance (prevalence) | 0.174 | 0.173 | 0.172 | |
| B0 prior | 0.134 | 0.133 | 0.134 | single run |
| B1 logistic | 0.147 | 0.142 | 0.139 | single run |
| B2 random forest | 0.123 | 0.124 | 0.121 | single run |
| B2b LightGBM+neighbours | 0.118 | 0.111 | 0.111 | single run |
| P persistence (sightings only) | 0.368 | 0.290 | 0.240 | single run |
| S2 random forest + sightings | 0.169 | 0.148 | 0.134 | single run |
| S2b LightGBM + sightings | 0.178 | 0.129 | 0.122 | single run |
| M ConvLSTM | 0.254 | 0.255 | 0.244 | single run |
| B3 per-cell LSTM | 0.195 | 0.343 | 0.353 | single run |
| B3 per-cell LSTM [sight_month] | 0.253 | 0.240 | 0.230 | mean ± sd over 1 seed(s) |

Notes: the three-seed runs use 'sight_month' inputs (6 anomaly channels + 3 recent-sightings channels + sin/cos month).
Bootstrap intervals for single runs are in output/rolling_nn_*.csv (columns pr_auc_lo, pr_auc_hi); they overlap persistence.