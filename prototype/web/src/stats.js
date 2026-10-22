import { mean, sd } from './util.js'

export const PERSISTENCE = 'P persistence (sightings only)'
export const CONVLSTM_SEEDS = 'M ConvLSTM [sight_month]'

/** One row of rolling_all_models.csv. */
export const modelRow = (results, target, model, h) =>
  results.models.find((r) => r.target === target && r.model === model && r.horizon === h)

/** Seed runs (rolling_nn_*_s{seed}.csv) of one model: individual values, mean and sd. */
export function seedStats(results, target, model, h) {
  const rows = results.seeds.filter((r) => r.target === target && r.model === model && r.horizon === h)
  const vals = rows.map((r) => r.pr_auc)
  return { rows, vals, n: vals.length, mean: vals.length ? mean(vals) : null, sd: sd(vals), seeds: rows.map((r) => r.seed) }
}

export const prevalence = (results, target, h) => modelRow(results, target, 'B0 prior', h)?.prevalence

/** Labels shown in the UI, keyed by the model name used in the CSVs. */
export const MODEL_INFO = {
  'B0 prior': { name: 'Climatological prior', code: 'B0', group: 'env' },
  'B1 logistic': { name: 'Logistic regression', code: 'B1', group: 'env' },
  'B2 random forest': { name: 'Random forest', code: 'B2', group: 'env' },
  'B2b LightGBM+neighbours': { name: 'LightGBM + neighbour cells', code: 'B2b', group: 'env' },
  [PERSISTENCE]: { name: 'Persistence rule', code: 'P', group: 'sight' },
  'S2 random forest + sightings': { name: 'Random forest + sightings', code: 'S2', group: 'sight' },
  'S2b LightGBM + sightings': { name: 'LightGBM + sightings', code: 'S2b', group: 'sight' },
  'M ConvLSTM': { name: 'ConvLSTM', code: 'M', group: 'nn-env' },
  'B3 per-cell LSTM': { name: 'Per-cell LSTM', code: 'B3', group: 'nn-env' },
  'M ConvLSTM [sight_month]': { name: 'ConvLSTM', code: 'M', group: 'nn-sight' },
  'B3 per-cell LSTM [sight_month]': { name: 'Per-cell LSTM', code: 'B3', group: 'nn-sight' },
}

export const GROUPS = {
  env: { title: 'Environment only, no neural network', note: 'satellite and climate anomalies' },
  sight: { title: 'Using recent sightings', note: 'what was recorded in the last months' },
  'nn-env': { title: 'Neural networks, environment only', note: 'single run' },
  'nn-sight': { title: 'Neural networks, sightings + month', note: 'seeds shown as dots' },
}
