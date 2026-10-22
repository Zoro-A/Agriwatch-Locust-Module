import { useEffect, useState } from 'react'
import { fetchForecast, getJSON } from './api.js'

/** Debounced loader: keeps the previous value on screen while the next one loads. */
function useLoader(key, load, delay = 90) {
  const [state, setState] = useState({ data: null, loading: true, err: null })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true }))
    const t = setTimeout(() => {
      load().then((data) => alive && setState({ data, loading: false, err: null }))
        .catch((e) => alive && setState((s) => ({ ...s, loading: false, err: e.message })))
    }, delay)
    return () => { alive = false; clearTimeout(t) }
  }, [key])
  return state
}

export const useForecast = (origin, mode) => useLoader(`f|${origin}|${mode}`, () => fetchForecast(origin, mode))
export const useInputs = (origin) => useLoader(`i|${origin}`, () => getJSON(`/api/inputs?origin=${origin}`))
export const useBacktest = () => useLoader('bt', () => getJSON('/api/backtest'), 0)

export async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`)
  return r.json()
}

/** Cells ranked by risk. Ties are broken by a fixed pseudo-random hash, so a rule with many equal scores
 *  (persistence is zero almost everywhere) does not favour the north-west corner. */
const jitter = (r, c) => (Math.imul(r + 1, 73856093) ^ Math.imul(c + 1, 19349663)) >>> 0
export function rankCells(grid2d) {
  const out = []
  grid2d.forEach((row, r) => row.forEach((v, c) => out.push({ r, c, v, j: jitter(r, c) })))
  return out.sort((a, b) => b.v - a.v || a.j - b.j)
}

/** Boolean grid flagging the top `frac` of all cells by risk. */
export function topFlags(grid2d, frac) {
  const ranked = rankCells(grid2d), n = Math.round(ranked.length * frac)
  const flags = grid2d.map((row) => row.map(() => false))
  ranked.slice(0, n).forEach(({ r, c }) => { flags[r][c] = true })
  return flags
}

/** How many recorded-locust cells a flag set captures (known cells only). */
export function captured(flags, outcome) {
  let pos = 0, hit = 0, knownFlagged = 0
  outcome.forEach((row, r) => row.forEach((o, c) => {
    if (o === 2) { pos++; if (flags[r][c]) hit++ }
    if (o !== 0 && flags[r][c]) knownFlagged++
  }))
  return { pos, hit, knownFlagged }
}
