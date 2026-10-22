import React, { useEffect, useMemo, useState } from 'react'
import GridMap, { CHANGE_STOPS, cellLat, cellLon, rampCss } from '../components/GridMap.jsx'
import OriginBar from '../components/OriginBar.jsx'
import { postJSON, rankCells, useForecast } from '../hooks.js'
import { adaptStops, useMapTheme } from '../mapTheme.js'
import RiskLegend from '../components/RiskLegend.jsx'
import { fmt } from '../util.js'

const SLIDERS = [
  ['rain', 'Rain anomaly'], ['sm', 'Soil-moisture anomaly'], ['ndvi', 'NDVI anomaly'], ['wi', 'Water-index anomaly'], ['temp', 'Temperature anomaly'],
]
const ZERO = { ndvi: 0, wi: 0, sm: 0, temp: 0, rain: 0 }
const PRESETS = [
  { id: 'dry', label: 'Dry spell', shifts: { rain: -2, sm: -2, ndvi: -1 }, n: 6 },
  { id: 'wet', label: 'Wet spell', shifts: { rain: 2, sm: 1.5, ndvi: 1 }, n: 3 },
  { id: 'hot', label: 'Heatwave', shifts: { temp: 2 }, n: 3 },
]

export default function WhatIf({ meta, sel, setSel }) {
  const g = meta.grid, hz = sel.hz
  const T = useMapTheme()
  const fc = useForecast(sel.origin, 'live')
  const [shifts, setShifts] = useState(ZERO)
  const [nSteps, setNSteps] = useState(3)
  const [scope, setScope] = useState('all')       // 'all' | 'patch'
  const [radius, setRadius] = useState(3)
  const [tool, setTool] = useState('sighting')    // 'sighting' | 'pick'
  const [cluster, setCluster] = useState(true)
  const [sights, setSights] = useState([])
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)

  const cell = sel.cell
  const box = scope === 'patch' && cell
    ? { r0: Math.max(0, cell.r - radius), r1: Math.min(g.rows - 1, cell.r + radius), c0: Math.max(0, cell.c - radius), c1: Math.min(g.cols - 1, cell.c + radius) }
    : null
  const edited = Object.values(shifts).some((v) => v) || sights.length > 0
  const key = JSON.stringify([sel.origin, shifts, nSteps, box, sights])

  useEffect(() => {
    let alive = true
    setBusy(true)
    const t = setTimeout(() => {
      postJSON('/api/whatif', { origin: sel.origin, shifts, n_steps: nSteps, box, sightings: sights })
        .then((r) => { if (alive) { setRes(r); setErr(null) } })
        .catch((e) => alive && setErr(e.message))
        .finally(() => alive && setBusy(false))
    }, 350)
    return () => { alive = false; clearTimeout(t) }
  }, [key])

  const patchFlags = useMemo(() => {
    if (!box) return null
    return Array.from({ length: g.rows }, (_, r) => Array.from({ length: g.cols }, (_, c) => r >= box.r0 && r <= box.r1 && c >= box.c0 && c <= box.c1))
  }, [box && `${box.r0}${box.r1}${box.c0}${box.c1}`, g])

  const delta = res?.delta?.[hz]
  const dmax = delta ? Math.max(0.05, Math.ceil(Math.max(...delta.flat().map(Math.abs)) * 20) / 20) : 0.1

  function addSighting(r, c) {
    const add = []
    const span = cluster ? 1 : 0
    for (let dr = -span; dr <= span; dr++) for (let dc = -span; dc <= span; dc++) {
      const rr = r + dr, cc = c + dc
      if (rr >= 0 && rr < g.rows && cc >= 0 && cc < g.cols) add.push({ r: rr, c: cc })
    }
    setSights((s) => [...s.filter((x) => !add.some((a) => a.r === x.r && a.c === x.c)), ...add])
  }
  const onPick = (r, c) => (tool === 'sighting' ? addSighting(r, c) : setSel({ cell: { r, c } }))
  const reset = () => { setShifts(ZERO); setSights([]); setScope('all') }
  const applyPreset = (p) => { setShifts({ ...ZERO, ...p.shifts }); setNSteps(p.n); setScope('all') }
  function seedAtPeak() {
    if (!fc.data) return
    const { r, c } = rankCells(fc.data.convlstm[hz])[0]
    setSights([]); setSel({ cell: { r, c } }); addSighting(r, c)
  }

  const describe = (which) => (r, c) => (
    <>
      <div className="mono" style={{ color: 'var(--gold)' }}>{cellLat(g, r).toFixed(3)}°N {cellLon(g, c).toFixed(3)}°E</div>
      <div>Before <b>{fmt(res.base[hz][r][c], 3)}</b> → after <b>{fmt(res.edited[hz][r][c], 3)}</b></div>
      <div>Change <b>{res.delta[hz][r][c] >= 0 ? '+' : ''}{fmt(res.delta[hz][r][c], 3)}</b></div>
    </>
  )

  return (
    <div className="screen">
      <div className="tool-title"><h1>What-if lab</h1><p>Edit the inputs the network receives, or report new sightings, and watch the same trained model respond. This shows what the model is sensitive to, not what causes outbreaks.</p></div>
      <OriginBar meta={meta} sel={sel} setSel={setSel} info={fc.data} lockMode="live" />

      <div className="lab-grid">
        <aside className="panel lab-controls" style={{ marginTop: 0 }}>
          <div className="lab-head"><h3>Edit the inputs</h3><button className="btn ghost sm" onClick={reset} disabled={!edited}>Reset</button></div>
          <div className="chips">
            {PRESETS.map((p) => <button key={p.id} className="chip" onClick={() => applyPreset(p)}>{p.label}</button>)}
            <button className="chip" onClick={seedAtPeak} disabled={!fc.data}>Sightings at peak-risk cell</button>
          </div>
          {SLIDERS.map(([k, label]) => (
            <label className="slide" key={k}>
              <span>{label}</span>
              <input type="range" min={-3} max={3} step={0.5} value={shifts[k]} onChange={(e) => setShifts((s) => ({ ...s, [k]: +e.target.value }))} />
              <b className="mono">{shifts[k] > 0 ? '+' : ''}{shifts[k].toFixed(1)}σ</b>
            </label>
          ))}
          <div className="row2">
            <label>Applied to the last
              <select value={nSteps} onChange={(e) => setNSteps(+e.target.value)}>
                {[1, 3, 6, 12].map((n) => <option key={n} value={n}>{n} step{n > 1 ? 's' : ''} (~{n * 16} days)</option>)}
              </select>
            </label>
          </div>
          <div className="row2">
            <div className="seg on-card" role="group" aria-label="Scope">
              <button aria-pressed={scope === 'all'} onClick={() => setScope('all')}>Whole map</button>
              <button aria-pressed={scope === 'patch'} onClick={() => { setScope('patch'); if (!cell) setSel({ cell: { r: 14, c: 12 } }) }}>Patch around cell</button>
            </div>
            {scope === 'patch' && <label>Radius <input type="range" min={1} max={8} value={radius} onChange={(e) => setRadius(+e.target.value)} style={{ width: 90 }} /> <b className="mono">{radius}</b></label>}
          </div>

          <h3 style={{ marginTop: 18 }}>Report sightings</h3>
          <div className="seg on-card" role="group" aria-label="Map click action">
            <button aria-pressed={tool === 'sighting'} onClick={() => setTool('sighting')}>Click adds sighting</button>
            <button aria-pressed={tool === 'pick'} onClick={() => setTool('pick')}>Click picks cell</button>
          </div>
          <label className="check-inline" style={{ marginTop: 8 }}><input type="checkbox" checked={cluster} onChange={(e) => setCluster(e.target.checked)} /> add a 3×3 cluster, not one cell</label>
          <p className="chart-note">{sights.length ? `${sights.length} cell${sights.length > 1 ? 's' : ''} with a new sighting in the latest step (also marked as surveyed).` : 'None added. Click the map to place a recent sighting.'}</p>
          {sights.length > 0 && <button className="btn ghost sm" onClick={() => setSights([])}>Clear sightings</button>}
        </aside>

        <div style={{ minWidth: 0 }}>
          <RiskLegend />
          {err ? <div className="callout"><p><b>Could not run the model.</b> {err}</p></div> : (
            <div className="maps two-up">
              <section className={`mapcard${busy ? ' loading' : ''}`}>
                <header><h3>Real inputs</h3></header>
                <div className="mc-sub">The final network on the actual data, {meta.horizons[hz]} days ahead.</div>
                {res ? <GridMap grid={g} values={res.base[hz]} selected={cell} onSelect={onPick} cursor={tool === 'sighting' ? 'crosshair' : 'pointer'} describe={describe('base')} /> : <div className="skeleton" style={{ aspectRatio: '24 / 28' }} />}
              </section>
              <section className={`mapcard${busy ? ' loading' : ''}`}>
                <header><h3>Edited inputs</h3></header>
                <div className="mc-sub">{edited ? 'Same network, your edits applied.' : 'No edits yet, so this equals the real inputs.'}</div>
                {res ? <GridMap grid={g} values={res.edited[hz]} selected={cell} flags={patchFlags} markers={sights} onSelect={onPick} cursor={tool === 'sighting' ? 'crosshair' : 'pointer'} describe={describe('edited')} /> : <div className="skeleton" style={{ aspectRatio: '24 / 28' }} />}
              </section>
              <section className={`mapcard${busy ? ' loading' : ''}`} style={{ gridColumn: '1 / -1' }}>
                <header><h3>Change in risk</h3><span className="mono" style={{ fontSize: 13, color: 'var(--ink-3)' }}>edited − real</span></header>
                <div className="change-row">
                  <div style={{ maxWidth: 420, flex: '0 0 auto', width: '46%' }}>
                    {res ? <GridMap grid={g} values={delta} kind="field" stops={CHANGE_STOPS} vmax={dmax} diverging onSelect={onPick} describe={describe('change')} regions={false} /> : <div className="skeleton" style={{ aspectRatio: '24 / 28' }} />}
                    <div className="legend" style={{ marginTop: 8, gap: 10 }}>
                      <span className="mono" style={{ fontSize: 12.5 }}>−{dmax.toFixed(2)}</span><span style={{ width: 150, height: 10, borderRadius: 5, border: '1px solid rgba(0,0,0,.14)', background: rampCss(adaptStops(CHANGE_STOPS, T)) }} /><span className="mono" style={{ fontSize: 12.5 }}>+{dmax.toFixed(2)}</span>
                    </div>
                  </div>
                  <div style={{ flex: 1, minWidth: 260 }}>
                    <table className="t compact">
                      <thead><tr><th>Horizon</th><th>Mean risk</th><th>Largest change</th></tr></thead>
                      <tbody>
                        {(res?.summary || []).map((s, j) => (
                          <tr key={s.h} className={j === hz ? 'hl' : ''} onClick={() => setSel({ hz: j, cell: { r: s.at[0], c: s.at[1] } })} style={{ cursor: 'pointer' }} title="Click to inspect that cell">
                            <td>{s.h} d</td>
                            <td>{fmt(s.mean_before, 3)} → {fmt(s.mean_after, 3)}</td>
                            <td>{s.max_change >= 0 ? '+' : ''}{fmt(s.max_change, 2)} <span className="mono" style={{ fontSize: 12.5, color: 'var(--ink-3)' }}>@ {cellLat(g, s.at[0]).toFixed(1)}°N {cellLon(g, s.at[1]).toFixed(1)}°E</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="chart-note">The model was trained on real histories. Large, uniform, or unrealistic edits push it outside what it has seen, so read the response as sensitivity, not as cause and effect.</p>
                  </div>
                </div>
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
