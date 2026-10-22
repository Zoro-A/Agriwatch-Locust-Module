import React, { useCallback, useEffect, useState } from 'react'
import { fetchMeta, fetchResults } from './api.js'
import { Mark } from './components/Icons.jsx'
import { THEMES, THEME_ORDER, rampCss, setMapTheme, useMapTheme } from './mapTheme.js'
import Forecast from './screens/Forecast.jsx'
import Inputs from './screens/Inputs.jsx'
import WhatIf from './screens/WhatIf.jsx'
import Backtest from './screens/Backtest.jsx'
import Models from './screens/Models.jsx'
import Labels from './screens/Labels.jsx'

const TOOLS = [
  { id: 'forecast', num: '1', label: 'Forecast', hint: 'Risk maps and alert planner', group: 'Run the model', C: Forecast },
  { id: 'inputs', num: '2', label: 'Inputs', hint: 'What the network sees', group: 'Run the model', C: Inputs },
  { id: 'whatif', num: '3', label: 'What-if lab', hint: 'Edit inputs, re-run it', group: 'Run the model', C: WhatIf },
  { id: 'backtest', num: '4', label: 'Backtest', hint: 'Scored against what happened', group: 'Check it', C: Backtest },
  { id: 'models', num: '5', label: 'Model comparison', hint: 'Every model, one yardstick', group: 'Check it', C: Models },
  { id: 'labels', num: '6', label: 'Labels & data', hint: 'Ground truth and coverage', group: 'Data', C: Labels },
]

const fromHash = () => {
  const id = window.location.hash.replace(/^#\/?/, '')
  return TOOLS.some((s) => s.id === id) ? id : 'forecast'
}

export default function App() {
  const [tool, setTool] = useState(fromHash())
  const [meta, setMeta] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [card, setCard] = useState(false)
  const mt = useMapTheme()
  const [sel, setSelState] = useState({ mode: 'replay', origin: '2020-04-06', hz: 0, cell: null })
  const setSel = useCallback((patch) => setSelState((s) => ({ ...s, ...patch })), [])

  useEffect(() => {
    Promise.all([fetchMeta(), fetchResults()]).then(([m, r]) => {
      setMeta(m); setResults(r)
      const rep = m.origins.filter((o) => o.replay).map((o) => o.date)
      setSelState((s) => (rep.includes(s.origin) ? s : { ...s, origin: rep[Math.floor(rep.length / 2)] }))
    }).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    const onHash = () => setTool(fromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const go = useCallback((id) => { window.location.hash = `/${id}`; window.scrollTo({ top: 0 }) }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return
      if (/^[1-6]$/.test(e.key)) go(TOOLS[+e.key - 1].id)
      if (e.key === 'Escape') setCard(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go])

  const groups = [...new Set(TOOLS.map((s) => s.group))]
  const Active = TOOLS.find((s) => s.id === tool).C

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <Mark />
          <div>
            <div className="brand-name">AgriWatch</div>
            <div className="brand-sub">Locust risk workbench</div>
          </div>
        </div>
        <nav aria-label="Tools">
          {groups.map((g) => (
            <div className="nav-group" key={g}>
              <div className="nav-group-label">{g}</div>
              {TOOLS.filter((s) => s.group === g).map((s) => (
                <button key={s.id} className={`nav-item${s.id === tool ? ' active' : ''}`} onClick={() => go(s.id)}
                  aria-current={s.id === tool ? 'page' : undefined} title={s.label}>
                  <span className="num">{s.num}</span>
                  <span className="lbl">{s.label}<span className="hint">{s.hint}</span></span>
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="side-foot">
          <div className="mapstyle" role="group" aria-label="Map style">
            <div className="ms-head"><span>Map style</span><b>{THEMES[mt.id].name}</b></div>
            <div className="ms-opts">
              {THEME_ORDER.map((id) => (
                <button key={id} aria-pressed={mt.id === id} title={THEMES[id].name} aria-label={`${THEMES[id].name} map style`}
                  onClick={() => setMapTheme(id)} style={{ background: rampCss(THEMES[id].risk) }} />
              ))}
            </div>
          </div>
          {meta && (
            <button className="model-chip" onClick={() => setCard(true)} aria-haspopup="dialog">
              <b>Final model · open card</b>
              <div className="row"><span>Network</span><span>ConvLSTM</span></div>
              <div className="row"><span>Labels to</span><span>{meta.model.cutoff}</span></div>
              <div className="row"><span>Grid</span><span>{meta.grid.rows}×{meta.grid.cols} · {meta.grid.cell}°</span></div>
            </button>
          )}
          <span className="tag-research">Research prototype</span>
          <div className="kbd-hint"><kbd>1</kbd>–<kbd>6</kbd> switch tool</div>
        </div>
      </aside>
      <main className="main">
        {error ? (
          <div className="screen" style={{ paddingTop: 80, maxWidth: 640 }}>
            <div className="eyebrow">Can't reach the model server</div>
            <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 38, margin: '10px 0 14px', letterSpacing: '-0.02em' }}>The locusts are quiet, and so is the server.</h1>
            <p className="prose">Start it from the project folder with <span className="code">python prototype/app.py</span>, then reload this page.</p>
            <p className="chart-note">{error}</p>
          </div>
        ) : !meta || !results ? (
          <div className="screen" style={{ paddingTop: 60 }}>
            <div className="skeleton" style={{ height: 24, width: 180 }} />
            <div className="skeleton" style={{ height: 64, width: '55%', marginTop: 18 }} />
            <div className="skeleton" style={{ height: 320, marginTop: 40 }} />
          </div>
        ) : (
          <Active key={tool} meta={meta} results={results} sel={sel} setSel={setSel} go={go} />
        )}
      </main>
      {card && meta && <ModelCard meta={meta} onClose={() => setCard(false)} />}
    </div>
  )
}

function ModelCard({ meta, onClose }) {
  const m = meta.model
  return (
    <div className="modal-back" onClick={onClose} role="presentation">
      <div className="modal" role="dialog" aria-modal="true" aria-label="Model card" onClick={(e) => e.stopPropagation()}>
        <div className="eyebrow">Model card</div>
        <h2>ConvLSTM, sightings + month</h2>
        <table className="t" style={{ marginTop: 10 }}>
          <tbody>
            <tr><td>Network</td><td>2 ConvLSTM layers (32, 64 filters, 3×3), three 1×1 horizon heads</td></tr>
            <tr><td>Input</td><td>12 steps × 11 channels × {meta.grid.rows}×{meta.grid.cols}</td></tr>
            <tr><td>Output</td><td>Risk of locust presence per cell, {meta.horizons.join(' / ')} days ahead, 16-day window</td></tr>
            <tr><td>Seed</td><td>{m.seed} (the median of three)</td></tr>
            <tr><td>Trained on labels ended by</td><td>{m.cutoff}</td></tr>
            <tr><td>Validation PR-AUC at that refit</td><td>{m.best_val.toFixed(3)}</td></tr>
          </tbody>
        </table>
        <p className="prose" style={{ marginTop: 14 }}>
          In the rolling-origin evaluation this model family did <b>not</b> beat the simple persistence rule. Live maps for dates before the
          cut-off were seen in training and look better than they should; use Replay for an honest history. Research prototype, not an operational warning.
        </p>
        <button className="btn" onClick={onClose} style={{ marginTop: 16 }}>Close</button>
      </div>
    </div>
  )
}
