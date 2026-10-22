import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play } from './Icons.jsx'
import { niceDate } from '../util.js'

/** Sticky control strip shared by the tools: forecast origin, replay/live mode, horizon. */
export default function OriginBar({ meta, sel: sel0, setSel, info, horizon = true, lockMode = null }) {
  const sel = lockMode ? { ...sel0, mode: lockMode } : sel0
  const lists = useMemo(() => ({
    replay: meta.origins.filter((o) => o.replay).map((o) => o.date),
    live: meta.origins.map((o) => o.date),
  }), [meta])
  const list = lists[sel.mode]
  const idx = Math.max(0, list.indexOf(sel.origin))
  const [playing, setPlaying] = useState(false)
  const idxRef = useRef(idx); idxRef.current = idx

  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => {
      if (idxRef.current >= list.length - 1) { setPlaying(false); return }
      setSel({ origin: list[idxRef.current + 1] })
    }, 1300)
    return () => clearInterval(t)
  }, [playing, list])

  function switchMode(m) {
    if (m === sel.mode) return
    setPlaying(false)
    const t = Date.parse(sel.origin)
    const next = lists[m].reduce((b, d) => (Math.abs(Date.parse(d) - t) < Math.abs(Date.parse(b) - t) ? d : b), lists[m][0])
    setSel({ mode: m, origin: next })
  }

  const cutIdx = list.findIndex((d) => d >= meta.model.cutoff)
  const n = Math.max(list.length - 1, 1)
  const at = (f) => `calc(11px + (100% - 22px) * ${f})`
  const fellBack = info && sel.mode === 'replay' && info.mode === 'live'

  return (
    <div className="originbar">
      <div className="studio-bar">
        <div className="origin-read">
          <div className="eyebrow">Forecast made on</div>
          <div className="d">{niceDate(sel.origin)}</div>
        </div>
        <div className="slider-wrap">
          {list.length <= 140 && (
            <div className="slider-ticks" aria-hidden="true">
              {list.map((d, i) => <i key={d} className={sel.mode === 'replay' ? 'rep' : ''} style={{ left: at(i / n) }} />)}
              {sel.mode === 'live' && cutIdx >= 0 && <i className="cut" style={{ left: at(cutIdx / n) }} />}
            </div>
          )}
          <input type="range" min={0} max={list.length - 1} value={idx} aria-label="Forecast origin"
            onChange={(e) => { setPlaying(false); setSel({ origin: list[+e.target.value] }) }} />
          <div className="slider-ends">
            <span>{niceDate(list[0])}</span>
            {sel.mode === 'live' && <span style={{ color: 'var(--accent-ink)' }}>▲ model cut-off {meta.model.cutoff}</span>}
            <span>{niceDate(list[list.length - 1])}</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {sel.mode === 'replay' && (
            <button className="btn ghost" onClick={() => { if (!playing && idx >= list.length - 1) setSel({ origin: list[0] }); setPlaying((p) => !p) }}
              aria-label={playing ? 'Pause' : 'Play through 2019–2020'}>
              {playing ? <Pause /> : <Play />}{playing ? 'Pause' : 'Play'}
            </button>
          )}
          {!lockMode && <div className="seg" role="group" aria-label="Mode">
            <button aria-pressed={sel.mode === 'replay'} onClick={() => switchMode('replay')} title="Stored forecast from the rolling evaluation">Replay</button>
            <button aria-pressed={sel.mode === 'live'} onClick={() => switchMode('live')} title="Run the final trained network now">Live</button>
          </div>}
        </div>
      </div>
      <div className="ctl-row">
        {horizon && (
          <div className="seg" role="group" aria-label="Horizon">
            {meta.horizons.map((h, j) => <button key={h} aria-pressed={sel.hz === j} onClick={() => setSel({ hz: j })}>{h} days ahead</button>)}
          </div>
        )}
        <span className="ctl-pills">
          {sel.mode === 'replay' && !fellBack && <span className="pill good">Out-of-sample forecast</span>}
          {fellBack && <span className="pill warn">No stored forecast for this date · showing live model</span>}
          {(sel.mode === 'live' || fellBack) && info?.in_training && <span className="pill warn">Seen in training · flattering</span>}
          {(sel.mode === 'live' || fellBack) && info && !info.in_training && <span className="pill dark">Final model · after cut-off</span>}
        </span>
      </div>
    </div>
  )
}
