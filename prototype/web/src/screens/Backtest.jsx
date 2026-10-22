import React, { useMemo, useState } from 'react'
import { useBacktest } from '../hooks.js'
import { dayNum, fmt, niceDate } from '../util.js'

export default function Backtest({ meta, sel, setSel, go }) {
  const { data, err } = useBacktest()
  const hz = sel.hz
  const H = data?.horizons?.[hz]
  const [hover, setHover] = useState(null)

  const scored = useMemo(() => (H ? H.origins.filter((o) => o.pr_convlstm != null) : []), [H])
  const wins = scored.filter((o) => o.pr_convlstm > o.pr_persistence).length
  const open = (o) => { setSel({ origin: o.origin, mode: 'replay', hz }); go('forecast') }

  return (
    <div className="screen">
      <div className="tool-title"><h1>Backtest</h1><p>Every forecast the rolling-origin protocol made in 2019–2020, scored against what surveys later recorded. Each point is one forecast date; click it to open that forecast.</p></div>
      <div className="ctl-row" style={{ marginTop: 20 }}>
        <div className="seg" role="group" aria-label="Horizon">
          {meta.horizons.map((h, j) => <button key={h} aria-pressed={hz === j} onClick={() => setSel({ hz: j })}>{h} days ahead</button>)}
        </div>
        <span className="pill">Stored forecasts of the exported model (seed {meta.model.seed})</span>
      </div>

      {err ? <div className="callout" style={{ marginTop: 20 }}><p><b>Could not load the backtest.</b> {err}</p></div> : !H ? (
        <div className="skeleton" style={{ height: 420, marginTop: 20 }} />
      ) : (
        <>
          <div className="metrics" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <div className="metric"><div className="eyebrow">ConvLSTM, pooled PR-AUC</div><div className="v" style={{ color: 'var(--accent-ink)' }}>{fmt(H.pooled.convlstm)}</div><div className="l">all {H.origins.length} forecast dates together</div></div>
            <div className="metric"><div className="eyebrow">Persistence, pooled PR-AUC</div><div className="v">{fmt(H.pooled.persistence)}</div><div className="l">the rule to beat</div></div>
            <div className="metric"><div className="eyebrow">Chance level</div><div className="v">{fmt(H.prevalence)}</div><div className="l">share of known cells with locust</div></div>
            <div className="metric"><div className="eyebrow">Dates ConvLSTM wins</div><div className="v">{wins}<small style={{ fontSize: 20, color: 'var(--ink-3)' }}> / {scored.length}</small></div><div className="l">of the dates with both locust and non-locust cells</div></div>
          </div>

          <div className="two-col">
            <section className="panel" style={{ marginTop: 0 }}>
              <h3>Score by forecast date</h3>
              <p className="sub">PR-AUC of each date’s map. Grey columns show how many known cells had locust, which is what a random map would score.</p>
              <TimeChart origins={H.origins} hover={hover} setHover={setHover} onOpen={open} />
              <div className="legend"><span><i className="sw" style={{ background: '#1f5fbf' }} /> ConvLSTM</span><span><i className="sw" style={{ background: '#14202b' }} /> persistence</span><span><i className="sw" style={{ background: '#e3e7ec' }} /> share of known cells with locust</span></div>
              <div className="chart-note" style={{ minHeight: 22 }}>
                {hover != null ? (() => { const o = H.origins[hover]; return <><b style={{ color: 'var(--ink)' }}>{niceDate(o.origin)}</b> · {o.known} known cells, {o.pos} with locust · ConvLSTM {fmt(o.pr_convlstm, 2)} · persistence {fmt(o.pr_persistence, 2)}</> })() : 'Hover a date for its numbers.'}
              </div>
            </section>

            <section className="panel" style={{ marginTop: 0 }}>
              <h3>Survey less, find more?</h3>
              <p className="sub">Share of recorded locust cells captured when only the highest-risk part of the map is surveyed, pooled over all dates.</p>
              <GainChart curve={H.curve} />
              <div className="legend"><span><i className="sw" style={{ background: '#1f5fbf' }} /> ConvLSTM</span><span><i className="sw" style={{ background: '#14202b' }} /> persistence</span><span><i className="sw" style={{ background: '#aab4c0' }} /> random</span></div>
            </section>
          </div>

          <section className="panel">
            <h3>All forecast dates</h3>
            <div className="scroll-tbl">
              <table className="t compact">
                <thead><tr><th>Forecast made</th><th>Known</th><th>Locust</th><th>ConvLSTM</th><th>Persistence</th><th>Difference</th><th /></tr></thead>
                <tbody>
                  {H.origins.map((o, i) => {
                    const d = o.pr_convlstm != null ? o.pr_convlstm - o.pr_persistence : null
                    return (
                      <tr key={o.origin} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} className={hover === i ? 'hl' : ''}>
                        <td>{niceDate(o.origin)}</td><td>{o.known}</td><td>{o.pos}</td>
                        <td>{fmt(o.pr_convlstm, 2)}</td><td>{fmt(o.pr_persistence, 2)}</td>
                        <td style={{ color: d == null ? 'var(--ink-3)' : d >= 0 ? 'var(--good)' : 'var(--accent-ink)' }}>{d == null ? '–' : `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(2)}`}</td>
                        <td><button className="btn ghost sm" onClick={() => open(o)}>Open</button></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="chart-note">Dates with no known locust cells, or only locust cells, cannot be scored (PR-AUC is undefined), so they show “–”.</p>
          </section>
        </>
      )}
    </div>
  )
}

function TimeChart({ origins, hover, setHover, onOpen }) {
  const W = 640, Ht = 300, L = 40, B = 34, T = 12, R = 12
  const t0 = dayNum(origins[0].origin), t1 = dayNum(origins[origins.length - 1].origin)
  const X = (d) => L + ((dayNum(d) - t0) / (t1 - t0 || 1)) * (W - L - R)
  const Y = (v) => T + (1 - v) * (Ht - T - B)
  const path = (key) => origins.filter((o) => o[key] != null).map((o, i) => `${i ? 'L' : 'M'}${X(o.origin).toFixed(1)} ${Y(o[key]).toFixed(1)}`).join(' ')
  const bw = Math.max(5, (W - L - R) / origins.length * 0.7)
  const months = origins.filter((o, i) => i === 0 || o.origin.slice(0, 7) !== origins[i - 1].origin.slice(0, 7) && ['01', '04', '07', '10'].includes(o.origin.slice(5, 7)))
  return (
    <svg viewBox={`0 0 ${W} ${Ht}`} width="100%" role="img" aria-label="PR-AUC by forecast date">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}><line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} stroke="#e3e7ec" /><text x={L - 7} y={Y(t) + 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">{t.toFixed(2)}</text></g>
      ))}
      {origins.map((o, i) => <rect key={`b${i}`} x={X(o.origin) - bw / 2} y={Y(o.known ? o.pos / o.known : 0)} width={bw} height={Y(0) - Y(o.known ? o.pos / o.known : 0)} fill="#e3e7ec" opacity={hover === i ? 1 : 0.85} />)}
      {months.map((o) => <text key={o.origin} x={X(o.origin)} y={Ht - 12} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">{o.origin.slice(0, 7)}</text>)}
      <path d={path('pr_persistence')} fill="none" stroke="#14202b" strokeWidth="2" />
      <path d={path('pr_convlstm')} fill="none" stroke="#1f5fbf" strokeWidth="2" />
      {origins.map((o, i) => (
        <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onClick={() => onOpen(o)} style={{ cursor: 'pointer' }}>
          <rect x={X(o.origin) - bw / 2 - 1} y={T} width={bw + 2} height={Ht - T - B} fill="transparent" />
          {o.pr_persistence != null && <circle cx={X(o.origin)} cy={Y(o.pr_persistence)} r={hover === i ? 5 : 3} fill="#14202b" />}
          {o.pr_convlstm != null && <circle cx={X(o.origin)} cy={Y(o.pr_convlstm)} r={hover === i ? 5.5 : 3.4} fill="#1f5fbf" />}
        </g>
      ))}
    </svg>
  )
}

function GainChart({ curve }) {
  const W = 420, Ht = 300, L = 40, B = 34, T = 12, R = 14
  const [at, setAt] = useState(null)
  const X = (a) => L + (a / 100) * (W - L - R)
  const Y = (v) => T + (1 - v) * (Ht - T - B)
  const pts = (arr) => [[0, 0], ...curve.area.map((a, i) => [a, arr[i]])].map(([a, v], i) => `${i ? 'L' : 'M'}${X(a).toFixed(1)} ${Y(v ?? 0).toFixed(1)}`).join(' ')
  const i = at == null ? curve.area.indexOf(10) : at
  return (
    <>
      <svg viewBox={`0 0 ${W} ${Ht}`} width="100%" role="img" aria-label="Recall against alerted area"
        onMouseMove={(e) => { const b = e.currentTarget.getBoundingClientRect(); const a = ((e.clientX - b.left) / b.width * W - L) / (W - L - R) * 100; setAt(Math.min(curve.area.length - 1, Math.max(0, Math.round(a / 5) - 1))) }}
        onMouseLeave={() => setAt(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}><line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} stroke="#e3e7ec" /><text x={L - 7} y={Y(t) + 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">{Math.round(t * 100)}%</text></g>
        ))}
        {[0, 25, 50, 75, 100].map((a) => <text key={a} x={X(a)} y={Ht - 12} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">{a}%</text>)}
        <text x={(L + W - R) / 2} y={Ht - 0} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">share of known cells surveyed</text>
        <line x1={X(0)} y1={Y(0)} x2={X(100)} y2={Y(1)} stroke="#aab4c0" strokeWidth="2" strokeDasharray="5 4" />
        <path d={pts(curve.persistence)} fill="none" stroke="#14202b" strokeWidth="2.2" />
        <path d={pts(curve.convlstm)} fill="none" stroke="#1f5fbf" strokeWidth="2.2" />
        <line x1={X(curve.area[i])} x2={X(curve.area[i])} y1={T} y2={Ht - B} stroke="#14202b" strokeWidth="1" strokeDasharray="3 3" />
        <circle cx={X(curve.area[i])} cy={Y(curve.convlstm[i] ?? 0)} r="4.5" fill="#1f5fbf" />
        <circle cx={X(curve.area[i])} cy={Y(curve.persistence[i] ?? 0)} r="4.5" fill="#14202b" />
      </svg>
      <div className="chart-note">
        Survey the top <b style={{ color: 'var(--ink)' }}>{curve.area[i]}%</b>: ConvLSTM finds <b style={{ color: 'var(--accent-ink)' }}>{Math.round((curve.convlstm[i] ?? 0) * 100)}%</b>,
        persistence <b style={{ color: 'var(--ink)' }}>{Math.round((curve.persistence[i] ?? 0) * 100)}%</b>, random {curve.area[i]}%.
      </div>
    </>
  )
}
