import React from 'react'
import { rampCss, useMapTheme } from '../mapTheme.js'

/** Legend for the risk scale: labelled classes when the style has them, otherwise a ramp with ticks. */
export default function RiskLegend() {
  const T = useMapTheme()
  if (T.bins) {
    return (
      <div className="risk-legend" aria-label="Risk classes">
        <span className="rl-title">Risk of locust presence</span>
        {T.bins.map((b) => (
          <span key={b.label} className="rl-item"><i className="sw" style={{ background: b.color }} />{b.label}</span>
        ))}
      </div>
    )
  }
  return (
    <div className="risk-legend" aria-label="Risk scale">
      <span className="rl-title">Risk of locust presence</span>
      <span className="rl-ramp">
        <i style={{ background: rampCss(T.risk) }} />
        <span className="rl-ticks">{[0, 0.25, 0.5, 0.75, 1].map((t) => <b key={t} style={{ left: `${t * 100}%` }}>{t}</b>)}</span>
      </span>
    </div>
  )
}
