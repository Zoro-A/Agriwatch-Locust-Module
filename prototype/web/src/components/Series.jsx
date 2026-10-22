import React from 'react'

/** Tiny line chart of one cell's input over the 12 steps, with a zero line. */
export function Sparkline({ values, w = 120, h = 26 }) {
  const lim = Math.max(1.5, ...values.map(Math.abs))
  const x = (i) => 2 + (i / (values.length - 1)) * (w - 4)
  const y = (v) => h / 2 - (v / lim) * (h / 2 - 3)
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <line x1="0" x2={w} y1={h / 2} y2={h / 2} stroke="#dde2e8" strokeDasharray="2 3" />
      <path d={d} fill="none" stroke="#14202b" strokeWidth="1.6" strokeLinejoin="round" />
      <circle cx={x(values.length - 1)} cy={y(values[values.length - 1])} r="2.8" fill="#1f5fbf" />
    </svg>
  )
}

/** One row of 12 squares for a 0/1 channel (sighting / survey) across the input steps. */
export function ChannelStrip({ label, values, faint }) {
  return (
    <div className="spark-row">
      <span>{label}</span>
      <svg width="120" height="16" aria-hidden="true">
        {values.map((v, i) => <rect key={i} x={i * 10} y="2" width="8" height="12" rx="1.5" fill={v ? (faint ? '#7cc4a9' : '#14202b') : '#eceff3'} />)}
      </svg>
      <b className="mono">{values.reduce((a, b) => a + b, 0)}/{values.length}</b>
    </div>
  )
}
