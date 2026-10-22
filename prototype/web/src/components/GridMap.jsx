import React, { useEffect, useRef, useState } from 'react'
import { adaptStops, useMapTheme } from '../mapTheme.js'

export const RISK_STOPS = [[0, '#fff8dc'], [0.15, '#fde08b'], [0.35, '#f9a03f'], [0.6, '#e0521e'], [0.8, '#a51d1d'], [1, '#4a0a12']]
export const POS_STOPS = [[0, '#fff8dc'], [0.15, '#fde08b'], [0.5, '#e0521e'], [1, '#4a0a12']]
export const NEG_STOPS = [[0, '#f1f8f4'], [0.2, '#cfe9dc'], [0.55, '#55ad8c'], [1, '#14573f']]
// diverging: t = 0 (negative end) .. 0.5 (zero) .. 1 (positive end)
export const WET_STOPS = [[0, '#8a4b12'], [0.25, '#d9a95f'], [0.5, '#f4f5f2'], [0.75, '#7fb39b'], [1, '#14573f']]
export const HEAT_STOPS = [[0, '#2d5e93'], [0.25, '#8fb4d6'], [0.5, '#f4f5f2'], [0.75, '#e69a62'], [1, '#8c2f0e']]
export const CHANGE_STOPS = [[0, '#2d5e93'], [0.25, '#9bbbd8'], [0.5, '#f6f6f3'], [0.75, '#e69a62'], [1, '#9c2d0b']]
export const BIN_STOPS = [[0, '#f1f3f6'], [1, '#14202b']]
export const OUTCOME = { 0: '#e6e9ee', 1: '#7cc4a9', 2: '#111827' }

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))
export function colorAt(stops, t) {
  t = Math.max(0, Math.min(1, t))
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i]
      const f = (t - t0) / (t1 - t0 || 1)
      const a = hex(c0), b = hex(c1)
      return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(',')})`
    }
  }
  return stops[stops.length - 1][1]
}
export const rampCss = (stops) => `linear-gradient(90deg, ${stops.map(([t, c]) => `${c} ${t * 100}%`).join(', ')})`
/** Theme-aware stops: the shared risk / count scales follow the chosen map style. */
export const resolveStops = (stops, T) => (stops === RISK_STOPS ? T.risk : stops === POS_STOPS ? T.pos : stops === NEG_STOPS ? T.neg : adaptStops(stops, T))

// Approximate positions, for orientation only.
const REGIONS = [
  { name: 'Cholistan', lat: 28.35, lon: 71.9 },
  { name: 'Tharparkar', lat: 24.95, lon: 70.15 },
]
const ML = 38, MB = 22, MT = 4, MR = 4

/**
 * Canvas map of the 28x24 grid.
 * kind: 'risk' (0..1, optional `mask` = outcome grid, unknown cells are greyed), 'outcome' (0/1/2),
 *       'count' (values / vmax through `stops`), 'field' (diverging when `diverging`, else sequential).
 * flags: boolean grid outlined (alert cells). selected: {r,c}. markers: [{r,c}] drawn as dots.
 */
export default function GridMap({ grid, values, kind = 'risk', stops = RISK_STOPS, vmax = 1, diverging = false, mask = null, flags = null,
  selected = null, markers = null, regions = true, describe, onSelect, cursor }) {
  const wrap = useRef(null), cv = useRef(null), shown = useRef(null), raf = useRef(0)
  const [w, setW] = useState(360)
  const [hover, setHover] = useState(null)
  const T = useMapTheme()
  const st0 = resolveStops(stops, T)
  const rows = grid.rows, cols = grid.cols

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(220, Math.round(e.contentRect.width))))
    ro.observe(wrap.current)
    return () => ro.disconnect()
  }, [])

  const cell = (w - ML - MR) / cols
  const H = Math.round(cell * rows + MT + MB)

  function paint(vals, hov) {
    const c = cv.current, dpr = window.devicePixelRatio || 1
    if (!c) return
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(H * dpr) }
    const g = c.getContext('2d')
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, H)
    g.fillStyle = T.bg; g.fillRect(0, 0, w, H)
    for (let r = 0; r < rows; r++) {
      for (let k = 0; k < cols; k++) {
        const v = vals ? vals[r][k] : null
        const x = ML + k * cell, y = MT + r * cell
        const unknown = (kind === 'outcome' && v === 0) || (kind === 'risk' && mask && mask[r][k] === 0)
        if (v == null) g.fillStyle = T.dark ? '#162330' : '#eceff3'
        else if (kind === 'outcome') g.fillStyle = T.outcome[v]
        else if (unknown) g.fillStyle = T.outcome[0]
        else g.fillStyle = colorAt(st0, diverging ? (v / vmax + 1) / 2 : v / vmax)
        g.fillRect(x, y, cell + 0.4, cell + 0.4)
        if (unknown) {
          g.strokeStyle = 'rgba(110,125,145,0.28)'; g.lineWidth = 1
          g.beginPath(); g.moveTo(x + 1, y + cell - 1); g.lineTo(x + cell - 1, y + 1); g.stroke()
        }
      }
    }
    g.strokeStyle = T.grid; g.lineWidth = 0.6
    for (let r = 0; r <= rows; r++) { g.beginPath(); g.moveTo(ML, MT + r * cell); g.lineTo(ML + cols * cell, MT + r * cell); g.stroke() }
    for (let k = 0; k <= cols; k++) { g.beginPath(); g.moveTo(ML + k * cell, MT); g.lineTo(ML + k * cell, MT + rows * cell); g.stroke() }
    g.strokeStyle = T.frame; g.lineWidth = 1; g.strokeRect(ML + 0.5, MT + 0.5, cols * cell - 1, rows * cell - 1)
    g.fillStyle = T.axis; g.font = '12px "JetBrains Mono Variable", monospace'
    g.strokeStyle = 'rgba(233,238,245,0.4)'; g.lineWidth = 1
    for (let lon = grid.lon[0]; lon <= grid.lon[1]; lon++) {
      const x = ML + ((lon - grid.lon[0]) / grid.cell) * cell
      g.beginPath(); g.moveTo(x, MT + rows * cell); g.lineTo(x, MT + rows * cell + 4); g.stroke()
      g.textAlign = lon === grid.lon[1] ? 'right' : lon === grid.lon[0] ? 'left' : 'center'
      g.fillText(`${lon}°E`, x, H - 5)
    }
    g.textAlign = 'right'
    for (let lat = grid.lat[0]; lat <= grid.lat[1]; lat++) {
      const y = MT + ((grid.lat[1] - lat) / grid.cell) * cell
      g.beginPath(); g.moveTo(ML - 4, y); g.lineTo(ML, y); g.stroke()
      g.fillText(`${lat}°N`, ML - 7, Math.min(Math.max(y + 3.5, MT + 9), MT + rows * cell - 2))
    }
    if (flags) {
      for (const [col, lw] of (T.flag[1] ? [[T.flag[1], 3.6], [T.flag[0], 1.6]] : [[T.flag[0], 1.7]])) {
        g.strokeStyle = col; g.lineWidth = lw
        for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) if (flags[r][k]) g.strokeRect(ML + k * cell + 1, MT + r * cell + 1, cell - 2, cell - 2)
      }
    }
    if (markers) {
      markers.forEach((m) => {
        const x = ML + (m.c + 0.5) * cell, y = MT + (m.r + 0.5) * cell
        g.beginPath(); g.arc(x, y, Math.max(3.5, cell * 0.28), 0, Math.PI * 2)
        g.fillStyle = T.dark ? '#0e1721' : '#ffffff'; g.fill(); g.lineWidth = 2; g.strokeStyle = T.dark ? '#ffffff' : '#14202b'; g.stroke()
      })
    }
    if (regions) {
      g.font = '600 12px "Inter Variable", sans-serif'; g.textAlign = 'center'
      REGIONS.forEach((q) => {
        const x = ML + ((q.lon - grid.lon[0]) / grid.cell) * cell, y = MT + ((grid.lat[1] - q.lat) / grid.cell) * cell
        g.lineWidth = 3.5; g.strokeStyle = 'rgba(255,255,255,0.85)'; g.strokeText(q.name, x, y)
        g.fillStyle = T.label; g.fillText(q.name, x, y)
      })
    }
    if (selected) {
      const x = ML + selected.c * cell, y = MT + selected.r * cell
      g.lineWidth = 4; g.strokeStyle = T.select[0]; g.strokeRect(x + 1, y + 1, cell - 2, cell - 2)
      g.lineWidth = 2; g.strokeStyle = T.select[1]; g.strokeRect(x + 1, y + 1, cell - 2, cell - 2)
    }
    if (hov) {
      g.strokeStyle = T.dark ? '#ffffff' : '#14202b'; g.lineWidth = 2
      g.strokeRect(ML + hov.c * cell + 1, MT + hov.r * cell + 1, cell - 2, cell - 2)
    }
  }

  // ease between successive risk maps when the data changes; every other change repaints immediately
  useEffect(() => {
    cancelAnimationFrame(raf.current)
    const to = values, from = shown.current
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (kind !== 'risk' || !to || !from || reduce || from === to) { shown.current = to; paint(to, hover); return }
    const t0 = performance.now()
    const step = (now) => {
      const f = Math.min(1, (now - t0) / 280), e = 1 - (1 - f) ** 3
      const cur = to.map((row, r) => row.map((v, k) => from[r][k] + (v - from[r][k]) * e))
      shown.current = cur; paint(cur, hover)
      if (f < 1) raf.current = requestAnimationFrame(step); else shown.current = to
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [values])

  useEffect(() => { paint(shown.current || values, hover) }, [hover, mask, w, kind, stops, vmax, flags, selected, markers, T.id])

  function locate(e) {
    const b = cv.current.getBoundingClientRect()
    const x = e.clientX - b.left - ML, y = e.clientY - b.top - MT
    const c = Math.floor(x / cell), r = Math.floor(y / cell)
    return r >= 0 && r < rows && c >= 0 && c < cols ? { r, c, px: e.clientX - b.left, py: e.clientY - b.top } : null
  }

  return (
    <div ref={wrap} className="mapcanvas" onMouseMove={(e) => setHover(locate(e))} onMouseLeave={() => setHover(null)}
      onClick={(e) => { const l = locate(e); if (l && onSelect) onSelect(l.r, l.c) }} style={{ cursor: onSelect ? cursor || 'pointer' : 'default' }}>
      <canvas ref={cv} style={{ width: w, height: H }} role="img" aria-label="Grid map" />
      {hover && describe && values && (
        <div className="tip" style={{ left: Math.min(hover.px, w - 210), top: hover.py }}>{describe(hover.r, hover.c)}</div>
      )}
    </div>
  )
}

export const cellLat = (grid, r) => grid.lat[1] - (r + 0.5) * grid.cell
export const cellLon = (grid, c) => grid.lon[0] + (c + 0.5) * grid.cell
