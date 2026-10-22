import React, { useEffect, useRef } from 'react'
import { colorAt, resolveStops } from './GridMap.jsx'
import { useMapTheme } from '../mapTheme.js'

/** Small static thumbnail of a grid (no axes), for filmstrips and side panels. */
export default function MiniMap({ values, stops, vmax = 1, diverging = false, width = 84, active = false, onClick, label }) {
  const ref = useRef(null)
  const T = useMapTheme()
  const st0 = resolveStops(stops, T)
  const rows = values.length, cols = values[0].length
  const cell = width / cols, height = Math.round(cell * rows)
  useEffect(() => {
    const c = ref.current, dpr = window.devicePixelRatio || 1
    c.width = Math.round(width * dpr); c.height = Math.round(height * dpr)
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0)
    for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
      const v = values[r][k]
      g.fillStyle = colorAt(st0, diverging ? (v / vmax + 1) / 2 : v / vmax)
      g.fillRect(k * cell, r * cell, cell + 0.4, cell + 0.4)
    }
  }, [values, stops, vmax, diverging, width, T.id])
  return (
    <button className={`mini${active ? ' active' : ''}`} onClick={onClick} aria-label={label} aria-pressed={active} disabled={!onClick}>
      <canvas ref={ref} style={{ width, height }} />
    </button>
  )
}
