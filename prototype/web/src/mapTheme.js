import { useSyncExternalStore } from 'react'

/**
 * Map styles. Each theme restyles every grid map (risk, outcome, counts, inputs) and the thumbnails.
 * risk / pos / neg are colour stops [t, hex]. `bins` (optional) makes the risk scale a set of labelled classes:
 * colour stops are then a step function and the legend lists each class.
 * outcome colours: 0 unknown, 1 survey found none, 2 locust recorded.
 */
const stepStops = (breaks, colors) => {
  const out = []
  colors.forEach((c, i) => { out.push([i === 0 ? 0 : breaks[i - 1], c]); out.push([i === colors.length - 1 ? 1 : breaks[i], c]) })
  return out
}
const BREAKS = [0.1, 0.25, 0.5, 0.75]
const binLabels = (colors) => ['Low  < 0.10', 'Guarded  0.10 – 0.25', 'Elevated  0.25 – 0.50', 'High  0.50 – 0.75', 'Very high  ≥ 0.75'].map((label, i) => ({ label, color: colors[i] }))

const OR = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']
const BL = ['#eff3ff', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
const GR = ['#f2f2f2', '#d0d0d0', '#969696', '#525252', '#111111']

const COMMON = {
  dark: false, bg: '#ffffff', frame: '#9aa5b1',
  neg: [[0, '#f1f8f4'], [0.2, '#cfe9dc'], [0.55, '#55ad8c'], [1, '#14573f']],
  grid: 'rgba(255,255,255,0.7)', axis: '#4a5868', label: '#14202b', halo: 'rgba(255,255,255,0.92)',
  flag: ['#111827', null], select: ['#ffffff', '#1f5fbf'],
}

export const THEMES = {
  classes: {
    ...COMMON, name: 'Risk classes', bins: binLabels(OR), risk: stepStops(BREAKS, OR),
    pos: [[0, '#fff8dc'], [0.15, '#fde08b'], [0.5, '#e0521e'], [1, '#4a0a12']],
    outcome: { 0: '#e6e9ee', 1: '#a8dcc2', 2: '#1f2937' },
  },
  heat: {
    ...COMMON, name: 'Smooth heat',
    risk: [[0, '#fef0d9'], [0.08, '#fdd49e'], [0.25, '#fc8d59'], [0.5, '#e34a33'], [0.75, '#b30000'], [1, '#600000']],
    pos: [[0, '#fff8dc'], [0.15, '#fde08b'], [0.5, '#e0521e'], [1, '#4a0a12']],
    outcome: { 0: '#e6e9ee', 1: '#a8dcc2', 2: '#1f2937' },
  },
  blues: {
    ...COMMON, name: 'Blues', bins: binLabels(BL), risk: stepStops(BREAKS, BL),
    pos: [[0, '#fdf1e4'], [0.2, '#f9d0a6'], [0.5, '#e07a2c'], [1, '#6b2a05']],
    outcome: { 0: '#e6e9ee', 1: '#a8dcc2', 2: '#c2410c' },
  },
  grey: {
    ...COMMON, name: 'Greyscale (print)', bins: binLabels(GR), risk: stepStops(BREAKS, GR),
    pos: [[0, '#f2f2f2'], [0.2, '#d0d0d0'], [0.55, '#737373'], [1, '#111111']],
    outcome: { 0: '#e6e9ee', 1: '#9ecae1', 2: '#d7301f' },
    flag: ['#d7301f', null],
  },
}
export const THEME_ORDER = ['classes', 'heat', 'blues', 'grey']

let current = (() => { try { const t = localStorage.getItem('agriwatch-map-theme-v2'); return THEMES[t] ? t : 'classes' } catch { return 'classes' } })()
const subs = new Set()
export function setMapTheme(id) {
  current = id
  try { localStorage.setItem('agriwatch-map-theme-v2', id) } catch { /* private mode */ }
  subs.forEach((f) => f())
}
const subscribe = (f) => { subs.add(f); return () => subs.delete(f) }
export const useMapTheme = () => {
  const id = useSyncExternalStore(subscribe, () => current)
  return { id, ...THEMES[id] }
}

/** Kept for the diverging scales: neutral ends become dark only in a dark theme (none shipped now). */
export function adaptStops(stops, theme) {
  if (!theme.dark) return stops
  return stops
}
export const rampCss = (stops) => `linear-gradient(90deg, ${stops.map(([t, c]) => `${c} ${t * 100}%`).join(', ')})`
