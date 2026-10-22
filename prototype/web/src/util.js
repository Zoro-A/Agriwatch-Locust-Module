export const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d))
export const pct = (v, d = 0) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`)
export const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length
export const sd = (a) => {
  if (a.length < 2) return 0
  const m = mean(a)
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1))
}
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export function niceDate(iso) {
  if (!iso) return ''
  const [y, m, d] = iso.split('-').map(Number)
  return `${d} ${MONTHS[m - 1]} ${y}`
}
export function niceRange(a, b) {
  const [y1, m1, d1] = a.split('-').map(Number)
  const [y2, m2, d2] = b.split('-').map(Number)
  if (y1 === y2 && m1 === m2) return `${d1}–${d2} ${MONTHS[m1 - 1]} ${y1}`
  if (y1 === y2) return `${d1} ${MONTHS[m1 - 1]} – ${d2} ${MONTHS[m2 - 1]} ${y1}`
  return `${niceDate(a)} – ${niceDate(b)}`
}
export const dayNum = (iso) => Date.UTC(...iso.split('-').map((x, i) => (i === 1 ? +x - 1 : +x))) / 86400000
