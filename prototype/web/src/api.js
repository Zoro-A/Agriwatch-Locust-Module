const cache = new Map()

export async function getJSON(url) {
  if (cache.has(url)) return cache.get(url)
  const r = await fetch(url)
  if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`)
  const data = await r.json()
  cache.set(url, data)
  return data
}

export const fetchMeta = () => getJSON('/api/meta')
export const fetchResults = () => getJSON('/api/results')
export const fetchForecast = (origin, mode) => getJSON(`/api/forecast?origin=${origin}&mode=${mode}`)
