import React, { useState } from 'react'
import GridMap, { POS_STOPS, NEG_STOPS, cellLat, cellLon } from '../components/GridMap.jsx'
import OriginBar from '../components/OriginBar.jsx'
import { useForecast } from '../hooks.js'
import { useMapTheme } from '../mapTheme.js'
import { niceRange, pct } from '../util.js'

const OUT_NAME = ['unknown', 'survey found none', 'locust recorded']

const valuesFor = (T) => [
  { c: T.outcome[2], t: 'Positive', r: 'A locust record lies in the cell during the 16-day window.' },
  { c: T.outcome[1], t: 'Negative', r: 'A survey found no locust there, and there is no positive record.' },
  { c: T.outcome[0], t: 'Unknown', r: 'Neither. Ignored by the loss and by every metric.', hatch: true },
]

export default function Labels({ meta, results, sel, setSel }) {
  const T = useMapTheme()
  const VALUES = valuesFor(T)
  const g = meta.grid, cov = results.coverage
  const { data } = useForecast(sel.origin, sel.mode)
  const [target, setTarget] = useState('any')
  const out = data?.outcome?.[sel.hz] ?? null
  const st = data?.stats?.[sel.hz]
  const counts = out ? out.flat().reduce((a, v) => { a[v]++; return a }, [0, 0, 0]) : null
  const rows = results.labels.filter((r) => r.target === target && r.horizon === 28)
  const train = results.labels.find((r) => r.target === 'any' && r.block === 'train' && r.horizon === 28)
  const maxN = Math.max(...rows.map((r) => r.positive + r.negative))
  const posMax = Math.max(...cov.positive.flat()), negMax = Math.max(...cov.negative.flat())
  const BLOCK = { train: 'Training', val: 'Validation', test: 'Test (development)', embargo: 'Embargoed' }

  return (
    <div className="screen">
      <div className="tool-title"><h1>Labels &amp; data</h1><p>The ground truth the model learns from and is scored on. Only cells where someone recorded something carry a label; everything else is unknown and is never treated as “no locusts”.</p></div>
      <OriginBar meta={meta} sel={sel} setSel={setSel} info={data} />

      <div className="two-col" style={{ marginTop: 20 }}>
        <section className="mapcard">
          <header><h3>Label map for the selected window</h3></header>
          <div className="mc-sub">{st?.window ? `${meta.horizons[sel.hz]}-day target window: ${niceRange(st.window[0], st.window[1])}` : 'No labels exist for this window yet.'}</div>
          {out ? (
            <GridMap grid={g} values={out} kind="outcome" regions
              describe={(r, c) => <><div className="mono" style={{ color: 'var(--gold)' }}>{cellLat(g, r).toFixed(3)}°N {cellLon(g, c).toFixed(3)}°E</div><div>{OUT_NAME[out[r][c]]}</div></>} />
          ) : <div className="empty-map"><div><div className="t">Not recorded yet</div><p>Choose an earlier date to see labels.</p></div></div>}
          {counts && (
            <div className="legend" style={{ marginTop: 10, gap: 18 }}>
              <span><i className="sw" style={{ background: T.outcome[2] }} /> <b>{counts[2]}</b> positive</span>
              <span><i className="sw" style={{ background: T.outcome[1] }} /> <b>{counts[1]}</b> negative</span>
              <span><i className="sw" style={{ background: T.outcome[0] }} /> <b>{counts[0]}</b> unknown ({pct(counts[0] / out.flat().length)})</span>
            </div>
          )}
        </section>
        <section className="panel" style={{ marginTop: 0 }}>
          <h3>How a cell gets its label</h3>
          {VALUES.map((v) => (
            <div key={v.t} style={{ display: 'grid', gridTemplateColumns: '34px 1fr', gap: 12, padding: '12px 0', borderTop: '1px solid var(--line)' }}>
              <span style={{ width: 34, height: 34, borderRadius: 7, background: v.c, border: '1px solid rgba(0,0,0,.14)', backgroundImage: v.hatch ? 'repeating-linear-gradient(135deg, transparent 0 5px, rgba(110,125,145,.3) 5px 6px)' : undefined }} />
              <div><b>{v.t}</b><div className="sub" style={{ fontSize: 14.5 }}>{v.r}</div></div>
            </div>
          ))}
          <div className="stat" style={{ marginTop: 14 }}>
            <div className="v" style={{ fontSize: 38 }}>{pct(train.known_frac)}<small> known</small></div>
            <div className="l">of training cell-samples have a label (28-day horizon). The other ~88% are masked out, not guessed.</div>
          </div>
          <h3 style={{ marginTop: 20 }}>Sources</h3>
          <table className="t compact">
            <tbody>
              <tr><td>FAO Desert Locust catalogue</td><td>10,364 locust cell-dates · 56,795 “none found”</td></tr>
              <tr><td>MODIS MOD13Q1 v061</td><td>NDVI, water index, valid-pixel mask</td></tr>
              <tr><td>ERA5-Land</td><td>temperature, rain, soil moisture</td></tr>
            </tbody>
          </table>
          <p className="chart-note">The FAO field <span className="code">area_ha</span> is treated area, not infested area, and is never used.</p>
        </section>
      </div>

      <div className="two-col">
        <figure className="mapcard" style={{ margin: 0 }}>
          <header><h3>Where locusts were recorded</h3></header>
          <div className="mc-sub">Positive windows per cell, training period 2008–2019 ({cov.samples} forecast dates)</div>
          <GridMap grid={g} values={cov.positive} kind="count" stops={POS_STOPS} vmax={posMax} describe={(r, c) => <><b>{cov.positive[r][c]}</b> positive windows</>} />
        </figure>
        <figure className="mapcard" style={{ margin: 0 }}>
          <header><h3>Where surveys found none</h3></header>
          <div className="mc-sub">Negative windows per cell, same period. Pale cells may simply be unobserved.</div>
          <GridMap grid={g} values={cov.negative} kind="count" stops={NEG_STOPS} vmax={negMax} describe={(r, c) => <><b>{cov.negative[r][c]}</b> negative windows</>} />
        </figure>
      </div>

      <section className="panel">
        <div className="lab-head">
          <h3>Class balance by period</h3>
          <div className="seg on-card" role="group" aria-label="Target">
            <button aria-pressed={target === 'any'} onClick={() => setTarget('any')}>Any life stage</button>
            <button aria-pressed={target === 'breeding'} onClick={() => setTarget('breeding')}>Breeding only</button>
          </div>
        </div>
        <table className="t compact">
          <thead><tr><th>Block</th><th>Origins</th><th style={{ width: '40%', textAlign: 'left' }}>Known cell-samples</th><th>Positive</th><th>Negative</th><th>Positive share</th></tr></thead>
          <tbody>
            {rows.map((r) => {
              const n = r.positive + r.negative
              return (
                <tr key={r.block}>
                  <td>{BLOCK[r.block] || r.block}</td><td>{r.origins}</td>
                  <td style={{ textAlign: 'left' }}>
                    <div style={{ display: 'flex', height: 12, width: `${(n / maxN) * 100}%`, minWidth: 40, borderRadius: 3, overflow: 'hidden', background: T.outcome[1] }}><i style={{ width: `${(r.positive / n) * 100}%`, background: T.outcome[2] }} /></div>
                  </td>
                  <td>{r.positive.toLocaleString()}</td><td>{r.negative.toLocaleString()}</td><td>{pct(r.positive / n, 1)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="chart-note">Horizon 28 d. The share of known cells that are positive reflects where and when surveys were recorded, not the true locust rate, and it moves a lot between periods (about 12% in training, about 37% in the 2019–2020 rolling evaluation). Scores are always read against that period’s chance level.</p>
      </section>
    </div>
  )
}
