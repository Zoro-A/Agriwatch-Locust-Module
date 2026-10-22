import React, { useEffect, useMemo, useState } from 'react'
import GridMap, { BIN_STOPS, HEAT_STOPS, RISK_STOPS, WET_STOPS, cellLat, cellLon, rampCss } from '../components/GridMap.jsx'
import MiniMap from '../components/MiniMap.jsx'
import OriginBar from '../components/OriginBar.jsx'
import { Pause, Play } from '../components/Icons.jsx'
import { useForecast, useInputs } from '../hooks.js'
import { adaptStops, useMapTheme } from '../mapTheme.js'
import { fmt, niceDate } from '../util.js'

const CH = {
  ndvi: { name: 'NDVI anomaly', unit: 'σ', stops: WET_STOPS, vmax: 3, div: true, src: 'MODIS', d: 'Vegetation greenness against the 2001–2019 norm for the same time of year. Green means greener than normal.' },
  wi: { name: 'Water-index anomaly', unit: 'σ', stops: WET_STOPS, vmax: 3, div: true, src: 'MODIS', d: 'A near-infrared / mid-infrared water index. Green means wetter surface than normal.' },
  sm: { name: 'Soil-moisture anomaly', unit: 'σ', stops: WET_STOPS, vmax: 3, div: true, src: 'ERA5-Land', d: 'Soil moisture against the norm. Wet soil is where locust eggs can develop.' },
  temp: { name: 'Temperature anomaly', unit: 'σ', stops: HEAT_STOPS, vmax: 3, div: true, src: 'ERA5-Land', d: '2 m air temperature against the norm. Red is warmer than normal.' },
  rain: { name: 'Rain anomaly', unit: 'σ', stops: WET_STOPS, vmax: 3, div: true, src: 'ERA5-Land', d: 'Rainfall against the norm for this time of year.' },
  modis_observed: { name: 'MODIS observed', unit: '', stops: BIN_STOPS, vmax: 1, src: 'MODIS', d: 'Dark where the satellite saw enough valid pixels. Where it did not, the anomaly is set to zero and this mask says so.' },
  sight_any: { name: 'Sightings, any stage', unit: '', stops: BIN_STOPS, vmax: 1, src: 'FAO', d: 'Dark where a locust record fell in the cell during that 16-day step.' },
  sight_breeding: { name: 'Sightings, hopper / band', unit: '', stops: BIN_STOPS, vmax: 1, src: 'FAO', d: 'Dark where a breeding-stage record (hoppers or bands) fell in the cell.' },
  surveyed: { name: 'Surveyed', unit: '', stops: BIN_STOPS, vmax: 1, src: 'FAO', d: 'Dark where any survey was made, whatever it found. Tells the network where observers were.' },
}

export default function Inputs({ meta, sel, setSel }) {
  const g = meta.grid
  const { data: inp, loading } = useInputs(sel.origin)
  const fc = useForecast(sel.origin, sel.mode)
  const [ch, setCh] = useState('ndvi')
  const [step, setStep] = useState(11)
  const [playing, setPlaying] = useState(false)
  const c = CH[ch]
  const T = useMapTheme()

  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setStep((s) => (s + 1) % 12), 650)
    return () => clearInterval(t)
  }, [playing])

  const ci = inp ? inp.channels.indexOf(ch) : -1
  const field = inp && ci >= 0 ? inp.data[step][ci] : null
  const stats = useMemo(() => {
    if (!field) return null
    const flat = field.flat()
    if (!c.div) return { a: `${flat.reduce((s, v) => s + v, 0)} of ${flat.length}`, l: 'cells flagged in this step' }
    const mean = flat.reduce((s, v) => s + v, 0) / flat.length
    return { a: `${mean >= 0 ? '+' : ''}${mean.toFixed(2)}σ`, l: `mean over the map · range ${Math.min(...flat).toFixed(1)} to ${Math.max(...flat).toFixed(1)}σ` }
  }, [field, c])

  const cell = sel.cell
  return (
    <div className="screen">
      <div className="tool-title"><h1>Inputs</h1><p>The 12 maps per channel the network receives for this forecast date. Step through six months of history, channel by channel.</p></div>
      <OriginBar meta={meta} sel={sel} setSel={setSel} info={fc.data} />

      <div className="inputs-grid">
        <nav className="chlist" aria-label="Input channels">
          {Object.entries(CH).map(([k, v], i) => (
            <React.Fragment key={k}>
              <button className={k === ch ? 'on' : ''} onClick={() => setCh(k)}>
                <span className="src">{v.src}</span>
                <span className="nm">{v.name}</span>
              </button>
              {i === 4 && <div className="divider" />}
            </React.Fragment>
          ))}
          <div className="chnote">Plus sin / cos of the month, so the network knows the season. 11 channels in all.</div>
        </nav>

        <section className="mapcard" style={{ minWidth: 0 }}>
          <header style={{ justifyContent: 'space-between' }}>
            <h3>{c.name}</h3>
            <span className="mono" style={{ fontSize: 13.5, color: 'var(--ink-3)' }}>{inp ? `step ${step + 1} of 12 · ends ${niceDate(inp.steps[step])}` : ''}</span>
          </header>
          <div className="mc-sub">{c.d}</div>
          {field ? (
            <GridMap grid={g} values={field} kind="field" stops={c.stops} vmax={c.vmax} diverging={!!c.div} selected={cell}
              onSelect={(r, cc) => setSel({ cell: { r, c: cc } })}
              describe={(r, cc) => (
                <>
                  <div className="mono" style={{ color: 'var(--gold)' }}>{cellLat(g, r).toFixed(3)}°N {cellLon(g, cc).toFixed(3)}°E</div>
                  <div>{c.name} <b>{c.div ? `${field[r][cc].toFixed(2)}σ` : field[r][cc] ? 'yes' : 'no'}</b></div>
                </>
              )} />
          ) : <div className="skeleton" style={{ aspectRatio: '24 / 28' }} />}
          <div className="legend" style={{ marginTop: 10, gap: 12 }}>
            {c.div ? (<>
              <span className="mono" style={{ fontSize: 12.5 }}>−{c.vmax}σ</span><span style={{ width: 170, height: 10, borderRadius: 5, border: '1px solid rgba(0,0,0,.14)', background: rampCss(adaptStops(c.stops, T)) }} /><span className="mono" style={{ fontSize: 12.5 }}>+{c.vmax}σ</span>
            </>) : (<><span><i className="sw" style={{ background: T.dark ? '#e8eef5' : '#14202b' }} /> yes</span><span><i className="sw" style={{ background: T.dark ? '#1d2a38' : '#f1f3f6' }} /> no</span></>)}
            {stats && <span style={{ marginLeft: 'auto' }}><b>{stats.a}</b>&nbsp;<span style={{ color: 'var(--ink-3)' }}>{stats.l}</span></span>}
          </div>
        </section>

        <aside className="side-col">
          <div className="panel" style={{ marginTop: 0 }}>
            <h3>Time</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button className="btn ghost" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause' : 'Play history'}>{playing ? <Pause /> : <Play />}</button>
              <input type="range" min={0} max={11} value={step} onChange={(e) => { setPlaying(false); setStep(+e.target.value) }} aria-label="Input step" />
            </div>
            <div className="slider-ends"><span>{inp ? niceDate(inp.steps[0]) : ''}</span><span>{inp ? niceDate(inp.steps[11]) : ''}</span></div>
            <div className="film" aria-label="Filmstrip of the 12 input steps">
              {inp && inp.data.map((s, i) => (
                <MiniMap key={i} values={s[ci]} stops={c.stops} vmax={c.vmax} diverging={!!c.div} width={56} active={i === step} onClick={() => { setPlaying(false); setStep(i) }} label={`Step ${i + 1}`} />
              ))}
            </div>
          </div>
          <div className="panel">
            <h3>→ What comes out</h3>
            <p className="sub">ConvLSTM risk, {meta.horizons[sel.hz]} days ahead, for the same date.</p>
            {fc.data ? <MiniMap values={fc.data.convlstm[sel.hz]} stops={RISK_STOPS} width={200} /> : <div className="skeleton" style={{ height: 230, width: 200 }} />}
            <p className="chart-note">Open <b>Forecast</b> for the full map, outcome and scoring.</p>
          </div>
        </aside>
      </div>
    </div>
  )
}
