import React, { useMemo, useState } from 'react'
import GridMap, { cellLat, cellLon } from '../components/GridMap.jsx'
import OriginBar from '../components/OriginBar.jsx'
import { Sparkline, ChannelStrip } from '../components/Series.jsx'
import { useMapTheme } from '../mapTheme.js'
import RiskLegend from '../components/RiskLegend.jsx'
import { captured, rankCells, topFlags, useForecast, useInputs } from '../hooks.js'
import { fmt, niceRange } from '../util.js'

const OUT_NAME = ['unknown', 'survey: none', 'locust recorded']

export default function Forecast({ meta, sel, setSel }) {
  const g = meta.grid
  const T = useMapTheme()
  const { data, loading, err } = useForecast(sel.origin, sel.mode)
  const inputs = useInputs(sel.origin)
  const [q, setQ] = useState(10)          // alert area, % of all cells
  const [grey, setGrey] = useState(false)
  const hz = sel.hz, cell = sel.cell
  const st = data?.stats?.[hz]
  const out = data?.outcome?.[hz] ?? null
  const risk = data?.convlstm?.[hz]
  const pers = data?.persistence

  const flagsC = useMemo(() => (risk ? topFlags(risk, q / 100) : null), [risk, q])
  const flagsP = useMemo(() => (pers ? topFlags(pers, q / 100) : null), [pers, q])
  const capC = out && flagsC ? captured(flagsC, out) : null
  const capP = out && flagsP ? captured(flagsP, out) : null
  const top = useMemo(() => (risk ? rankCells(risk).slice(0, 8) : []), [risk])

  const select = (r, c) => setSel({ cell: { r, c } })
  const describe = (r, c) => (
    <>
      <div className="mono" style={{ color: 'var(--gold)' }}>{cellLat(g, r).toFixed(3)}°N {cellLon(g, c).toFixed(3)}°E</div>
      <div>ConvLSTM risk <b>{fmt(risk[r][c], 2)}</b></div>
      <div>Persistence <b>{fmt(pers[r][c], 2)}</b></div>
      {out && <div>Outcome <b>{OUT_NAME[out[r][c]]}</b></div>}
    </>
  )
  const winLabel = st?.window ? niceRange(st.window[0], st.window[1]) : null
  const diff = st && st.pr_convlstm != null ? st.pr_convlstm - st.pr_persistence : null
  const chance = st && st.known ? st.pos / st.known : null

  return (
    <div className="screen">
      <div className="tool-title"><h1>Forecast</h1><p>Pick a date. The network reads the previous six months of satellite, climate and sighting maps and returns a risk map for each horizon.</p></div>
      <OriginBar meta={meta} sel={sel} setSel={setSel} info={data} />

      {err ? <div className="callout" style={{ marginTop: 20 }}><p><b>Could not load this forecast.</b> {err}</p></div> : (
        <>
          <div className="maps">
            <MapCard title="ConvLSTM forecast" sub={`Risk of locust presence, ${meta.horizons[hz]} days ahead · click a cell`} loading={loading}>
              {data ? <GridMap grid={g} values={risk} mask={grey ? out : null} flags={flagsC} selected={cell} onSelect={select} describe={describe} /> : <Skel />}
            </MapCard>
            <MapCard title="Persistence rule" sub="Recent sightings only, no learning. Same map at every horizon." loading={loading}>
              {data ? <GridMap grid={g} values={pers} mask={grey ? out : null} flags={flagsP} selected={cell} onSelect={select} describe={describe} /> : <Skel />}
            </MapCard>
            <MapCard title="Recorded outcome" sub={winLabel ? `What surveys found, ${winLabel}` : 'Not yet recorded'} loading={loading}>
              {data && out ? <GridMap grid={g} values={out} kind="outcome" selected={cell} onSelect={select} describe={describe} /> : (
                <div className="empty-map">
                  {data ? <div><div className="t">Nothing to compare yet</div><p>This window has not happened, or has no records. It is a prediction, not a result.</p></div> : <Skel />}
                </div>
              )}
            </MapCard>
          </div>

          <RiskLegend />
          <div className="legend" style={{ marginTop: 10, gap: 26 }}>
            <span><i className="sw" style={{ background: T.outcome[2] }} /> locust recorded</span>
            <span><i className="sw" style={{ background: T.outcome[1] }} /> survey found none</span>
            <span><i className="sw" style={{ background: T.outcome[0], backgroundImage: 'repeating-linear-gradient(135deg, transparent 0 4px, rgba(110,125,145,.4) 4px 5px)' }} /> unknown, not scored</span>
            <label className="check-inline"><input type="checkbox" checked={grey} onChange={(e) => setGrey(e.target.checked)} disabled={!out} /> grey out unknown cells on the risk maps</label>
          </div>

          <div className="metrics" aria-live="polite">
            <div className="metric">
              <div className="eyebrow">{meta.horizons[hz]}-day forecast for</div>
              <div className="v" style={{ fontSize: 26, marginTop: 4 }}>{winLabel ?? 'future window'}</div>
              <div className="l">{st?.known ? <><b>{st.known}</b> known cells · <b>{st.pos}</b> with locust ({Math.round(chance * 100)}%)</> : 'no recorded outcome for this window'}</div>
            </div>
            <Score label="ConvLSTM PR-AUC" v={st?.pr_convlstm} chance={chance} color="var(--accent)" />
            <Score label="Persistence PR-AUC" v={st?.pr_persistence} chance={chance} color="var(--ink)" />
            <div className="metric">
              <div className="eyebrow">This forecast alone</div>
              <div className="v" style={{ color: diff == null ? 'var(--ink-3)' : diff >= 0 ? 'var(--good)' : 'var(--accent-ink)' }}>
                {diff == null ? '–' : `${diff >= 0 ? '+' : '−'}${Math.abs(diff).toFixed(2)}`}
              </div>
              <div className="l">
                {diff == null ? 'needs both locust and non-locust cells' : diff >= 0 ? 'ConvLSTM ahead of persistence' : 'persistence ahead of ConvLSTM'}
                {data?.mode === 'live' && data?.in_training && diff != null && <> · <b>not a fair test:</b> this date was in the training data</>}
              </div>
            </div>
          </div>

          <div className="panels">
            <section className="panel">
              <h3>Alert planner</h3>
              <p className="sub">Suppose teams can only survey the top <b>{q}%</b> of the map ({Math.round((q / 100) * g.rows * g.cols)} cells). Flagged cells are outlined on the maps.</p>
              <input type="range" min={1} max={50} value={q} onChange={(e) => setQ(+e.target.value)} aria-label="Alert area percent" style={{ marginTop: 8 }} />
              {capC ? (
                capC.pos ? (
                  <div style={{ marginTop: 10 }}>
                    <Capture label="ConvLSTM" hit={capC.hit} pos={capC.pos} color="var(--accent)" />
                    <Capture label="Persistence" hit={capP.hit} pos={capP.pos} color="var(--ink)" />
                    <Capture label="Random pick" hit={Math.round((q / 100) * capC.pos)} pos={capC.pos} color="var(--line-2)" approx />
                    <p className="chart-note">Share of the {capC.pos} cells with a recorded locust that fall inside the flagged area. Ties in the risk are broken at random.</p>
                  </div>
                ) : <p className="chart-note" style={{ marginTop: 12 }}>No recorded locust cells in this window, so there is nothing to capture.</p>
              ) : <p className="chart-note" style={{ marginTop: 12 }}>No recorded outcome for this window. The outlined cells are where the model would send surveyors.</p>}
            </section>

            <section className="panel">
              <h3>Highest-risk cells</h3>
              <p className="sub">{meta.horizons[hz]}-day forecast. Click a row to inspect the cell.</p>
              <table className="t compact">
                <thead><tr><th>#</th><th>Location</th><th>Conv</th><th>Pers</th><th>Found</th></tr></thead>
                <tbody>
                  {top.map(({ r, c, v }, i) => (
                    <tr key={`${r}-${c}`} className={cell && cell.r === r && cell.c === c ? 'hl' : ''} onClick={() => select(r, c)} style={{ cursor: 'pointer' }}>
                      <td>{i + 1}</td>
                      <td className="mono" style={{ fontSize: 13.5 }}>{cellLat(g, r).toFixed(2)}°N {cellLon(g, c).toFixed(2)}°E</td>
                      <td>{fmt(v, 2)}</td><td>{fmt(pers[r][c], 2)}</td>
                      <td>{out ? <i className="sw" title={OUT_NAME[out[r][c]]} style={{ background: T.outcome[out[r][c]] }} /> : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="panel">
              <h3>Cell inspector</h3>
              {!cell ? (
                <div className="empty-panel"><div className="t">No cell selected</div><p>Click any cell on a map, or a row in the list, to see its forecasts and the last six months of inputs behind them.</p></div>
              ) : data && (
                <>
                  <p className="sub mono" style={{ color: 'var(--ink)' }}>{cellLat(g, cell.r).toFixed(3)}°N, {cellLon(g, cell.c).toFixed(3)}°E · row {cell.r}, col {cell.c}</p>
                  <table className="t compact" style={{ marginTop: 6 }}>
                    <thead><tr><th>Horizon</th><th>Conv</th><th>Pers</th><th>Found</th></tr></thead>
                    <tbody>
                      {meta.horizons.map((h, j) => (
                        <tr key={h} className={j === hz ? 'hl' : ''}>
                          <td>{h} d</td><td>{fmt(data.convlstm[j][cell.r][cell.c], 2)}</td><td>{fmt(data.persistence[cell.r][cell.c], 2)}</td>
                          <td>{data.outcome?.[j] ? OUT_NAME[data.outcome[j][cell.r][cell.c]] : '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {inputs.data ? <CellSeries inputs={inputs.data} cell={cell} /> : <div className="skeleton" style={{ height: 150, marginTop: 12 }} />}
                </>
              )}
            </section>
          </div>
        </>
      )}
    </div>
  )
}

const UNITS = { ndvi: 'NDVI anomaly', wi: 'water index anomaly', sm: 'soil moisture anomaly', temp: 'temperature anomaly', rain: 'rain anomaly' }

function CellSeries({ inputs, cell }) {
  const at = (ci) => inputs.data.map((step) => step[ci][cell.r][cell.c])
  return (
    <div style={{ marginTop: 12 }}>
      <div className="eyebrow" style={{ marginBottom: 4 }}>Inputs, {inputs.steps[0]} → {inputs.steps[inputs.steps.length - 1]}</div>
      {Object.keys(UNITS).map((k) => (
        <div className="spark-row" key={k}>
          <span>{UNITS[k]}</span>
          <Sparkline values={at(inputs.channels.indexOf(k))} />
          <b className="mono">{at(inputs.channels.indexOf(k)).slice(-1)[0].toFixed(1)}σ</b>
        </div>
      ))}
      <ChannelStrip label="sighted (any stage)" values={at(inputs.channels.indexOf('sight_any'))} />
      <ChannelStrip label="surveyed" values={at(inputs.channels.indexOf('surveyed'))} faint />
    </div>
  )
}

const Skel = () => <div className="skeleton" style={{ aspectRatio: '24 / 28' }} />

function MapCard({ title, sub, loading, children }) {
  return (
    <section className={`mapcard${loading ? ' loading' : ''}`}>
      <header><h3>{title}</h3></header>
      <div className="mc-sub">{sub}</div>
      {children}
    </section>
  )
}

function Score({ label, v, chance, color }) {
  return (
    <div className="metric">
      <div className="eyebrow">{label}</div>
      <div className="v">{v == null ? '–' : v.toFixed(2)}</div>
      <div className="bar" style={{ position: 'relative' }}>
        {v != null && <i style={{ width: `${v * 100}%`, background: color }} />}
        {chance != null && <b style={{ position: 'absolute', left: `${chance * 100}%`, top: 0, bottom: 0, width: 2, background: 'var(--ink-3)' }} />}
      </div>
    </div>
  )
}

function Capture({ label, hit, pos, color, approx }) {
  const f = pos ? hit / pos : 0
  return (
    <div className="capture">
      <span>{label}</span>
      <div className="track"><i style={{ transform: `scaleX(${f})`, background: color }} /></div>
      <b className="mono">{approx ? '≈ ' : ''}{Math.round(f * 100)}%</b>
      <small className="mono">{approx ? '' : `${hit}/${pos}`}</small>
    </div>
  )
}
