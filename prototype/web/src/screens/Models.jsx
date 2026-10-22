import React, { useMemo, useState } from 'react'
import { Info } from '../components/Icons.jsx'
import { CONVLSTM_SEEDS, GROUPS, MODEL_INFO, PERSISTENCE, prevalence } from '../stats.js'
import { fmt, mean, sd } from '../util.js'

const COLOR = { env: '#b4bcc7', sight: '#8da2c0', 'nn-env': '#8fb0e6', 'nn-sight': '#1f5fbf' }
const ORDER = Object.keys(MODEL_INFO)

/** Rows for one target and horizon: single runs from rolling_all_models.csv, multi-seed models from the seed CSVs. */
function buildRows(results, target, h) {
  const rows = []
  for (const model of ORDER) {
    const info = MODEL_INFO[model]
    const seeds = results.seeds.filter((r) => r.target === target && r.model === model && r.horizon === h)
    if (model.includes('[sight_month]')) {
      if (!seeds.length) continue
      const v = seeds.map((s) => s.pr_auc)
      const avg = (k) => mean(seeds.map((s) => s[k]).filter((x) => x != null))
      rows.push({ model, ...info, name: seeds.length > 1 ? info.name : `${info.name} (1 seed)`, value: mean(v), sd: sd(v), dots: v, n: seeds.length,
        lift: avg('lift'), recall: avg('recall@10%'), brier: avg('brier') })
    } else {
      const r = results.models.find((x) => x.target === target && x.model === model && x.horizon === h)
      if (!r) continue
      rows.push({ model, ...info, value: r.pr_auc, lo: r.pr_auc_lo, hi: r.pr_auc_hi, lift: r.lift, recall: r['recall@10%'], brier: r.brier, n: 1 })
    }
  }
  return rows
}

function BarChart({ rows, chance, pers, onHover, hovered }) {
  const W = 1000, LBL = 262, R = 24, rowH = 34, gh = 40
  const hiMax = Math.max(...rows.map((r) => r.hi ?? Math.max(r.value, ...(r.dots ?? [0]))), pers)
  const max = Math.ceil((hiMax + 0.02) * 10) / 10
  const X = (v) => LBL + (v / max) * (W - LBL - R)
  let y = 30
  const items = []
  let lastGroup = null
  rows.forEach((r) => {
    if (r.group !== lastGroup) {
      items.push({ type: 'group', y, g: r.group }); y += gh; lastGroup = r.group
    }
    items.push({ type: 'row', y, r }); y += rowH
  })
  const H = y + 26
  const ticks = []
  for (let t = 0; t <= max + 1e-9; t += 0.1) ticks.push(+t.toFixed(1))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="PR-AUC by model">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={22} y2={H - 22} stroke="#e3e7ec" strokeWidth="1" />
          <text x={X(t)} y={H - 6} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12.5" fill="#5b6877">{t.toFixed(1)}</text>
        </g>
      ))}
      <line x1={X(chance)} x2={X(chance)} y1={22} y2={H - 22} stroke="#5b6877" strokeWidth="1.5" strokeDasharray="5 4" />
      <text x={X(chance) + 5} y={16} fontFamily="var(--font-mono)" fontSize="12.5" fill="#3b4958">chance {chance.toFixed(2)}</text>
      <line x1={X(pers)} x2={X(pers)} y1={22} y2={H - 22} stroke="#14202b" strokeWidth="2" />
      <text x={X(pers) + 5} y={16} fontFamily="var(--font-mono)" fontSize="12.5" fontWeight="600" fill="#14202b">persistence {pers.toFixed(2)}</text>
      {items.map((it, i) => it.type === 'group' ? (
        <text key={i} x={0} y={it.y + 22} fontFamily="var(--font-mono)" fontSize="12.5" letterSpacing="1.4" fill="#5b6877">{GROUPS[it.g].title.toUpperCase()}</text>
      ) : (
        <g key={i} tabIndex={0} onMouseEnter={() => onHover(it.r)} onFocus={() => onHover(it.r)} onMouseLeave={() => onHover(null)} onBlur={() => onHover(null)}
          style={{ outline: 'none', cursor: 'default' }} opacity={hovered && hovered.model !== it.r.model ? 0.55 : 1}>
          <rect x={0} y={it.y} width={W} height={rowH} fill={hovered?.model === it.r.model ? 'rgba(31,95,191,.08)' : 'transparent'} />
          <text x={8} y={it.y + 21} fontFamily="var(--font-mono)" fontSize="12.5" fill="#5b6877">{it.r.code}</text>
          <text x={52} y={it.y + 22} fontFamily="var(--font-body)" fontSize="15" fontWeight={it.r.model === PERSISTENCE ? 700 : 500} fill="#14202b">{it.r.name}</text>
          <rect x={LBL} y={it.y + 8} width={Math.max(2, X(it.r.value) - LBL)} height={18} rx="3" fill={it.r.model === PERSISTENCE ? '#14202b' : COLOR[it.r.group]} />
          {it.r.lo != null && (
            <g stroke="#14202b" strokeWidth="1.4"><line x1={X(it.r.lo)} x2={X(it.r.hi)} y1={it.y + 17} y2={it.y + 17} /><line x1={X(it.r.lo)} x2={X(it.r.lo)} y1={it.y + 12} y2={it.y + 22} /><line x1={X(it.r.hi)} x2={X(it.r.hi)} y1={it.y + 12} y2={it.y + 22} /></g>
          )}
          {it.r.dots?.map((d, k) => <circle key={k} cx={X(d)} cy={it.y + 17} r="4.6" fill="#ffffff" stroke="#14202b" strokeWidth="1.5" />)}
          <text x={X(it.r.hi ?? Math.max(it.r.value, ...(it.r.dots ?? [0]))) + 10} y={it.y + 22} fontFamily="var(--font-mono)" fontSize="13.5" fontWeight="600" fill="#14202b">
            {it.r.value.toFixed(3)}{it.r.n > 1 ? ` ±${it.r.sd.toFixed(3)}` : ''}
          </text>
        </g>
      ))}
    </svg>
  )
}

const PERIODS = ['2019H1', '2019H2', '2020H1', '2020H2']

function PeriodChart({ results, h }) {
  const data = results.by_period.filter((r) => r.horizon === h && PERIODS.includes(r.period))
  const get = (model, p, seed) => data.find((r) => r.model === model && r.period === p && (seed == null || r.seed === seed))
  const seedIds = [...new Set(data.filter((r) => r.model === CONVLSTM_SEEDS).map((r) => r.seed))].sort()
  const W = 560, Ht = 316, L = 44, B = 68, T = 14, R = 14
  const X = (i) => L + ((i + 0.5) / PERIODS.length) * (W - L - R)
  const Y = (v) => T + (1 - v) * (Ht - T - B)
  const line = (pts) => pts.filter((p) => p.v != null).map((p, i) => `${i ? 'L' : 'M'}${X(p.i)} ${Y(p.v)}`).join(' ')
  const persPts = PERIODS.map((p, i) => ({ i, v: get(PERSISTENCE, p)?.pr_auc }))
  const seedLines = seedIds.map((s) => PERIODS.map((p, i) => ({ i, v: get(CONVLSTM_SEEDS, p, s)?.pr_auc })))
  const meanPts = PERIODS.map((p, i) => {
    const vs = seedIds.map((s) => get(CONVLSTM_SEEDS, p, s)?.pr_auc).filter((x) => x != null)
    return { i, v: vs.length ? mean(vs) : null }
  })
  const bw = 62
  return (
    <svg viewBox={`0 0 ${W} ${Ht}`} width="100%" role="img" aria-label="PR-AUC by half-year">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}><line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} stroke="#e3e7ec" /><text x={L - 8} y={Y(t) + 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="12.5" fill="#5b6877">{t.toFixed(2)}</text></g>
      ))}
      {PERIODS.map((p, i) => {
        const prev = get(PERSISTENCE, p)?.prevalence ?? 0
        return (
          <g key={p}>
            <rect x={X(i) - bw / 2} y={Y(prev)} width={bw} height={Y(0) - Y(prev)} fill="#e3e7ec" />
            <text x={X(i)} y={Ht - 34} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="13" fill="#14202b">{p}</text>
            <text x={X(i)} y={Ht - 15} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">{get(PERSISTENCE, p)?.origins} origins</text>
            <text x={X(i)} y={Ht - 2} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12" fill="#5b6877">{Math.round(prev * 100)}% positive</text>
          </g>
        )
      })}
      {seedLines.map((pts, k) => <path key={k} d={line(pts)} fill="none" stroke="#1f5fbf" strokeWidth="1.3" opacity="0.4" />)}
      <path d={line(persPts)} fill="none" stroke="#14202b" strokeWidth="2.6" />
      <path d={line(meanPts)} fill="none" stroke="#1f5fbf" strokeWidth="2.6" />
      {persPts.map((p) => p.v != null && <circle key={`p${p.i}`} cx={X(p.i)} cy={Y(p.v)} r="4.5" fill="#14202b" />)}
      {meanPts.map((p) => p.v != null && <circle key={`m${p.i}`} cx={X(p.i)} cy={Y(p.v)} r="4.5" fill="#1f5fbf" />)}
    </svg>
  )
}

export default function Models({ results }) {
  const [target, setTarget] = useState('any')
  const [hi, setHi] = useState(0)
  const h = [28, 42, 56][hi]
  const rows = useMemo(() => buildRows(results, target, h), [results, target, h])
  const [hover, setHover] = useState(null)
  const chance = prevalence(results, target, h)
  const pers = rows.find((r) => r.model === PERSISTENCE)
  const neural = rows.filter((r) => r.group.startsWith('nn'))
  const best = neural.reduce((b, r) => (b == null || r.value > b.value ? r : b), null)
  const detail = hover || pers

  const verdict = best.value > pers.value
    ? <>The best neural row, <b>{best.name}{best.group === 'nn-sight' ? ' (sightings + month)' : ' (environment only)'}</b>, scores {fmt(best.value)} against persistence {fmt(pers.value)}, but this is {best.n > 1 ? `a ${best.n}-seed spread` : 'a single run'} whose interval overlaps persistence. <b>Not a claimable win.</b></>
    : <>Persistence ({fmt(pers.value)}) is ahead of every neural model here; the best, <b>{best.name}</b>{best.group === 'nn-sight' ? ' (sightings + month)' : ' (environment only)'}, reaches {fmt(best.value)}{best.n > 1 ? ` ± ${fmt(best.sd)}` : ''}.</>

  return (
    <div className="screen">
      <div className="tool-title"><h1>Model comparison</h1><p>Every model on the same 45 rolling-origin forecasts, scored with the same yardstick. The dark line is the rule to beat.</p></div>

      <div className="ctl-row" style={{ marginTop: 20 }}>
        <div className="seg" role="group" aria-label="Target">
          <button aria-pressed={target === 'any'} onClick={() => setTarget('any')}>Any life stage</button>
          <button aria-pressed={target === 'breeding'} onClick={() => setTarget('breeding')}>Breeding only</button>
        </div>
        <div className="seg" role="group" aria-label="Horizon">
          {[28, 42, 56].map((x, i) => <button key={x} aria-pressed={hi === i} onClick={() => setHi(i)}>{x} days</button>)}
        </div>
        <span className="eyebrow" style={{ marginLeft: 'auto' }}>PR-AUC on known cells · higher is better</span>
      </div>

      <div className="card" style={{ marginTop: 16, padding: '18px 22px 12px' }}>
        <BarChart rows={rows} chance={chance} pers={pers.value} onHover={setHover} hovered={hover} />
        <div className="legend" style={{ marginTop: 4 }}>
          <span><i className="sw" style={{ background: '#14202b' }} /> persistence rule</span>
          <span><svg width="26" height="12"><line x1="2" x2="24" y1="6" y2="6" stroke="#14202b" strokeWidth="1.4" /><line x1="2" x2="2" y1="1" y2="11" stroke="#14202b" strokeWidth="1.4" /><line x1="24" x2="24" y1="1" y2="11" stroke="#14202b" strokeWidth="1.4" /></svg> bootstrap interval (single runs)</span>
          <span><svg width="14" height="14"><circle cx="7" cy="7" r="4.6" fill="#ffffff" stroke="#14202b" strokeWidth="1.5" /></svg> one seed (bar = mean)</span>
        </div>
        <div className="chart-note" style={{ minHeight: 24 }}>
          <b style={{ color: 'var(--ink)' }}>{detail.name}</b> ({detail.code}): PR-AUC {fmt(detail.value)}
          {detail.lo != null && <> · interval {fmt(detail.lo, 2)}–{fmt(detail.hi, 2)}</>}
          {detail.n > 1 && <> · {detail.n} seeds, sd {fmt(detail.sd)}</>}
          {' '}· lift {fmt(detail.lift, 2)}× · recall in top 10% area {fmt(detail.recall, 2)} · Brier {fmt(detail.brier, 3)}
        </div>
      </div>

      <section className="section">
        <div className="callout dark">
          <Info className="ico" style={{ color: 'var(--accent-ink)' }} />
          <p><b>Reading this panel.</b> {verdict}</p>
        </div>
      </section>

      <section className="section">
        <div className="split" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
          <div className="card">
            <h3>It depends on the season</h3>
            <div className="sub" style={{ marginBottom: 8 }}>PR-AUC by half-year, any life stage, {h}-day horizon</div>
            {target === 'any' ? <PeriodChart results={results} h={h} /> : <p className="prose">The half-year breakdown is shown for the any-life-stage target.</p>}
            <div className="legend">
              <span><i className="sw" style={{ background: '#14202b' }} /> persistence</span>
              <span><i className="sw" style={{ background: '#1f5fbf' }} /> ConvLSTM (thick = mean of seeds)</span>
              <span><i className="sw" style={{ background: '#e3e7ec' }} /> share of positive cells</span>
            </div>
          </div>
          <div>
            <div className="eyebrow">What the chart says</div>
            <ul className="check" style={{ marginTop: 8 }}>
              <li><span style={{ width: 18 }} /><span><b>Prevalence changes the game.</b> When most known cells are positive (2019 H2), almost any ranking scores high; when almost none are (2020 H2), every model scores near zero.</span></li>
              <li><span style={{ width: 18 }} /><span><b>Seeds disagree.</b> The thin orange lines are separate random initialisations of the same network. Their spread is comparable to the gap to persistence.</span></li>
              <li><span style={{ width: 18 }} /><span><b>Few forecasts per bar.</b> Each half-year holds only about ten origins, so single points are indicative, not conclusive.</span></li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
