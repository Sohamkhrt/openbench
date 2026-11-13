import { type Row, colorFor } from '../lib/metrics'
import { CLASS_LABEL } from './Theater'

const SHORT = ['S', 'R', 'P', 'B']

function rgba(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a.toFixed(3)})`
}

/** Row-normalised confusion matrices: each row shows where windows of one true class ended up. */
export function ConfusionGrid({ rows, classes }: { rows: Row[]; classes: string[] }) {
  const withCm = rows.filter((r) => r.meta.heldOutConfusion)
  if (!withCm.length) return null
  return (
    <div className="card mt-4 p-4">
      <h3 className="text-[15px] font-semibold text-zinc-100">Where each model goes wrong</h3>
      <p className="mt-1 max-w-3xl text-xs leading-relaxed text-zinc-400">
        Each row is what the road really was; each column is what the model said. A bright diagonal is good. Bright cells off the diagonal are the mistakes to worry about, especially in the <b className="text-zinc-200">P</b> (pothole) and <b className="text-zinc-200">B</b> (speed bump) rows, because those are missed hazards.
      </p>
      <div className="mt-4 grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        {withCm.map((r) => {
          const cm = r.meta.heldOutConfusion!
          const norm = cm.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => v / s) })
          let worst = { t: 0, p: 0, v: 0 }
          norm.forEach((row, t) => row.forEach((v, p) => { if (t !== p && v > worst.v) worst = { t, p, v } }))
          const color = colorFor(r.meta.id)
          return (
            <figure key={r.meta.id}>
              <figcaption className="mb-1.5 flex items-center gap-1.5 text-xs text-zinc-200">
                <i className="h-2 w-2 rounded-full" style={{ background: color }} />{r.meta.name}
                {r.meta.custom && <span className="text-[10px] text-zinc-500">(20 fixtures)</span>}
              </figcaption>
              <div className="grid grid-cols-[auto_repeat(4,1fr)] gap-[3px] font-mono text-[10px]">
                <span />
                {SHORT.map((s) => <span key={s} className="text-center text-zinc-600">{s}</span>)}
                {norm.map((row, t) => (
                  <div key={t} className="contents">
                    <span className="pr-1 text-zinc-600">{SHORT[t]}</span>
                    {row.map((v, p) => (
                      <span
                        key={p}
                        title={`True ${CLASS_LABEL[classes[t]]}, predicted ${CLASS_LABEL[classes[p]]}: ${(v * 100).toFixed(1)}%`}
                        className="grid h-7 place-items-center rounded-[3px] text-zinc-100"
                        style={{ background: rgba(t === p ? color : '#f43f5e', 0.1 + 0.9 * v) }}
                      >
                        {v >= 0.005 ? Math.round(v * 100) : ''}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
              <p className="mt-1.5 text-[11px] text-zinc-500">
                {worst.v < 0.005 ? 'No systematic mistake.' : <>Biggest mistake: calls {(worst.v * 100).toFixed(0)}% of <span className="text-zinc-300">{CLASS_LABEL[classes[worst.t]].toLowerCase()}</span> windows <span className="text-zinc-300">{CLASS_LABEL[classes[worst.p]].toLowerCase()}</span>.</>}
              </p>
            </figure>
          )
        })}
      </div>
    </div>
  )
}
